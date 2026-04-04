from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VECTOR_DIR = PROJECT_ROOT / "knowledge" / ".chroma"
SUPPORTED_SUFFIXES = {".md", ".txt"}

# 进程级缓存：避免每次请求都重建索引。
_kb_lock = RLock()
_cached_store: Any | None = None
_cached_fingerprint = ""
_cached_vector_backend = ""
_cached_embedding_backend = ""
_cached_chunk_count = 0


def _resolve_docs_dir(raw_path: str) -> Path:
    # 允许绝对路径和相对项目根目录路径两种写法。
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class SentenceTransformerEmbeddings(Embeddings):
    """本地开源 embedding：基于 sentence-transformers。"""

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name_or_path=model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return vector.tolist()


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _load_documents(docs_dir: Path) -> list[Document]:
    if not docs_dir.exists() or not docs_dir.is_dir():
        return []

    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        content = _normalize_text(_read_file(path))
        if not content:
            continue
        # 保留来源路径，便于前端展示与结果溯源。
        source = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        docs.append(Document(page_content=content, metadata={"source": source}))
    return docs


def _split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    return splitter.split_documents(docs)


def _fingerprint(paths: list[Path]) -> str:
    digest = hashlib.md5()
    for path in paths:
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _build_embeddings(settings: Settings) -> tuple[Embeddings, str]:
    # 仅支持本地开源 embedding；加载失败直接报错。
    try:
        embeddings = SentenceTransformerEmbeddings(
            model_name=settings.rag_embedding_model,
            device=settings.rag_embedding_device,
        )
        return embeddings, "local_open_source"
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load local embedding model. "
            "Please install sentence-transformers and check model/device settings."
        ) from exc


def _build_vector_store(
    docs: list[Document],
    fingerprint: str,
    settings: Settings,
    embeddings: Embeddings,
) -> tuple[Any, str]:
    if settings.rag_vector_backend == "chroma":
        try:
            from langchain_community.vectorstores import Chroma

            DEFAULT_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
            collection_name = f"merchant_ops_{fingerprint[:10]}"
            store = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=str(DEFAULT_VECTOR_DIR),
                collection_name=collection_name,
            )
            return store, "chroma"
        except Exception:
            # 如果 Chroma 不可用或配置异常，回退到内存向量库。
            pass

    store = InMemoryVectorStore(embeddings)
    store.add_documents(docs)
    return store, "in_memory"


def _get_or_build_store(settings: Settings) -> tuple[Any | None, str, str, int]:
    global _cached_store
    global _cached_fingerprint
    global _cached_vector_backend
    global _cached_embedding_backend
    global _cached_chunk_count

    docs_dir = _resolve_docs_dir(settings.rag_docs_dir)
    files = (
        sorted(
            [
                path
                for path in docs_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            ]
        )
        if docs_dir.exists()
        else []
    )

    if not files:
        return None, "none", "none", 0

    # 文档内容或检索配置变化时，让缓存失效并触发重建。
    fp = "|".join(
        [
            _fingerprint(files),
            settings.rag_vector_backend,
            settings.rag_embedding_model,
            settings.rag_embedding_device,
        ]
    )
    with _kb_lock:
        if _cached_store is not None and _cached_fingerprint == fp:
            # 热路径：命中缓存，直接复用已构建索引。
            return _cached_store, _cached_vector_backend, _cached_embedding_backend, _cached_chunk_count

        raw_docs = _load_documents(docs_dir)
        chunks = _split_documents(raw_docs)
        if not chunks:
            return None, "none", "none", 0

        # 冷路径：重建 embedding 与向量库，然后写入缓存。
        embeddings, embedding_backend = _build_embeddings(settings)
        store, vector_backend = _build_vector_store(chunks, fp, settings, embeddings)
        _cached_store = store
        _cached_fingerprint = fp
        _cached_vector_backend = vector_backend
        _cached_embedding_backend = embedding_backend
        _cached_chunk_count = len(chunks)
        return _cached_store, _cached_vector_backend, _cached_embedding_backend, _cached_chunk_count


def retrieve_knowledge(query: str, context: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """从本地知识库检索相关片段。"""

    if not settings.rag_enabled:
        return {
            "tool": "retrieve_knowledge",
            "status": "disabled",
            "summary": "RAG is disabled by settings.",
            "data": {},
        }

    if not query.strip():
        return {
            "tool": "retrieve_knowledge",
            "status": "error",
            "summary": "Query is empty.",
            "data": {},
        }

    try:
        store, vector_backend, embedding_backend, chunk_count = _get_or_build_store(settings)
        if store is None:
            return {
                "tool": "retrieve_knowledge",
                "status": "empty",
                "summary": "Knowledge base is empty. Add files under RAG_DOCS_DIR.",
                "data": {"docs_dir": settings.rag_docs_dir},
            }

        retrieval_query = f"{query}\ncontext={json.dumps(context, ensure_ascii=False)}"
        top_k = max(1, settings.rag_top_k)
        matches = store.similarity_search(retrieval_query, k=top_k)

        # 返回紧凑结构，便于前端流式展示和步骤可视化。
        snippets: list[dict[str, Any]] = []
        for idx, doc in enumerate(matches, start=1):
            source = str(doc.metadata.get("source", "unknown"))
            snippet = _normalize_text(doc.page_content)
            snippets.append(
                {
                    "rank": idx,
                    "source": source,
                    "snippet": snippet[:260],
                }
            )

        return {
            "tool": "retrieve_knowledge",
            "status": "ok",
            "summary": f"Retrieved {len(snippets)} snippets from {vector_backend} store using {embedding_backend}.",
            "data": {
                "vector_backend": vector_backend,
                "embedding_backend": embedding_backend,
                "embedding_model": settings.rag_embedding_model,
                "indexed_chunks": chunk_count,
                "top_k": top_k,
                "matches": snippets,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "tool": "retrieve_knowledge",
            "status": "error",
            "summary": "Knowledge retrieval failed.",
            "data": {"error": str(exc)},
        }
