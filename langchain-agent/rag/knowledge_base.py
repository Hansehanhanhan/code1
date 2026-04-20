from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.settings import Settings

# 项目根目录，用于解析相对路径配置。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Chroma 默认持久化目录。
DEFAULT_VECTOR_DIR = PROJECT_ROOT / "knowledge" / ".chroma"
SUPPORTED_SUFFIXES = {".md", ".txt"}
# 允许用于检索过滤的 metadata 字段。
METADATA_FILTER_KEYS = ("merchant_id", "category", "time_range")

# 下面这组全局缓存用于复用索引与切片，避免每次请求都重建。
_kb_lock = RLock()
_cached_store: Any | None = None
_cached_fingerprint = ""
_cached_vector_backend = ""
_cached_embedding_backend = ""
_cached_chunk_count = 0
_cached_chunks: list[Document] = []
_cached_sparse_tf: list[Counter[str]] = []
_cached_sparse_df: dict[str, int] = {}
_cached_sparse_avgdl: float = 0.0
# RRF 融合常数，值越大，排名差异的影响越平滑。
RRF_K = 60


def _resolve_docs_dir(raw_path: str) -> Path:
    """把配置中的知识库目录解析为绝对路径。"""
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _normalize_text(text: str) -> str:
    """压缩多余空白，方便输出短摘要。"""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_value_for_match(value: Any) -> str:
    """统一 metadata 比较口径（去空白 + 小写）。"""
    return str(value).strip().lower()


def _extract_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """从 Markdown front matter 中提取 metadata，并返回正文。"""
    stripped = content.lstrip()
    if not stripped.startswith("---\n"):
        return {}, content

    lines = stripped.splitlines()
    if len(lines) < 3:
        return {}, content

    end_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break
    if end_index is None:
        return {}, content

    metadata: dict[str, Any] = {}
    for line in lines[1:end_index]:
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key or not value:
            continue
        if "," in value:
            metadata[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            metadata[key] = value

    body = "\n".join(lines[end_index + 1 :]).strip()
    return metadata, body


class SentenceTransformerEmbeddings(Embeddings):
    """Local open-source embedding via sentence-transformers."""

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
    """优先按 UTF-8 读取；遇到编码异常时忽略非法字符兜底。"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _load_documents(docs_dir: Path) -> list[Document]:
    """加载知识库文档，并把 front matter 合并进 metadata。"""
    if not docs_dir.exists() or not docs_dir.is_dir():
        return []

    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        raw_content = _read_file(path)
        metadata_extra, body = _extract_front_matter(raw_content)
        content = body.strip()
        if not content:
            continue

        source = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        metadata = {
            "source": source,
            "topic": path.stem,
            **metadata_extra,
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def _split_documents(docs: list[Document]) -> list[Document]:
    """中文友好切片：固定长度 + overlap，分隔符优先按段落/句子。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            ".",
            ";",
            ",",
            " ",
            "",
        ],
    )
    return splitter.split_documents(docs)


def _fingerprint(paths: list[Path]) -> str:
    """计算文件指纹，用于判断缓存是否失效。"""
    digest = hashlib.md5()
    for path in paths:
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _build_embeddings(settings: Settings) -> tuple[Embeddings, str]:
    """构建本地 embedding 模型；失败时直接抛错，不做 hash 回退。"""
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
    """优先使用 Chroma；失败时回退内存向量库。"""
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
            pass

    store = InMemoryVectorStore(embeddings)
    store.add_documents(docs)
    return store, "in_memory"


def _get_or_build_store(settings: Settings) -> tuple[Any | None, str, str, int]:
    """返回可复用的向量索引与稀疏索引，必要时重建并写入缓存。"""
    global _cached_store
    global _cached_fingerprint
    global _cached_vector_backend
    global _cached_embedding_backend
    global _cached_chunk_count
    global _cached_chunks
    global _cached_sparse_tf
    global _cached_sparse_df
    global _cached_sparse_avgdl

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

    fp = "|".join(
        [
            _fingerprint(files),
            settings.rag_vector_backend,
            settings.rag_embedding_model,
            settings.rag_embedding_device,
            str(settings.rag_top_k),
            str(settings.rag_fetch_k),
        ]
    )
    with _kb_lock:
        if _cached_store is not None and _cached_fingerprint == fp:
            return _cached_store, _cached_vector_backend, _cached_embedding_backend, _cached_chunk_count

        raw_docs = _load_documents(docs_dir)
        chunks = _split_documents(raw_docs)
        if not chunks:
            return None, "none", "none", 0

        embeddings, embedding_backend = _build_embeddings(settings)
        store, vector_backend = _build_vector_store(chunks, fp, settings, embeddings)
        sparse_tf, sparse_df, sparse_avgdl = _build_sparse_index(chunks)
        _cached_store = store
        _cached_fingerprint = fp
        _cached_vector_backend = vector_backend
        _cached_embedding_backend = embedding_backend
        _cached_chunk_count = len(chunks)
        _cached_chunks = list(chunks)
        _cached_sparse_tf = sparse_tf
        _cached_sparse_df = sparse_df
        _cached_sparse_avgdl = sparse_avgdl
        return _cached_store, _cached_vector_backend, _cached_embedding_backend, _cached_chunk_count


def _tokenize_for_rerank(text: str) -> set[str]:
    """重排分词：英文 token + 中文单字 + 中文双字。"""
    lowered = text.lower()
    en_tokens = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    zh_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    zh_unigram = set(zh_chars)
    zh_bigram = {"".join(zh_chars[idx : idx + 2]) for idx in range(len(zh_chars) - 1)}
    return en_tokens | zh_unigram | zh_bigram


def _tokenize_for_sparse(text: str) -> list[str]:
    """稀疏检索分词：用于 BM25（保留词频信息）。"""
    lowered = text.lower()
    en_tokens = re.findall(r"[a-z0-9_]{2,}", lowered)
    zh_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    zh_bigrams = ["".join(zh_chars[idx : idx + 2]) for idx in range(len(zh_chars) - 1)]
    return en_tokens + zh_chars + zh_bigrams


def _document_to_sparse_text(doc: Document) -> str:
    """把正文和 metadata 拼接，供 BM25 统一建索引。"""
    return f"{doc.page_content}\n{json.dumps(doc.metadata, ensure_ascii=False)}"


def _build_sparse_index(docs: list[Document]) -> tuple[list[Counter[str]], dict[str, int], float]:
    """构建 BM25 所需统计量：每文档 tf、全局 df、平均文档长度。"""
    tf_list: list[Counter[str]] = []
    df: Counter[str] = Counter()
    total_len = 0

    for doc in docs:
        tokens = _tokenize_for_sparse(_document_to_sparse_text(doc))
        tf = Counter(tokens)
        tf_list.append(tf)
        total_len += len(tokens)
        for term in tf.keys():
            df[term] += 1

    avgdl = (total_len / len(docs)) if docs else 0.0
    return tf_list, dict(df), avgdl


def _bm25_search_with_index(
    query: str,
    context: dict[str, Any],
    docs: list[Document],
    tf_list: list[Counter[str]],
    df: dict[str, int],
    avgdl: float,
    top_k: int,
) -> list[Document]:
    """基于已构建索引执行 BM25 排序，返回 top_k 候选。"""
    if not docs or not tf_list:
        return []

    query_text = f"{query}\n{json.dumps(context, ensure_ascii=False)}"
    query_terms = Counter(_tokenize_for_sparse(query_text))
    if not query_terms:
        return docs[:top_k]

    n_docs = len(docs)
    k1 = 1.5
    b = 0.75
    safe_avgdl = max(1.0, avgdl)
    scored: list[tuple[float, int, Document]] = []

    for idx, doc in enumerate(docs):
        tf = tf_list[idx] if idx < len(tf_list) else Counter(_tokenize_for_sparse(_document_to_sparse_text(doc)))
        dl = max(1, sum(tf.values()))
        score = 0.0
        for term, qf in query_terms.items():
            f = tf.get(term, 0)
            if f <= 0:
                continue
            n_t = df.get(term, 0)
            idf = math.log(1.0 + ((n_docs - n_t + 0.5) / (n_t + 0.5)))
            denom = f + (k1 * (1.0 - b + b * (dl / safe_avgdl)))
            score += qf * (idf * ((f * (k1 + 1.0)) / max(1e-9, denom)))
        scored.append((score, idx, doc))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:top_k]]


def _doc_key(doc: Document) -> str:
    """生成文档去重键：source + 内容哈希。"""
    source = str(doc.metadata.get("source", "unknown"))
    snippet_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{source}|{snippet_hash}"


def _fuse_ranked_candidates(
    vector_candidates: list[Document],
    bm25_candidates: list[Document],
    *,
    top_k: int,
) -> list[Document]:
    """用 RRF 融合向量召回与 BM25 召回结果。"""
    fused: dict[str, dict[str, Any]] = {}

    for rank, doc in enumerate(vector_candidates, start=1):
        key = _doc_key(doc)
        row = fused.setdefault(key, {"doc": doc, "score": 0.0, "best_rank": rank})
        row["score"] += 1.0 / (RRF_K + rank)
        row["best_rank"] = min(int(row["best_rank"]), rank)

    for rank, doc in enumerate(bm25_candidates, start=1):
        key = _doc_key(doc)
        row = fused.setdefault(key, {"doc": doc, "score": 0.0, "best_rank": rank})
        row["score"] += 1.0 / (RRF_K + rank)
        row["best_rank"] = min(int(row["best_rank"]), rank)

    ranked = sorted(fused.values(), key=lambda row: (-float(row["score"]), int(row["best_rank"])))
    return [row["doc"] for row in ranked[:top_k]]


def _metadata_matches_context(doc: Document, context: dict[str, Any]) -> bool:
    """metadata 匹配规则：仅在文档存在该字段时做严格匹配。"""
    for key in METADATA_FILTER_KEYS:
        if key not in context:
            continue
        ctx_value = context.get(key)
        meta_value = doc.metadata.get(key)
        if meta_value is None:
            continue

        target = _normalize_value_for_match(ctx_value)
        if isinstance(meta_value, list):
            normalized = {_normalize_value_for_match(item) for item in meta_value}
            if target not in normalized:
                return False
        else:
            if _normalize_value_for_match(meta_value) != target:
                return False
    return True


def _apply_metadata_filter(candidates: list[Document], context: dict[str, Any]) -> tuple[list[Document], int]:
    """应用 metadata 过滤，并返回被过滤数量。"""
    filtered = [doc for doc in candidates if _metadata_matches_context(doc, context)]
    removed_count = len(candidates) - len(filtered)
    return filtered, removed_count


def _has_filter_context(context: dict[str, Any]) -> bool:
    """判断请求上下文是否携带了可过滤字段。"""
    return any(key in context for key in METADATA_FILTER_KEYS)


def _rerank_documents(query: str, context: dict[str, Any], candidates: list[Document], top_k: int) -> list[Document]:
    """轻量重排：词重叠得分 + metadata 命中加分。"""
    if not candidates:
        return []

    query_tokens = _tokenize_for_rerank(query)
    context_tokens = _tokenize_for_rerank(json.dumps(context, ensure_ascii=False))

    scored: list[tuple[float, int, Document]] = []
    for idx, doc in enumerate(candidates):
        doc_text = f"{doc.page_content} {json.dumps(doc.metadata, ensure_ascii=False)}"
        doc_tokens = _tokenize_for_rerank(doc_text)

        q_overlap = len(query_tokens & doc_tokens)
        c_overlap = len(context_tokens & doc_tokens)

        meta_boost = 0.0
        for key in METADATA_FILTER_KEYS:
            if key in context and key in doc.metadata and _metadata_matches_context(doc, context):
                meta_boost += 1.5

        score = (2.0 * q_overlap) + (1.0 * c_overlap) + meta_boost - (idx * 0.01)
        scored.append((score, idx, doc))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:top_k]]


def retrieve_knowledge(query: str, context: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """RAG 主入口：双路召回（向量+BM25）-> RRF 融合 -> 过滤 -> 重排 -> 返回片段。"""

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

        top_k = max(1, settings.rag_top_k)
        fetch_k = max(top_k, settings.rag_fetch_k)

        # 向量召回：先取较大的候选池，后续再融合/重排截断。
        retrieval_query = f"{query}\ncontext={json.dumps(context, ensure_ascii=False)}"
        vector_candidates = store.similarity_search(retrieval_query, k=fetch_k)

        with _kb_lock:
            sparse_pool = list(_cached_chunks) if _cached_chunks else list(vector_candidates)
            sparse_tf = list(_cached_sparse_tf)
            sparse_df = dict(_cached_sparse_df)
            sparse_avgdl = _cached_sparse_avgdl

        if sparse_pool and len(sparse_tf) == len(sparse_pool) and sparse_df:
            # 稀疏召回：优先复用缓存统计量，减少重复计算。
            bm25_candidates = _bm25_search_with_index(
                query,
                context,
                sparse_pool,
                sparse_tf,
                sparse_df,
                sparse_avgdl,
                fetch_k,
            )
        else:
            # 兜底：缓存不完整时现场重建稀疏索引。
            fallback_tf, fallback_df, fallback_avgdl = _build_sparse_index(sparse_pool)
            bm25_candidates = _bm25_search_with_index(
                query,
                context,
                sparse_pool,
                fallback_tf,
                fallback_df,
                fallback_avgdl,
                fetch_k,
            )

        candidates = _fuse_ranked_candidates(vector_candidates, bm25_candidates, top_k=fetch_k)

        # 若请求带过滤条件，则优先在过滤后的集合重排。
        metadata_filtered, filtered_out = _apply_metadata_filter(candidates, context)
        rerank_input = metadata_filtered if _has_filter_context(context) else candidates
        reranked = _rerank_documents(query, context, rerank_input, top_k)

        snippets: list[dict[str, Any]] = []
        for idx, doc in enumerate(reranked, start=1):
            source = str(doc.metadata.get("source", "unknown"))
            snippet = _normalize_text(doc.page_content)
            snippets.append(
                {
                    "rank": idx,
                    "source": source,
                    "snippet": snippet[:260],
                    "metadata": {
                        key: doc.metadata.get(key)
                        for key in METADATA_FILTER_KEYS
                        if doc.metadata.get(key) is not None
                    },
                }
            )

        return {
            "tool": "retrieve_knowledge",
            "status": "ok",
            "summary": (
                f"Hybrid retrieved {len(snippets)} snippets "
                f"(vector={len(vector_candidates)}, bm25={len(bm25_candidates)}, fused={len(candidates)}). "
                f"(backend={vector_backend}, embedding={embedding_backend}, fetch_k={fetch_k}, top_k={top_k}, metadata_filtered_out={filtered_out})"
            ),
            "data": {
                "retrieval_mode": "hybrid_rrf",
                "vector_backend": vector_backend,
                "embedding_backend": embedding_backend,
                "embedding_model": settings.rag_embedding_model,
                "indexed_chunks": chunk_count,
                "top_k": top_k,
                "fetch_k": fetch_k,
                "vector_candidates_count": len(vector_candidates),
                "bm25_candidates_count": len(bm25_candidates),
                "fused_candidates_count": len(candidates),
                "metadata_filter_keys": list(METADATA_FILTER_KEYS),
                "metadata_filtered_out": filtered_out,
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
