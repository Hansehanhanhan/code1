from __future__ import annotations

from langchain_core.documents import Document

from rag.knowledge_base import (
    _apply_metadata_filter,
    _bm25_search_with_index,
    _extract_front_matter,
    _fuse_ranked_candidates,
    _has_filter_context,
    _build_sparse_index,
    _rerank_documents,
    _split_documents,
    retrieve_knowledge,
)
from backend.settings import Settings


def test_extract_front_matter() -> None:
    content = """---
merchant_id: demo-001
category: retail
tags: ads,sop
---
这是正文内容。"""
    metadata, body = _extract_front_matter(content)
    assert metadata["merchant_id"] == "demo-001"
    assert metadata["category"] == "retail"
    assert metadata["tags"] == ["ads", "sop"]
    assert body == "这是正文内容。"


def test_metadata_filter_prefers_matching_docs() -> None:
    docs = [
        Document(page_content="A", metadata={"source": "a.md", "category": "retail"}),
        Document(page_content="B", metadata={"source": "b.md", "category": "fashion"}),
        Document(page_content="C", metadata={"source": "c.md"}),
    ]
    kept, removed = _apply_metadata_filter(docs, {"category": "retail"})
    assert removed == 1
    assert [doc.metadata.get("source") for doc in kept] == ["a.md", "c.md"]


def test_rerank_documents_prefers_query_overlap() -> None:
    candidates = [
        Document(page_content="库存周转正常", metadata={"source": "inventory.md"}),
        Document(page_content="广告ROI下降，需要优化投放", metadata={"source": "ads.md"}),
    ]
    reranked = _rerank_documents("广告ROI怎么优化", {"category": "retail"}, candidates, top_k=1)
    assert len(reranked) == 1
    assert reranked[0].metadata.get("source") == "ads.md"


def test_bm25_search_prefers_query_match() -> None:
    docs = [
        Document(page_content="库存健康，周转正常", metadata={"source": "inventory.md"}),
        Document(page_content="广告ROI下降，需要优化关键词和出价", metadata={"source": "ads.md"}),
    ]
    tf_list, df, avgdl = _build_sparse_index(docs)
    ranked = _bm25_search_with_index("广告ROI怎么优化", {"category": "retail"}, docs, tf_list, df, avgdl, top_k=1)
    assert len(ranked) == 1
    assert ranked[0].metadata.get("source") == "ads.md"


def test_hybrid_fusion_keeps_vector_and_bm25_signals() -> None:
    vector_candidates = [
        Document(page_content="流量分析结果", metadata={"source": "traffic.md"}),
    ]
    bm25_candidates = [
        Document(page_content="广告优化SOP", metadata={"source": "ads.md"}),
    ]
    fused = _fuse_ranked_candidates(vector_candidates, bm25_candidates, top_k=2)
    sources = [doc.metadata.get("source") for doc in fused]
    assert "traffic.md" in sources
    assert "ads.md" in sources


def test_split_documents_supports_chinese_punctuation() -> None:
    docs = [
        Document(
            page_content="第一段。第二段！第三段？第四段；第五段，继续说明。" * 40,
            metadata={"source": "zh.md"},
        )
    ]
    chunks = _split_documents(docs)
    assert len(chunks) > 1


def test_has_filter_context() -> None:
    assert _has_filter_context({"merchant_id": "demo-001"}) is True
    assert _has_filter_context({"foo": "bar"}) is False


def test_retrieve_knowledge_respects_strict_metadata_filter(monkeypatch) -> None:
    class FakeStore:
        def similarity_search(self, retrieval_query: str, k: int) -> list[Document]:
            del retrieval_query, k
            return [
                Document(page_content="广告策略文档", metadata={"source": "ads.md", "category": "fashion"}),
                Document(page_content="库存策略文档", metadata={"source": "inventory.md", "category": "fashion"}),
            ]

    def fake_get_or_build_store(settings: Settings):
        del settings
        return FakeStore(), "in_memory", "local_open_source", 2

    monkeypatch.setattr("rag.knowledge_base._get_or_build_store", fake_get_or_build_store)

    settings = Settings(
        openai_api_key="k",
        openai_base_url="u",
        openai_model="m",
        allow_rule_fallback=True,
        rag_enabled=True,
        rag_docs_dir="knowledge/seed",
        rag_vector_backend="in_memory",
        rag_top_k=3,
        rag_fetch_k=12,
        rag_embedding_model="BAAI/bge-small-zh-v1.5",
        rag_embedding_device="cpu",
        session_backend="memory",
        session_ttl_seconds=3600,
        redis_url=None,
        rate_limit_enabled=True,
        rate_limit_window_seconds=60,
        rate_limit_max_requests=30,
        rate_limit_max_requests_run=20,
        rate_limit_max_requests_stream=10,
        rate_limit_max_requests_ip=60,
        trust_x_forwarded_for=False,
        trusted_proxy_ips=[],
        request_timeout_seconds=120,
        request_timeout_seconds_stream=150,
        run_retry_attempts=1,
        retry_backoff_ms=300,
        degrade_on_timeout=True,
        degrade_on_error=True,
        app_auth_enabled=False,
        app_api_key=None,
        max_query_chars=2000,
        max_context_chars=8000,
        prompt_injection_guard_enabled=True,
        app_cors_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        app_cors_allow_credentials=False,
        agent_verbose=False,
    )

    out = retrieve_knowledge("广告优化", {"category": "retail"}, settings)
    assert out["status"] == "ok"
    assert out["data"]["retrieval_mode"] == "hybrid_rrf"
    assert out["data"]["metadata_filtered_out"] == 2
    assert out["data"]["matches"] == []
