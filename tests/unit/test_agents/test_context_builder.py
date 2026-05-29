"""
tests/unit/test_agents/test_context_builder.py
Unit tests for ContextBuilder and Context Engineering modules.
"""
import pytest
from src.agents.context.builder import ContextBuilder, AgentContext
from src.agents.context.compressor import ContextCompressor
from src.agents.context.reranker import DocumentReranker


class TestContextBuilder:

    def test_analyst_context_structure(self):
        builder = ContextBuilder(agent_type="analyst")
        context = builder.build_analyst_context(
            metrics={"rmse": 0.18, "consecutive_periods": 6},
            model_config={"feedback_error_threshold": 0.15, "feedback_consecutive_periods": 5},
            rag_documents=["Document about RMSE optimization"],
            anomaly_history=[{"timestamp": "2024-01-01", "anomaly_score": 0.9}],
            previous_suggestions=[],
        )
        assert isinstance(context, AgentContext)
        assert context.critical_data != ""
        assert context.task != ""
        assert context.token_estimate > 0

    def test_lost_in_middle_ordering(self):
        """Critical data must appear at start and end, not just middle."""
        builder = ContextBuilder(agent_type="analyst")
        context = builder.build_analyst_context(
            metrics={"rmse": 0.20},
            model_config={"feedback_error_threshold": 0.15, "feedback_consecutive_periods": 5},
            rag_documents=[],
            anomaly_history=[],
            previous_suggestions=[],
        )
        full_context = context.build()
        # Critical data (metrics) should appear at the start
        assert full_context.startswith("CURRENT MODEL METRICS")

    def test_researcher_context_high_confidence(self):
        builder = ContextBuilder(agent_type="researcher")
        context = builder.build_researcher_context(
            query="high RMSE causes",
            rag_documents=["doc1", "doc2"],
            confidence_scores=[0.9, 0.85],
        )
        assert "RAG documents (high confidence)" in context.background

    def test_researcher_context_low_confidence(self):
        builder = ContextBuilder(agent_type="researcher")
        context = builder.build_researcher_context(
            query="obscure query",
            rag_documents=["doc1"],
            confidence_scores=[0.2],
        )
        assert "Web results only" in context.background

    def test_unsupported_agent_type(self):
        with pytest.raises(ValueError):
            ContextBuilder(agent_type="unknown_agent")

    def test_reporter_context_build(self):
        builder = ContextBuilder(agent_type="reporter")
        context = builder.build_reporter_context(
            analysis_result={"suggestions": [{"hyperparameter": "lr"}]},
            research_result={"synthesis": "test"},
            anomaly_data={"score": 0.95},
            report_type="anomaly_analysis",
        )
        full = context.build()
        assert "REPORT TYPE: anomaly_analysis" in full
        assert "Do NOT invent data" in full


class TestContextCompressor:

    def test_no_compression_needed(self):
        compressor = ContextCompressor(max_tokens=10000)
        messages = [{"role": "user", "content": "short message"}]
        result = compressor.compress(messages)
        assert result == messages

    def test_compression_preserves_recent(self):
        compressor = ContextCompressor(max_tokens=100, keep_last=2)
        messages = [
            {"role": "user", "content": "old message " * 20},
            {"role": "assistant", "content": "old response " * 20},
            {"role": "user", "content": "recent message 1"},
            {"role": "user", "content": "recent message 2"},
        ]
        result = compressor.compress(messages)
        # Recent messages should be preserved
        contents = [m["content"] for m in result]
        assert "recent message 1" in contents
        assert "recent message 2" in contents

    def test_compress_rag_documents(self):
        compressor = ContextCompressor()
        docs = ["a" * 1000, "b" * 1000, "c" * 1000, "d" * 1000]
        result = compressor.compress_rag_documents(docs, max_docs=3, max_chars_per_doc=100)
        assert len(result) == 3
        assert all(len(d) <= 120 for d in result)

    def test_extract_critical_data(self):
        compressor = ContextCompressor()
        messages = [
            {"role": "user", "content": "normal message"},
            {"role": "assistant", "content": "rmse exceeded threshold approved"},
            {"role": "user", "content": "another normal message"},
        ]
        critical = compressor.extract_critical_data(messages)
        assert len(critical) == 1
        assert "rmse" in critical[0]["content"]


class TestDocumentReranker:

    def test_rerank_returns_sorted(self):
        reranker = DocumentReranker()
        ranked = reranker.rerank(
            query="rmse anomaly telecom",
            documents=["rmse high anomaly in telecom network", "unrelated document"],
        )
        assert len(ranked) == 2
        assert ranked[0].relevance_score >= ranked[1].relevance_score

    def test_crag_high_confidence(self):
        reranker = DocumentReranker()
        from src.agents.context.reranker import RankedDocument
        ranked = [
            RankedDocument(content="doc", relevance_score=0.9, source="d1", confidence_level="high"),
            RankedDocument(content="doc2", relevance_score=0.85, source="d2", confidence_level="high"),
        ]
        strategy = reranker.get_crag_strategy(ranked)
        assert strategy["strategy"] == "rag_only"
        assert strategy["use_rag"] is True
        assert strategy["use_web"] is False

    def test_crag_low_confidence(self):
        reranker = DocumentReranker()
        from src.agents.context.reranker import RankedDocument
        ranked = [
            RankedDocument(content="doc", relevance_score=0.2, source="d1", confidence_level="low"),
        ]
        strategy = reranker.get_crag_strategy(ranked)
        assert strategy["strategy"] == "web_only"
        assert strategy["use_rag"] is False
        assert strategy["use_web"] is True

    def test_crag_empty_docs(self):
        reranker = DocumentReranker()
        strategy = reranker.get_crag_strategy([])
        assert strategy["strategy"] == "no_docs"
        assert strategy["confidence"] == 0.0

    def test_filter_by_confidence(self):
        reranker = DocumentReranker()
        from src.agents.context.reranker import RankedDocument
        ranked = [
            RankedDocument(content="good", relevance_score=0.8, source="d1", confidence_level="high"),
            RankedDocument(content="bad", relevance_score=0.1, source="d2", confidence_level="low"),
        ]
        filtered = reranker.filter_by_confidence(ranked, min_score=0.5)
        assert len(filtered) == 1
        assert filtered[0].content == "good"
