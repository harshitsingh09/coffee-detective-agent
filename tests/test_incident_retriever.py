import json
import tempfile
import unittest
from pathlib import Path

from incident_assistant.infrastructure.incident_retriever import (
    LexicalHistoricalIncidentRetriever,
    PersistentSemanticIncidentRetriever,
    ResilientHistoricalIncidentRetriever,
)
from incident_assistant.infrastructure.sqlite_repository import SqliteRepository
from seed_data import seed


class KeywordEmbeddingProvider:
    model_id = "test-keyword-embeddings-v1"
    features = ("beans", "milk", "overheat", "cleaning", "healthy")

    def __init__(self) -> None:
        self.document_embedding_calls = 0

    def _embed(self, text: str):
        normalized = text.casefold()
        return tuple(float(feature in normalized) for feature in self.features)

    def embed_documents(self, texts):
        self.document_embedding_calls += 1
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text: str):
        return self._embed(text)


class BrokenEmbeddingProvider(KeywordEmbeddingProvider):
    def embed_query(self, text: str):
        raise RuntimeError("embedding provider unavailable")


class HistoricalIncidentRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._temporary_directory.name)
        database_path, _, _ = seed(self.data_dir, brews_per_machine=50)
        self.repository = SqliteRepository(database_path)
        self.index_path = self.data_dir / "incident_embeddings.json"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_persists_index_and_ranks_semantically_matching_incident(self) -> None:
        provider = KeywordEmbeddingProvider()
        retriever = PersistentSemanticIncidentRetriever(
            self.repository,
            provider,
            self.index_path,
        )
        self.assertEqual(retriever.rebuild_index(), 5)
        self.assertTrue(self.index_path.is_file())
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.assertEqual(index["embedding_dimension"], len(provider.features))
        self.assertEqual(index["version"], 2)

        matches = retriever.search("boiler overheat safety warning", top_k=1)
        self.assertEqual(matches[0].incident_id, "CAF-0003")
        self.assertEqual(matches[0].retrieval_method, "semantic")

    def test_filters_matches_below_configured_similarity_threshold(self) -> None:
        provider = KeywordEmbeddingProvider()
        retriever = PersistentSemanticIncidentRetriever(
            self.repository,
            provider,
            self.index_path,
            similarity_threshold=0.8,
        )
        retriever.rebuild_index()

        matches = retriever.search("words with no configured keyword", top_k=3)

        self.assertEqual(matches, ())

    def test_reuses_persisted_document_vectors_without_reembedding_corpus(self) -> None:
        builder_provider = KeywordEmbeddingProvider()
        PersistentSemanticIncidentRetriever(
            self.repository,
            builder_provider,
            self.index_path,
        ).rebuild_index()
        query_provider = KeywordEmbeddingProvider()
        retriever = PersistentSemanticIncidentRetriever(
            self.repository,
            query_provider,
            self.index_path,
        )

        retriever.search("milk foam failed", top_k=1)
        self.assertEqual(query_provider.document_embedding_calls, 0)

    def test_rejects_index_built_with_a_different_model(self) -> None:
        provider = KeywordEmbeddingProvider()
        PersistentSemanticIncidentRetriever(
            self.repository,
            provider,
            self.index_path,
        ).rebuild_index()
        provider.model_id = "different-model"
        retriever = PersistentSemanticIncidentRetriever(
            self.repository,
            provider,
            self.index_path,
        )
        with self.assertRaises(ValueError):
            retriever.search("boiler overheat", top_k=1)

    def test_falls_back_to_lexical_retrieval_when_semantics_fail(self) -> None:
        semantic = PersistentSemanticIncidentRetriever(
            self.repository,
            BrokenEmbeddingProvider(),
            self.index_path,
        )
        fallback = LexicalHistoricalIncidentRetriever(self.repository)
        retriever = ResilientHistoricalIncidentRetriever(semantic, fallback)

        matches = retriever.search("boiler overheating", top_k=1)
        self.assertEqual(matches[0].incident_id, "CAF-0003")
        self.assertEqual(matches[0].retrieval_method, "lexical_fallback")


if __name__ == "__main__":
    unittest.main()
