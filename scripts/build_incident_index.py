"""Build the persisted local embedding index for historical incidents."""

from incident_assistant.bootstrap import ensure_demo_data
from incident_assistant.config import Settings
from incident_assistant.infrastructure.incident_retriever import (
    PersistentSemanticIncidentRetriever,
    SentenceTransformerEmbeddingProvider,
)
from incident_assistant.infrastructure.sqlite_repository import SqliteRepository


def main() -> None:
    settings = Settings.from_environment()
    ensure_demo_data(settings)
    repository = SqliteRepository(settings.database_path)
    provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model,
        settings.embedding_cache_path,
    )
    retriever = PersistentSemanticIncidentRetriever(
        repository,
        provider,
        settings.embedding_index_path,
        settings.similarity_threshold,
    )
    document_count = retriever.rebuild_index()
    print(
        f"Indexed {document_count} incidents with {settings.embedding_model} "
        f"at {settings.embedding_index_path}"
    )


if __name__ == "__main__":
    main()
