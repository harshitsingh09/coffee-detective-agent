"""Persisted local semantic retrieval with an explicit lexical fallback."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from incident_assistant.domain.agent_models import (
    HistoricalIncidentDocument,
    SimilarIncident,
)
from incident_assistant.domain.ports import (
    EmbeddingProvider,
    HistoricalIncidentDocumentRepository,
    HistoricalIncidentRetriever,
)


class _PersistedIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 2
    model_id: str = Field(min_length=1)
    embedding_dimension: int = Field(gt=0)
    documents: tuple[HistoricalIncidentDocument, ...]
    embeddings: tuple[tuple[float, ...], ...]


class SentenceTransformerEmbeddingProvider:
    """Lazy local Sentence Transformers provider with normalized NumPy output."""

    def __init__(self, model_id: str, cache_path: Path) -> None:
        self._model_id = model_id
        self._cache_path = Path(cache_path)
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _active_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._cache_path.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                self._model_id,
                cache_folder=str(self._cache_path),
                device="cpu",
            )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors = self._active_model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return tuple(tuple(float(value) for value in vector) for vector in vectors)

    def embed_query(self, text: str) -> tuple[float, ...]:
        vector = self._active_model().encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return tuple(float(value) for value in vector)


class PersistentSemanticIncidentRetriever:
    """Search a validated on-disk embedding index without rebuilding per request."""

    def __init__(
        self,
        documents: HistoricalIncidentDocumentRepository,
        embeddings: EmbeddingProvider,
        index_path: Path,
        similarity_threshold: float = 0.35,
    ) -> None:
        self._documents = documents
        self._embeddings = embeddings
        self._index_path = Path(index_path)
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1.")
        self._similarity_threshold = similarity_threshold
        self._index: _PersistedIndex | None = None

    def rebuild_index(self) -> int:
        documents = tuple(self._documents.list_incident_documents())
        if not documents:
            raise ValueError("Cannot build a semantic index without incident documents.")
        vectors = tuple(
            tuple(float(value) for value in vector)
            for vector in self._embeddings.embed_documents(
                [document.embedding_text() for document in documents]
            )
        )
        self._validate_vectors(documents, vectors)
        index = _PersistedIndex(
            model_id=self._embeddings.model_id,
            embedding_dimension=len(vectors[0]),
            documents=documents,
            embeddings=vectors,
        )
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        temporary_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
        temporary_path.replace(self._index_path)
        self._index = index
        return len(documents)

    def search(self, description: str, top_k: int = 3) -> tuple[SimilarIncident, ...]:
        query = description.strip()
        if not query:
            raise ValueError("A historical-incident search description is required.")
        index = self._load_index()
        query_vector = tuple(float(value) for value in self._embeddings.embed_query(query))
        if not query_vector or len(query_vector) != len(index.embeddings[0]):
            raise ValueError("Query embedding dimensions do not match the incident index.")
        ranked = self._rank_with_numpy(query_vector, index)
        eligible = [item for item in ranked if item[0] >= self._similarity_threshold]
        bounded_top_k = max(1, min(top_k, 10, len(ranked)))
        return tuple(
            SimilarIncident(
                incident_id=document.incident_id,
                similarity_score=max(0.0, min(1.0, score)),
                description=document.description,
                root_cause=document.root_cause,
                resolution=document.resolution,
                retrieval_method="semantic",
            )
            for score, document in eligible[:bounded_top_k]
        )

    def _load_index(self) -> _PersistedIndex:
        if self._index is not None:
            return self._index
        if not self._index_path.is_file():
            raise FileNotFoundError(
                f"Semantic index not found at {self._index_path}. "
                "Run scripts/build_incident_index.py first."
            )
        index = _PersistedIndex.model_validate_json(self._index_path.read_text(encoding="utf-8"))
        if index.model_id != self._embeddings.model_id:
            raise ValueError(
                "Semantic index model does not match EMBEDDING_MODEL; rebuild the index."
            )
        self._validate_vectors(index.documents, index.embeddings)
        if index.embedding_dimension != len(index.embeddings[0]):
            raise ValueError("Semantic index dimension metadata does not match vectors.")
        self._index = index
        return index

    @staticmethod
    def _rank_with_numpy(
        query_vector: Sequence[float],
        index: _PersistedIndex,
    ) -> list[tuple[float, HistoricalIncidentDocument]]:
        query = np.asarray(query_vector, dtype=np.float32)
        matrix = np.asarray(index.embeddings, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        document_norms = np.linalg.norm(matrix, axis=1)
        denominators = document_norms * query_norm
        scores = np.divide(
            matrix @ query,
            denominators,
            out=np.zeros(len(matrix), dtype=np.float32),
            where=denominators != 0,
        )
        return sorted(
            zip((float(score) for score in scores), index.documents, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )

    @staticmethod
    def _validate_vectors(
        documents: Sequence[HistoricalIncidentDocument],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(documents) != len(vectors):
            raise ValueError("Semantic index document and vector counts do not match.")
        if not vectors or not vectors[0]:
            raise ValueError("Semantic index contains no embedding vectors.")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("Semantic index vectors have inconsistent dimensions.")


class LexicalHistoricalIncidentRetriever:
    """Deterministic retrieval fallback used when embeddings are unavailable."""

    def __init__(self, documents: HistoricalIncidentDocumentRepository) -> None:
        self._documents = documents

    def search(self, description: str, top_k: int = 3) -> tuple[SimilarIncident, ...]:
        query_tokens = self._tokens(description)
        ranked = sorted(
            (
                (
                    self._overlap(
                        query_tokens,
                        self._tokens(document.embedding_text()),
                    ),
                    document,
                )
                for document in self._documents.list_incident_documents()
            ),
            key=lambda item: (item[0], item[1].incident_id),
            reverse=True,
        )
        bounded_top_k = max(1, min(top_k, 10, len(ranked)))
        return tuple(
            SimilarIncident(
                incident_id=document.incident_id,
                similarity_score=score,
                description=document.description,
                root_cause=document.root_cause,
                resolution=document.resolution,
                retrieval_method="lexical_fallback",
            )
            for score, document in ranked[:bounded_top_k]
        )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.strip(".,:;!?()[]{}\"'").casefold()
            for token in text.split()
            if len(token.strip(".,:;!?()[]{}\"'")) >= 3
        }

    @staticmethod
    def _overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / math.sqrt(len(left) * len(right))


class ResilientHistoricalIncidentRetriever:
    """Use lexical retrieval if the local model or persisted index is unavailable."""

    def __init__(
        self,
        primary: HistoricalIncidentRetriever,
        fallback: HistoricalIncidentRetriever,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    def search(self, description: str, top_k: int = 3) -> Sequence[SimilarIncident]:
        try:
            return self._primary.search(description, top_k)
        except Exception:
            return self._fallback.search(description, top_k)
