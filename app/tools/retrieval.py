from functools import lru_cache
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import get_settings


COLLECTION_NAME = "research_papers"
VECTOR_SIZE = 384


class RetrievalPipeline:
    def __init__(
        self,
        *,
        embedding_model: Any | None = None,
        reranker: Any | None = None,
        qdrant: QdrantClient | Any | None = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = COLLECTION_NAME
        self.embedding_model = embedding_model or SentenceTransformer(
            settings.embedding_model
        )
        self.reranker = reranker or CrossEncoder(settings.reranker_model)
        self.qdrant = qdrant or self._create_qdrant_client(settings)
        self.create_collection()

    @staticmethod
    def _create_qdrant_client(settings: Any) -> QdrantClient:
        if settings.qdrant_url:
            return QdrantClient(url=settings.qdrant_url)
        return QdrantClient(path=settings.qdrant_path)

    def create_collection(self) -> None:
        if self.qdrant.collection_exists(self.collection_name):
            return
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    def index_documents(
        self,
        chunks: list[dict[str, Any]],
        *,
        paper_id: str,
        paper_title: str,
    ) -> int:
        if not chunks:
            raise ValueError("Cannot index a paper without usable text chunks")

        embeddings = self.embedding_model.encode(
            [chunk["text"] for chunk in chunks],
            show_progress_bar=False,
        )
        points = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            payload: dict[str, Any] = {
                "paper_id": paper_id,
                "paper": paper_title,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            }
            if chunk.get("page") is not None:
                payload["page"] = chunk["page"]
            point_id = str(uuid5(NAMESPACE_URL, f"{paper_id}:{chunk['chunk_id']}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def delete_paper(self, paper_id: str) -> None:
        if not self.qdrant.collection_exists(self.collection_name):
            return
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchValue(value=paper_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def search(self, query: str, *, top_k: int = 5, candidate_limit: int = 50) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Search query must not be empty")
        query_embedding = self.embedding_model.encode(query)
        candidates = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=candidate_limit,
            with_payload=True,
        ).points
        candidates = [
            candidate
            for candidate in candidates
            if candidate.payload
            and isinstance(candidate.payload.get("paper_id"), str)
            and isinstance(candidate.payload.get("paper"), str)
            and isinstance(candidate.payload.get("chunk_id"), str)
            and isinstance(candidate.payload.get("text"), str)
        ]
        if not candidates:
            return []

        pairs = [[query, candidate.payload["text"]] for candidate in candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        results = []
        for candidate, score in reranked[:top_k]:
            payload = candidate.payload
            results.append(
                {
                    "paper_id": payload["paper_id"],
                    "paper": payload["paper"],
                    "chunk_id": payload["chunk_id"],
                    "page": payload.get("page"),
                    "text": payload["text"],
                    "score": float(score),
                }
            )
        return results


@lru_cache(maxsize=1)
def get_retrieval_pipeline() -> RetrievalPipeline:
    return RetrievalPipeline()
