from typing import List, Dict
import json

try:
    from sentence_transformers import (
        SentenceTransformer,
        CrossEncoder,
    )

    from qdrant_client import QdrantClient

    from qdrant_client.models import (
        VectorParams,
        Distance,
        PointStruct,
    )

except ImportError as e:

    raise ImportError(
        f"Import failed:\n{e}"
    )


# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------

COLLECTION_NAME = "research_papers"

QDRANT_PATH = "./qdrant_db"


# ------------------------------------------------------
# RETRIEVAL PIPELINE
# ------------------------------------------------------

class RetrievalPipeline:

    def __init__(self):

        self.collection_name = COLLECTION_NAME

        print("\n[INFO] Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("[SUCCESS] Embedding model loaded!")

        print("\n[INFO] Loading reranker model...")

        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-base"
        )

        print("[SUCCESS] Reranker loaded!")

        print("\n[INFO] Initializing Qdrant...")

        self.qdrant = QdrantClient(
            path=QDRANT_PATH
        )

        print("[SUCCESS] Qdrant initialized!")

        self.vector_size = 384

    # --------------------------------------------------
    # CREATE COLLECTION
    # --------------------------------------------------

    def create_collection(self):

        existing = [
            c.name
            for c in self.qdrant.get_collections().collections
        ]

        if self.collection_name in existing:

            print(
                "[INFO] Collection already exists."
            )

            return

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

        print("[SUCCESS] Collection created!")

    # --------------------------------------------------
    # INDEX DOCUMENTS
    # --------------------------------------------------

    def index_documents(
        self,
        chunks: List[Dict]
    ):

        print("\n[INFO] Creating embeddings...")

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
        )

        points = []

        for idx, (chunk, vector) in enumerate(
            zip(chunks, embeddings)
        ):

            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                    },
                )
            )

        print(
            "\n[INFO] Uploading vectors to Qdrant..."
        )

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(
            "[SUCCESS] Documents indexed!"
        )

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:

        print(f"\n[INFO] Searching for: {query}")

        query_embedding = self.embedding_model.encode(
            query
        )

        # ----------------------------------------------
        # STAGE 1: Dense Retrieval
        # ----------------------------------------------

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=50,
        ).points

        print(
            f"[SUCCESS] Dense retrieval got "
            f"{len(results)} candidates."
        )

        # ----------------------------------------------
        # STAGE 2: Reranking
        # ----------------------------------------------

        candidate_texts = [
            r.payload["text"]
            for r in results
        ]

        rerank_pairs = [
            [query, text]
            for text in candidate_texts
        ]

        scores = self.reranker.predict(
            rerank_pairs
        )

        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        final_results = []

        for result, score in reranked[:top_k]:

            final_results.append(
                {
                    "chunk_id": result.payload["chunk_id"],
                    "text": result.payload["text"],
                    "score": float(score),
                }
            )

        return final_results


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------

if __name__ == "__main__":

    pipeline = RetrievalPipeline()

    # --------------------------------------------------
    # LOAD CHUNKS
    # --------------------------------------------------

    try:

        with open(
            "chunks.json",
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

        print(
            "\n[SUCCESS] Loaded chunks "
            "from chunks.json"
        )

    except Exception as e:

        raise RuntimeError(
            f"Failed to load chunks:\n{e}"
        )

    # --------------------------------------------------
    # DELETE OLD COLLECTION
    # --------------------------------------------------

    try:

        pipeline.qdrant.delete_collection(
            collection_name=pipeline.collection_name
        )

        print(
            "\n[INFO] Old collection deleted."
        )

    except Exception:

        print(
            "\n[INFO] No old collection found."
        )

    # --------------------------------------------------
    # RECREATE COLLECTION
    # --------------------------------------------------

    pipeline.create_collection()

    # --------------------------------------------------
    # INDEX NEW DOCUMENTS
    # --------------------------------------------------

    pipeline.index_documents(chunks)

    print(
        "\n[SUCCESS] New paper indexed!"
    )

    # --------------------------------------------------
    # TEST QUERY
    # --------------------------------------------------

    query = input(
        "\nEnter test query: "
    )

    results = pipeline.search(query)

    print(
        "\n========== TOP RESULTS ==========\n"
    )

    for result in results:

        print(
            f"\n{result['chunk_id']}"
        )

        print(
            f"\nScore: {result['score']}"
        )

        print("\n")

        print(
            result["text"][:1000]
        )

        print(
            "\n" + "=" * 80 + "\n"
        )

    pipeline.qdrant.close()