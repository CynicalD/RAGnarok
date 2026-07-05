"""Hybrid retrieval: turn a question into the top-k most relevant wiki chunks.

Embeds the query (dense) + BM25-encodes it (sparse), blends them with `alpha`,
then searches the Pinecone index. Imported by the Lambda in a later milestone.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

from openai import OpenAI
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder


INDEX_NAME = "ragnarok"
EMBED_MODEL = "text-embedding-3-small"
BM25_PATH = Path(__file__).resolve().parents[2] / "data" / "bm25_encoder.json"
DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.5

openai_client = OpenAI()
pinecone_index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(INDEX_NAME)
bm25 = BM25Encoder().load(str(BM25_PATH))


def retrieve(query, top_k=DEFAULT_TOP_K, alpha=DEFAULT_ALPHA):
    dense = openai_client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    sparse = bm25.encode_queries(query)

    dense = [value * alpha for value in dense]
    sparse = {"indices": sparse["indices"], "values": [value * (1 - alpha) for value in sparse["values"]]}

    query_args = {"vector": dense, "top_k": top_k, "include_metadata": True}
    if sparse["indices"] and alpha < 1.0:
        query_args["sparse_vector"] = sparse

    matches = pinecone_index.query(**query_args)["matches"]
    return [
        {
            "score": match["score"],
            "title": match["metadata"]["title"],
            "url": match["metadata"]["url"],
            "text": match["metadata"]["text"],
        }
        for match in matches
    ]


SAMPLE_QUESTIONS = [
    "What saddle do I need to ride a Rex?",
    "How do I tame an Ankylosaurus?",
    "Which creature is best for gathering metal?",
    "What does the Argentavis eat?",
    "How do I knock out a Carnotaurus?",
    "Which dinos are good for carrying heavy loads?",
    "What is the Giganotosaurus known for?",
    "How much torpor does a Rex have?",
    "Which creature can gather a lot of berries?",
    "What is a good early game mount?",
    "Tell me about the Quetzal platform saddle.",
    "Which creatures live in the water?",
]


if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ALPHA
    print(f"=== alpha = {alpha} ===\n")
    for question in SAMPLE_QUESTIONS:
        print(f"Q: {question}")
        for hit in retrieve(question, alpha=alpha):
            snippet = hit["text"][:100].replace("\n", " ")
            print(f"  [{hit['score']:.3f}] {hit['title']}: {snippet}...")
        print()