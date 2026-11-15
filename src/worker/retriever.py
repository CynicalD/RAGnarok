"""Hybrid retrieval: turn a question into the top-k most relevant wiki chunks.

Embeds the query (dense) + BM25-encodes it (sparse), blends them with `alpha`,
then searches the Pinecone index.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

import nltk

from openai import OpenAI
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# BM25's tokenizer needs nltk data; we bundle it beside this file for Lambda.
nltk.data.path.insert(0, str(Path(__file__).parent / "nltk_data"))

INDEX_NAME = "ragnarok"
EMBED_MODEL = "text-embedding-3-small"
BM25_PATH = Path(__file__).parent / "bm25_encoder.json"
DEFAULT_TOP_K = 8
# Measured, not guessed: scripts/eval_retrieval.py scores 38 questions at each alpha.
# On the 3,876-chunk corpus only 0.8-0.9 gets every question right, so 0.85 sits in the
# middle. Lower values lose misspelled names, which score zero on the BM25 side.
DEFAULT_ALPHA = 0.85

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
    "Which creature is best for gathering metal?",
    "How can I tame a dreadnoughtus",
    "How can I breed a Rex?",
    "What does a Rex Saddle cost to craft?",
    "Which armor is best against the cold?",
    "How much damage does a Longneck Rifle do?",
    "What can destroy a metal wall?",
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