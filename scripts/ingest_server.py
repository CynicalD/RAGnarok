"""Load one Discord server's own notes into its private Pinecone namespace.

The wiki sits in the index's default namespace and every server shares it. A server's
rules, rates and rank perks go into the namespace "guild-<guild_id>", which the worker
searches only for questions asked in that server. Runs locally, like ingest.py.

    python scripts/ingest_server.py 123456789012345678 servers/123456789012345678
"""

import os
import sys
from pathlib import Path

import tiktoken
from openai import OpenAI
from pinecone import Pinecone

from ingest import BATCH_SIZE, EMBED_MODEL, INDEX_NAME, chunk_text, embed_texts

NOTE_SUFFIXES = {".md", ".txt"}


def note_title(path):
    """rank-perks.md -> "Rank perks", FAQ.txt -> "FAQ". Answers show this title as their source."""
    title = path.stem.replace("-", " ").replace("_", " ")
    return title[:1].upper() + title[1:]


def read_notes(folder):
    encoder = tiktoken.encoding_for_model(EMBED_MODEL)
    chunks = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in NOTE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        file_chunks = chunk_text(text, note_title(path), "server", encoder)
        print(f"  {path.name}: {len(file_chunks)} chunks")
        chunks.extend(file_chunks)
    return chunks


def main(guild_id, folder):
    if not guild_id.isdigit():
        sys.exit(f"{guild_id!r} is not a Discord server ID. Usage: ingest_server.py <guild_id> <folder>")

    print(f"Reading notes from {folder}...")
    chunks = read_notes(folder)
    if not chunks:
        sys.exit("No .md or .txt notes found. Nothing was changed in Pinecone.")

    # Dense vectors only. The BM25 encoder is fitted on wiki text, so its word
    # statistics would be wrong for a server's notes.
    vectors = embed_texts(OpenAI(), [chunk["text"] for chunk in chunks])
    records = [
        {"id": chunk["id"], "values": vector, "metadata": {"text": chunk["text"], "title": chunk["title"]}}
        for chunk, vector in zip(chunks, vectors)
    ]

    namespace = f"guild-{guild_id}"  # must match guild_namespace() in src/worker/retriever.py
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(INDEX_NAME)

    # Start from empty so a deleted or shortened note leaves nothing behind.
    # Pinecone raises on deleting a namespace that doesn't exist yet, so check first.
    if namespace in index.describe_index_stats()["namespaces"]:
        index.delete(delete_all=True, namespace=namespace)
    for start in range(0, len(records), BATCH_SIZE):
        index.upsert(vectors=records[start:start + BATCH_SIZE], namespace=namespace)
    print(f"Upserted {len(records)} chunks into namespace {namespace}.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python scripts/ingest_server.py <guild_id> <folder>")
    main(sys.argv[1], sys.argv[2])
