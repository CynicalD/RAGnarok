"""Offline data pipeline for the ARK wiki. Runs locally, never deployed to Lambda.

Phase A (rn): scrape -> clean-> chunk -> save to data/chunks.jsonl.
Phase B (added once OpenAI/Pinecone keys exist): embed + upsert into Pinecone.
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

load_dotenv()

API_URL = "https://ark.wiki.gg/api.php"
USER_AGENT = "RAGnarok/0.1 (ARK wiki RAG learning project)"
OUTPUT_PATH = Path("data/chunks.jsonl")

CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50
MIN_PAGE_WORDS = 50  # skip stub pages

INDEX_NAME = "ragnarok"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
BM25_PATH = Path("data/bm25_encoder.json")
BATCH_SIZE = 100

# sMall iconic test slice. Swap for a full category once quality is verified.
CREATURES = [
    "Rex", "Raptor", "Ankylosaurus", "Argentavis", "Triceratops", "Parasaur",
    "Dodo", "Pteranodon", "Spino", "Giganotosaurus", "Carnotaurus", "Dilophosaur",
    "Sarco", "Megalodon", "Quetzal", "Therizinosaurus", "Stegosaurus", "Brontosaurus",
    "Doedicurus", "Castoroides", "Direwolf", "Sabertooth", "Woolly Rhino", "Mammoth",
    "Beelzebufo", "Baryonyx", "Allosaurus", "Megatherium", "Iguanodon", "Pachy",
]

# HTML elements that are navigation/styling/tables, not article prose.
NOISE_SELECTORS = [
    "script", "style", "table", "sup.reference", "span.mw-editsection",
    "div.navbox", "div.thumb", "figure", "#toc", ".noprint", ".mw-empty-elt",
    "div.info-arkitex", "ul.itemlist",  # template widgets: variant lists, not prose
]


def page_url(title):
    return f"https://ark.wiki.gg/wiki/{title.replace(' ', '_')}"


def fetch_page_html(title):
    params = {"action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": "2"}
    resp = requests.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        return None
    return data["parse"]["text"]


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output") or soup
    for selector in NOISE_SELECTORS:
        for tag in content.select(selector):
            tag.decompose()

    parts = []
    for element in content.find_all(["h2", "h3", "p", "li"]):
        text = element.get_text(" ", strip=True)
        if text:
            parts.append(re.sub(r"\s+", " ", text))
    return "\n".join(parts)


def chunk_text(text, title, encoder):
    tokens = encoder.encode(text)
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    chunks = []
    for index, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start:start + CHUNK_TOKENS]
        chunks.append({
            "id": f"{title}-{index}",
            "title": title,
            "url": page_url(title),
            "text": encoder.decode(window).strip(),
        })
        if start + CHUNK_TOKENS >= len(tokens):
            break
    return chunks


def scrape_pages(titles):
    rows = []
    for title in titles:
        try:
            html = fetch_page_html(title)
        except requests.RequestException as error:
            print(f"  ! {title}: request failed ({error})")
            continue
        if not html:
            print(f"  ! {title}: page not found, skipping")
            continue
        text = html_to_text(html)
        rows.append({"title": title, "url": page_url(title), "text": text, "words": len(text.split())})
        print(f"  - {title}: {len(text.split())} words")
        time.sleep(0.5)  # be polite to the wiki
    return rows


def embed_texts(client, texts):
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        response = client.embeddings.create(model=EMBED_MODEL, input=texts[start:start + BATCH_SIZE])
        vectors.extend(item.embedding for item in response.data)
    return vectors


def ensure_index(pc):
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="dotproduct",  # required for dense + sparse hybrid search
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    while True:
        status = pc.describe_index(INDEX_NAME).status
        ready = status["ready"] if isinstance(status, dict) else getattr(status, "ready", True)
        if ready:
            break
        time.sleep(1)
    return pc.Index(INDEX_NAME)


def upsert_chunks(index, chunks, dense_vectors, sparse_vectors):
    vectors = []
    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
        vector = {
            "id": chunk["id"],
            "values": dense,
            "metadata": {"text": chunk["text"], "title": chunk["title"], "url": chunk["url"]},
        }
        if sparse["indices"]:  # skip empty sparse vectors (chunks with only stopwords)
            vector["sparse_values"] = {"indices": sparse["indices"], "values": sparse["values"]}
        vectors.append(vector)

    for start in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[start:start + BATCH_SIZE])


def main():
    print(f"Scraping {len(CREATURES)} pages from the ARK wiki...")
    rows = scrape_pages(CREATURES)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="text")
    df = df[df["words"] >= MIN_PAGE_WORDS].reset_index(drop=True)
    print(f"\nKept {len(df)} pages after dropping stubs/duplicates.")

    encoder = tiktoken.encoding_for_model(EMBED_MODEL)
    all_chunks = []
    for row in df.itertuples():
        all_chunks.extend(chunk_text(row.text, row.title, encoder))

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w") as file:
        for chunk in all_chunks:
            file.write(json.dumps(chunk) + "\n")
    print(f"Wrote {len(all_chunks)} chunks to {OUTPUT_PATH}")

    texts = [chunk["text"] for chunk in all_chunks]

    print("\nEmbedding chunks with OpenAI (dense vectors)...")
    dense_vectors = embed_texts(OpenAI(), texts)

    print("Fitting BM25 on the corpus (sparse vectors)...")
    bm25 = BM25Encoder()
    bm25.fit(texts)
    bm25.dump(str(BM25_PATH))
    sparse_vectors = bm25.encode_documents(texts)
    print(f"Saved fitted BM25 encoder to {BM25_PATH}")

    print("Creating index (if needed) and upserting into Pinecone...")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = ensure_index(pc)
    upsert_chunks(index, all_chunks, dense_vectors, sparse_vectors)
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
