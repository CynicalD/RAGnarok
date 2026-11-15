"""Offline data pipeline for the ARK wiki. Runs locally, never deployed to Lambda.

scrape -> clean -> chunk -> embed (dense) + BM25 (sparse) -> upsert into Pinecone.

    python scripts/ingest.py                  # full run: rebuilds the whole index
    python scripts/ingest.py "Rex Saddle"     # preview one page's extracted text, upserts nothing
"""

import json
import os
import re
import sys
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
# Written straight into the worker package: the worker must load the same fit the index was built with.
BM25_PATH = Path("src/worker/bm25_encoder.json")
BATCH_SIZE = 100

# Wiki categories to ingest. Order is priority: a page in two categories keeps the first.
CATEGORIES = ["Creatures", "Weapons", "Armor"]

# Primitive Plus was an ASE-only mod. Its pages reuse real item names with mod stats.
EXCLUDED_TITLE_MARKERS = ["(Primitive Plus)"]

# Abort before touching Pinecone if more than this share of pages failed to scrape.
MAX_FAILED_SHARE = 0.02

# General "how does X work" pages. Category pages cover one creature or item each,
# these cover the mechanics those pages assume you already know.
MECHANICS_PAGES = [
    "Taming", "Breeding", "Incubation", "Imprinting", "Mutations", "Kibble",
    "Saddles", "Cryopod", "Eggs", "Torpidity", "Leveling", "Experience",
    "Creature stats calculation", "Cloning Chamber", "Tek Tier", "Bosses", "Weight",
]

# HTML elements that are navigation/styling/tables, not article prose.
NOISE_SELECTORS = [
    "script", "style", "table", "sup.reference", "span.mw-editsection",
    "div.navbox", "div.thumb", "figure", "#toc", ".noprint", ".mw-empty-elt",
    "div.info-arkitex", "ul.itemlist",  # template widgets: variant lists, not prose
]


def page_url(title):
    return f"https://ark.wiki.gg/wiki/{title.replace(' ', '_')}"


def slug(title):
    """Pinecone vector IDs must be printable ASCII, but titles aren't (Fenrisúlfr)."""
    return re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()


def api_query(**params):
    """Yield each page of a MediaWiki API response, following the continuation cursor."""
    params.update(format="json", formatversion="2")
    while True:
        resp = requests.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        yield data
        if "continue" not in data:
            return
        params.update(data["continue"])


def fetch_category_titles(category):
    """Every real page in one wiki category.

    Over half of Category:Creatures is redirects: variants like "Aberrant Rex" or
    "Corrupted Rex" point at the base creature page rather than holding their own
    content, so dropping redirects collapses them. Subpages (e.g. "Ankylosaurus/tr")
    are translations, so they go too. Gear categories have almost no redirects, but
    the same filters still apply.
    """
    titles = []
    for page in api_query(
        action="query",
        generator="categorymembers",
        gcmtitle=f"Category:{category}",
        gcmnamespace=0,
        gcmlimit=500,
        prop="info",
    ):
        for entry in page["query"]["pages"]:
            title = entry["title"]
            if "redirect" in entry or "/" in title:
                continue
            if any(marker in title for marker in EXCLUDED_TITLE_MARKERS):
                continue
            titles.append(title)
    return sorted(titles)


def collect_titles():
    """Map each page title to its category label, so every page is scraped once."""
    labels = {}
    for category in CATEGORIES:
        titles = fetch_category_titles(category)
        new = [title for title in titles if title not in labels]
        for title in new:
            labels[title] = category.lower()
        print(f"  {category}: {len(titles)} pages ({len(new)} new)")
    for title in MECHANICS_PAGES:
        labels.setdefault(title, "mechanics")

    # Two titles with the same slug would silently overwrite each other's chunks.
    slugs = {}
    for title in labels:
        if slugs.setdefault(slug(title), title) != title:
            sys.exit(f"Slug collision: {title!r} and {slugs[slug(title)]!r}")
    return labels


def fetch_page_html(title):
    params = {"action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": "2"}
    resp = requests.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        return None
    return data["parse"]["text"]


def infobox_facts(content, title):
    """Pull the infobox out as plain "Label Value" facts.

    The infobox holds the facts players actually ask for: a creature's saddle, taming
    method and incubation time, or an item's damage, armor rating, unlock level and
    crafting station. It renders as nested widget divs, so we read only the leaf rows
    (`info-unit-row`) and dedupe: the parent wrappers repeat their children's text,
    which would otherwise triple the content.
    """
    facts = []
    for row in content.select("div.info-unit-row"):
        text = re.sub(r"\s+", " ", row.get_text(" ", strip=True))
        if text and text != title and text not in facts:
            facts.append(text)

    # Crafting cost has no leaf rows: it is a list of bold "380 × Hide" entries.
    for unit in content.select("div.info-unit"):
        caption = unit.select_one("div.info-unit-caption")
        if caption and caption.get_text(strip=True) == "Ingredients":
            items = [re.sub(r"\s+", " ", b.get_text(" ", strip=True)) for b in unit.select("b")]
            if items:
                facts.append("Ingredients " + " + ".join(items))
            break
    return facts


def cell_text(cell):
    """A table cell's text, falling back to its image's alt text (icon-only headers)."""
    text = re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
    if not text and cell.find("img"):
        text = cell.find("img").get("alt", "").removesuffix(".png")
    return text


def table_lines(table):
    """Flatten one data table into text lines, one per row.

    When every row lines up with the header we write "Header: value" pairs, so a row
    still makes sense if chunking separates it from the header. Tables with merged
    cells don't line up, so those fall back to plain " | " rows under the header.
    Changelog tables are skipped: long, and nobody asks the bot for patch history.
    """
    rows = []
    for tr in table.find_all("tr"):
        cells = [cell_text(cell) for cell in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    if not rows or rows[0][0].startswith("Patch"):
        return []

    header, body = rows[0], rows[1:]
    if body and all(len(row) == len(header) for row in body):
        return ["; ".join(f"{name}: {value}" for name, value in zip(header, row) if value) for row in body]
    return [" | ".join(row) for row in rows]


def flatten_tables(soup, content):
    """Swap each data table for a <p> of its rows, so it survives the noise pass in place."""
    for table in content.select("table.wikitable"):
        if table.find_parent("table"):
            continue
        lines = table_lines(table)
        if lines:
            paragraph = soup.new_tag("p")
            paragraph.string = "\n".join(lines)
            table.replace_with(paragraph)


def html_to_text(html, title):
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output") or soup

    # Read the infobox and tables before the noise pass below strips them.
    facts = infobox_facts(content, title)
    flatten_tables(soup, content)

    for selector in NOISE_SELECTORS:
        for tag in content.select(selector):
            tag.decompose()

    parts = [f"{title} key facts: " + "; ".join(facts)] if facts else []
    for element in content.find_all(["h2", "h3", "p", "li"]):
        text = element.get_text(" ", strip=True)
        if text:
            parts.append(re.sub(r"[^\S\n]+", " ", text))  # collapse spaces, keep table row breaks
    return "\n".join(parts)


def chunk_text(text, title, category, encoder):
    tokens = encoder.encode(text)
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    chunks = []
    for index, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start:start + CHUNK_TOKENS]
        chunks.append({
            "id": f"{slug(title)}-{index}",
            "title": title,
            "category": category,
            "url": page_url(title),
            "text": encoder.decode(window).strip(),
        })
        if start + CHUNK_TOKENS >= len(tokens):
            break
    return chunks


def fetch_with_retry(title, attempts=3):
    for attempt in range(attempts):
        try:
            return fetch_page_html(title)
        except requests.RequestException as error:
            print(f"  ! {title}: request failed ({error}), attempt {attempt + 1}/{attempts}")
            time.sleep(2 * (attempt + 1))
    return None


def scrape_pages(labels):
    rows, failed = [], []
    for position, (title, category) in enumerate(labels.items(), start=1):
        html = fetch_with_retry(title)
        if not html:
            failed.append(title)
            continue
        text = html_to_text(html, title)
        rows.append({"title": title, "category": category, "text": text, "words": len(text.split())})
        print(f"  [{position}/{len(labels)}] {title}: {len(text.split())} words")
        time.sleep(0.3)  # be polite to the wiki
    return rows, failed


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
            "metadata": {
                "text": chunk["text"],
                "title": chunk["title"],
                "category": chunk["category"],
                "url": chunk["url"],
            },
        }
        if sparse["indices"]:  # skip empty sparse vectors (chunks with only stopwords)
            vector["sparse_values"] = {"indices": sparse["indices"], "values": sparse["values"]}
        vectors.append(vector)

    for start in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[start:start + BATCH_SIZE])


def main():
    print("Listing pages...")
    labels = collect_titles()
    print(f"Found {len(labels)} pages in total.\n")
    rows, failed = scrape_pages(labels)

    if failed:
        print(f"\n{len(failed)} pages failed to scrape: {failed}")
    if len(failed) > len(labels) * MAX_FAILED_SHARE:
        sys.exit("Too many failures. The live index was not touched; fix the scrape and re-run.")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="text")
    stubs = df[df["words"] < MIN_PAGE_WORDS]["title"].tolist()
    df = df[df["words"] >= MIN_PAGE_WORDS].reset_index(drop=True)
    print(f"\nKept {len(df)} pages. Dropped {len(stubs)} stubs: {stubs}")
    print(df.groupby("category")["title"].count().to_string())

    encoder = tiktoken.encoding_for_model(EMBED_MODEL)
    all_chunks = []
    for row in df.itertuples():
        all_chunks.extend(chunk_text(row.text, row.title, row.category, encoder))

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

    # Upsert only overwrites IDs it sends. Chunks from pages that shrank or were dropped
    # would linger with old text and old BM25 weights, so start from an empty index.
    if index.describe_index_stats().get("total_vector_count", 0):
        print("  wiping the old vectors...")
        index.delete(delete_all=True)

    upsert_chunks(index, all_chunks, dense_vectors, sparse_vectors)
    print(index.describe_index_stats())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for page_title in sys.argv[1:]:
            print(html_to_text(fetch_page_html(page_title), page_title), "\n")
    else:
        main()
