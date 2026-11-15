"""Measure retrieval quality across alpha values.

Each case is a question, the wiki page(s) that should answer it, and a fact that must
appear in the retrieved text. Two scores come out of that:

    page  - did a right page come back? (is search finding the right place)
    fact  - is the answer itself in the retrieved text? (did ingest keep the data)

The misspelled cases matter most for alpha: BM25 is exact-token matching, so a typo
scores zero on the sparse side and only the dense vector can save it.

    python scripts/eval_retrieval.py            # sweep alpha 0.0 -> 1.0
    python scripts/eval_retrieval.py 0.7        # every case at one alpha
    python scripts/eval_retrieval.py 0.7 5      # same, with top_k = 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "worker"))

from retriever import DEFAULT_TOP_K, retrieve

# (group, question, acceptable page titles, fact that must be in the retrieved text)
CASES = [
    ("creatures", "What saddle do I need to ride an Ankylosaurus?", ["Ankylosaurus", "Ankylo Saddle"], "Level 36"),
    ("creatures", "How do I tame a Rex?", ["Rex"], "Knockout"),
    ("creatures", "What does an Argentavis eat?", ["Argentavis"], "Carnivore"),
    ("creatures", "How much torpor does a Carnotaurus have?", ["Carnotaurus"], "Torpidity"),
    ("creatures", "What is the Giganotosaurus known for?", ["Giganotosaurus"], "Rage"),
    ("creatures", "Which creature gathers the most metal?", ["Ankylosaurus"], "Metal"),
    ("creatures", "How long does a Rex egg take to hatch?", ["Rex"], "Incubation"),
    ("creatures", "What is the platform saddle on a Quetzal used for?", ["Quetzal", "Quetzal Platform Saddle"], "platform"),
    ("creatures", "How do I tame a Rock Drake?", ["Rock Drake"], "egg"),
    ("creatures", "What does a Shadowmane do?", ["Shadowmane"], "Shadowmane"),
    ("creatures", "How much base health does a Rex have?", ["Rex"], "1100"),
    ("creatures", "How much damage does a Rex bite do?", ["Rex"], "62 damage"),
    ("mechanics", "How does imprinting work?", ["Imprinting"], "imprint"),
    ("mechanics", "What is kibble used for in taming?", ["Kibble"], "Kibble"),
    ("mechanics", "How do stat mutations work when breeding?", ["Mutations"], "mutation"),
    ("mechanics", "How do cryopods work?", ["Cryopod"], "Cryopod"),
    ("mechanics", "What temperature does an egg need to incubate?", ["Incubation"], "temperature"),
    ("gear", "How much armor does a Flak Helmet give?", ["Flak Helmet"], "Armor rating 100"),
    ("gear", "What level unlocks the Compound Bow?", ["Compound Bow"], "Level 79"),
    ("gear", "What does a Rex Saddle cost to craft?", ["Rex Saddle"], "380 × Hide"),
    ("gear", "How much damage does an Assault Rifle do?", ["Assault Rifle"], "Ranged damage 47"),
    ("gear", "What ammo does the Longneck Rifle use?", ["Longneck Rifle"], "Simple Rifle Ammo"),
    ("gear", "Where do I craft a Pump-Action Shotgun?", ["Pump-Action Shotgun"], "Fabricator"),
    ("gear", "Which armor set gives the most cold protection?", ["Armor"], "Cold protection: 249"),
    ("gear", "How much heat protection does a Ghillie Chestpiece give?", ["Ghillie Chestpiece"], "Heat protection"),
    ("gear", "How much damage does C4 do to a metal wall?", ["C4 Charge"], "Metal: 3544"),
    ("gear", "What do I need to make Tranquilizer Darts?", ["Tranquilizer Dart"], "Narcotic"),
    ("gear", "What level is the Argentavis Saddle?", ["Argentavis Saddle", "Argentavis"], "Level 62"),
    ("gear", "How much durability does a Metal Pick have?", ["Metal Pick"], "Durability"),
    # Deliberately misspelled: these lean entirely on the dense vector.
    ("typos", "how do i tame a rhyniognata", ["Rhyniognatha"], "Rhyniognatha"),
    ("typos", "what does a gigantoraptr eat", ["Gigantoraptor"], "Gigantoraptor"),
    ("typos", "theriznosaurus saddle level", ["Therizinosaur", "Therizinosaurus Saddle"], "Level 69"),
    ("typos", "how to tame a beelzebufo frog", ["Beelzebufo"], "Beelzebufo"),
    ("typos", "taming a carcarodontosaurus", ["Carcharodontosaurus"], "Carcharodontosaurus"),
    ("typos", "fabricated snipr rifle damage", ["Fabricated Sniper Rifle"], "Ranged damage"),
    ("typos", "flack chestpeice armor rating", ["Flak Chestpiece"], "Armor rating 100"),
    ("typos", "compund bow unlock level", ["Compound Bow"], "Level 79"),
    ("typos", "how to craft a riot helmit", ["Riot Helmet"], "Ingredients"),
]

GROUPS = ["creatures", "mechanics", "gear", "typos"]


def score(question, pages, fact, alpha, top_k):
    chunks = retrieve(question, top_k=top_k, alpha=alpha)
    page_hit = any(chunk["title"] in pages for chunk in chunks)
    fact_hit = any(fact.lower() in chunk["text"].lower() for chunk in chunks)
    return page_hit, fact_hit


def run(alpha, top_k=DEFAULT_TOP_K, verbose=False):
    results = []
    for group, question, pages, fact in CASES:
        page_hit, fact_hit = score(question, pages, fact, alpha, top_k)
        results.append((group, page_hit, fact_hit))
        if verbose:
            print(f"  page {'HIT ' if page_hit else 'MISS'}  fact {'HIT ' if fact_hit else 'MISS'}  {pages[0]:24s} {question}")
    return results


def share(results, group=None, index=1):
    picked = [result[index] for result in results if group in (None, result[0])]
    return sum(picked) / len(picked)


def main():
    if len(sys.argv) > 1:
        alpha = float(sys.argv[1])
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOP_K
        print(f"=== alpha = {alpha}, top_k = {top_k} ===")
        results = run(alpha, top_k, verbose=True)
        print(f"\npage {share(results):.0%}  |  fact {share(results, index=2):.0%}")
        return

    print(f"top_k = {DEFAULT_TOP_K}, {len(CASES)} cases. Columns are page-hit rate, last is fact-hit rate.\n")
    print(f"{'alpha':>6}  {'overall':>8}  " + "  ".join(f"{group:>10}" for group in GROUPS) + f"  {'fact':>6}")
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        results = run(alpha)
        groups = "  ".join(f"{share(results, group):>10.0%}" for group in GROUPS)
        print(f"{alpha:>6.1f}  {share(results):>8.0%}  {groups}  {share(results, index=2):>6.0%}")


if __name__ == "__main__":
    main()
