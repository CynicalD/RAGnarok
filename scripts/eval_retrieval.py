"""Measure retrieval quality across alpha values.

Each case is a question plus the wiki page that should answer it. We score how often
that page shows up in the retrieved chunks, so alpha stops being a guess.

The misspelled cases are the point of the exercise: BM25 is exact-token matching, so a
typo scores zero on the sparse side and only the dense vector can save it. If accuracy
on those collapses as alpha drops, alpha is too lexical.

    python scripts/eval_retrieval.py            # sweep 0.0 -> 1.0
    python scripts/eval_retrieval.py 0.4        # detail for one alpha
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "worker"))

from retriever import DEFAULT_TOP_K, retrieve

# (question, page title that should answer it)
CASES = [
    ("What saddle do I need to ride an Ankylosaurus?", "Ankylosaurus"),
    ("How do I tame a Rex?", "Rex"),
    ("What does an Argentavis eat?", "Argentavis"),
    ("How much torpor does a Carnotaurus have?", "Carnotaurus"),
    ("What is the Giganotosaurus known for?", "Giganotosaurus"),
    ("Which creature gathers the most metal?", "Ankylosaurus"),
    ("How long does a Rex egg take to hatch?", "Rex"),
    ("What is the platform saddle on a Quetzal used for?", "Quetzal"),
    ("How do I tame a Rock Drake?", "Rock Drake"),
    ("What does a Shadowmane do?", "Shadowmane"),
    ("How does imprinting work?", "Imprinting"),
    ("What is kibble used for in taming?", "Kibble"),
    ("How do stat mutations work when breeding?", "Mutations"),
    ("How do cryopods work?", "Cryopod"),
    ("What temperature does an egg need to incubate?", "Incubation"),
    # Deliberately misspelled — these lean entirely on the dense vector.
    ("how do i tame a rhyniognata", "Rhyniognatha"),
    ("what does a gigantoraptr eat", "Gigantoraptor"),
    ("theriznosaurus saddle level", "Therizinosaur"),
    ("how to tame a beelzebufo frog", "Beelzebufo"),
    ("taming a carcarodontosaurus", "Carcharodontosaurus"),
]

MISSPELLED_FROM = 15  # index where the misspelled cases start


def hit(question, expected, alpha):
    titles = {chunk["title"] for chunk in retrieve(question, top_k=DEFAULT_TOP_K, alpha=alpha)}
    return expected in titles


def run(alpha, verbose=False):
    results = [hit(question, expected, alpha) for question, expected in CASES]
    if verbose:
        for (question, expected), ok in zip(CASES, results):
            print(f"  {'HIT ' if ok else 'MISS'}  {expected:20s}  {question}")
    clean = results[:MISSPELLED_FROM]
    typos = results[MISSPELLED_FROM:]
    return sum(results) / len(results), sum(clean) / len(clean), sum(typos) / len(typos)


def main():
    if len(sys.argv) > 1:
        alpha = float(sys.argv[1])
        print(f"=== alpha = {alpha}, top_k = {DEFAULT_TOP_K} ===")
        overall, clean, typos = run(alpha, verbose=True)
        print(f"\noverall {overall:.0%}  |  clean {clean:.0%}  |  misspelled {typos:.0%}")
        return

    print(f"top_k = {DEFAULT_TOP_K}\n")
    print(f"{'alpha':>6}  {'overall':>8}  {'clean':>8}  {'misspelled':>11}")
    for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        overall, clean, typos = run(alpha)
        print(f"{alpha:>6.1f}  {overall:>7.0%}  {clean:>8.0%}  {typos:>11.0%}")


if __name__ == "__main__":
    main()
