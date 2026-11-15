"""Score the answers one Discord server gets once it has its own notes.

Two checks. Questions about the server must be answered from its notes. And the wiki
questions from eval_answers.py must still pass with those notes added to the context.

The server's questions sit beside its notes in servers/<guild_id>/eval_cases.json, so
they stay out of the repo too:

    [{"question": "What is the taming rate here?", "must_contain": ["5x", "server notes (rates)"]}]

Listing the note's label checks the source line as well as the fact. An empty list
means the bot must refuse.

    python scripts/eval_server.py 123456789012345678
    python scripts/eval_server.py 123456789012345678 -v
"""

import json
import sys
from pathlib import Path

import eval_answers


def main(guild_id, verbose):
    path = Path("servers") / guild_id / "eval_cases.json"
    cases = [(case["question"], case["must_contain"]) for case in json.loads(path.read_text(encoding="utf-8"))]

    print(f"Server questions from {path}")
    eval_answers.summarize(len(cases), *eval_answers.run(cases, guild_id, verbose))

    print("\nWiki questions, with the server's notes in the context")
    wiki_cases = eval_answers.CASES
    eval_answers.summarize(len(wiki_cases), *eval_answers.run(wiki_cases, guild_id, verbose))


if __name__ == "__main__":
    arguments = [argument for argument in sys.argv[1:] if argument != "-v"]
    if len(arguments) != 1:
        sys.exit("Usage: python scripts/eval_server.py <guild_id> [-v]")
    main(arguments[0], "-v" in sys.argv)
