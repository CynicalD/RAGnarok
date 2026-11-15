"""Score the bot's final answers, to compare chat models or prompts.

eval_retrieval.py checks that search finds the right text. This checks the last step:
given that text, does the model state the right fact, and does it refuse questions the
wiki can't answer? It also reports tokens, cost and speed, so "cheaper" can be weighed
against "worse" with numbers.

    python scripts/eval_answers.py                                   # the model in generator.py
    python scripts/eval_answers.py gpt-5-mini reasoning_effort=low   # try another model
    python scripts/eval_answers.py gpt-5.4-mini reasoning_effort=none temperature=0.2 -v
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "worker"))

import generator

# $ per 1M tokens (input, output), from OpenAI's pricing page. Add a row to try a new model.
PRICES = {
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5-mini": (0.25, 2.00),
}

# (question, words the answer must contain). An empty list means the bot must refuse.
CASES = [
    ("What saddle do I need to ride an Ankylosaurus?", ["ankylo saddle", "36"]),
    ("How long does a Rex egg take to hatch?", ["4", "59"]),
    ("What kibble does an Argentavis prefer?", ["superior"]),
    ("Which creature is best for gathering metal?", ["ankylo"]),
    ("How much base health does a Rex have?", ["1100"]),
    ("How does imprinting work?", ["care"]),
    ("how do i tame a rhyniognata", ["pheromone"]),
    ("theriznosaurus saddle level", ["69"]),
    ("What does a Rex Saddle cost to craft?", ["380", "200", "50"]),
    ("How much damage does an Assault Rifle do?", ["47"]),
    ("What level unlocks the Compound Bow?", ["79"]),
    ("Which armor set gives the most cold protection?", ["fur"]),
    ("How much armor does a Flak Helmet give?", ["100"]),
    ("How much damage does C4 do to a metal wall?", ["3544"]),
    ("What ammo does the Longneck Rifle use?", ["simple rifle ammo"]),
    ("flack chestpeice armor rating", ["100"]),
    ("How do I bake a chocolate cake?", []),
    ("Who won the 2022 World Cup?", []),
    ("What is the best graphics card for ARK?", []),
]


def parse_option(text):
    name, value = text.split("=")
    try:
        return name, float(value)
    except ValueError:
        return name, value


def main():
    arguments = [argument for argument in sys.argv[1:] if argument != "-v"]
    verbose = "-v" in sys.argv
    if arguments:
        generator.CHAT_MODEL = arguments[0]
        generator.CHAT_OPTIONS = dict(parse_option(option) for option in arguments[1:])
        generator.MAX_ANSWER_TOKENS = 2000  # room for reasoning tokens, which count as output
    input_price, output_price = PRICES[generator.CHAT_MODEL]

    correct = tokens_in = tokens_out = 0
    seconds = []
    for question, must_contain in CASES:
        started = time.time()
        result = generator.answer(question)
        seconds.append(time.time() - started)
        tokens_in += result["input_tokens"]
        tokens_out += result["output_tokens"]

        text = result["text"].lower()
        if must_contain:
            passed = result["answered"] and all(word in text for word in must_contain)
        else:
            passed = not result["answered"]
        correct += passed
        if verbose or not passed:
            print(f"{'PASS' if passed else 'FAIL'}  {question}\n      {result['text'][:300]!r}\n")

    count = len(CASES)
    cost = (tokens_in * input_price + tokens_out * output_price) / 1e6 / count
    print(f"model {generator.CHAT_MODEL} {generator.CHAT_OPTIONS}")
    print(f"correct {correct}/{count}  |  ${cost:.5f} per question  |  {tokens_in // count} in / {tokens_out // count} out")
    print(f"seconds per question: average {sum(seconds) / count:.1f}, slowest {max(seconds):.1f}")


if __name__ == "__main__":
    main()
