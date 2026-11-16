"""Generate a grounded answer, with its sources, from retrieved wiki chunks and server notes."""

import re
import sys

from openai import OpenAI

from retriever import retrieve

CHAT_MODEL = "gpt-5.4-mini"
# gpt-5 models are reasoning-capable. The answer is already in the context, so we
# turn reasoning off: no reasoning tokens billed, and Discord users aren't left waiting.
CHAT_OPTIONS = {"reasoning_effort": "none", "temperature": 0.2}
MAX_ANSWER_TOKENS = 600
MAX_SOURCES = 3
# Discord allows 2000 characters. This leaves room for the question the worker quotes above the answer.
MAX_REPLY_CHARS = 1600

# The model replies with this exact word when the context has no answer. A fixed marker
# lets code swap in a friendly message, skip the source links, and log the miss.
NO_ANSWER = "NO_ANSWER"
NO_ANSWER_MESSAGE = (
    "I couldn't find that in the ARK wiki. Try rephrasing, "
    "or name the creature or item you mean."
)
# The last two sentences were picked by counting. The old ending, "Only if the passages
# hold nothing that answers the question, reply with just NO_ANSWER", made the model
# refuse every yes/no question whose answer was no, such as "Can you cryopod a Titanosaur?".
SYSTEM_PROMPT = (
    "You are RAGnarok, an assistant for the game Ark: Survival Ascended. "
    "Answer the question using only the numbered passages from the ARK wiki that come with it. "
    "Players misspell and shorten names, so treat a close match as the same creature or item. "
    "Answer in at most four sentences or five short bullet points, using the exact numbers "
    "from the passages. State only what the passages say, without quoting them or "
    "mentioning passage numbers in the answer. "
    "Then add a final line 'Sources: ' with the numbers of the passages you used, "
    "for example 'Sources: 1, 3'. "
    "A passage that shows something is not possible does answer the question: say no and give "
    f"the reason. Reply with just {NO_ANSWER} only when the passages do not cover what was asked."
)

# Added to the system prompt only when the context holds server notes, so a server
# without notes sends the model exactly the prompt it always has. The wording was picked
# by counting: without the "forbids" sentence the model answered NO_ANSWER to rule
# questions whose answer is "no", and when left to divide a wiki time by a server rate
# it sometimes multiplied.
SERVER_LABEL = "Server notes"
SERVER_RULE = (
    f" Passages marked '{SERVER_LABEL}' were written by this Discord server's admins about "
    "its own rules, rates and perks. Answer from them like any other passage. Servers change "
    "the game's default settings, so when a server note and a wiki passage disagree, for "
    "example on taming or breeding rates, go with the server note. A rule that forbids "
    "something does answer the question: say it is not allowed and give the rule. When a "
    "server rate applies to a wiki number, such as hatch, mature or taming speed to a wiki "
    "time, give both the wiki number and the server's rate, and leave the maths to the player."
)

openai_client = OpenAI(max_retries=4)  # retries 429s and timeouts with backoff


def is_server(chunk):
    return chunk["source"] == "server"


def passage_name(chunk):
    return f"{SERVER_LABEL} ({chunk['title']})" if is_server(chunk) else chunk["title"]


def system_prompt(chunks):
    return SYSTEM_PROMPT + (SERVER_RULE if any(is_server(chunk) for chunk in chunks) else "")


def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[{number}] {passage_name(chunk)}: {chunk['text']}" for number, chunk in enumerate(chunks, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {query}"


def split_sources(reply, chunks):
    """Split the model's reply into the answer text and the chunks it says it used."""
    match = re.search(r"\n*Sources:\s*([\d,\s]*)$", reply)
    if not match:
        return reply, chunks  # no source line: fall back to retrieval order
    numbers = [int(number) for number in re.findall(r"\d+", match.group(1))]
    used = [chunks[number - 1] for number in numbers if 1 <= number <= len(chunks)]
    return reply[:match.start()].strip(), used or chunks


def source_links(chunks):
    """Markdown links to the pages behind an answer. <url> stops Discord embedding previews.

    A server note has no public page, so it shows as a plain label.
    """
    links = []
    for chunk in chunks:
        link = passage_name(chunk) if is_server(chunk) else f"[{chunk['title']}](<{chunk['url']}>)"
        if link not in links:
            links.append(link)
    return "Sources: " + " · ".join(links[:MAX_SOURCES])


def answer(question, guild_id=None):
    """Run the full RAG pipeline. Returns the reply text plus what the worker logs."""
    chunks = retrieve(question, guild_id)
    result = {"text": NO_ANSWER_MESSAGE, "answered": False, "sources": [], "input_tokens": 0, "output_tokens": 0}
    if not chunks:
        return result

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        max_completion_tokens=MAX_ANSWER_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt(chunks)},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        **CHAT_OPTIONS,
    )
    result["input_tokens"] = response.usage.prompt_tokens
    result["output_tokens"] = response.usage.completion_tokens

    reply = (response.choices[0].message.content or "").strip()
    if not reply or NO_ANSWER in reply:
        return result

    text, used = split_sources(reply, chunks)
    links = source_links(used)
    result["text"] = f"{text[:MAX_REPLY_CHARS - len(links) - 2]}\n\n{links}"
    result["answered"] = True
    result["sources"] = list(dict.fromkeys(passage_name(chunk) for chunk in used))[:MAX_SOURCES]
    return result


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "How do I tame an Ankylosaurus?",
        "What does a Rex Saddle cost to craft?",
        "Which armor set gives the most cold protection?",
        "How do I bake a chocolate cake?",
    ]
    for question in questions:
        result = answer(question)
        print(f"Q: {question}")
        print(f"A: {result['text']}")
        print(f"   [{result['input_tokens']} in / {result['output_tokens']} out]\n")
