"""Generate a grounded answer from retrieved wiki chunks using gpt-4o-mini."""

import sys

from openai import OpenAI

from retriever import retrieve

CHAT_MODEL = "gpt-4o-mini"
NO_CONTEXT_MESSAGE = (
    "I couldn't find anything in the ARK wiki to answer that. Try rephrasing, "
    "or ask about a specific creature."
)
SYSTEM_PROMPT = (
    "You are RAGnarok, an assistant for the game Ark: Survival Ascended. "
    "Answer the user's question using ONLY the context provided. "
    "If the context does not contain the answer, say you don't know rather than guessing. "
    "Be concise and specific."
)

openai_client = OpenAI()


def build_prompt(query, chunks):
    context = "\n\n".join(f"[{chunk['title']}] {chunk['text']}" for chunk in chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def generate_answer(query, chunks):
    if not chunks:
        return NO_CONTEXT_MESSAGE
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, chunks)},
        ],
    )
    return response.choices[0].message.content.strip()


def answer(question):
    return generate_answer(question, retrieve(question))


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "How do I tame an Ankylosaurus?",
        "What is the Giganotosaurus known for?",
        "What does the Argentavis eat?",
        "How do I bake a chocolate cake?",
    ]
    for question in questions:
        print(f"Q: {question}")
        print(f"A: {answer(question)}\n")
