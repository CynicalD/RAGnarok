"""Processing Lambda: runs the RAG pipeline and posts the answer back to Discord.

Invoked asynchronously by the handler Lambda (main.py). The event is the raw Discord
interaction payload. We answer the question, then PATCH the deferred "thinking" message
with the result via Discord's follow-up webhook.
"""

import requests

from generator import answer

DISCORD_API = "https://discord.com/api/v10"
DISCORD_MAX_CHARS = 2000


def extract_question(interaction):
    for option in interaction.get("data", {}).get("options", []):
        if option.get("name") == "question":
            return option.get("value", "")
    return ""


def send_followup(application_id, token, content):
    url = f"{DISCORD_API}/webhooks/{application_id}/{token}/messages/@original"
    requests.patch(url, json={"content": content[:DISCORD_MAX_CHARS]}, timeout=20)


def lambda_handler(event, context):
    try:
        question = extract_question(event)
        reply = answer(question) if question else "I didn't catch a question."
    except Exception as error:
        print(f"worker error: {error}")  # goes to CloudWatch
        reply = "Something went wrong answering that. Please try again in a moment."
    send_followup(event["application_id"], event["token"], reply)
    return {"ok": True}
