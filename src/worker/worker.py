"""Processing Lambda: runs the RAG pipeline and posts the answer back to Discord.

Invoked asynchronously by the handler Lambda (main.py). The event is the raw Discord
interaction payload. We check the user's daily quota, answer the question, then PATCH
the deferred "thinking" message with the result via Discord's follow-up webhook.
"""

import json
import time

import requests
from openai import RateLimitError

import quota
from generator import answer

DISCORD_API = "https://discord.com/api/v10"
MAX_QUESTION_CHARS = 300

BUSY_MESSAGE = "I'm getting a lot of questions right now. Please ask again in a minute."
ERROR_MESSAGE = "Something went wrong answering that. Please try again in a moment."


def extract_question(interaction):
    for option in interaction.get("data", {}).get("options", []):
        if option.get("name") == "question":
            return option.get("value", "")[:MAX_QUESTION_CHARS]
    return ""


def who_asked(interaction):
    """Guild interactions carry `member` (with roles). DMs carry only `user`."""
    member = interaction.get("member") or {}
    user = member.get("user") or interaction.get("user") or {}
    return interaction.get("guild_id", "dm"), user.get("id", "unknown"), member.get("roles", [])


def send_followup(application_id, token, content):
    url = f"{DISCORD_API}/webhooks/{application_id}/{token}/messages/@original"
    response = requests.patch(url, json={"content": content}, timeout=20)
    if not response.ok:
        print(f"discord followup failed: {response.status_code} {response.text[:200]}")


def lambda_handler(event, context):
    started = time.time()
    question = extract_question(event)
    guild_id, user_id, role_ids = who_asked(event)
    log = {"event": "ask", "guild": guild_id, "user": user_id, "question": question, "outcome": "answered"}

    try:
        limit = quota.daily_limit(guild_id, role_ids)
        used = quota.count_question(guild_id, user_id)
        log.update(used=used, limit=limit)
        if not question:
            reply, log["outcome"] = "I didn't catch a question.", "empty"
        elif used > limit:
            reply = f"You've used all {limit} of your questions for today. They reset at midnight UTC."
            log["outcome"] = "over_quota"
        else:
            result = answer(question)
            reply = result.pop("text")
            log.update(result)
            if not result["answered"]:
                log["outcome"] = "no_answer"
    except RateLimitError:
        reply, log["outcome"] = BUSY_MESSAGE, "rate_limited"
    except Exception as error:
        reply, log["outcome"] = ERROR_MESSAGE, "error"
        log["error"] = repr(error)

    send_followup(event["application_id"], event["token"], reply)
    log["seconds"] = round(time.time() - started, 2)
    print(json.dumps(log))  # one line per question in CloudWatch, the raw data for usage reports
    return {"ok": True}
