"""Daily question quota per user, counted in DynamoDB.

A user's limit comes from their Discord roles, so a server's paid ranks can each carry
a bigger allowance. Discord sends the member's role IDs with every guild interaction,
which means no accounts, logins or billing data are needed: the role is the proof.
"""

import os
import time
from datetime import datetime, timezone

DEFAULT_DAILY_LIMIT = 10

# guild ID -> {role ID -> questions per day}. Role IDs are unique to each server, so a
# server's entry is filled in when it is onboarded. A member with several listed roles
# gets the highest limit. Anyone without a listed role gets DEFAULT_DAILY_LIMIT.
GUILD_ROLE_LIMITS = {
    # "123456789012345678": {
    #     "111111111111111111": 15,   # Apprentice
    #     "222222222222222222": 30,   # Journeyman
    #     "333333333333333333": 50,   # Mastercraft
    #     "444444444444444444": 100,  # Ascendant
    # },
}

_table = None


def daily_limit(guild_id, role_ids):
    role_limits = GUILD_ROLE_LIMITS.get(guild_id, {})
    limits = [role_limits[role_id] for role_id in role_ids if role_id in role_limits]
    return max(limits, default=DEFAULT_DAILY_LIMIT)


def usage_table():
    global _table
    if _table is None:
        import boto3  # ships with the Lambda runtime, not needed for local runs

        _table = boto3.resource("dynamodb").Table(os.environ["USAGE_TABLE"])
    return _table


def count_question(guild_id, user_id):
    """Add one to this user's count for today and return the new total.

    ADD is atomic, so two questions sent in the same instant can't both read the old
    count. Each day gets its own item, and the TTL lets DynamoDB delete it afterwards.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    response = usage_table().update_item(
        Key={"pk": f"{guild_id}#{user_id}#{today}"},
        UpdateExpression="ADD used :one SET expires_at = :expires",
        ExpressionAttributeValues={":one": 1, ":expires": int(time.time()) + 2 * 86400},
        ReturnValues="UPDATED_NEW",
    )
    return int(response["Attributes"]["used"])
