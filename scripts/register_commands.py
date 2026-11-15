"""Register the /ask slash command with Discord. Run locally: python scripts/register_commands.py

The command is registered globally, so it appears in every server the bot is added to.
Re-running overwrites the existing definition. Changes can take a while to show up.
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.environ.get("DISCORD_APP_ID")
BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")

if not APP_ID or not BOT_TOKEN:
    sys.exit("Set DISCORD_APP_ID and DISCORD_BOT_TOKEN in your environment/.env first.")

command = {
    "name": "ask",
    "description": "Ask about Ark: Survival Ascended creatures, weapons, armor and mechanics",
    "type": 1,
    # Servers only, no DMs: daily quotas depend on the member's roles in a server.
    "integration_types": [0],
    "contexts": [0],
    "options": [{
            "name": "question",
            "description": "Your question about ASA",
            "type": 3,
            "required": True,
            "max_length": 300,
        }
    ],
}

response = requests.post(
    f"https://discord.com/api/v10/applications/{APP_ID}/commands",
    headers={"Authorization": f"Bot {BOT_TOKEN}"},
    json=command,
    timeout=15,
)
response.raise_for_status()
print(response.status_code, response.json())
