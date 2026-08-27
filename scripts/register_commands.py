"""Register the /ask slash command with Discord. Run locally: python scripts/register_commands.py"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.environ.get("DISCORD_APP_ID")
BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
GUILD_ID = os.environ.get("DISCORD_GUILD_IiD")

if not APP_ID or not BOT_TOKEN:
    sys.exit("Set DISCORD_APP_ID and DISCORD_BOT_TOKEN in your environment/.env first.")

command = {
    "name": "ask",
    "description": "Ask a question about: Ark Survival Ascended",
    "type": 1,
    "options": [{
            "name": "question",
            "description": "Your question about ASA",
            "type": 3,
            "required": True
        }
    ],
}

# A guild-scoped command updates instantly; a global one can take ~1 hour.
if GUILD_ID:
    url = f"https://discord.com/api/v10/applications/{APP_ID}/guilds/{GUILD_ID}/commands"
else:
    url = f"https://discord.com/api/v10/applications/{APP_ID}/commands"

response = requests.post(
    url,
    headers={"Authorization": f"Bot {BOT_TOKEN}"},
    json=command,
    timeout=15,
)
response.raise_for_status()
print(response.status_code, response.json())
