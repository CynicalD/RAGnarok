import os

# dotenv is a local-only dependency; in Lambda the env vars are already set.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

DISCORD_PUBLIC_KEY = os.environ.get("DISCORD_PUBLIC_KEY", "")
