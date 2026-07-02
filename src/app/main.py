"""Discord interactions endpoint and the Lambda entrypoint."""

import config
from fastapi import FastAPI, HTTPException, Request, status
from mangum import Mangum
from models import Interaction, InteractionResponseType, InteractionType
from verification import verify_signature

app = FastAPI(title="RAGnarok")


@app.post("/interactions")
async def interactions(request: Request):
    signature = request.headers.get("X-Signature-Ed25519")
    timestamp = request.headers.get("X-Signature-Timestamp")
    body = await request.body()  # raw bytes: the signature is computed over these

    if (
        signature is None
        or timestamp is None
        or not verify_signature(config.DISCORD_PUBLIC_KEY, signature, timestamp, body)
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid request signature")

    interaction = Interaction.model_validate_json(body)

    if interaction.type == InteractionType.PING:
        return {"type": InteractionResponseType.PONG}

    return {"type": InteractionResponseType.DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE}


handler = Mangum(app)
