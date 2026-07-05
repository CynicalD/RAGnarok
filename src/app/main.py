"""Discord interactions endpoint and the Lambda entrypoint (the fast front door).

Verifies the request, answers PING with PONG, and for a slash command fires the worker
Lambda asynchronously, then immediately returns a deferred ack. The async invoke MUST
happen before returning: once this function returns, API Gateway ends its execution.
"""

import json
import os

import config
from fastapi import FastAPI, HTTPException, Request, status
from mangum import Mangum
from models import Interaction, InteractionResponseType, InteractionType
from verification import verify_signature

app = FastAPI(title="RAGnarok")


def invoke_worker(interaction):
    import boto3

    boto3.client("lambda").invoke(
        FunctionName=os.environ["WORKER_FUNCTION_NAME"],
        InvocationType="Event",  # async: returns immediately, worker runs on its own
        Payload=json.dumps(interaction).encode(),
    )


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

    invoke_worker(json.loads(body))
    return {"type": InteractionResponseType.DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE}


handler = Mangum(app)
