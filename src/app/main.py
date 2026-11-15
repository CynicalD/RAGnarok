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
from pydantic import ValidationError
from verification import verify_signature

app = FastAPI(title="RAGnarok")
lambda_client = None


def invoke_worker(interaction):
    # Built on first use and kept for warm invocations. Importing boto3 at the top
    # would slow down every cold start, including Discord's PING check.
    global lambda_client
    if lambda_client is None:
        import boto3

        lambda_client = boto3.client("lambda")

    lambda_client.invoke(
        FunctionName=os.environ["WORKER_FUNCTION_NAME"],
        InvocationType="Event",  # async: returns immediately, worker runs on its own
        Payload=json.dumps(interaction).encode(),  # the data sent to the worker
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

    try:
        interaction = Interaction.model_validate_json(body)
    except ValidationError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="malformed interaction")

    if interaction.type == InteractionType.PING:
        return {"type": InteractionResponseType.PONG}

    invoke_worker(json.loads(body))
    return {"type": InteractionResponseType.DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE}


handler = Mangum(app)
