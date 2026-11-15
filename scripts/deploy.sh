#!/bin/bash
# Build both Lambdas for Linux arm64 and deploy the stack. Needs Docker running.
# The API keys are read from .env at run time, so they never land in shell history.
set -euo pipefail
cd "$(dirname "$0")/.."

read_key() { grep "^$1=" .env | cut -d= -f2- | tr -d ' '; }

sam build --use-container
sam deploy --no-confirm-changeset --no-fail-on-empty-changeset \
  --parameter-overrides \
    OpenAIApiKey="$(read_key OPENAI_API_KEY)" \
    PineconeApiKey="$(read_key PINECONE_API_KEY)"
