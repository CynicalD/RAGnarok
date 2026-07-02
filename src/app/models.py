"""Discord protocol enums and the incoming interaction payload."""

from enum import IntEnum

from pydantic import BaseModel


class InteractionType(IntEnum):
    PING = 1
    APPLICATION_COMMAND = 2
    MESSAGE_COMPONENT = 3
    APPLICATION_COMMAND_AUTOCOMPLETE = 4
    MODAL_SUBMIT = 5


class InteractionResponseType(IntEnum):
    PONG = 1
    CHANNEL_MESSAGE_WITH_SOURCE = 4
    DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE = 5


class Interaction(BaseModel):
    type: InteractionType
    id: str | None = None
    token: str | None = None
    data: dict | None = None
