from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey


def verify_signature(public_key: str, signature: str, timestamp: str, body: bytes) -> bool:
    """Confirm a request really came from Discord. Signed message is timestamp + raw body."""
    if not public_key or not signature:
        return False
    try:
        verify_key = VerifyKey(bytes.fromhex(public_key))
        verify_key.verify(timestamp.encode() + body, bytes.fromhex(signature))
        return True
    except (BadSignatureError, ValueError):
        return False
