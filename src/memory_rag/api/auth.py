"""可信宿主签发的单一用户/环境服务凭据；与 Java wac1 协议一致。"""
import base64
import hashlib
import hmac
import json
import re

from starlette.responses import JSONResponse


def verify_scope(token: str, key: str) -> tuple[str, int] | None:
    if not key or len(token) > 4096:
        return None
    try:
        version, payload, signature = token.split(".")
        if version != "wac1":
            return None
        expected = hmac.new(key.encode(), f"wac1.{payload}".encode("ascii"), hashlib.sha256).digest()
        actual = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
        if not hmac.compare_digest(expected, actual):
            return None
        audience, user, env = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if audience != "memory-rag" or not isinstance(user, str) or not user.strip() or type(env) is not int or env < 1:
            return None
        return user, env
    except (ValueError, TypeError, UnicodeError):
        return None


class ScopedMemoryAuth:
    def __init__(self, app, signing_key: str):
        if signing_key and len(signing_key) < 32:
            raise ValueError("作用域签名密钥至少需要 32 字符")
        self.app = app
        self.signing_key = signing_key

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.signing_key or scope["path"] in {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}:
            await self.app(scope, receive, send)
            return
        authorization = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
        token = authorization[7:] if authorization.startswith("Bearer ") else ""
        owner = verify_scope(token, self.signing_key)
        if owner is None:
            await JSONResponse({"detail": "需要有效的记忆服务凭据"}, status_code=401)(scope, receive, send)
            return
        match = re.fullmatch(r"/users/([^/]+)/environments/([1-9][0-9]*)/memories(?:/[^;]*)?", scope["path"])
        if not match or "%" in scope["path"] or owner != (match[1], int(match[2])):
            await JSONResponse({"detail": "凭据不允许该用户环境"}, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)
