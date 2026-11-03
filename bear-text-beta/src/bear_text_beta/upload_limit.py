"""ASGI request-size guard before multipart is spooled by FastAPI."""

from __future__ import annotations

import json


class RequestSizeLimitMiddleware:
    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def reject():
            body = json.dumps({"error": "PayloadTooLarge", "detail": "HTTP body exceeds limit"}).encode()
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})

        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if declared > self.max_body_bytes:
            return await reject()

        total = 0

        class BodyTooLarge(Exception):
            pass

        async def limited_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_body_bytes:
                    raise BodyTooLarge
            return message

        try:
            return await self.app(scope, limited_receive, send)
        except BodyTooLarge:
            return await reject()
