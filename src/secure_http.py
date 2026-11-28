#!/usr/bin/env python3
"""
ASGI-middleware: принимает зашифрованные конверты и отвечает ими же.

Правило включения: запросы с `X-Secure: 1` расшифровываются всегда (любой
путь). Незашифрованные запросы к API воркера извне loopback отклоняются —
иначе шифрование было бы добровольным. Локальные клиенты (панель, кошелёк
на той же машине) работают как раньше: они не в сети.
"""

import json
import threading

import secure

# Пути, которые обязаны быть зашифрованы при обращении не с loopback.
#
# /submit-state здесь НЕТ намеренно: это многогигабайтный поток
# (-RESTART.wfn), конверт вокруг него невозможен без буферизации файла в
# памяти. Вместо шифрования маршрут закрыт одноразовым тикетом, который
# выдаёт /state-ticket — а он в списке, то есть тикет выдаётся ТОЛЬКО по
# зашифрованному запросу. Сам wfn не секретен: целостность проверяется
# state_sha256, а подделать содержимое бесполезно — мастер всё равно
# пересчитывает SCF от этой плотности.
SECURE_REQUIRED_PREFIXES = (
    "/register-challenge", "/register-worker", "/get-task", "/heartbeat",
    "/submit-result", "/state-ticket", "/work/height",
)
# Маршруты, где вместо конверта работает одноразовый тикет (см. выше).
# Нужен, чтобы это решение было видно в одном месте, а не только в коде.
TICKET_AUTHORIZED_PATHS = ("/submit-state",)
LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}


# Конверт — это JSON небольшого размера. /submit-result может нести
# устаревший base64-состояния, для него лимит выше.
CONVERT_MAX_BYTES = 8 * 1024 * 1024
CONVERT_MAX_BYTES_BIG = 96 * 1024 * 1024


class SecureASGIMiddleware:
    def __init__(self, app, seed_provider, require: bool = True):
        self.app = app
        self.seed_provider = seed_provider      # () -> bytes (32)
        self.require = require
        self.guard = secure.ReplayGuard()
        self._lock = threading.Lock()
        self._priv = None
        self._own_pub = None

    # ---------------------------------------------------------------- keys
    def _keys(self):
        with self._lock:
            if self._priv is None:
                seed = self.seed_provider()
                self._priv, self._own_pub = secure.transport_keypair(seed)
            return self._priv, self._own_pub

    # ---------------------------------------------------------------- asgi
    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}
        path = scope.get("path") or "/"
        client = (scope.get("client") or ("", 0))[0]
        encrypted = headers.get("x-secure") == "1"

        if not encrypted:
            if self.require and client not in LOOPBACK and \
                    any(path.startswith(p) for p in SECURE_REQUIRED_PREFIXES):
                return await _reply(send, 426,
                                    {"detail": "нужен зашифрованный запрос "
                                               "(X-Secure: 1); см. src/secure.py"})
            return await self.app(scope, receive, send)

        # ---- расшифровка запроса (с пределом до чтения всего тела)
        cap = (CONVERT_MAX_BYTES_BIG if path.rstrip("/").endswith("/submit-result")
               else CONVERT_MAX_BYTES)
        declared = headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > cap:
            return await _reply(send, 413, {"detail": "конверт слишком большой"})
        body = await _read_body(receive, cap)
        try:
            env = json.loads(body.decode("utf-8"))
        except Exception:
            return await _reply(send, 400, {"detail": "битый конверт"})

        priv, own_pub = self._keys()
        try:
            from_hex, method, meta_path, query, plain = secure.open_request(
                priv, own_pub, env, path, self.guard)
        except ValueError as e:
            return await _reply(send, 413, {"detail": str(e)})
        except Exception as e:  # noqa: BLE001
            return await _reply(send, 403, {"detail": f"конверт: {e}"})
        if meta_path != path:
            return await _reply(send, 400, {"detail": "конверт: путь не совпадает"})

        # ---- подзапрос к настоящему приложению
        inner = dict(scope)
        inner["method"] = method.upper()
        inner["query_string"] = "&".join(
            f"{_q(k)}={_q(v)}" for k, v in query.items()).encode("ascii")
        inner["headers"] = [(k, v) for k, v in inner.get("headers") or []
                            if k.lower() not in (b"content-length",)]
        inner["headers"].append((b"content-length", str(len(plain)).encode()))

        sent = {"status": 500, "headers": [], "body": b""}

        async def inner_receive():
            if sent.get("fed"):
                return {"type": "http.disconnect"}
            sent["fed"] = True
            return {"type": "http.request", "body": plain, "more_body": False}

        async def inner_send(msg):
            if msg["type"] == "http.response.start":
                sent["status"] = msg["status"]
                sent["headers"] = msg.get("headers") or []
            elif msg["type"] == "http.response.body":
                sent["body"] += msg.get("body", b"")
                if not msg.get("more_body"):
                    sent["done"] = True

        await self.app(inner, inner_receive, inner_send)

        # ---- шифрование ответа
        peer_pub = bytes.fromhex(from_hex)
        key = secure.derive_key(priv, peer_pub, peer_pub, own_pub)
        import base64
        payload = json.dumps({
            "status": sent["status"],
            "b": base64.b64encode(sent["body"]).decode("ascii"),
        }, ensure_ascii=False).encode("utf-8")
        sealed = secure.seal(key, payload, secure.aad_bytes(
            own_pub.hex(), from_hex, "RESP", path, int(env.get("ts") or 0)))
        out = {"v": secure.SECURE_VERSION, "from": own_pub.hex(),
               "ts": env.get("ts"), **sealed}
        await _reply(send, 200, out, content_type="application/json")


def _q(value: str) -> str:
    from urllib.parse import quote
    return quote(str(value), safe="")


async def _read_body(receive, max_bytes: int = 8 * 1024 * 1024) -> bytes:
    body = b""
    while True:
        msg = await receive()
        if msg["type"] != "http.request":
            break
        body += msg.get("body", b"")
        if len(body) > max_bytes:
            raise ValueError("тело больше лимита")
        if not msg.get("more_body"):
            break
    return body


async def _reply(send, status: int, obj, content_type: str = "application/json"):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", content_type.encode()),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
