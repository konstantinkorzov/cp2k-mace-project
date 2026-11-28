#!/usr/bin/env python3
"""
HTTP-клиент узла: говорит с мастером зашифрованными конвертами.

API повторяет requests настолько, насколько нужно вызывающему коду:
`.get(url, params=..., timeout=...)`, `.post(url, params=..., json=...,
timeout=...)`, объект ответа с `.status_code`, `.json()`, `.text`,
`.raise_for_status()`.

Ключ мастера запоминается при первом обращении (TOFU, как known_hosts):
если он потом изменится — запрос не пойдёт, пока запись не удалят.
"""

import base64
import hashlib
import json
from urllib.parse import urlsplit, urlencode, quote

import requests

import secure


class SecureError(RuntimeError):
    pass


class SecureResponse:
    def __init__(self, status: int, body: bytes):
        self.status_code = status
        self.content = body
        self.headers = {}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")

    def json(self):
        return json.loads(self.content.decode("utf-8"))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTP {self.status_code}", response=self)


class SecureClient:
    def __init__(self, seed: bytes, trust_path: str, timeout: int = 30,
                 transport=None):
        self.seed = seed
        self.trust = secure.TrustStore(trust_path)
        self.timeout = timeout
        self._peers: dict[str, str] = {}
        self._http = transport or requests

    # ---------------------------------------------------------------- hello
    def peer_key(self, base: str) -> bytes:
        base = base.rstrip("/")
        if base in self._peers:
            return bytes.fromhex(self._peers[base])
        r = self._http.get(base + "/secure/hello", timeout=10)
        r.raise_for_status()
        payload = r.json()
        why = secure.verify_hello(payload)
        if why:
            raise SecureError(why)
        pub = str(payload["x"]).lower()
        if self.trust.get(base) is None:
            self.trust.pin(base, pub)
        else:
            bad = self.trust.pin(base, pub)
            if bad:
                raise SecureError(bad)
        self._peers[base] = pub
        return bytes.fromhex(pub)

    # ---------------------------------------------------------------- core
    def request(self, method: str, url: str, params=None, json_body=None,
                body: bytes | None = None, timeout: int | None = None):
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        path = parts.path or "/"
        query = dict(params or {})
        if parts.query:
            for kv in parts.query.split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    query.setdefault(k, v)
        if json_body is not None:
            body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        env = secure.seal_request(self.seed, self.peer_key(base), method, path,
                                  query=query, body=body)
        r = self._http.post(base + path, json=env, timeout=timeout or self.timeout,
                            headers={"X-Secure": "1"})
        if r.status_code != 200:
            # 426/403 от мастера — это не конверт, а текст ошибки
            return SecureResponse(r.status_code, r.content)
        return self._open(base, r.json(), path, env.get("ts"))

    def _open(self, base: str, env: dict, path: str, ts) -> SecureResponse:
        from_hex = str(env.get("from") or "").lower()
        own_pub = secure.transport_public(self.seed)
        if from_hex != self._peers.get(base.rstrip("/")):
            # ответ должен быть подписан тем же ключом, что мы запомнили
            raise SecureError("ответ пришёл не от того узла")
        priv, _ = secure.transport_keypair(self.seed)
        key = secure.derive_key(priv, bytes.fromhex(from_hex),
                                bytes.fromhex(from_hex), own_pub)
        raw = secure.open_sealed(key, env, secure.aad_bytes(
            from_hex, own_pub.hex(), "RESP", path, int(ts or 0)))
        data = json.loads(raw.decode("utf-8"))
        return SecureResponse(int(data.get("status") or 0),
                              base64.b64decode(data.get("b") or ""))

    # ---------------------------------------------------------------- api
    def get(self, url: str, params=None, timeout=None, **_kw):
        return self.request("GET", url, params=params, timeout=timeout)

    def post(self, url: str, params=None, json=None, data=None, timeout=None,
             headers=None, **_kw):
        if data is not None and not isinstance(data, (bytes, bytearray)):
            raise SecureError("потоковую загрузку делайте через upload_state()")
        return self.request("POST", url, params=params, json_body=json,
                            body=data, timeout=timeout)

    # ---------------------------------------------------------------- upload
    def upload_state(self, url: str, params: dict, fileobj, timeout: int = 1800):
        """
        Загрузка -RESTART.wfn: шифруется только запрос тикета, сам файл идёт
        открытым (он не секретен, целостность проверяется state_sha256), а
        одноразовый тикет не даёт светить сессию в URL.
        """
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        ticket_r = self.request("POST", base + "/state-ticket", params=params)
        ticket_r.raise_for_status()
        ticket = str((ticket_r.json() or {}).get("ticket") or "")
        if not ticket:
            raise SecureError("мастер не выдал тикет")
        r = self._http.post(
            base + "/submit-state",
            params={"ticket": ticket, "worker_id": params.get("worker_id"),
                    "struct_id": params.get("struct_id")},
            data=fileobj, timeout=timeout,
            headers={"Content-Type": "application/octet-stream"})
        return SecureResponse(r.status_code, r.content)
