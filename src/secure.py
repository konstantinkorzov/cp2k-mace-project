#!/usr/bin/env python3
"""
Транспортное шифрование между узлами: мастер ↔ воркер, мастер ↔ мастер.

HTTPS не используем (сертификаты, CA, продление — лишняя возня), а свой TLS не
пишем. Вместо этого — прикладной слой поверх обычного HTTP:

  ключи        X25519 (статический обмен), выведены из seed кошелька;
  согласование X25519(свой приватный, ключ пира) → HKDF-SHA256 → ключ сессии;
  шифр         AES-256-GCM (аутентифицированное шифрование, nonce на сообщение);
  привязка     AAD = версия | от кого | кому | метод | путь | время;
  защита       метка времени ±120 с + кэш nonce (повтор не пройдёт).

Что шифруется: управляющие сообщения (регистрация/сессии, задания, приём
результата, синхронизация высот/анонсы). Большие бинарные данные (блоки,
файлы состояния) идут открыто: они не секретны, а целостность у них уже есть —
`block_id`/`tx_root` и `state_sha256`. Сессия воркера при загрузке .wfn
передаётся одноразовым тикетом, чтобы не светить её в URL.

Доверие к ключу мастера — TOFU: при первом обращении ключ запоминается
(как в SSH), при смене — отказ, пока не удалить запись вручную.
"""

import base64
import hashlib
import json
import os
import re
import secrets
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import crypto

SECURE_VERSION = 1
AAD_PREFIX = b"cp2k-secure-v1"
HELLO_SIGN_PREFIX = b"cp2k-secure-hello-v1"
CLOCK_SKEW_SEC = 120
REPLAY_TTL_SEC = 300
PUB_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------- ключи
def transport_keypair(seed: bytes) -> tuple[bytes, bytes]:
    """(приватный, публичный) X25519 из seed кошелька (32 байта)."""
    if len(seed) != 32:
        raise ValueError("seed должен быть 32 байта")
    raw = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
               info=b"cp2k-transport-x25519-v1").derive(seed)
    priv = X25519PrivateKey.from_private_bytes(raw)
    pub = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    return raw, pub


def transport_public(seed: bytes) -> bytes:
    return transport_keypair(seed)[1]


def _priv_obj(raw: bytes) -> X25519PrivateKey:
    return X25519PrivateKey.from_private_bytes(raw)


def derive_key(priv: bytes, peer_pub: bytes,
               from_pub: bytes, to_pub: bytes) -> bytes:
    """Общий ключ сессии, привязанный к паре участников."""
    if not PUB_RE.match(peer_pub.hex()):
        raise ValueError("плохой публичный ключ пира")
    ss = _priv_obj(priv).exchange(X25519PublicKey.from_public_bytes(peer_pub))
    # порядок участников не должен влиять: соль симметрична (сортировка)
    pair = sorted([from_pub.hex(), to_pub.hex()])
    salt = hashlib.sha256((pair[0] + pair[1]).encode("ascii")).digest()
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt,
                info=b"cp2k-secure-session-v1").derive(ss)


def hello_message(pub: bytes) -> bytes:
    return HELLO_SIGN_PREFIX + b"|" + pub.hex().encode("ascii")


def hello_payload(seed: bytes) -> dict:
    """Публичный пакет узла: X25519-ключ + Ed25519-подпись на него."""
    _priv, pub = transport_keypair(seed)
    ed_pub = crypto.public_key_from_seed(seed)
    sig = crypto.sign(seed, hello_message(pub))
    return {"v": SECURE_VERSION, "x": pub.hex(),
            "ed": ed_pub.hex(), "sig": sig.hex()}


def verify_hello(payload: dict) -> str | None:
    """Проверяет самоподпись hello. Возвращает None или причину отказа."""
    try:
        pub = bytes.fromhex(str(payload.get("x") or ""))
        ed_pub = bytes.fromhex(str(payload.get("ed") or ""))
        sig = bytes.fromhex(str(payload.get("sig") or ""))
    except ValueError:
        return "hello: не hex"
    if len(pub) != 32 or len(ed_pub) != 32 or len(sig) != 64:
        return "hello: неверная длина"
    if not crypto.verify(ed_pub, hello_message(pub), sig):
        return "hello: подпись неверна"
    return None


# --------------------------------------------------------------- конверт
def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text) -> bytes:
    return base64.b64decode(str(text or ""), validate=True)


def aad_bytes(from_hex: str, to_hex: str, method: str, path: str, ts: int) -> bytes:
    return (AAD_PREFIX + b"|" + from_hex.encode("ascii") + b"|"
            + to_hex.encode("ascii") + b"|" + method.upper().encode("ascii")
            + b"|" + path.encode("utf-8") + b"|" + str(int(ts)).encode("ascii"))


def seal(key: bytes, plaintext: bytes, aad: bytes) -> dict:
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext, aad)
    return {"n": _b64(nonce), "c": _b64(ct)}


def open_sealed(key: bytes, sealed: dict, aad: bytes) -> bytes:
    nonce = _unb64(sealed.get("n"))
    ct = _unb64(sealed.get("c"))
    if len(nonce) != 12 or not ct:
        raise ValueError("конверт: неверная длина")
    return AESGCM(key).decrypt(nonce, ct, aad)


def seal_request(seed: bytes, peer_pub: bytes, method: str, path: str,
                 query: dict | None = None, body: bytes | None = None,
                 ts: int | None = None) -> dict:
    """
    Конверт запроса: всё (метод, путь, query, тело) внутри шифра.

    По проводу конверт всегда уходит POST-ом на тот же путь, поэтому AAD
    привязан к "POST" и пути — их знают обе стороны до расшифровки.
    """
    _priv, own_pub = transport_keypair(seed)
    ts = int(ts if ts is not None else time.time())
    key = derive_key(_priv, peer_pub, own_pub, peer_pub)
    meta = {"m": method.upper(), "p": path,
            "q": {str(k): str(v) for k, v in (query or {}).items()},
            "b": (body or b"").decode("utf-8", "surrogateescape")}
    sealed = seal(key, json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                  aad_bytes(own_pub.hex(), peer_pub.hex(), "POST", path, ts))
    return {"v": SECURE_VERSION, "from": own_pub.hex(), "ts": ts, **sealed}


class ReplayGuard:
    """Кэш (от кого, nonce) на REPLAY_TTL_SEC: повторный конверт не пройдёт."""

    def __init__(self, ttl: int = REPLAY_TTL_SEC):
        self.ttl = ttl
        self._seen: dict[tuple[str, str], float] = {}
        self._lock = None  # ленивый threading.Lock

    def check(self, from_hex: str, nonce_b64: str, ts: int) -> str | None:
        import threading
        if self._lock is None:
            self._lock = threading.Lock()
        now = time.time()
        key = (from_hex, nonce_b64)
        with self._lock:
            if len(self._seen) > 100_000:
                for k, t in list(self._seen.items()):
                    if now - t > self.ttl:
                        self._seen.pop(k, None)
            if key in self._seen:
                return "конверт: повтор"
            self._seen[key] = now
        return None


def open_request(priv: bytes, own_pub: bytes, env: dict, path: str,
                 guard: ReplayGuard | None = None):
    """
    Расшифровывает конверт. Возвращает (from_hex, method, path, query, body).
    Бросает ValueError с причиной — вызывающий отвечает 400/403.
    """
    from_hex = str(env.get("from") or "").lower()
    if not PUB_RE.match(from_hex):
        raise ValueError("конверт: нет ключа отправителя")
    try:
        ts = int(env.get("ts"))
    except (TypeError, ValueError):
        raise ValueError("конверт: нет времени")
    if abs(time.time() - ts) > CLOCK_SKEW_SEC:
        raise ValueError("конверт: время вне допуска")
    if guard is not None:
        why = guard.check(from_hex, str(env.get("n") or ""), ts)
        if why:
            raise ValueError(why)
    peer_pub = bytes.fromhex(from_hex)
    key = derive_key(priv, peer_pub, peer_pub, own_pub)
    raw = open_sealed(key, env, aad_bytes(from_hex, own_pub.hex(), "POST", path, ts))
    meta = json.loads(raw.decode("utf-8"))
    return (from_hex, str(meta.get("m") or "GET"), str(meta.get("p") or "/"),
            dict(meta.get("q") or {}), str(meta.get("b") or "").encode(
                "utf-8", "surrogateescape"))


# --------------------------------------------------------------- доверие (TOFU)
class TrustStore:
    """Запомненные ключи узлов: как known_hosts в SSH."""

    def __init__(self, path: str):
        self.path = path
        self._cache: dict[str, str] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._cache = {str(k): str(v) for k, v in data.items()}
        except Exception:
            self._cache = {}

    def get(self, host: str) -> str | None:
        return self._cache.get(host)

    def pin(self, host: str, pub_hex: str, allow_change: bool = False) -> str | None:
        """None — ок; иначе причина отказа (ключ сменился)."""
        old = self._cache.get(host)
        if old and old != pub_hex and not allow_change:
            return (f"ключ {host} изменился: был {old[:12]}…, стал {pub_hex[:12]}…; "
                    f"если это правда, удалите строку из {self.path}")
        if old != pub_hex:
            self._cache[host] = pub_hex
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.path)
            except Exception:
                pass
        return None
