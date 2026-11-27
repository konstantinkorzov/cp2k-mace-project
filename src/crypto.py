#!/usr/bin/env python3
"""
Криптопримитивы без внешних зависимостей.

  * Ed25519 (RFC 8032) — подпись и проверка. Реализация эталонная, медленная,
    но для нескольких транзакций на блок её достаточно.
  * Шифрование приватных ключей: scrypt (hashlib) для KDF + HMAC-SHA256
    в режиме счётчика (encrypt-then-MAC). Формат: nonce(16) | tag(32) | ciphertext.
  * Хэш пароля: scrypt(password, salt).

Только stdlib: hashlib, hmac, os, secrets.
"""

import hashlib
import hmac
import os

# ============================================================
# Ed25519 (RFC 8032, reference implementation)
# ============================================================
_p = 2 ** 255 - 19
_q = 2 ** 252 + 27742317777372353535851937790883648493


def _H(m: bytes) -> bytes:
    return hashlib.sha512(m).digest()


def _inv(x: int) -> int:
    return pow(x, _p - 2, _p)


_d = -121665 * _inv(121666) % _p
_I = pow(2, (_p - 1) // 4, _p)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * _inv(_d * y * y + 1)
    x = pow(xx, (_p + 3) // 8, _p)
    if (x * x - xx) % _p != 0:
        x = (x * _I) % _p
    if x % 2 != 0:
        x = _p - x
    return x


_By = 4 * _inv(5) % _p
_Bx = _xrecover(_By)
_B = (_Bx % _p, _By % _p, 1, (_Bx * _By) % _p)


def _edwards_add(P, Q):
    x1, y1, z1, t1 = P
    x2, y2, z2, t2 = Q
    a = (y1 - x1) * (y2 - x2) % _p
    b = (y1 + x1) * (y2 + x2) % _p
    c = t1 * 2 * _d * t2 % _p
    dd = z1 * 2 * z2 % _p
    e = b - a
    f = dd - c
    g = dd + c
    h = b + a
    return (e * f % _p, g * h % _p, f * g % _p, e * h % _p)


def _edwards_double(P):
    x1, y1, z1, _ = P
    a = x1 * x1 % _p
    b = y1 * y1 % _p
    c = 2 * z1 * z1 % _p
    h = a + b
    e = h - (x1 + y1) * (x1 + y1) % _p
    g = a - b
    f = c + g
    return (e * f % _p, g * h % _p, f * g % _p, e * h % _p)


def _scalarmult(P, e: int):
    if e == 0:
        return (0, 1, 1, 0)
    Q = _scalarmult(P, e // 2)
    Q = _edwards_double(Q)
    if e & 1:
        Q = _edwards_add(Q, P)
    return Q


def _encode_point(P) -> bytes:
    x, y, z, _ = P
    zi = _inv(z)
    x = x * zi % _p
    y = y * zi % _p
    bits = [(y >> i) & 1 for i in range(255)] + [x & 1]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(32))


def _decode_point(s: bytes):
    if len(s) != 32:
        raise ValueError("point must be 32 bytes")
    y = int.from_bytes(s, "little") & ((1 << 255) - 1)
    x = _xrecover(y)
    if x & 1 != (s[31] >> 7):
        x = _p - x
    P = (x, y, 1, x * y % _p)
    if (-x * x + y * y - 1 - _d * x * x * y * y) % _p != 0:
        raise ValueError("point is not on curve")
    return P


def public_key_from_seed(seed: bytes) -> bytes:
    if len(seed) != 32:
        raise ValueError("seed must be 32 bytes")
    h = _H(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return _encode_point(_scalarmult(_B, a))


def sign(seed: bytes, msg: bytes) -> bytes:
    h = _H(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    A = _encode_point(_scalarmult(_B, a))
    r = int.from_bytes(_H(h[32:] + msg), "little") % _q
    R = _scalarmult(_B, r)
    Rs = _encode_point(R)
    S = (r + int.from_bytes(_H(Rs + A + msg), "little") * a) % _q
    return Rs + S.to_bytes(32, "little")


def _is_small_order(encoded: bytes) -> bool:
    """Точка малого порядка (в т.ч. единица) — источник подделок в наивных
    реализациях Ed25519. Проверяем [8]P == identity."""
    try:
        P = _decode_point(encoded)
    except Exception:  # noqa: BLE001
        return True
    Q = P
    for _ in range(3):            # 2P, 4P, 8P
        Q = _edwards_add(Q, Q)
    # Точки в расширенных координатах (X : Y : Z : T), identity — это
    # X == 0 и Y == Z (а не только (0, 1)): без нормировки по Z проверка
    # не срабатывала, и identity проходила как валидный публичный ключ.
    return Q[0] % _p == 0 and (Q[1] - Q[2]) % _p == 0


def verify(public_key: bytes, msg: bytes, signature: bytes) -> bool:
    if len(signature) != 64 or len(public_key) != 32:
        return False
    try:
        # R и A малого порядка отбрасываем до арифметики: иначе возможны
        # подписи, валидные для нескольких сообщений (классическая ошибка).
        if _is_small_order(signature[:32]) or _is_small_order(public_key):
            return False
        R = _decode_point(signature[:32])
        S = int.from_bytes(signature[32:], "little")
        if S >= _q:
            return False
        A = _decode_point(public_key)
        h = int.from_bytes(_H(signature[:32] + public_key + msg), "little") % _q
        return _encode_point(_scalarmult(_B, S)) == \
            _encode_point(_edwards_add(R, _scalarmult(A, h)))
    except Exception:  # noqa: BLE001
        return False


# ============================================================
# KDF и шифрование приватных ключей
# ============================================================
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LEN = 32
NONCE_LEN = 16
TAG_LEN = 32


def kdf(password: str, salt: bytes, length: int = KEY_LEN) -> bytes:
    """scrypt(password, salt) — для хэша пароля и ключа шифрования."""
    return hashlib.scrypt(password.encode("utf-8"), salt=salt,
                          n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=length)


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Возвращает (salt, hash). Соль случайная — при одинаковом пароле хэши разные."""
    salt = salt or os.urandom(16)
    return salt, kdf(password, salt)


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = b""
    counter = 0
    while len(out) < length:
        out += hashlib.sha256(key + nonce + counter.to_bytes(8, "big")).digest()
        counter += 1
    return out[:length]


def encrypt(key: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
    """nonce | tag | ciphertext (encrypt-then-MAC)."""
    nonce = os.urandom(NONCE_LEN)
    stream = _keystream(key, nonce, len(plaintext))
    ct = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag = hmac.new(key, aad + nonce + ct, hashlib.sha256).digest()
    return nonce + tag + ct


def decrypt(key: bytes, blob: bytes, aad: bytes = b"") -> bytes:
    if len(blob) < NONCE_LEN + TAG_LEN:
        raise ValueError("ciphertext too short")
    nonce, tag, ct = blob[:NONCE_LEN], blob[NONCE_LEN:NONCE_LEN + TAG_LEN], blob[NONCE_LEN + TAG_LEN:]
    expect = hmac.new(key, aad + nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expect):
        raise ValueError("bad tag (wrong password or corrupted key)")
    stream = _keystream(key, nonce, len(ct))
    return bytes(a ^ b for a, b in zip(ct, stream))


def address_from_public(public_key: bytes) -> str:
    return hashlib.sha256(b"cp2k-addr-v1|" + public_key).hexdigest()


if __name__ == "__main__":
    # Тестовый вектор RFC 8032, TEST 1
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    pub = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    sig = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555f"
        "b8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
    assert public_key_from_seed(seed) == pub, "pubkey mismatch"
    assert sign(seed, b"") == sig, "signature mismatch"
    assert verify(pub, b"", sig), "verify failed"
    assert not verify(pub, b"x", sig), "verify should fail"
    print("Ed25519 OK (RFC 8032 TEST 1)")
