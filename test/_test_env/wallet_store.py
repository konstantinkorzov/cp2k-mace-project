#!/usr/bin/env python3
"""
Единое хранилище кошельков мастера и воркеров (SQLite).

Одна база (по умолчанию results/keys.db) содержит и мастер, и воркеров:

    wallets(id, role, name UNIQUE, password_salt, password_hash, enc_salt, created_at)
    wallet_keys(id, wallet_id, created_at, public_key, address UNIQUE, enc_private)

* Пароль проверяется по scrypt(password, password_salt).
* Приватные ключи (Ed25519 seed) шифруются ключом scrypt(password, enc_salt)
  (encrypt-then-MAC, см. src/crypto.py) — в базе открытого ключа нет.
* Соли случайные, поэтому у мастера и воркера с ОДИНАКОВЫМ паролем разные
  password_hash, разные ключи шифрования и, соответственно, разные адреса.
* Первый пароль по умолчанию — «root»; меняется командой master-cli passwd
  (мастер) или wallet.py (воркер).

Криптография только stdlib (scrypt/HMAC/SHA-256/Ed25519 из crypto.py).
"""

import os
import secrets
import sqlite3
import time

import crypto

DEFAULT_PASSWORD = "root"

SCHEMA = """
CREATE TABLE IF NOT EXISTS wallets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    role          TEXT    NOT NULL,          -- 'master' | 'worker'
    name          TEXT    NOT NULL UNIQUE,
    password_salt BLOB    NOT NULL,
    password_hash BLOB    NOT NULL,
    enc_salt      BLOB    NOT NULL,
    created_at    TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS wallet_keys (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_id  INTEGER NOT NULL REFERENCES wallets(id),
    created_at TEXT    NOT NULL,
    public_key TEXT    NOT NULL,
    address    TEXT    NOT NULL UNIQUE,
    enc_private BLOB   NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_keys_wallet ON wallet_keys(wallet_id);
"""


class WalletError(Exception):
    pass


class Wallet:
    """Разблокированный кошелёк: умеет создавать ключи и подписывать."""

    def __init__(self, conn: sqlite3.Connection, row: sqlite3.Row, password: str):
        self.conn = conn
        self.id = row["id"]
        self.role = row["role"]
        self.name = row["name"]
        self._enc_key = crypto.kdf(password, row["enc_salt"])

    # ---- ключи ----
    def keys(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, created_at, public_key, address FROM wallet_keys "
            "WHERE wallet_id = ? ORDER BY id ASC", (self.id,)).fetchall()
        return [{"id": r["id"], "created_at": r["created_at"],
                 "public_key": r["public_key"], "address": r["address"]} for r in rows]

    def addresses(self) -> set[str]:
        return {k["address"] for k in self.keys()}

    def latest_key(self) -> dict | None:
        rows = self.conn.execute(
            "SELECT id, created_at, public_key, address FROM wallet_keys "
            "WHERE wallet_id = ? ORDER BY id DESC LIMIT 1", (self.id,)).fetchone()
        return None if rows is None else {"id": rows["id"], "created_at": rows["created_at"],
                                          "public_key": rows["public_key"],
                                          "address": rows["address"]}

    def add_key(self) -> dict:
        seed = secrets.token_bytes(32)
        public = crypto.public_key_from_seed(seed)
        address = crypto.address_from_public(public)
        enc = crypto.encrypt(self._enc_key, seed, address.encode("ascii"))
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cur = self.conn.execute(
            "INSERT INTO wallet_keys (wallet_id, created_at, public_key, address, enc_private) "
            "VALUES (?, ?, ?, ?, ?)",
            (self.id, created_at, public.hex(), address, enc))
        self.conn.commit()
        return {"id": cur.lastrowid, "created_at": created_at,
                "public_key": public.hex(), "address": address}

    def private_seed(self, address: str) -> bytes:
        row = self.conn.execute(
            "SELECT enc_private FROM wallet_keys WHERE wallet_id = ? AND address = ?",
            (self.id, address)).fetchone()
        if row is None:
            raise WalletError(f"адрес {address} не принадлежит кошельку")
        return crypto.decrypt(self._enc_key, row["enc_private"], address.encode("ascii"))

    def public_for(self, address: str) -> bytes:
        row = self.conn.execute(
            "SELECT public_key FROM wallet_keys WHERE wallet_id = ? AND address = ?",
            (self.id, address)).fetchone()
        if row is None:
            raise WalletError(f"адрес {address} не принадлежит кошельку")
        return bytes.fromhex(row["public_key"])

    def sign(self, address: str, message: bytes) -> bytes:
        return crypto.sign(self.private_seed(address), message)

    def change_password(self, old: str, new: str):
        row = self.conn.execute("SELECT * FROM wallets WHERE id = ?", (self.id,)).fetchone()
        if not _check_password(row, old):
            raise WalletError("старый пароль неверен")
        # перешифровываем все приватные ключи новым ключом
        new_enc_salt = os.urandom(16)
        new_enc_key = crypto.kdf(new, new_enc_salt)
        pw_salt, pw_hash = crypto.hash_password(new)

        keys = self.conn.execute(
            "SELECT address, enc_private FROM wallet_keys WHERE wallet_id = ?",
            (self.id,)).fetchall()
        for k in keys:
            seed = crypto.decrypt(self._enc_key, k["enc_private"], k["address"].encode("ascii"))
            enc = crypto.encrypt(new_enc_key, seed, k["address"].encode("ascii"))
            self.conn.execute("UPDATE wallet_keys SET enc_private = ? WHERE wallet_id = ? AND address = ?",
                              (enc, self.id, k["address"]))

        self.conn.execute(
            "UPDATE wallets SET password_salt = ?, password_hash = ?, enc_salt = ? WHERE id = ?",
            (pw_salt, pw_hash, new_enc_salt, self.id))
        self.conn.commit()
        self._enc_key = new_enc_key


# ---------- база ----------
def open_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _check_password(row: sqlite3.Row, password: str) -> bool:
    _, digest = crypto.hash_password(password, row["password_salt"])
    return digest == row["password_hash"]


def _create(conn, role: str, name: str, password: str) -> Wallet:
    pw_salt, pw_hash = crypto.hash_password(password)
    enc_salt = os.urandom(16)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        cur = conn.execute(
            "INSERT INTO wallets (role, name, password_salt, password_hash, enc_salt, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (role, name, pw_salt, pw_hash, enc_salt, created_at))
        conn.commit()
    except sqlite3.IntegrityError:
        raise WalletError(f"кошелёк {role}/{name} уже существует")
    row = conn.execute("SELECT * FROM wallets WHERE id = ?", (cur.lastrowid,)).fetchone()
    wallet = Wallet(conn, row, password)
    wallet.add_key()  # первый ключ создаётся сразу
    return wallet


def unlock(conn, role: str, name: str, password: str) -> Wallet | None:
    row = conn.execute("SELECT * FROM wallets WHERE role = ? AND name = ?",
                       (role, name)).fetchone()
    if row is None or not _check_password(row, password):
        return None
    return Wallet(conn, row, password)


def find_by_password(conn, password: str) -> list[Wallet]:
    """Все кошельки, к которым подходит пароль (мастер и/или воркеры)."""
    out = []
    for row in conn.execute("SELECT * FROM wallets ORDER BY role, id").fetchall():
        if _check_password(row, password):
            out.append(Wallet(conn, row, password))
    return out


def ensure_master(conn, password: str | None = None) -> tuple[Wallet, bool]:
    """
    Возвращает (кошелёк мастера, создан_ли_сейчас).
    Если мастера ещё нет — создаёт с переданным паролем (или 'root').
    """
    row = conn.execute("SELECT * FROM wallets WHERE role = 'master' ORDER BY id LIMIT 1").fetchone()
    if row is None:
        wallet = _create(conn, "master", "master", password or DEFAULT_PASSWORD)
        return wallet, True
    if password is None:
        raise WalletError("мастер уже существует — нужен пароль")
    wallet = unlock(conn, "master", row["name"], password)
    if wallet is None:
        raise WalletError("неверный пароль мастера")
    return wallet, False


def ensure_worker(conn, name: str, password: str | None = None) -> tuple[Wallet, bool]:
    row = conn.execute("SELECT * FROM wallets WHERE role = 'worker' AND name = ?",
                       (name,)).fetchone()
    if row is None:
        wallet = _create(conn, "worker", name, password or DEFAULT_PASSWORD)
        return wallet, True
    if password is None:
        raise WalletError(f"воркер {name} уже существует — нужен пароль")
    wallet = unlock(conn, "worker", name, password)
    if wallet is None:
        raise WalletError(f"неверный пароль воркера {name}")
    return wallet, False


if __name__ == "__main__":
    import json

    conn = open_db(os.path.join("results", "keys.db"))
    print(json.dumps([dict(r) for r in conn.execute(
        "SELECT id, role, name, created_at FROM wallets").fetchall()],
        ensure_ascii=False, indent=2))
