#!/usr/bin/env python3
"""
Единое хранилище кошельков мастера и воркеров (SQLite).

Одна база (по умолчанию results/keys.db) хранит кошельки — БЕЗ ролей:

    wallets(id, password_salt, password_hash, enc_salt, created_at)
    wallet_keys(id, wallet_id, created_at, public_key, address UNIQUE, enc_private)

Кошелёк — самостоятельная сущность: мастер это или воркер, решает процесс,
который им пользуется, а не деньги. Пароль и есть кошелёк: один пароль —
один кошелёк со всеми его адресами.

* Пароль проверяется по scrypt(password, password_salt).
* Приватные ключи (Ed25519 seed) шифруются ключом scrypt(password, enc_salt)
  (encrypt-then-MAC, см. src/crypto.py) — в базе открытого ключа нет.
* Соли случайные: один и тот же пароль в двух разных базах (на разных
  машинах) даёт разные ключи шифрования и разные адреса.
* Пароль по умолчанию — «root»; меняется командой master-cli passwd или
  wallet.py (в панели — «Сменить пароль кошелька»).

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
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """
    Старые базы хранили у кошелька роль и имя. Роль — свойство процесса, а не
    денег, поэтому колонки убираем. Ключи, пароли и id сохраняются: записи
    wallet_keys продолжают ссылаться на те же кошельки.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(wallets)")}
    if "role" not in cols and "name" not in cols:
        return
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript(
        """
        CREATE TABLE wallets_v2 (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            password_salt BLOB    NOT NULL,
            password_hash BLOB    NOT NULL,
            enc_salt      BLOB    NOT NULL,
            created_at    TEXT    NOT NULL
        );
        INSERT INTO wallets_v2 (id, password_salt, password_hash, enc_salt, created_at)
            SELECT id, password_salt, password_hash, enc_salt, created_at FROM wallets;
        DROP TABLE wallets;
        ALTER TABLE wallets_v2 RENAME TO wallets;
        """)
    conn.commit()


def _check_password(row: sqlite3.Row, password: str) -> bool:
    _, digest = crypto.hash_password(password, row["password_salt"])
    return digest == row["password_hash"]


def _create(conn, password: str) -> Wallet:
    pw_salt, pw_hash = crypto.hash_password(password)
    enc_salt = os.urandom(16)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cur = conn.execute(
        "INSERT INTO wallets (password_salt, password_hash, enc_salt, created_at) "
        "VALUES (?, ?, ?, ?)", (pw_salt, pw_hash, enc_salt, created_at))
    conn.commit()
    row = conn.execute("SELECT * FROM wallets WHERE id = ?", (cur.lastrowid,)).fetchone()
    wallet = Wallet(conn, row, password)
    wallet.add_key()  # первый ключ создаётся сразу
    return wallet


def find_by_password(conn, password: str) -> list[Wallet]:
    """Все записи, которые открывает пароль (старые базы могли наплодить их)."""
    out = []
    for row in conn.execute("SELECT * FROM wallets ORDER BY id").fetchall():
        if _check_password(row, password):
            out.append(Wallet(conn, row, password))
    return out


class WalletSet:
    """
    Кошелёк = пароль. В базе под одним паролем может оказаться несколько
    записей (старые версии создавали отдельную запись на мастера и на каждого
    воркера). Для владельца это ОДИН кошелёк: общий набор адресов, общий
    баланс, общая история. Роли здесь нет — ею пользуется процесс.
    """

    def __init__(self, wallets: list):
        self.wallets = list(wallets)
        self.primary = max(self.wallets, key=lambda w: w.id) if self.wallets else None

    def keys(self) -> list[dict]:
        out = []
        for w in self.wallets:
            out += w.keys()
        return sorted(out, key=lambda k: k["id"])

    def addresses(self) -> set[str]:
        return {k["address"] for k in self.keys()}

    def latest_key(self) -> dict | None:
        keys = self.keys()
        return keys[-1] if keys else None

    def add_key(self) -> dict:
        if self.primary is None:
            raise WalletError("пустой кошелёк")
        return self.primary.add_key()

    def _owner(self, address: str) -> Wallet:
        for w in self.wallets:
            if address in w.addresses():
                return w
        raise WalletError(f"адрес {address} не принадлежит кошельку")

    def public_for(self, address: str) -> bytes:
        return self._owner(address).public_for(address)

    def sign(self, address: str, message: bytes) -> bytes:
        return self._owner(address).sign(address, message)

    def change_password(self, old: str, new: str):
        # Пароль один на все записи кошелька — меняем сразу у всех, иначе
        # часть адресов осталась бы за старым паролем.
        for w in self.wallets:
            w.change_password(old, new)


def ensure_wallet(conn, password: str | None = None) -> tuple[WalletSet, bool]:
    """
    (кошелёк, создан_ли_сейчас). Пароль и есть кошелёк: если он уже есть —
    возвращаем его, иначе создаём новый пустой.

    Опечатка в пароле неотличима от нового кошелька — поэтому вызывающий
    обязан сказать пользователю, что кошелёк создан с нуля (иначе «пропавший»
    баланс будет искать в поддержке).
    """
    pw = password or DEFAULT_PASSWORD
    matched = find_by_password(conn, pw)
    if matched:
        return WalletSet(matched), False
    return WalletSet([_create(conn, pw)]), True


if __name__ == "__main__":
    import json

    conn = open_db(os.path.join("results", "keys.db"))
    rows = conn.execute(
        "SELECT w.id, w.created_at, "
        "(SELECT COUNT(*) FROM wallet_keys k WHERE k.wallet_id = w.id) AS keys "
        "FROM wallets w ORDER BY w.id").fetchall()
    print(json.dumps([dict(r) for r in rows], ensure_ascii=False, indent=2))
