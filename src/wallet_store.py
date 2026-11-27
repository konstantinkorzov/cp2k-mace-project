#!/usr/bin/env python3
"""
Единое хранилище кошельков мастера и воркеров (SQLite).

Одна база (по умолчанию results/keys.db) хранит кошельки:

    wallets(id, name UNIQUE, password_salt, password_hash, enc_salt, created_at)
    wallet_keys(id, wallet_id, created_at, public_key, address UNIQUE, enc_private)

Кошелёк — самостоятельная сущность: мастер это или воркер, решает процесс,
который им пользуется, а не деньги. Кошелёк задаётся парой «имя + пароль»:
имена разные, а пароль может быть общим — один пароль на несколько кошельков.

По умолчанию в базе НЕТ ни одного кошелька: он появляется только по явной
команде `wallet.py create` (или кнопкой «Создать» в панели). Ни мастер, ни
воркер кошельки не создают.

* Пароль проверяется по scrypt(password, password_salt).
* Приватные ключи (Ed25519 seed) шифруются ключом scrypt(password, enc_salt)
  (encrypt-then-MAC, см. src/crypto.py) — в базе открытого ключа нет.
* Соли случайные: один и тот же пароль в двух разных кошельках (или базах)
  даёт разные ключи шифрования и разные адреса.
* Пароль меняется командой wallet.py --change-to (в панели — «Сменить пароль»).

Криптография только stdlib (scrypt/HMAC/SHA-256/Ed25519 из crypto.py).
"""

import os
import re
import secrets
import sqlite3
import time

import crypto

DEFAULT_PASSWORD = "root"

SCHEMA = """
CREATE TABLE IF NOT EXISTS wallets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
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
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """
    Приводим старые базы к текущей схеме. Ключи, пароли и id не трогаем:
    записи wallet_keys продолжают ссылаться на те же кошельки.

      * v1 (role + name): роль — свойство процесса, а не денег; убираем её,
        имя оставляем (оно и есть идентификатор кошелька);
      * v2 (без имени): имя потерялось при миграции — подставляем техническое
        «wallet-<id>», пользователь может задать своё командой rename.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(wallets)")}
    if "role" in cols:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.executescript(
            """
            CREATE TABLE wallets_v3 (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL UNIQUE,
                password_salt BLOB    NOT NULL,
                password_hash BLOB    NOT NULL,
                enc_salt      BLOB    NOT NULL,
                created_at    TEXT    NOT NULL
            );
            INSERT INTO wallets_v3 (id, name, password_salt, password_hash, enc_salt, created_at)
                SELECT id, name, password_salt, password_hash, enc_salt, created_at FROM wallets;
            DROP TABLE wallets;
            ALTER TABLE wallets_v3 RENAME TO wallets;
            """)
        conn.commit()
        return
    if "name" not in cols:
        conn.execute("ALTER TABLE wallets ADD COLUMN name TEXT")
        conn.execute("UPDATE wallets SET name = 'wallet-' || id")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_wallets_name ON wallets(name)")
        conn.commit()


def _check_password(row: sqlite3.Row, password: str) -> bool:
    _, digest = crypto.hash_password(password, row["password_salt"])
    return digest == row["password_hash"]


MIN_PASSWORD = 4
NAME_RE = re.compile(r"^[\w.+-]{1,64}$")


def _check_name(name: str) -> str:
    name = str(name or "").strip()
    if not NAME_RE.match(name):
        raise WalletError("имя кошелька: 1–64 символа, буквы/цифры/точка/дефис/подчёркивание, без пробелов")
    return name


def _create(conn, name: str, password: str) -> Wallet:
    name = _check_name(name)
    pw_salt, pw_hash = crypto.hash_password(password)
    enc_salt = os.urandom(16)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        cur = conn.execute(
            "INSERT INTO wallets (name, password_salt, password_hash, enc_salt, created_at) "
            "VALUES (?, ?, ?, ?, ?)", (name, pw_salt, pw_hash, enc_salt, created_at))
        conn.commit()
    except sqlite3.IntegrityError:
        raise WalletError(f"кошелёк «{name}» уже существует")
    row = conn.execute("SELECT * FROM wallets WHERE id = ?", (cur.lastrowid,)).fetchone()
    wallet = Wallet(conn, row, password)
    wallet.add_key()  # первый ключ создаётся сразу
    return wallet


def create_wallet(conn, name: str, password: str) -> Wallet:
    """
    Явное создание кошелька. Автосоздания нет нигде: ни в мастере, ни в
    воркере, ни в панели — только по команде пользователя.
    """
    name = _check_name(name)
    password = str(password or "")
    if len(password) < MIN_PASSWORD:
        raise WalletError(f"пароль кошелька короче {MIN_PASSWORD} символов")
    return _create(conn, name, password)


def unlock(conn, name: str, password: str) -> Wallet | None:
    """Кошелёк по имени и паролю (имя — идентификатор, пароль — ключ)."""
    row = conn.execute("SELECT * FROM wallets WHERE name = ?",
                       (str(name or "").strip(),)).fetchone()
    if row is None or not _check_password(row, password):
        return None
    return Wallet(conn, row, password)


def list_wallets(conn) -> list[dict]:
    """Имена кошельков для панели (секретов здесь нет)."""
    rows = conn.execute(
        "SELECT w.id, w.name, w.created_at, "
        "(SELECT COUNT(*) FROM wallet_keys k WHERE k.wallet_id = w.id) AS keys "
        "FROM wallets w ORDER BY w.name COLLATE NOCASE").fetchall()
    return [dict(r) for r in rows]


def open_wallet(conn, name: str, password: str) -> Wallet:
    """
    Открыть существующий кошелёк. Если его нет — понятная ошибка, а не
    создание пустого: опечатка в имени не должна стоить потерянных монет.
    """
    name = str(name or "").strip()
    if not name:
        raise WalletError("нужно имя кошелька (--wallet NAME); список: wallet.py list")
    row = conn.execute("SELECT * FROM wallets WHERE name = ?", (name,)).fetchone()
    if row is None:
        names = [w["name"] for w in list_wallets(conn)]
        hint = ("кошельков пока нет — создайте: "
                f"python3 src/wallet.py create --name {name}")
        if names:
            hint = "есть: " + ", ".join(names)
        raise WalletError(f"кошелёк «{name}» не найден ({hint})")
    if not _check_password(row, password):
        raise WalletError(f"неверный пароль кошелька «{name}»")
    return Wallet(conn, row, password)


def rename_wallet(conn, name: str, new_name: str) -> dict:
    """Переименовать кошелёк: id, ключи и пароль не меняются."""
    name = str(name or "").strip()
    new_name = _check_name(new_name)
    row = conn.execute("SELECT * FROM wallets WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise WalletError(f"кошелёк «{name}» не найден")
    if conn.execute("SELECT 1 FROM wallets WHERE name = ?", (new_name,)).fetchone():
        raise WalletError(f"кошелёк «{new_name}» уже существует")
    conn.execute("UPDATE wallets SET name = ? WHERE id = ?", (new_name, row["id"]))
    conn.commit()
    return {"id": row["id"], "name": new_name}


if __name__ == "__main__":
    import json

    conn = open_db(os.path.join("results", "keys.db"))
    rows = conn.execute(
        "SELECT w.id, w.name, w.created_at, "
        "(SELECT COUNT(*) FROM wallet_keys k WHERE k.wallet_id = w.id) AS keys "
        "FROM wallets w ORDER BY w.name").fetchall()
    print(json.dumps([dict(r) for r in rows], ensure_ascii=False, indent=2))
