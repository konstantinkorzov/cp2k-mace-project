#!/usr/bin/env python3
"""
Администраторы веб-панели: имя + scrypt-хэш в SQLite (results/webui.db).

Пароль никогда не хранится и не передаётся в аргументах командной строки —
только через переменные окружения (их не видно в `ps`).

Команды (вызываются из Node через execFile):
    init   --root DIR [--name admin]     создать admin, если база пуста
    check  --root DIR [--name admin]     WEBUI_PASSWORD -> {"ok": bool}
    passwd --root DIR [--name admin]     WEBUI_OLD_PASSWORD/WEBUI_NEW_PASSWORD
    status --root DIR                    {"ok": True, "count": N}

Все команды печатают один JSON в stdout.
"""

import argparse
import hmac
import json
import os
import secrets
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS web_admins (
    id         INTEGER PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    salt       BLOB NOT NULL,
    hash       BLOB NOT NULL,
    updated_at REAL NOT NULL
)
"""


def _conn(root: str) -> sqlite3.Connection:
    os.makedirs(root, exist_ok=True)
    conn = sqlite3.connect(os.path.join(root, "webui.db"))
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def _row(conn, name):
    return conn.execute("SELECT * FROM web_admins WHERE name = ?", (name,)).fetchone()


def _verify(row, password: str) -> bool:
    if row is None:
        return False
    digest = crypto.kdf(password, row["salt"])
    return hmac.compare_digest(digest, row["hash"])


def _set_password(conn, name: str, password: str) -> None:
    salt, digest = crypto.hash_password(password)
    if _row(conn, name) is None:
        conn.execute(
            "INSERT INTO web_admins(name, salt, hash, updated_at) VALUES (?,?,?,?)",
            (name, salt, digest, time.time()))
    else:
        conn.execute(
            "UPDATE web_admins SET salt=?, hash=?, updated_at=? WHERE name=?",
            (salt, digest, time.time(), name))
    conn.commit()


def cmd_init(args) -> dict:
    conn = _conn(args.root)
    count = conn.execute("SELECT COUNT(*) AS c FROM web_admins").fetchone()["c"]
    if count:
        return {"ok": True, "created": False, "count": count}
    password = os.environ.get("WEBUI_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
    _set_password(conn, args.name, password)
    out = {"ok": True, "created": True, "name": args.name, "count": 1}
    if not os.environ.get("WEBUI_ADMIN_PASSWORD"):
        # Пароль показать один раз (первый запуск): его надо сменить в панели.
        out["password"] = password
    return out


def cmd_check(args) -> dict:
    password = os.environ.get("WEBUI_PASSWORD") or ""
    conn = _conn(args.root)
    ok = _verify(_row(conn, args.name), password)
    return {"ok": ok}


def cmd_passwd(args) -> dict:
    old = os.environ.get("WEBUI_OLD_PASSWORD") or ""
    new = os.environ.get("WEBUI_NEW_PASSWORD") or ""
    if len(new) < 8:
        return {"ok": False, "error": "новый пароль короче 8 символов"}
    conn = _conn(args.root)
    if not _verify(_row(conn, args.name), old):
        return {"ok": False, "error": "неверный текущий пароль"}
    _set_password(conn, args.name, new)
    return {"ok": True}


def cmd_status(args) -> dict:
    conn = _conn(args.root)
    count = conn.execute("SELECT COUNT(*) AS c FROM web_admins").fetchone()["c"]
    return {"ok": True, "count": count}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "check", "passwd", "status"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--name", default="admin")
    args = ap.parse_args()

    fn = {"init": cmd_init, "check": cmd_check,
          "passwd": cmd_passwd, "status": cmd_status}[args.cmd]
    try:
        out = fn(args)
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
