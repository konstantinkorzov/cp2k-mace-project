#!/usr/bin/env python3
"""
Идентичность воркера на Ed25519 (утилита; в самом воркере ключи хранит
wallet_store в общей базе results/keys.db).

Приватный ключ — 32 случайных байта (seed), публичный — Ed25519,
pubkey_hash — адрес = sha256("cp2k-addr-v1|" + public_key).
"""

import secrets

import crypto


def make_identity(name: str) -> dict:
    """Новая пара ключей для имени (в БД не пишет — только утилита/тесты)."""
    if not name:
        raise ValueError("Имя воркера не может быть пустым")
    seed = secrets.token_bytes(32)
    public = crypto.public_key_from_seed(seed)
    return {
        "name": name,
        "private_key": seed.hex(),
        "public_key": public.hex(),
        "pubkey_hash": crypto.address_from_public(public),
    }


def new_worker_name() -> str:
    return f"worker_{secrets.token_hex(4)}"


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Сгенерировать идентичность воркера (Ed25519)")
    ap.add_argument("--name", default=None, help="имя воркера (по умолчанию случайное)")
    args = ap.parse_args()

    ident = make_identity(args.name or new_worker_name())
    print(json.dumps(ident, ensure_ascii=False, indent=2))
