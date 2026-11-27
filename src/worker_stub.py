#!/usr/bin/env python3
"""
Скрипт-заглушка воркера. Регистрируется с подписью ключа, получает задачу
и отправляет назад заведомо неправильный результат.

Нужен, чтобы проверять приёмку: мастер обязан отбросить такой результат.
Сам по себе он ничего не считает и ничего не доказывает, кроме владения
ключом (этого мастер и требует для выдачи задачи).
"""

import os
import sys
import secrets
import argparse

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto  # noqa: E402


def register(base_url: str) -> tuple[str, str, str] | None:
    """Регистрация с доказательством ключа → (worker_id, session, адрес)."""
    seed = secrets.token_bytes(32)
    public_key = crypto.public_key_from_seed(seed)
    pubkey_hash = crypto.address_from_public(public_key)

    ch = requests.get(base_url + "/register-challenge", timeout=10)
    ch.raise_for_status()
    nonce = str(ch.json()["nonce"])
    sig = crypto.sign(seed, b"cp2k-mace-register:" + bytes.fromhex(nonce)).hex()

    reg = requests.post(base_url + "/register-worker", json={
        "name": "stub",
        "pubkey_hash": pubkey_hash,
        "public_key": public_key.hex(),
        "nonce": nonce,
        "signature": sig,
    }, timeout=10)
    if reg.status_code != 200:
        print(f"Ошибка регистрации: HTTP {reg.status_code} {reg.text}")
        return None
    data = reg.json()
    return data["worker_id"], data["session"], pubkey_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Порт сервера main.py")
    parser.add_argument("--server", type=str, default="127.0.0.1")
    args = parser.parse_args()

    base_url = f"http://{args.server}:{args.port}"

    reg = register(base_url)
    if not reg:
        return
    worker_id, session, pubkey_hash = reg
    print(f"Зарегистрирован: {worker_id} (адрес {pubkey_hash[:12]}…)")

    print(f"[{worker_id}] Запрашиваю задачу...")
    try:
        response = requests.get(base_url + "/get-task",
                                params={"worker_id": worker_id, "session": session},
                                timeout=10)
        res_data = response.json()

        if res_data.get("status") == "empty":
            print(f"[{worker_id}] Очередь пуста.")
            return

        task = res_data.get("task")
        struct_id = task["struct_id"]
        print(f"[{worker_id}] Получена структура: {struct_id}. Имитирую расчет с ошибкой...")

        submit_resp = requests.post(base_url + "/submit-result", json={
            "worker_id": worker_id,
            "session": session,
            "struct_id": struct_id,
            "success": False,   # расчёт «не удался» — мастер обязан вернуть задачу
            "data": {"error_reason": "SCF convergence failed / bad forces"},
        }, timeout=10)
        print(f"[{worker_id}] Ответ сервера на сдачу фальшивого результата:")
        print(submit_resp.json())

    except Exception as e:  # noqa: BLE001
        print(f"[{worker_id}] Ошибка: {e}")


if __name__ == "__main__":
    main()
