#!/usr/bin/env python3
"""
webwallet.py — JSON-обёртка над кошельком для веб-панели.

Повторяет логику wallet.py (find_by_password → адреса → балансы → транзакции
с числом подтверждений), но отдаёт результат в JSON, чтобы страница могла
рисовать таблицы сама.

Примеры:
    python3 src/webwallet.py info --password master --results-root results
    python3 src/webwallet.py info --password-env WEBUI_WALLET_PASSWORD
    python3 src/webwallet.py send --password master --to <addr> --amount 5 \
            --master http://127.0.0.1:8000
"""

import argparse
import json
import os
import sys

import requests

import chain
import wallet_store
from wallet import _element, _transactions

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "results")


def _root(path: str) -> str:
    return os.path.abspath(path or DEFAULT_ROOT)


def _confirmations(root: str) -> int:
    try:
        with open(os.path.join(PROJECT_ROOT, "src", "config.json"), encoding="utf-8") as f:
            raw = json.load(f)
        entry = raw.get("coinbase_confirmations")
        if isinstance(entry, dict):
            return int(entry.get("value", 1))
        return int(entry or 1)
    except Exception:  # noqa: BLE001
        return 1


def _password(args) -> str:
    if args.password:
        return args.password
    env_name = args.password_env or "WEBUI_WALLET_PASSWORD"
    return os.environ.get(env_name, "")


def _mempool(masters: list[str] | None) -> dict:
    """Объединённый мемпул всех известных мастеров: txid -> tx."""
    out = {}
    for url in masters or []:
        url = str(url).rstrip("/")
        if not url:
            continue
        try:
            r = requests.get(url + "/tx/mempool", timeout=5)
            if r.status_code != 200:
                continue
            for tx in (r.json().get("txs") or []):
                tid = tx.get("txid") or chain.txid_of(tx)
                out[tid] = tx
        except Exception:  # noqa: BLE001
            continue
    return out


def _mempool_rows(txs: dict, index: dict, addresses: set[str]) -> list[dict]:
    """Неподтверждённые транзакции, затрагивающие адреса кошелька."""
    rows = []
    for tid, tx in txs.items():
        received = sum(float(o.get("value") or 0.0)
                       for o in tx.get("outputs", [])
                       if o.get("address") in addresses)
        sent = 0.0
        total_in = 0.0
        for i in tx.get("inputs", []):
            if i.get("coinbase"):
                continue
            key = (i.get("txid"), int(i.get("vout", -1)))
            addr, value = index.get(key, (None, None))
            if value:
                total_in += value
            if addr in addresses and value:
                sent += value
        if not received and not sent:
            continue
        total_out = sum(float(o.get("value") or 0.0) for o in tx.get("outputs", []))
        rows.append({
            "block": None, "type": tx.get("type"), "txid": tid,
            "received": round(received, 8), "sent": round(sent, 8),
            "fee": round(max(0.0, total_in - total_out), 8),
            "confirmations": 0, "mempool": True,
        })
    rows.sort(key=lambda r: r["txid"])
    return rows


def _masters_info(masters: list[str] | None) -> list[dict]:
    out = []
    for url in masters or []:
        url = str(url).rstrip("/")
        if not url:
            continue
        try:
            r = requests.get(url + "/chain/height", timeout=3)
            h = r.json() if r.status_code == 200 else None
            out.append({"url": url, "ok": bool(h),
                        "height": (h or {}).get("height")})
        except Exception:  # noqa: BLE001
            out.append({"url": url, "ok": False, "height": None})
    return out


def collect(root: str, password: str, confirmations: int,
            role: str | None = None, name: str | None = None,
            masters: list[str] | None = None) -> dict:
    root = _root(root)
    db_path = os.path.join(root, "keys.db")
    if not os.path.exists(db_path):
        return {"ok": False, "error": f"нет базы кошельков: {db_path}",
                "root": root, "wallets": []}

    conn = wallet_store.open_db(db_path)
    element = _element()
    element_dir = os.path.join(root, element)
    tip = (chain.next_seq(element_dir) - 1) if os.path.isdir(element_dir) else 0
    tip = max(0, tip)

    # Пароль может подходить сразу к нескольким кошелькам (у мастера и воркера
    # разные соли и ключи, но пароль один). Поэтому фильтруем по роли/имени:
    # без этого вкладка «мастер» показывала бы и балансы воркеров.
    matched = wallet_store.find_by_password(conn, password)
    wallets = [w for w in matched
               if (role is None or w.role == role)
               and (name is None or w.name == name)]
    others = [f"{w.role}/{w.name}" for w in matched if w not in wallets]

    if not wallets:
        hint = ("пароль не подходит ни к одному кошельку" if not matched
                else f"для роли {role or 'любой'} кошельков нет; пароль подходит к: "
                     + ", ".join(others))
        return {"ok": False, "root": root, "element": element, "height": tip,
                "error": hint, "wallets": [], "role": role,
                "also_matched": others}

    out = []
    mempool = _mempool(masters)
    index = _outputs_index(element_dir) if mempool else {}
    for w in wallets:
        addresses = w.addresses()
        balances = {}
        totals = {"available": 0.0, "pending": 0.0, "total": 0.0}
        txs = []
        if addresses and os.path.isdir(element_dir):
            try:
                balances = chain.balances(element_dir, addresses, confirmations)
            except Exception:  # noqa: BLE001
                balances = {}
            try:
                txs = _transactions(element_dir, addresses)
            except Exception:  # noqa: BLE001
                txs = []
            # неподтверждённые — сверху
            txs = _mempool_rows(mempool, index, addresses) + txs
        for b in balances.values():
            for key in totals:
                totals[key] += float(b.get(key) or 0.0)

        # Мемпул блокирует потраченные выходы и добавляет будущие поступления.
        locked = sum(t["sent"] for t in txs if t.get("mempool"))
        incoming = sum(t["received"] for t in txs if t.get("mempool"))
        if locked or incoming:
            totals["available"] = max(0.0, totals["available"] - locked)
            totals["pending"] += incoming
            totals["total"] = totals["available"] + totals["pending"]
        out.append({
            "role": w.role,
            "name": w.name,
            "addresses": [{"address": k["address"], "created_at": k["created_at"]}
                          for k in w.keys()],
            "balances": balances,
            "totals": {k: round(v, 8) for k, v in totals.items()},
            "transactions": txs,
        })

    return {"ok": True, "root": root, "element": element, "height": tip,
            "role": role, "also_matched": others, "wallets": out,
            "masters": _masters_info(masters),
            "mempool_count": len(mempool)}


def _outputs_index(element_dir: str) -> dict:
    """(txid, vout) -> (адрес, сумма) по всем блокам — чтобы показать, откуда пришли входы."""
    index = {}
    for b in chain.list_blocks(element_dir):
        try:
            txs = chain.read_block_txs(b["path"])
        except Exception:  # noqa: BLE001
            continue
        for tx in txs:
            tid = tx.get("txid") or chain.txid_of(tx)
            for n, out in enumerate(tx.get("outputs", [])):
                index[(tid, n)] = (out.get("address"), float(out.get("value") or 0.0))
    return index


def _tx_payload(tx: dict, index: dict, block: int | None,
                confirmations: int) -> dict:
    inputs, total_in = [], 0.0
    for i in tx.get("inputs", []):
        if i.get("coinbase"):
            inputs.append({"coinbase": True, "address": None,
                           "value": None, "txid": None, "vout": None})
            continue
        key = (i.get("txid"), int(i.get("vout", -1)))
        address, value = index.get(key, (None, None))
        if value:
            total_in += value
        inputs.append({"coinbase": False, "address": address,
                       "value": value, "txid": i.get("txid"),
                       "vout": i.get("vout")})

    outputs = [{"vout": n, "address": o.get("address"),
                "value": float(o.get("value") or 0.0)}
               for n, o in enumerate(tx.get("outputs", []))]
    total_out = sum(o["value"] for o in outputs)

    fee = None
    if tx.get("type") != "coinbase" and total_in:
        fee = round(total_in - total_out, 8)

    return {
        "ok": True, "block": block, "confirmations": confirmations,
        "mempool": block is None,
        "type": tx.get("type"),
        "txid": tx.get("txid") or chain.txid_of(tx),
        "fee": fee, "inputs": inputs, "outputs": outputs,
        "total_in": round(total_in, 8), "total_out": round(total_out, 8),
        "raw": {k: v for k, v in tx.items() if k != "_file"},
    }


def tx_details(root: str, txid: str, masters: list[str] | None = None) -> dict:
    """Полная информация о транзакции из блока: входы с адресами, выходы, комиссия."""
    root = _root(root)
    element = _element()
    element_dir = os.path.join(root, element)
    if not os.path.isdir(element_dir):
        return {"ok": False, "error": f"нет каталога {element_dir}"}

    tip = chain.next_seq(element_dir) - 1
    index = _outputs_index(element_dir)

    for b in chain.list_blocks(element_dir):
        try:
            txs = chain.read_block_txs(b["path"])
        except Exception:  # noqa: BLE001
            continue
        for tx in txs:
            tid = tx.get("txid") or chain.txid_of(tx)
            if tid != txid:
                continue

            inputs, total_in = [], 0.0
            for i in tx.get("inputs", []):
                if i.get("coinbase"):
                    inputs.append({"coinbase": True, "address": None,
                                   "value": None, "txid": None, "vout": None})
                    continue
                key = (i.get("txid"), int(i.get("vout", -1)))
                address, value = index.get(key, (None, None))
                if value:
                    total_in += value
                inputs.append({"coinbase": False, "address": address,
                               "value": value, "txid": i.get("txid"),
                               "vout": i.get("vout")})

            outputs = [{"vout": n, "address": o.get("address"),
                        "value": float(o.get("value") or 0.0)}
                       for n, o in enumerate(tx.get("outputs", []))]
            total_out = sum(o["value"] for o in outputs)

            fee = None
            if tx.get("type") != "coinbase" and total_in:
                fee = round(total_in - total_out, 8)

            return _tx_payload(tx, index, b["seq"], max(0, tip - b["seq"] + 1))

    # не нашли в блоках — ищем в мемпулах известных мастеров
    for tid, tx in _mempool(masters).items():
        if tid == txid:
            payload = _tx_payload(tx, index, None, 0)
            payload["element"] = element
            return payload

    return {"ok": False,
            "error": "транзакция не найдена ни в блоках, ни в мемпуле известных мастеров"}


def locked_utxos(masters: list[str] | None) -> set[tuple[str, int]]:
    """Выходы, уже потраченные в мемпулах: их нельзя тратить второй раз."""
    locked = set()
    for tx in _mempool(masters).values():
        for i in tx.get("inputs", []):
            if i.get("coinbase"):
                continue
            locked.add((i.get("txid"), int(i.get("vout", -1))))
    return locked


def send(root: str, password: str, to: str, amount: float, fee: float,
         masters: list[str], confirmations: int, role: str | None,
         name: str | None) -> dict:
    root = _root(root)
    conn = wallet_store.open_db(os.path.join(root, "keys.db"))
    element = _element()
    element_dir = os.path.join(root, element)

    wallets = [w for w in wallet_store.find_by_password(conn, password)
               if (role is None or w.role == role)
               and (name is None or w.name == name)]
    if len(wallets) != 1:
        return {"ok": False,
                "error": f"подходящих кошельков: {len(wallets)} — уточните роль/имя",
                "candidates": [f"{w.role}/{w.name}" for w in wallets]}
    wallet = wallets[0]

    tx = chain.build_wallet_transfer(wallet, element_dir, to, amount, fee=fee,
                                     confirmations=confirmations,
                                     exclude=locked_utxos(masters))
    if tx is None:
        return {"ok": False,
                "error": "недостаточно подтверждённых средств (часть может быть "
                         "заблокирована неподтверждёнными транзакциями)"}

    accepted_by = []
    attempts = []
    for url in masters:
        url = url.rstrip("/")
        try:
            import requests
            r = requests.post(url + "/tx/submit", json={"tx": tx}, timeout=30)
            body = {}
            try:
                body = r.json()
            except Exception:  # noqa: BLE001
                body = {"raw": r.text[:200]}
            attempts.append({"url": url, "status": r.status_code, "body": body})
            if r.status_code == 200 and body.get("status") == "accepted":
                accepted_by.append(url)
        except Exception as e:  # noqa: BLE001
            attempts.append({"url": url, "error": str(e)})

    return {"ok": bool(accepted_by), "txid": tx["txid"], "tx": tx,
            "accepted_by": accepted_by, "attempts": attempts,
            "wallet": f"{wallet.role}/{wallet.name}"}


def change_password(root: str, old: str, new: str,
                    role: str | None, name: str | None) -> dict:
    root = _root(root)
    conn = wallet_store.open_db(os.path.join(root, "keys.db"))
    wallets = [w for w in wallet_store.find_by_password(conn, old)
               if (role is None or w.role == role)
               and (name is None or w.name == name)]
    if len(wallets) != 1:
        return {"ok": False,
                "error": f"подходящих кошельков: {len(wallets)} — уточните роль/имя"}
    wallets[0].change_password(old, new)
    return {"ok": True, "wallet": f"{wallets[0].role}/{wallets[0].name}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="JSON-кошелёк для веб-панели")
    ap.add_argument("action", choices=["info", "send", "passwd", "tx"])
    ap.add_argument("--password", default=None)
    ap.add_argument("--password-env", default="WEBUI_WALLET_PASSWORD")
    ap.add_argument("--results-root", default=DEFAULT_ROOT)
    ap.add_argument("--role", choices=["master", "worker"], default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--confirmations", type=int, default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--amount", type=float, default=0.0)
    ap.add_argument("--fee", type=float, default=chain.TX_FEE)
    ap.add_argument("--change-to", default=None)
    ap.add_argument("--txid", default=None)
    ap.add_argument("--master", action="append", default=None)
    args = ap.parse_args()

    password = _password(args)
    confirmations = args.confirmations if args.confirmations is not None \
        else _confirmations(args.results_root)

    if args.action == "info":
        masters = [m for m in (args.master or []) if m]
        result = collect(args.results_root, password, confirmations,
                         role=args.role, name=args.name, masters=masters)
    elif args.action == "tx":
        if not args.txid:
            print(json.dumps({"ok": False, "error": "нужен --txid"}, ensure_ascii=False))
            return 2
        result = tx_details(args.results_root, args.txid,
                            masters=[m for m in (args.master or []) if m])
    elif args.action == "send":
        if not args.to or args.amount <= 0:
            print(json.dumps({"ok": False, "error": "нужны --to и --amount > 0"},
                             ensure_ascii=False))
            return 2
        masters = [m for m in (args.master or [os.environ.get(
            "MASTER_URL", "http://127.0.0.1:8000")]) if m]
        result = send(args.results_root, password, args.to, args.amount,
                      args.fee, masters, confirmations, args.role, args.name)
    else:
        if not args.change_to:
            print(json.dumps({"ok": False, "error": "нужен --change-to"},
                             ensure_ascii=False))
            return 2
        result = change_password(args.results_root, password, args.change_to,
                                 args.role, args.name)

    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
