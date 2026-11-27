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
import re
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


def _master_chain(url: str) -> str | None:
    """Имя цепочки (элемента) мастера или None, если он не ответил."""
    try:
        r = requests.get(url + "/chain/height", timeout=3)
        if r.status_code != 200:
            return None
        return (r.json() or {}).get("element") or None
    except Exception:  # noqa: BLE001
        return None


def _mempool(masters: list[str] | None, element: str | None = None) -> dict:
    """
    Объединённый мемпул мастеров ОДНОЙ цепочки: txid -> tx.

    Один адрес живёт во всех цепочках (Cu, Fe, Al2O3), поэтому без фильтра по
    элементу в кошелёк меди попадали бы неподтверждённые переводы из цепочки
    железа — как будто медь можно тратить железом.
    """
    out = {}
    for url in masters or []:
        url = str(url).rstrip("/")
        if not url:
            continue
        if element is not None:
            remote = _master_chain(url)
            if remote is not None and remote != element:
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
        received = sum(int(o.get("value") or 0)
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
        total_out = sum(int(o.get("value") or 0) for o in tx.get("outputs", []))
        rows.append({
            "block": None, "type": tx.get("type"), "txid": tid,
            "received": int(received), "sent": int(sent),
            "fee": int(max(0, total_in - total_out)),
            "confirmations": 0, "mempool": True,
        })
    rows.sort(key=lambda r: r["txid"])
    return rows


def _masters_info(masters: list[str] | None,
                  element: str | None = None) -> list[dict]:
    """Живые мастера выбранной цепочки (мастера других элементов не считаем)."""
    out = []
    for url in masters or []:
        url = str(url).rstrip("/")
        if not url:
            continue
        try:
            r = requests.get(url + "/chain/height", timeout=3)
            h = r.json() if r.status_code == 200 else None
            if element is not None and h and h.get("element") not in (None, element):
                continue
            out.append({"url": url, "ok": bool(h),
                        "height": (h or {}).get("height")})
        except Exception:  # noqa: BLE001
            out.append({"url": url, "ok": False, "height": None})
    return out


ELEMENT_RE = re.compile(r"^[A-Za-z0-9_.+-]{1,64}$")
# Чем каталог отличается от служебного (worker_logs и т.п.): это цепочка.
CHAIN_MARKERS = ("blocks", "blobs", "frames", "checkup", "mempool.json")


def _chain_dir(root: str, element: str | None) -> tuple[str, str]:
    """
    Каталог цепочки выбранного элемента: results/<element>.

    Один элемент (система) — одна цепочка со своим генезисом, UTXO и высотой.
    Пусто — берём систему из config.json (system.name).
    """
    name = str(element or _element()).strip()
    if not ELEMENT_RE.match(name) or name in (".", ".."):
        raise ValueError(f"недопустимое имя элемента: {name!r}")
    return name, os.path.join(_root(root), name)


def list_chains(root: str) -> dict:
    """Доступные цепочки (элементы) в каталоге результатов и их высота."""
    root = _root(root)
    current = _element()
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        names = []
    for name in names:
        d = os.path.join(root, name)
        if name.startswith(".") or not os.path.isdir(d):
            continue
        if not any(os.path.exists(os.path.join(d, m)) for m in CHAIN_MARKERS):
            continue
        try:
            height = max(0, chain.next_seq(d) - 1)
        except Exception:  # noqa: BLE001
            height = 0
        out.append({"name": name, "height": height, "current": name == current})
    out.sort(key=lambda c: (not c["current"], c["name"]))
    return {"ok": True, "root": root, "element": current, "chains": out}


def list_wallets(root: str) -> dict:
    """Имена кошельков для панели (без секретов: имя — не тайна)."""
    root = _root(root)
    db_path = os.path.join(root, "keys.db")
    if not os.path.exists(db_path):
        return {"ok": True, "root": root, "wallets": []}
    conn = wallet_store.open_db(db_path)
    return {"ok": True, "root": root, "wallets": wallet_store.list_wallets(conn)}


def create_wallet(root: str, name: str, password: str) -> dict:
    """Явное создание кошелька из панели. Автосоздания нигде нет."""
    root = _root(root)
    conn = wallet_store.open_db(os.path.join(root, "keys.db"))
    try:
        wallet = wallet_store.create_wallet(conn, name, password)
    except wallet_store.WalletError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "name": wallet.name,
            "address": wallet.latest_key()["address"],
            "wallets": wallet_store.list_wallets(conn)}


def collect(root: str, name: str, password: str, confirmations: int,
            masters: list[str] | None = None,
            element: str | None = None) -> dict:
    """
    Кошелёк по имени и паролю в выбранной цепочке.

    Кошелёк — самостоятельная сущность: роли у него нет. Имя — идентификатор
    (можно иметь несколько кошельков с одним паролем), пароль — ключ.
    """
    root = _root(root)
    db_path = os.path.join(root, "keys.db")
    if not os.path.exists(db_path):
        return {"ok": False, "error": f"нет базы кошельков: {db_path}",
                "root": root, "wallet": None}

    try:
        element, element_dir = _chain_dir(root, element)
    except ValueError as e:
        return {"ok": False, "error": str(e), "root": root, "wallet": None}

    conn = wallet_store.open_db(db_path)
    tip = (chain.next_seq(element_dir) - 1) if os.path.isdir(element_dir) else 0
    tip = max(0, tip)

    try:
        wallet = wallet_store.open_wallet(conn, name, password)
    except wallet_store.WalletError as e:
        return {"ok": False, "root": root, "element": element, "height": tip,
                "error": str(e), "wallet": None,
                "chains": list_chains(root)["chains"]}

    keys = wallet.keys()
    addresses = wallet.addresses()

    balances, txs = {}, []
    mempool = _mempool(masters, element)
    index = _outputs_index(element_dir) if mempool else {}
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

    totals = {"available": 0.0, "pending": 0.0, "total": 0.0}
    for b in balances.values():
        for key in totals:
            totals[key] += int(b.get(key) or 0)

    # Мемпул блокирует потраченные выходы и добавляет будущие поступления.
    locked = sum(t["sent"] for t in txs if t.get("mempool"))
    incoming = sum(t["received"] for t in txs if t.get("mempool"))
    if locked or incoming:
        totals["available"] = max(0.0, totals["available"] - locked)
        totals["pending"] += incoming
        totals["total"] = totals["available"] + totals["pending"]

    return {"ok": True, "root": root, "element": element, "height": tip,
            "wallet": {
                "name": wallet.name,
                "addresses": [{"address": k["address"],
                               "created_at": k["created_at"]} for k in keys],
                "balances": balances,
                "totals": {k: int(v) for k, v in totals.items()},
                "transactions": txs,
            },
            "chains": list_chains(root)["chains"],
            "masters": _masters_info(masters, element),
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
                index[(tid, n)] = (out.get("address"), int(out.get("value") or 0))
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
                "value": int(o.get("value") or 0)}
               for n, o in enumerate(tx.get("outputs", []))]
    total_out = sum(o["value"] for o in outputs)

    fee = None
    if tx.get("type") != "coinbase" and total_in:
        fee = int(total_in - total_out)

    return {
        "ok": True, "block": block, "confirmations": confirmations,
        "mempool": block is None,
        "type": tx.get("type"),
        "txid": tx.get("txid") or chain.txid_of(tx),
        "fee": fee, "inputs": inputs, "outputs": outputs,
        "total_in": int(total_in), "total_out": int(total_out),
        "raw": {k: v for k, v in tx.items() if k != "_file"},
    }


def tx_details(root: str, txid: str, masters: list[str] | None = None,
               element: str | None = None) -> dict:
    """Полная информация о транзакции из блока: входы с адресами, выходы, комиссия."""
    root = _root(root)
    try:
        element, element_dir = _chain_dir(root, element)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
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
                        "value": int(o.get("value") or 0)}
                       for n, o in enumerate(tx.get("outputs", []))]
            total_out = sum(o["value"] for o in outputs)

            fee = None
            if tx.get("type") != "coinbase" and total_in:
                fee = int(total_in - total_out)

            return _tx_payload(tx, index, b["seq"], max(0, tip - b["seq"] + 1))

    # не нашли в блоках — ищем в мемпулах известных мастеров
    for tid, tx in _mempool(masters, element).items():
        if tid == txid:
            payload = _tx_payload(tx, index, None, 0)
            payload["element"] = element
            return payload

    return {"ok": False,
            "error": "транзакция не найдена ни в блоках, ни в мемпуле известных мастеров"}


def locked_utxos(masters: list[str] | None,
                 element: str | None = None) -> set[tuple[str, int]]:
    """Выходы, уже потраченные в мемпулах: их нельзя тратить второй раз."""
    locked = set()
    for tx in _mempool(masters, element).values():
        for i in tx.get("inputs", []):
            if i.get("coinbase"):
                continue
            locked.add((i.get("txid"), int(i.get("vout", -1))))
    return locked


def send(root: str, name: str, password: str, to: str, amount: int, fee: int,
         masters: list[str], confirmations: int,
         element: str | None = None) -> dict:
    root = _root(root)
    conn = wallet_store.open_db(os.path.join(root, "keys.db"))
    try:
        element, element_dir = _chain_dir(root, element)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    if not os.path.isdir(element_dir):
        return {"ok": False, "error": f"нет цепочки {element}: {element_dir}"}

    try:
        wallet = wallet_store.open_wallet(conn, name, password)
    except wallet_store.WalletError as e:
        return {"ok": False, "error": str(e)}

    tx = chain.build_wallet_transfer(wallet, element_dir, to, amount, fee=fee,
                                     confirmations=confirmations,
                                     exclude=locked_utxos(masters, element))
    if tx is None:
        return {"ok": False,
                "error": "недостаточно подтверждённых средств (часть может быть "
                         "заблокирована неподтверждёнными транзакциями)"}

    accepted_by = []
    attempts = []
    for url in masters:
        url = url.rstrip("/")
        try:
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
            "element": element, "wallet": wallet.name,
            "from": sorted(wallet.addresses())}


def change_password(root: str, name: str, old: str, new: str) -> dict:
    """Смена пароля одного кошелька (другие кошельки не затрагиваются)."""
    root = _root(root)
    conn = wallet_store.open_db(os.path.join(root, "keys.db"))
    try:
        wallet = wallet_store.open_wallet(conn, name, old)
    except wallet_store.WalletError as e:
        return {"ok": False, "error": str(e)}
    wallet.change_password(old, new)
    return {"ok": True, "name": wallet.name, "keys": len(wallet.keys())}


def new_address(root: str, name: str, password: str) -> dict:
    """Новый адрес (ключ) в указанном кошельке."""
    conn = wallet_store.open_db(os.path.join(root, "keys.db"))
    try:
        wallet = wallet_store.open_wallet(conn, name, password)
    except wallet_store.WalletError as e:
        return {"ok": False, "error": str(e)}
    key = wallet.add_key()
    return {
        "ok": True,
        "name": wallet.name,
        "address": key["address"],
        "key_id": key["id"],
        "addresses": [k["address"] for k in wallet.keys()],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="JSON-кошелёк для веб-панели")
    ap.add_argument("action",
                    choices=["list", "create", "info", "send", "passwd",
                             "tx", "newkey", "chains"])
    ap.add_argument("--name", default=None,
                    help="имя кошелька (нужно для info/send/passwd/newkey)")
    ap.add_argument("--password", default=None)
    ap.add_argument("--password-env", default="WEBUI_WALLET_PASSWORD")
    ap.add_argument("--results-root", default=DEFAULT_ROOT)
    ap.add_argument("--element", "--system", dest="element", default=None,
                    help="элемент/система: цепочка results/<element> "
                         "(по умолчанию system.name из config.json)")
    ap.add_argument("--confirmations", type=int, default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--amount", type=int, default=0)
    ap.add_argument("--fee", type=int, default=chain.TX_FEE)
    ap.add_argument("--change-to", default=None)
    ap.add_argument("--txid", default=None)
    ap.add_argument("--master", action="append", default=None)
    args = ap.parse_args()

    password = _password(args)
    confirmations = args.confirmations if args.confirmations is not None \
        else _confirmations(args.results_root)

    if args.action == "list":
        result = list_wallets(args.results_root)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("ok") else 1

    if args.action == "create":
        if not args.name:
            print(json.dumps({"ok": False, "error": "нужен --name"}, ensure_ascii=False))
            return 2
        result = create_wallet(args.results_root, args.name, password)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("ok") else 1

    if args.action == "chains":
        result = list_chains(args.results_root)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("ok") else 1

    if args.action == "info":
        masters = [m for m in (args.master or []) if m]
        result = collect(args.results_root, args.name, password, confirmations,
                         masters=masters, element=args.element)
    elif args.action == "tx":
        if not args.txid:
            print(json.dumps({"ok": False, "error": "нужен --txid"}, ensure_ascii=False))
            return 2
        result = tx_details(args.results_root, args.txid,
                            masters=[m for m in (args.master or []) if m],
                            element=args.element)
    elif args.action == "newkey":
        result = new_address(args.results_root, args.name, password)
    elif args.action == "send":
        if not args.to or args.amount <= 0:
            print(json.dumps({"ok": False, "error": "нужны --to и --amount > 0"},
                             ensure_ascii=False))
            return 2
        masters = [m for m in (args.master or [os.environ.get(
            "MASTER_URL", "http://127.0.0.1:8000")]) if m]
        result = send(args.results_root, args.name, password, args.to, args.amount,
                      args.fee, masters, confirmations, element=args.element)
    else:
        if not args.change_to:
            print(json.dumps({"ok": False, "error": "нужен --change-to"},
                             ensure_ascii=False))
            return 2
        result = change_password(args.results_root, args.name, password,
                                 args.change_to)

    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
