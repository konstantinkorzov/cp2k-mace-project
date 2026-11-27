#!/usr/bin/env python3
"""
wallet.py — кошелёк по паролю.

Запрашивает пароль, находит в общей базе (results/keys.db) все кошельки,
к которым он подходит (мастер и/или воркеры — соли разные, поэтому даже
одинаковый пароль даёт разные записи), и показывает:

  * элемент (система) — отдельная цепочка, каталог results/<element>;
  * адреса и балансы (доступно / pending);
  * транзакции по этим адресам (приход/расход, комиссия, подтверждения).

Роли у кошелька нет: мастер и воркер — это процессы, которые им пользуются.
Пароль — это кошелёк, все его адреса дают один баланс и одну историю.

Пример:
    python3 src/wallet.py
    python3 src/wallet.py --password root
"""

import argparse
import json
import os
import sqlite3
import sys

import requests

import chain
import crypto
import wallet_store

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(PROJECT_ROOT, "results", "keys.db")


def _save_confirmations(value: int) -> None:
    """Пишет coinbase_confirmations в src/config.json, сохраняя владельца файла."""
    path = os.path.join(PROJECT_ROOT, "src", "config.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    entry = raw.get("coinbase_confirmations")
    if isinstance(entry, dict) and "value" in entry:
        entry["value"] = int(value)
    else:
        raw["coinbase_confirmations"] = {"value": int(value)}
    st = os.stat(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    try:
        os.chown(tmp, st.st_uid, st.st_gid)
    except OSError:
        pass
    os.replace(tmp, path)


def _element() -> str:
    """
    Имя текущей системы (элемента) из config.json.

    Раньше читался ключ `element`, теперь система описывается в `system.name`
    (там же состав). Старый формат поддерживаем как запасной вариант — иначе
    кошелёк молча смотрел бы в results/Cu при системе Al2O3.
    """
    try:
        with open(os.path.join(PROJECT_ROOT, "src", "config.json"), encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:  # noqa: BLE001
        return "Cu"
    system = raw.get("system")
    if isinstance(system, dict):
        value = system.get("value") if isinstance(system.get("value"), dict) else system
        name = str((value or {}).get("name") or "").strip()
        if name:
            return name
    legacy = raw.get("element")
    if isinstance(legacy, dict):
        legacy = legacy.get("value")
    return str(legacy or "Cu").strip() or "Cu"


def _address_map(element_dir: str) -> dict:
    """(txid, vout) -> адрес по всем блокам (чтобы понять, чьи входы)."""
    out = {}
    for b in chain.list_blocks(element_dir):
        try:
            for tx in chain.read_block_txs(b["path"]):
                tid = tx.get("txid") or chain.txid_of(tx)
                for vout, o in enumerate(tx.get("outputs", [])):
                    out[(tid, vout)] = o["address"]
        except Exception:  # noqa: BLE001
            continue
    return out


def _transactions(element_dir: str, addresses: set[str]) -> list[dict]:
    addr_map = _address_map(element_dir)
    tip = chain.next_seq(element_dir) - 1
    rows = []
    for b in chain.list_blocks(element_dir):
        try:
            txs = chain.read_block_txs(b["path"])
        except Exception:  # noqa: BLE001
            continue
        for tx in txs:
            tid = tx.get("txid") or chain.txid_of(tx)
            received = sum(o["value"] for o in tx.get("outputs", [])
                           if o["address"] in addresses)
            sent = 0.0
            for i in tx.get("inputs", []):
                if i.get("coinbase"):
                    continue
                if addr_map.get((i["txid"], int(i["vout"]))) in addresses:
                    # сумму входа берём из соответствующего выхода
                    sent += _output_value(element_dir, i["txid"], int(i["vout"]))
            fee = 0.0
            if tx.get("type") != "coinbase":
                total_in = sum(_output_value(element_dir, i["txid"], int(i["vout"]))
                               for i in tx.get("inputs", []) if not i.get("coinbase"))
                total_out = sum(o["value"] for o in tx.get("outputs", []))
                fee = int(total_in - total_out)
            if received or sent:
                rows.append({
                    "block": b["seq"], "type": tx.get("type"), "txid": tid,
                    "received": received, "sent": sent, "fee": fee,
                    "confirmations": max(0, tip - b["seq"] + 1),
                })
    rows.sort(key=lambda r: (r["block"], r["type"]))
    return rows


_value_cache: dict = {}   # (каталог цепочки, txid, vout) -> сумма
_value_loaded: set = set()


def _output_value(element_dir: str, txid: str, vout: int) -> int:
    # Кэш привязан к каталогу цепочки: один и тот же txid бывает в разных
    # цепочках (Cu/Fe), и суммы оттуда нельзя смешивать. Грузим каждый
    # каталог один раз — иначе второй элемент увидел бы пустой кэш первого.
    dir_key = os.path.abspath(element_dir)
    key = (dir_key, txid, vout)
    if dir_key not in _value_loaded:
        for b in chain.list_blocks(element_dir):
            try:
                for tx in chain.read_block_txs(b["path"]):
                    tid = tx.get("txid") or chain.txid_of(tx)
                    for i, o in enumerate(tx.get("outputs", [])):
                        _value_cache[(dir_key, tid, i)] = int(o["value"])
            except Exception:  # noqa: BLE001
                continue
        _value_loaded.add(dir_key)
    return _value_cache.get(key, 0.0)


def _prompt(label: str) -> str:
    import getpass
    return getpass.getpass(label)


def show(wallet, element_dir: str, confirmations: int,
         element_name: str = "") -> int:
    chain_name = element_name or os.path.basename(os.path.abspath(element_dir))
    keys = wallet.keys()
    addrs = wallet.addresses()
    print("=" * 70)
    print(f"Кошелёк «{wallet.name}»   ·   цепочка: {chain_name}")
    print(f"Адресов: {len(keys)}")
    for k in keys:
        print(f"  {k['address']}  (создан {k['created_at']})")

    bal = chain.balances(element_dir, addrs, confirmations)
    print("\nБалансы:")
    if not bal:
        print("  (пусто)")
    for a, b in bal.items():
        print(f"  {a[:16]}…  доступно {b['available']:>10.2f}  "
              f"pending {b['pending']:>8.2f}  всего {b['total']:>10.2f}")

    print("\nТранзакции:")
    rows = _transactions(element_dir, addrs)
    if not rows:
        print("  (нет)")
    for r in rows:
        if r["received"]:
            sign = f"+{r['received']:.2f}"
        else:
            sign = f"-{r['sent']:.2f}"
        print(f"  блок {r['block']:>3}  {r['type']:<8} {sign:>10}  "
              f"подтв. {r['confirmations']:>3}  {r['txid'][:16]}…"
              + (f"  комиссия {r['fee']:.8f}" if r["fee"] else ""))
    return 0


def backup_db(db_path: str, dest: str) -> str:
    """Копия базы кошельков (sqlite backup API) — страховка единственной копии ключей."""
    dest = os.path.abspath(dest)
    if os.path.exists(dest):
        raise FileExistsError(dest)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    src = sqlite3.connect(db_path)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass
    return dest


def sign_text(wallet, address: str, text: str) -> str:
    return wallet.sign(address, text.encode("utf-8")).hex()


def verify_text(wallet, address: str, text: str, signature: str) -> bool:
    return crypto.verify(wallet.public_for(address), text.encode("utf-8"),
                         bytes.fromhex(signature))


def _totals(element_dir: str, addresses: set[str], confirmations: int) -> dict:
    totals = {"available": 0, "pending": 0, "total": 0}
    for b in chain.balances(element_dir, addresses, confirmations).values():
        for k in totals:
            totals[k] += int(b.get(k) or 0)
    return totals


def show_info(wallet, element: str, element_dir: str, confirmations: int) -> int:
    totals = _totals(element_dir, wallet.addresses(), confirmations)
    print(f"Кошелёк «{wallet.name}»")
    print(f"  цепочка : {element} (results/{element})")
    print(f"  высота  : {max(0, chain.next_seq(element_dir) - 1)}")
    print(f"  адресов : {len(wallet.keys())}")
    print(f"  доступно: {totals['available']}  pending: {totals['pending']}  "
          f"всего: {totals['total']}")
    return 0


def show_balance(wallet, element_dir: str, confirmations: int) -> int:
    totals = _totals(element_dir, wallet.addresses(), confirmations)
    print(f"available={totals['available']} pending={totals['pending']} "
          f"total={totals['total']}")
    return 0


def show_unspent(wallet, element_dir: str, confirmations: int) -> int:
    """Выходы кошелька: сколько подтверждений и можно ли их тратить сейчас."""
    next_height = chain.next_seq(element_dir)
    rows = []
    for (txid, vout), info in sorted(chain.utxo_set(element_dir).items()):
        if info["address"] not in wallet.addresses():
            continue
        need = chain.COINBASE_MATURITY if info.get("coinbase") else max(0, confirmations)
        ready = info["height"] + need <= next_height
        rows.append(f"  {txid[:16]}…:{vout:<3} {int(info['value']):>12}  "
                    f"подтв. {max(0, next_height - 1 - info['height']):>3}  "
                    f"{info['address'][:16]}…  "
                    + ("доступен" if ready else "ждёт подтверждений")
                    + ("  coinbase" if info.get("coinbase") else ""))
    if not rows:
        print("  (непотраченных выходов нет)")
    for r in rows:
        print(r)
    return 0


def show_tx(element_dir: str, wallet, txid: str) -> int:
    """Детали транзакции из блоков: входы (с адресами), выходы, комиссия."""
    addrs = wallet.addresses()
    owners = _address_map(element_dir)
    for b in chain.list_blocks(element_dir):
        try:
            txs = chain.read_block_txs(b["path"])
        except Exception:  # noqa: BLE001
            continue
        for tx in txs:
            tid = tx.get("txid") or chain.txid_of(tx)
            if tid != txid:
                continue
            print(f"Транзакция {tid}")
            print(f"  блок {b['seq']}   тип {tx.get('type')}")
            total_in = 0
            for i in tx.get("inputs", []):
                if i.get("coinbase"):
                    print("  ← coinbase (новые монеты)")
                    continue
                value = _output_value(element_dir, i["txid"], int(i["vout"]))
                owner = owners.get((i["txid"], int(i["vout"])))
                total_in += value
                mark = " (наш)" if owner in addrs else ""
                print(f"  ← {int(value):>12}  {str(owner)[:16]}…{mark}")
            total_out = 0
            for n, o in enumerate(tx.get("outputs", [])):
                total_out += int(o["value"])
                mark = " (наш)" if o["address"] in addrs else ""
                print(f"  → #{n:<2} {int(o['value']):>9}  {o['address'][:16]}…{mark}")
            if tx.get("type") != "coinbase" and total_in:
                print(f"  комиссия: {total_in - total_out}")
            return 0
    print(f"✗ транзакция {txid} не найдена в блоках цепочки "
          f"{os.path.basename(os.path.normpath(element_dir))}")
    print("  если перевод только что отправлен — он ещё в мемпуле у мастера")
    return 1


def show_address(wallet, element_dir: str, address: str, confirmations: int) -> int:
    key = next((k for k in wallet.keys() if k["address"] == address), None)
    print(f"Адрес {address}")
    if key:
        print(f"  наш: ключ #{key['id']} (создан {key['created_at']})")
    else:
        print("  наш: нет (чужой адрес — только баланс по цепочке)")
    bal = chain.balances(element_dir, {address}, confirmations).get(address)
    if bal:
        print(f"  доступно {int(bal['available'])}, pending {int(bal['pending'])}, "
              f"всего {int(bal['total'])}")
    else:
        print("  баланс: пусто")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Кошелёк: создание, балансы, переводы, смена пароля")
    ap.add_argument("--db", default=DEFAULT_DB,
                    help=f"база кошельков (по умолчанию {DEFAULT_DB})")
    ap.add_argument("--list", action="store_true", help="список кошельков (имена)")
    ap.add_argument("--create", metavar="NAME", default=None,
                    help="создать кошелёк с этим именем (пароль — --password или запрос)")
    ap.add_argument("--wallet", "--name", dest="wallet", default=None,
                    help="имя кошелька для операций с балансом")
    ap.add_argument("--password", default=None, help="пароль кошелька (иначе запрос)")
    ap.add_argument("--change-to", default=None,
                    help="новый пароль для этого кошелька")
    ap.add_argument("--element", "--system", dest="element", default=None,
                    help="элемент/система: цепочка results/<element> "
                         "(по умолчанию system.name из config.json)")
    ap.add_argument("--confirmations", type=int, default=None,
                    help="подтверждений до доступности баланса (по умолчанию из config.json)")
    ap.add_argument("--set-confirmations", type=int, default=None,
                    help="сохранить число подтверждений в config.json "
                         "(для мастеров применится при следующем запуске)")
    ap.add_argument("--send-to", default=None, help="адрес получателя")
    ap.add_argument("--amount", type=int, default=None, help="сумма перевода (целые монеты)")
    ap.add_argument("--fee", type=int, default=chain.TX_FEE, help="комиссия (целые монеты)")
    ap.add_argument("--master", action="append", default=None,
                    help="URL мастера для отправки tx (можно несколько)")
    ap.add_argument("--new-address", action="store_true",
                    help="добавить новый адрес (ключ) в кошелёк")
    ap.add_argument("--rename", metavar="NEW", default=None,
                    help="переименовать кошелёк (ключи и пароль не меняются)")
    ap.add_argument("--info", action="store_true",
                    help="сводка: имя, цепочка, высота, адресов, баланс")
    ap.add_argument("--balance", action="store_true",
                    help="только числа баланса (available=/pending=/total=)")
    ap.add_argument("--unspent", action="store_true",
                    help="непотраченные выходы кошелька (listunspent)")
    ap.add_argument("--tx", metavar="TXID", default=None,
                    help="детали транзакции из блоков")
    ap.add_argument("--address", metavar="ADDR", default=None,
                    help="информация об адресе (или адрес для подписи)")
    ap.add_argument("--backup", metavar="PATH", default=None,
                    help="копия базы кошельков (пароль не нужен)")
    ap.add_argument("--sign-message", metavar="TEXT", default=None,
                    help="подписать текст ключом адреса (--address или последний)")
    ap.add_argument("--verify-message", metavar="TEXT", default=None,
                    help="проверить подпись для адреса из этого кошелька")
    ap.add_argument("--signature", default=None,
                    help="подпись в hex для --verify-message")
    args = ap.parse_args()

    if args.set_confirmations is not None:
        value = int(args.set_confirmations)
        if value < 0:
            print("✗ --set-confirmations не может быть отрицательным", file=sys.stderr)
            sys.exit(2)
        _save_confirmations(value)
        print(f"✓ coinbase_confirmations = {value} "
              f"(применится к следующему запуску мастера)")
        return

    conn = wallet_store.open_db(args.db)

    if args.list:
        rows = wallet_store.list_wallets(conn)
        if not rows:
            print("кошельков нет — создайте: python3 src/wallet.py --create NAME")
            return
        for w in rows:
            print(f"  {w['name']:<24} адресов {w['keys']:>3}   создан {w['created_at']}")
        return

    if args.create:
        password = args.password or _prompt("Новый пароль: ")
        if not args.password:
            if _prompt("Повторите пароль: ") != password:
                print("✗ пароли не совпадают", file=sys.stderr)
                sys.exit(2)
        try:
            made = wallet_store.create_wallet(conn, args.create, password)
        except wallet_store.WalletError as e:
            print(f"✗ {e}", file=sys.stderr)
            sys.exit(2)
        print(f"✓ кошелёк «{made.name}» создан, адрес {made.latest_key()['address']}")
        return

    if args.backup:
        try:
            dest = backup_db(args.db, args.backup)
        except FileExistsError:
            print(f"✗ файл уже существует: {args.backup}", file=sys.stderr)
            sys.exit(2)
        print(f"✓ копия базы кошельков: {dest}")
        return

    if not args.wallet:
        print("✗ нужно имя кошелька: --wallet NAME "
              "(список — --list, создать — --create NAME)", file=sys.stderr)
        sys.exit(2)

    password = args.password or _prompt(f"Пароль кошелька «{args.wallet}»: ")
    try:
        wallet = wallet_store.open_wallet(conn, args.wallet, password)
    except wallet_store.WalletError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)

    if args.rename:
        try:
            renamed = wallet_store.rename_wallet(conn, wallet.name, args.rename)
        except wallet_store.WalletError as e:
            print(f"✗ {e}", file=sys.stderr)
            sys.exit(2)
        print(f"✓ кошелёк «{wallet.name}» переименован в «{renamed['name']}»")
        return

    if args.new_address:
        key = wallet.add_key()
        print(f"✓ новый адрес: {key['address']} (ключ #{key['id']})")
        return

    if args.sign_message is not None:
        addr = args.address or (wallet.latest_key() or {}).get("address")
        if not addr:
            print("✗ в кошельке нет адресов", file=sys.stderr)
            sys.exit(2)
        print(f"адрес: {addr}")
        print(f"подпись: {sign_text(wallet, addr, args.sign_message)}")
        return

    if args.verify_message is not None:
        if not args.signature:
            print("✗ нужна --signature HEX", file=sys.stderr)
            sys.exit(2)
        addr = args.address or (wallet.latest_key() or {}).get("address")
        ok = verify_text(wallet, addr, args.verify_message, args.signature)
        print("✓ подпись верна" if ok else "✗ подпись не совпадает")
        sys.exit(0 if ok else 1)

    if args.change_to:
        wallet.change_password(password, args.change_to)
        print(f"✓ Пароль кошелька «{wallet.name}» изменён")
        return

    if args.confirmations is not None:
        confirmations = args.confirmations
    else:
        try:
            with open(os.path.join(PROJECT_ROOT, "src", "config.json"), encoding="utf-8") as f:
                confirmations = int(json.load(f).get("coinbase_confirmations", {}).get("value", 1))
        except Exception:  # noqa: BLE001
            confirmations = 1

    element = args.element or _element()
    element_dir = os.path.join(PROJECT_ROOT, "results", element)

    if args.info:
        sys.exit(show_info(wallet, element, element_dir, confirmations))

    if args.balance:
        sys.exit(show_balance(wallet, element_dir, confirmations))

    if args.unspent:
        sys.exit(show_unspent(wallet, element_dir, confirmations))

    if args.tx:
        sys.exit(show_tx(element_dir, wallet, args.tx))

    if args.address:
        sys.exit(show_address(wallet, element_dir, args.address, confirmations))

    if args.send_to:
        if args.amount is None or args.amount <= 0:
            print("✗ Нужна положительная сумма: --amount", file=sys.stderr)
            sys.exit(2)
        if not chain.ADDRESS_RE.match(str(args.send_to)):
            print("✗ Адрес получателя должен быть 64 hex (sha256 ключа)",
                  file=sys.stderr)
            sys.exit(2)
        masters = [m.rstrip("/") for m in (args.master or
                   [os.environ.get("MASTER_URL", "http://127.0.0.1:8000")]) if m]
        sys.exit(send_coins(wallet, element_dir, args.send_to, args.amount,
                            args.fee, masters, confirmations))

    sys.exit(show(wallet, element_dir, confirmations, element))


if __name__ == "__main__":
    main()
