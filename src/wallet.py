#!/usr/bin/env python3
"""
wallet.py — кошелёк по паролю.

Запрашивает пароль, находит в общей базе (results/keys.db) все кошельки,
к которым он подходит (мастер и/или воркеры — соли разные, поэтому даже
одинаковый пароль даёт разные записи), и показывает:

  * роль кошелька (имя записи в базе — служебное, не показывается);
  * элемент (система) — отдельная цепочка, каталог results/<element>;
  * адреса и балансы (доступно / pending);
  * транзакции по этим адресам (приход/расход, комиссия, подтверждения).

Все записи, которые открыл пароль (например worker/smoke и worker/w_props),
для владельца — ОДИН кошелёк: общий баланс и общая история.

Пример:
    python3 src/wallet.py
    python3 src/wallet.py --password root
"""

import argparse
import json
import os
import sys

import requests

import chain
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


def group_wallets(matched, role: str | None = None,
                  name: str | None = None) -> dict:
    """
    {роль: [Wallet, ...]} — кошельки, которые открыл пароль.

    Пароль и есть кошелёк: если под него подошло несколько записей одной роли
    (в базе они названы по-разному — worker/smoke, worker/w_props), для
    владельца это ОДИН кошелёк с несколькими наборами адресов: общий баланс,
    общая история. Имя записи — служебная метка, наружу её не выводим.
    """
    groups: dict[str, list] = {}
    for w in matched:
        if role is not None and w.role != role:
            continue
        if name is not None and w.name != name:
            continue
        groups.setdefault(w.role, []).append(w)
    return groups


class WalletView:
    """
    Несколько записей кошелька как один кошелёк.

    Умеет то же, что Wallet и что нужно chain.sign_transaction:
    addresses()/latest_key()/public_for()/sign(). Ключи каждой записи
    расшифрованы своим паролем, но пароль у них общий — поэтому подписывать
    входы можно из любой записи.
    """

    def __init__(self, wallets):
        self.wallets = list(wallets)
        self.role = self.wallets[0].role if self.wallets else "worker"

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

    def _owner(self, address: str):
        for w in self.wallets:
            if address in w.addresses():
                return w
        raise wallet_store.WalletError(f"адрес {address} не принадлежит кошельку")

    def public_for(self, address: str) -> bytes:
        return self._owner(address).public_for(address)

    def sign(self, address: str, message: bytes) -> bytes:
        return self._owner(address).sign(address, message)


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


def show(conn, password: str, element_dir: str, confirmations: int,
         element_name: str = "") -> int:
    groups = group_wallets(wallet_store.find_by_password(conn, password))
    if not groups:
        print("✗ Пароль не подходит ни к одному кошельку")
        return 1

    chain_name = element_name or os.path.basename(os.path.abspath(element_dir))
    for role, wallets in groups.items():
        view = WalletView(wallets)
        keys = view.keys()
        addrs = view.addresses()
        print("=" * 70)
        print(f"Кошелёк: {role}   ·   цепочка: {chain_name}")
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


def send_coins(wallet, element_dir: str, to: str, amount: int, fee: int,
               masters: list[str], confirmations: int) -> int:
    tx = chain.build_wallet_transfer(wallet, element_dir, to, amount, fee=fee,
                                     confirmations=confirmations)
    if tx is None:
        print("✗ Недостаточно подтверждённых средств на адресах кошелька")
        return 1

    tid = tx["txid"]
    print(f"Транзакция {tid}")
    print(f"  входов: {len(tx['inputs'])}, выходов: {len(tx['outputs'])}, "
          f"комиссия: {fee:g}")
    for o in tx["outputs"]:
        print(f"  → {o['address'][:16]}…  {o['value']:g}")

    for url in masters:
        try:
            r = requests.post(url.rstrip("/") + "/tx/submit", json={"tx": tx},
                              timeout=30)
            if r.status_code == 200 and r.json().get("status") == "accepted":
                print(f"✓ Принята мастером {url}")
                return 0
            print(f"  {url}: {r.status_code} {r.text[:120]}")
        except Exception as e:  # noqa: BLE001
            print(f"  {url}: {e}")
    print("✗ Ни один мастер не принял транзакцию")
    return 1


def main():
    ap = argparse.ArgumentParser(description="Кошелёк: балансы, транзакции и смена пароля")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"база кошельков (по умолчанию {DEFAULT_DB})")
    ap.add_argument("--password", default=None, help="пароль (иначе запрос)")
    ap.add_argument("--change-to", default=None,
                    help="новый пароль (для смены; уточните --role/--name, если кошельков несколько)")
    ap.add_argument("--role", choices=["master", "worker"], default=None)
    ap.add_argument("--name", default=None,
                    help="служебная метка записи (обычно не нужна)")
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

    password = args.password
    if not password:
        import getpass
        password = getpass.getpass("Пароль: ")

    conn = wallet_store.open_db(args.db)

    if args.change_to:
        wallets = wallet_store.find_by_password(conn, password)
        targets = [w for w in wallets
                   if (args.role is None or w.role == args.role)
                   and (args.name is None or w.name == args.name)]
        groups = group_wallets(wallets, role=args.role, name=args.name)
        targets = [w for ws in groups.values() for w in ws]
        if not targets:
            print("✗ Подходящих кошельков нет")
            sys.exit(1)
        for w in targets:
            w.change_password(password, args.change_to)
        print(f"✓ Пароль изменён ({len(targets)} записей, роли: "
              f"{', '.join(sorted(groups))})")
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

    if args.send_to:
        if args.amount is None or args.amount <= 0:
            print("✗ Нужна положительная сумма: --amount", file=sys.stderr)
            sys.exit(2)
        groups = group_wallets(wallet_store.find_by_password(conn, password),
                               role=args.role, name=args.name)
        if len(groups) != 1:
            print(f"✗ Пароль открывает {len(groups)} кошелька — укажите --role")
            for role in sorted(groups):
                print(f"  {role}")
            sys.exit(1)
        wallet = WalletView(next(iter(groups.values())))
        masters = [m.rstrip("/") for m in (args.master or
                   [os.environ.get("MASTER_URL", "http://127.0.0.1:8000")]) if m]
        sys.exit(send_coins(wallet, element_dir, args.send_to, args.amount,
                            args.fee, masters, confirmations))

    sys.exit(show(conn, password, element_dir, confirmations, element))


if __name__ == "__main__":
    main()
