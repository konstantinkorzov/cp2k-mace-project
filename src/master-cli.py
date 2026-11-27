#!/usr/bin/env python3
"""
master-cli.py — управление работающим мастером (main.py) из консоли.

Мастер работает в фоне (FastAPI), CLI ходит в его админ-API:

    status            состояние очереди, воркеров и настроек
    queue             краткая сводка очереди
    workers           список воркеров и их активных задач
    config list       показать изменяемые настройки
    config get KEY    значение настройки
    config set KEY V  изменить настройку (и сохранить в config.json)
    pause             перестать выдавать задачи
    resume            возобновить выдачу
    chain [--verify]  блоки цепочки элемента (с проверкой целостности)

Примеры:
    python3 src/master-cli.py status
    python3 src/master-cli.py config set verbose_cp2k false
    python3 src/master-cli.py pause
    python3 src/master-cli.py chain --verify

Адрес мастера: --url (или MASTER_URL), по умолчанию http://127.0.0.1:8000.
Если мастер запущен с MASTER_ADMIN_TOKEN, передайте --token (или тот же env).
"""

import argparse
import json
import os
import sys

import requests

DEFAULT_URL = os.environ.get("MASTER_URL", "http://127.0.0.1:8000")
DEFAULT_TOKEN = os.environ.get("MASTER_ADMIN_TOKEN", "")


def _request(args, method: str, path: str, **kwargs):
    url = args.url.rstrip("/") + path
    params = dict(kwargs.pop("params", {}))
    if args.token:
        # заголовок, а не ?token=: query попадает в логи и историю прокси
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["X-Admin-Token"] = args.token
        kwargs["headers"] = headers
    timeout = kwargs.pop("timeout", args.timeout)
    try:
        resp = requests.request(method, url, params=params, timeout=timeout, **kwargs)
    except requests.RequestException as e:
        print(f"✗ Мастер недоступен по {url}: {e}", file=sys.stderr)
        sys.exit(2)
    if resp.status_code >= 400:
        print(f"✗ HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    try:
        return resp.json()
    except ValueError:
        return resp.text


def _print_table(rows, headers):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


def cmd_status(args):
    data = _request(args, "GET", "/admin/status")
    print(f"Элемент     : {data['element']} ({data['potential']})")
    print(f"Пауза       : {'ДА' if data['paused'] else 'нет'}")
    print(f"Очередь     : pending={data['pending']} active={data['active']} "
          f"verifying={data['verifying']} completed={data['completed']} failed={data['failed']}")
    print(f"Воркеров    : {len(data['workers'])}")
    if data["workers"]:
        rows = [(w["worker_id"], w["name"] or "-", (w["pubkey"] + "…") if w["pubkey"] else "-",
                 w["active_task"] or "-", w["age_sec"] if w["age_sec"] is not None else "-")
                for w in data["workers"]]
        print()
        _print_table(rows, ["worker", "имя", "pubkey", "задача", "возраст,с"])
    print("\nНастройки:")
    for k, v in data["config"].items():
        print(f"  {k} = {v!r}")


def cmd_queue(args):
    data = _request(args, "GET", "/queue-status")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_workers(args):
    data = _request(args, "GET", "/admin/status")
    if not data["workers"]:
        print("Воркеров нет")
        return
    rows = [(w["worker_id"], w["name"] or "-",
             (w["pubkey"] + "…") if w["pubkey"] else "-",
             w["active_task"] or "-", w["age_sec"] if w["age_sec"] is not None else "-")
            for w in data["workers"]]
    _print_table(rows, ["worker", "имя", "pubkey", "задача", "возраст,с"])


def cmd_config(args):
    if args.action == "list":
        data = _request(args, "GET", "/admin/config")
        rows = [(k, v["value"], v["type"]) for k, v in data.items()]
        _print_table(rows, ["ключ", "значение", "тип"])
    elif args.action == "get":
        data = _request(args, "GET", "/admin/config")
        if args.key not in data:
            print(f"✗ Нет настройки {args.key!r}; доступно: {sorted(data)}", file=sys.stderr)
            sys.exit(1)
        print(data[args.key]["value"])
    elif args.action == "set":
        if args.value is None:
            print("✗ Для set нужно значение: config set KEY VALUE", file=sys.stderr)
            sys.exit(2)
        res = _request(args, "POST", "/admin/config", json={"key": args.key, "value": args.value})
        print(f"✓ {res['key']} = {res['value']!r}")


def cmd_pause(args):
    res = _request(args, "POST", "/admin/pause")
    print("⏸ Задачи больше не выдаются" if res.get("paused") else str(res))


def cmd_resume(args):
    res = _request(args, "POST", "/admin/resume")
    print("▶ Выдача задач возобновлена" if not res.get("paused") else str(res))


def cmd_chain(args):
    data = _request(args, "GET", "/admin/chain", params={"verify": str(args.verify).lower()})
    blocks = data.get("blocks", [])
    print(f"Система {data.get('element') or data.get('system')}: блоков {len(blocks)}")
    if blocks:
        _print_table([(b["seq"], b["system"], b["n_atoms"], b["name"]) for b in blocks],
                     ["seq", "элемент", "атомов", "файл"])
    if args.verify:
        chk = data.get("chain", {})
        print()
        if chk.get("ok"):
            print(f"✓ Цепочка целая ({chk.get('blocks')} блоков)")
        else:
            print("✗ Найдены проблемы:")
            for p in chk.get("problems", []):
                print("  -", p)


def cmd_keys(args):
    if args.action == "new":
        res = _request(args, "POST", "/admin/keys/new")
        k = res["key"]
        print(f"✓ Создан ключ #{k['id']}: {k['address']}")
        print(f"  public_key : {k['public_key']}")
        print("  (приватный ключ зашифрован паролем и лежит в SQLite — не выводим)")
        return

    data = _request(args, "GET", "/admin/keys")
    keys = data.get("keys", [])
    if args.action == "current":
        cur = [k for k in keys if k["id"] == data.get("current")]
        print(json.dumps(cur[0] if cur else {}, ensure_ascii=False, indent=2))
        return

    rows = [(k["id"], k["created_at"], k["address"],
             "← coinbase" if k["id"] == data.get("current") else "") for k in keys]
    _print_table(rows, ["id", "создан", "адрес", ""])


def cmd_balances(args):
    data = _request(args, "GET", "/admin/balances")
    print(f"Подтверждений: {data.get('confirmations')}")
    print("\nМастер:")
    if not data.get("master"):
        print("  (пусто)")
    for addr, b in data.get("master", {}).items():
        print(f"  {addr[:16]}…  доступно {b['available']:>10.2f}  "
              f"pending {b['pending']:>7.2f}  всего {b['total']:>10.2f}")
    print("\nВоркеры:")
    if not data.get("workers"):
        print("  (пусто)")
    for addr, b in data.get("workers", {}).items():
        print(f"  {addr[:16]}…  доступно {b['available']:>10.2f}  "
              f"pending {b['pending']:>7.2f}  всего {b['total']:>10.2f}")


def cmd_passwd(args):
    import getpass
    old = getpass.getpass("Текущий пароль мастера: ")
    new = getpass.getpass("Новый пароль: ")
    again = getpass.getpass("Повторите новый пароль: ")
    if new != again:
        print("✗ Пароли не совпадают")
        return
    res = _request(args, "POST", "/admin/passwd", json={"old": old, "new": new})
    print("✓ Пароль мастера изменён" if res.get("status") == "ok" else str(res))


def cmd_peers(args):
    if args.action == "list":
        data = _request(args, "GET", "/admin/peers")
        print(f"Высота: {data.get('height')}")
        if not data.get("peers"):
            print("  (пиров нет)")
        for p in data.get("peers", []):
            print(f"  {p}")
    elif args.action == "add":
        if not args.peer_url:
            print("✗ Нужен URL: peers add http://host:port", file=sys.stderr)
            sys.exit(2)
        res = _request(args, "POST", "/admin/peers", json={"url": args.peer_url})
        print("✓ пиры:", ", ".join(res.get("peers", [])))
    else:
        if not args.peer_url:
            print("✗ Нужен URL: peers remove http://host:port", file=sys.stderr)
            sys.exit(2)
        res = _request(args, "DELETE", "/admin/peers", params={"url": args.peer_url})
        print("✓ пиры:", ", ".join(res.get("peers", [])) or "(нет)")


def cmd_sync(args):
    data = _request(args, "POST", "/admin/sync", timeout=max(float(args.timeout), 600.0))
    cu = data.get("catch_up") or {}
    if not cu:
        print("Пустой ответ")
        return 1
    print(f"Догон: высота {cu.get('height')}, раундов {cu.get('rounds')}, "
          f"принято блоков {cu.get('accepted')}")
    if not cu.get("ok"):
        print("✗ не удалось догнать самую длинную цепочку")
        return 1
    print("✓ локальная цепочка не короче, чем у доступных пиров")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Управление работающим мастером CP2K+MACE",
        epilog=("Примеры: status | config set verbose_cp2k false | pause | resume | "
                "keys new | balances | chain --verify | peers add http://host:8001 | sync"),
    )
    ap.add_argument("--url", default=DEFAULT_URL, help=f"адрес мастера (по умолчанию {DEFAULT_URL})")
    ap.add_argument("--token", default=DEFAULT_TOKEN, help="admin-token (если задан MASTER_ADMIN_TOKEN)")
    ap.add_argument("--timeout", type=float, default=10, help="таймаут запроса, с")

    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="очередь, воркеры, настройки").set_defaults(func=cmd_status)
    sub.add_parser("queue", help="сырой /queue-status (JSON)").set_defaults(func=cmd_queue)
    sub.add_parser("workers", help="список воркеров и их задач").set_defaults(func=cmd_workers)

    p_conf = sub.add_parser("config", help="просмотр и изменение настроек")
    p_conf.add_argument("action", choices=["list", "get", "set"])
    p_conf.add_argument("key", nargs="?", help="имя настройки")
    p_conf.add_argument("value", nargs="?", help="новое значение (для set)")
    p_conf.set_defaults(func=cmd_config)

    sub.add_parser("pause", help="приостановить выдачу задач").set_defaults(func=cmd_pause)
    sub.add_parser("resume", help="возобновить выдачу задач").set_defaults(func=cmd_resume)

    p_chain = sub.add_parser("chain", help="блоки цепочки элемента")
    p_chain.add_argument("--verify", action="store_true", help="проверить целостность цепочки")
    p_chain.set_defaults(func=cmd_chain)

    p_keys = sub.add_parser("keys", help="ключи мастера (SQLite)")
    p_keys.add_argument("action", choices=["list", "new", "current"])
    p_keys.set_defaults(func=cmd_keys)

    sub.add_parser("balances", help="балансы мастера и воркеров").set_defaults(func=cmd_balances)
    sub.add_parser("passwd", help="сменить пароль мастера").set_defaults(func=cmd_passwd)

    p_peers = sub.add_parser("peers", help="другие мастера (пиры)")
    p_peers.add_argument("action", choices=["list", "add", "remove"])
    p_peers.add_argument("peer_url", nargs="?", help="http://host:port")
    p_peers.set_defaults(func=cmd_peers)

    sub.add_parser("sync", help="подтянуть блоки от пиров (с DFT-проверкой)").set_defaults(
        func=cmd_sync)

    return ap


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
