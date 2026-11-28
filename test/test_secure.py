#!/usr/bin/env python3
"""
Тесты транспортного шифрования (offline, без сети).

    python3 test/test_secure.py
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import secure  # noqa: E402

RESULTS = []


def check(name, ok, extra=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'OK ' if ok else 'FAIL'}] {name} {extra}")


def main():
    a = os.urandom(32)
    b = os.urandom(32)
    priv_a, pub_a = secure.transport_keypair(a)
    priv_b, pub_b = secure.transport_keypair(b)

    # ключи детерминированы и различны
    check("ключ узла детерминирован", secure.transport_public(a) == pub_a)
    check("ключи разных узлов различаются", pub_a != pub_b)

    # hello: самоподпись проверяется
    hello = secure.hello_payload(a)
    check("hello проходит проверку подписи", secure.verify_hello(hello) is None)
    bad = dict(hello, x=pub_b.hex())
    check("подмена ключа в hello ловится", secure.verify_hello(bad) is not None)

    # запрос: расшифровывается сервером
    env = secure.seal_request(a, pub_b, "GET", "/get-task",
                              query={"worker_id": "w1", "session": "s3cret"})
    guard = secure.ReplayGuard()
    from_hex, method, path, query, body = secure.open_request(
        priv_b, pub_b, env, "/get-task", guard)
    check("конверт расшифрован", (method, path) == ("GET", "/get-task"))
    check("query (включая session) внутри шифра", query.get("session") == "s3cret")
    check("отправитель определён по ключу", from_hex == pub_a.hex())

    # повтор не проходит
    try:
        secure.open_request(priv_b, pub_b, env, "/get-task", guard)
        check("повтор отклонён", False)
    except ValueError as e:
        check("повтор отклонён", "повтор" in str(e), str(e))

    # чужой ключ не расшифрует
    env2 = secure.seal_request(a, pub_b, "POST", "/submit-result", body=b'{"x":1}')
    try:
        secure.open_request(secure.transport_keypair(os.urandom(32))[0],
                            pub_a, env2, "/submit-result")
        check("чужой узел не расшифровывает", False)
    except Exception:
        check("чужой узел не расшифровывает", True)

    # подмена шифртекста
    env3 = json.loads(json.dumps(env2))
    raw = bytearray(bytes.fromhex("00") if False else b"")
    import base64
    ct = bytearray(base64.b64decode(env3["c"]))
    ct[0] ^= 0xFF
    env3["c"] = base64.b64encode(bytes(ct)).decode()
    try:
        secure.open_request(priv_b, pub_b, env3, "/submit-result")
        check("подмена ciphertext ловится", False)
    except Exception:
        check("подмена ciphertext ловится", True)

    # метка времени вне допуска
    old = secure.seal_request(a, pub_b, "GET", "/x", ts=int(time.time()) - 3600)
    try:
        secure.open_request(priv_b, pub_b, old, "/x")
        check("старый конверт отклонён", False)
    except ValueError as e:
        check("старый конверт отклонён", "время" in str(e), str(e))

    # TOFU: ключ запоминается, смена — отказ
    trust = os.path.join(tempfile.mkdtemp(), "trust.json")
    store = secure.TrustStore(trust)
    check("первое знакомство пинует ключ",
          store.pin("http://m1:8000", pub_a.hex()) is None
          and secure.TrustStore(trust).get("http://m1:8000") == pub_a.hex())
    check("смена ключа ловится",
          secure.TrustStore(trust).pin("http://m1:8000", pub_b.hex()) is not None)

    ok = sum(1 for _n, passed in RESULTS if passed)
    print(f"\nИТОГО: {ok}/{len(RESULTS)}")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
