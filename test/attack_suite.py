#!/usr/bin/env python3
"""
Стресс-тесты протокола: атаки на блоки, транзакции и приём данных.

Принцип: каждый тест строит атаку и проверяет, что ПРОТОКОЛ ЕЁ ОТКЛОНЯЕТ.
Если атака проходит — это уязвимость, и тест падает (exit 1).

    python3 test/attack_suite.py

CP2K не нужен: проверяются правила блоков/транзакций и валидаторы.
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chain          # noqa: E402
import crypto         # noqa: E402
import cp2k_runner    # noqa: E402

RESULTS = []


def check(name, ok, extra="", attack=False):
    RESULTS.append((name, bool(ok), attack, extra))
    tag = "OK  " if ok else "FAIL"
    kind = "атака отбита" if (attack and ok) else ("АТАКА ПРОШЛА" if attack else "")
    print(f"[{tag}] {name} {kind} {extra}")


def new_element():
    """Каталог элемента с именем Cu (verify_block сверяет имя каталога)."""
    d = os.path.join(tempfile.mkdtemp(prefix="attack_"), "Cu")
    os.makedirs(d, exist_ok=True)
    return d


def make_state(element_dir, data=b"fake-wfn-state" * 200):
    p = os.path.join(element_dir, "state.wfn")
    with open(p, "wb") as f:
        f.write(data)
    return chain.store_blob(element_dir, p)


def make_dft(element_dir, n_atoms=2, nonce=0, energy=-42.0):
    """Геометрия должна следовать из (prev_id, nonce) — правило v2."""
    prev = chain.previous_hash(element_dir, chain.next_seq(element_dir))
    geom = chain.expected_geometry({
        "prev_id": prev, "nonce": nonce, "n_atoms": n_atoms, "dim": 3,
        "atom_diameter": 2.0, "element": "Cu",
    })
    return {"energy_ev": energy,
            "forces_ev_ang": [[0.0, 0.0, 0.0]] * n_atoms,
            "geometry": np.asarray(geom).tolist()}


CP2K = {"cp2k_version": "2024.3",
        "cp2k_build": cp2k_runner.CP2K_BUILD_CPU,
        "cp2k_params": dict(cp2k_runner.CP2K_PARAMS)}


def write_block(element_dir, txs, dft=None, sha=None, rel=None, n_atoms=2, nonce=0):
    sha = sha or make_state(element_dir)[0]
    rel = rel or f"blobs/{sha}.wfn"
    return chain.write_block(element_dir, "Cu", n_atoms, sha, rel, txs,
                             dft=dft or make_dft(element_dir, n_atoms, nonce),
                             nonce=nonce, **CP2K)


def keypair():
    seed = os.urandom(32)
    pub = crypto.public_key_from_seed(seed)
    return seed, pub, crypto.address_from_public(pub)


def sign_tx(tx, seed, pub):
    tid = chain.txid_of(tx)
    for i in tx["inputs"]:
        i["public_key"] = pub.hex()
        i["signature"] = crypto.sign(seed, bytes.fromhex(tid)).hex()
    tx["txid"] = chain.txid_of(tx)
    return tx


def transfer(inputs, outputs):
    return {"version": 1, "type": "transfer",
            "inputs": [{"txid": t, "vout": v} for t, v in inputs],
            "outputs": outputs}


# ---------------------------------------------------------------- атаки

def test_valid_chain_sanity():
    d = new_element()
    try:
        seed, pub, addr = keypair()
        cb = chain.make_coinbase(1, addr, chain.COINBASE_VALUE)
        write_block(d, [cb])
        v = chain.verify_chain(d, "Cu")
        check("эталон: корректная цепочка проходит проверку", v["ok"], str(v["problems"][:3]))
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_coinbase_overpay():
    d = new_element()
    try:
        _, _, addr = keypair()
        cb = chain.make_coinbase(1, addr, 10 ** 9)
        write_block(d, [cb])
        v = chain.verify_chain(d, "Cu")
        check("coinbase 1 000 000 вместо 50+комиссии", not v["ok"], attack=True)
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_negative_output_inflation():
    """Выход с отрицательной суммой + большой положительный = эмиссия."""
    d = new_element()
    try:
        seed, pub, addr = keypair()
        cb = chain.make_coinbase(1, addr, chain.COINBASE_VALUE)
        write_block(d, [cb])
        tx = sign_tx(transfer([(chain.txid_of(cb), 0)],
                              [{"value": -1_000_000, "address": addr},
                               {"value": chain.COINBASE_VALUE + 1_000_000, "address": addr}]), seed, pub)
        write_block(d, [chain.make_coinbase(2, addr, chain.COINBASE_VALUE), tx])
        v = chain.verify_chain(d, "Cu")
        check("отрицательный выход (эмиссия 1 000 000)", not v["ok"],
              str(v["problems"][:2]), attack=True)
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_nan_output():
    d = new_element()
    try:
        seed, pub, addr = keypair()
        cb = chain.make_coinbase(1, addr, chain.COINBASE_VALUE)
        write_block(d, [cb])
        tx = sign_tx(transfer([(chain.txid_of(cb), 0)],
                              [{"value": float("nan"), "address": addr},
                               {"value": chain.COINBASE_VALUE, "address": addr}]), seed, pub)
        write_block(d, [chain.make_coinbase(2, addr, chain.COINBASE_VALUE), tx])
        v = chain.verify_chain(d, "Cu")
        check("NaN в сумме выхода", not v["ok"], str(v["problems"][:2]), attack=True)
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_coinbase_maturity_policy():
    """coinbase нельзя тратить в блоке его создания (зрелость — консенсус)."""
    d = new_element()
    try:
        seed, pub, addr = keypair()
        cb = chain.make_coinbase(1, addr, chain.COINBASE_VALUE)
        write_block(d, [cb])
        tx = sign_tx(transfer([(chain.txid_of(cb), 0)],
                              [{"value": 40_000_000, "address": addr}]), seed, pub)
        ok, probs = chain.validate_tx(tx, d, height=1, confirmations=1)
        check("трата coinbase в том же блоке (height=1)", not ok,
              str(probs[:2]), attack=True)
        # зрелость — консенсус-константа, а не настройка конфига
        check("зрелость coinbase зафиксирована в протоколе",
              isinstance(chain.COINBASE_MATURITY, int) and chain.COINBASE_MATURITY >= 1,
              f"COINBASE_MATURITY={chain.COINBASE_MATURITY}")
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_double_spend_same_block():
    d = new_element()
    try:
        seed, pub, addr = keypair()
        cb = chain.make_coinbase(1, addr, chain.COINBASE_VALUE)
        write_block(d, [cb])
        t1 = sign_tx(transfer([(chain.txid_of(cb), 0)],
                              [{"value": 49_900_000, "address": addr}]), seed, pub)
        t2 = sign_tx(transfer([(chain.txid_of(cb), 0)],
                              [{"value": 49_800_000, "address": addr}]), seed, pub)
        write_block(d, [chain.make_coinbase(2, addr, chain.COINBASE_VALUE), t1, t2])
        v = chain.verify_chain(d, "Cu")
        check("двойная трата одного входа в блоке", not v["ok"],
              str(v["problems"][:2]), attack=True)
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_wrong_key_signature():
    d = new_element()
    try:
        seed_a, pub_a, addr_a = keypair()
        seed_b, pub_b, _ = keypair()
        cb = chain.make_coinbase(1, addr_a, 50.0)
        write_block(d, [cb])
        tx = sign_tx(transfer([(chain.txid_of(cb), 0)],
                              [{"value": 40_000_000, "address": addr_a}]), seed_b, pub_b)
        write_block(d, [chain.make_coinbase(2, addr_a, 50.0), tx])
        v = chain.verify_chain(d, "Cu")
        check("подпись чужим ключом", not v["ok"], str(v["problems"][:2]), attack=True)
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_tampered_tx_after_signature():
    d = new_element()
    try:
        seed, pub, addr = keypair()
        cb = chain.make_coinbase(1, addr, chain.COINBASE_VALUE)
        write_block(d, [cb])
        tx = sign_tx(transfer([(chain.txid_of(cb), 0)],
                              [{"value": 40_000_000, "address": addr}]), seed, pub)
        tx["outputs"][0]["value"] = 49_900_000        # подменили сумму после подписи
        write_block(d, [chain.make_coinbase(2, addr, chain.COINBASE_VALUE), tx])
        v = chain.verify_chain(d, "Cu")
        check("подмена выхода после подписи (txid)", not v["ok"],
              str(v["problems"][:2]), attack=True)
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_rewrite_history_needs_work():
    """Подмена блока с пересчётом всех хэшей, но без нового DFT — отбита.

    Атакующий правит dft первого блока, пересчитывает block_id и prev_id
    второго, но оставляет старую геометрию/wfn. Хэши сходятся, а геометрия
    не следует из нового seed(prev_id, nonce) — блок отклоняется.
    """
    d = new_element()
    try:
        _seed, _pub, addr = keypair()
        cb1 = chain.make_coinbase(1, addr, chain.COINBASE_VALUE)
        p1 = write_block(d, [cb1])
        cb2 = chain.make_coinbase(2, addr, chain.COINBASE_VALUE)
        p2 = write_block(d, [cb2])

        def rewrite(path, mutate):
            with zipfile.ZipFile(path) as z:
                entries = {n: z.read(n) for n in z.namelist()}
            mutate(entries)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for name, data in entries.items():
                    z.writestr(name, data)

        new_id = {}

        def tamper_first(entries):
            header = json.loads(entries["header.json"])
            dft = json.loads(entries["dft.json"])
            dft["energy_ev"] = float(dft["energy_ev"]) + 1.0
            entries["dft.json"] = json.dumps(dft).encode("utf-8")
            header["dft_sha256"] = chain.dft_sha256(dft)
            header["block_id"] = chain.block_id_of(header)
            new_id["id"] = header["block_id"]
            entries["header.json"] = json.dumps(header).encode("utf-8")

        rewrite(p1, tamper_first)

        def reparent_second(entries):
            header = json.loads(entries["header.json"])
            header["prev_id"] = new_id["id"]
            header["block_id"] = chain.block_id_of(header)
            entries["header.json"] = json.dumps(header).encode("utf-8")

        rewrite(p2, reparent_second)

        v = chain.verify_chain(d, "Cu")
        check("переписаны хэши, но не DFT → геометрия не сходится",
              (not v["ok"]) and any("геометрия не следует" in p for p in v["problems"]),
              v["problems"][:3])
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_blob_path_traversal():
    """state_path вне каталога элемента — блок ссылается на чужой файл."""
    d = new_element()
    try:
        outside = os.path.join(os.path.dirname(d), "outside.bin")
        with open(outside, "wb") as f:
            f.write(b"not-a-wfn" * 10)
        sha = chain.sha256_file(outside)
        _, _, addr = keypair()
        path = chain.write_block(d, "Cu", 2, sha, "../outside.bin",
                                 [chain.make_coinbase(1, addr, 50.0)],
                                 dft=make_dft(d), **CP2K)
        v = chain.verify_block(d, path)
        check("state_path выходит за каталог элемента", not v["ok"],
              str(v["problems"][:3]), attack=True)
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_duplicate_seq_different_n():
    """Второй блок с тем же seq (другой n_atoms) — коллизия имён."""
    d = new_element()
    try:
        seed, pub, addr = keypair()
        cb = chain.make_coinbase(1, addr, chain.COINBASE_VALUE)
        p1 = write_block(d, [cb])
        dup = os.path.join(os.path.dirname(p1), "000001_Cu_3.zip")
        shutil.copyfile(p1, dup)
        v = chain.verify_chain(d, "Cu")
        check("дубль seq с другим n_atoms", not v["ok"], str(v["problems"][:3]), attack=True)
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def test_xyz_comment_injection():
    """struct_id попадает в XYZ-комментарий без экранирования (ready-путь)."""
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]
    forces = [[0.0, 0.0, 0.0]] * 2
    cfg = {"lattice_str": "20 0 0 0 20 0 0 0 20", "species": ["Cu", "Cu"],
           "element": "Cu"}
    rec = cp2k_runner.build_xyz_record(coords, -42.0, forces, 2, 3,
                                       "x\n3\nEvil 0 0 0", cfg)
    lines = rec.strip().splitlines()
    check("инъекция переводом строки в struct_id (порча датасета)",
          len(lines) == 4, f"строк {len(lines)} вместо 4", attack=True)


def test_worker_registration_nonce():
    """Регистрация без подписи nonce не должна давать чужой worker_id."""
    try:
        import main  # noqa: F401
    except Exception as e:  # noqa: BLE001
        check("регистрация: импорт main", False, str(e))
        return

    main.registered_workers.clear()
    main.worker_by_pubkey.clear()
    victim = keypair()
    victim_id = main._register_worker("victim", victim[2], victim[1].hex(),
                                      verified=True)["worker_id"]
    # атакующий заявляет тот же адрес без подписи
    attack_id = main._register_worker("evil", victim[2], victim[1].hex())["worker_id"]
    check("чужой адрес без подписи не занимает worker_id",
          attack_id != victim_id and victim_id in main.registered_workers,
          f"{attack_id} vs {victim_id}")

    # nonce: верная подпись — ок, повтор — отказ, чужой ключ — отказ
    nonce = main._issue_registration_nonce()
    sig = crypto.sign(victim[0], main.REG_SIGN_PREFIX + bytes.fromhex(nonce)).hex()
    check("nonce: верная подпись", main._verify_registration(
        nonce, victim[1].hex(), victim[2], sig) is None)
    check("nonce: повторное использование отклонено",
          main._verify_registration(nonce, victim[1].hex(), victim[2], sig) is not None)

    nonce2 = main._issue_registration_nonce()
    other = keypair()
    sig2 = crypto.sign(other[0], main.REG_SIGN_PREFIX + bytes.fromhex(nonce2)).hex()
    check("nonce: подпись другим ключом отклонена",
          main._verify_registration(nonce2, victim[1].hex(), victim[2], sig2) is not None)

    main.registered_workers.clear()
    main.worker_by_pubkey.clear()


def test_admin_api_without_token():
    """Админ-API при незаданном MASTER_ADMIN_TOKEN."""
    try:
        import main  # noqa: F401
    except Exception as e:  # noqa: BLE001
        check("admin: импорт main", False, str(e))
        return
    saved = main.ADMIN_TOKEN
    try:
        main.ADMIN_TOKEN = ""
        check("админ-API без токена: локально можно, снаружи нельзя",
              main._admin_allowed("127.0.0.1")
              and not main._admin_allowed("203.0.113.5"),
              attack=True)
        main.ADMIN_TOKEN = "s3cret-token"
        check("админ-API с токеном: снаружи по верному токену",
              main._admin_allowed("203.0.113.5", "s3cret-token"), attack=True)
        check("админ-API с токеном: без токена и с неверным — отказ",
              not main._admin_allowed("203.0.113.5")
              and not main._admin_allowed("203.0.113.5", "bad"), attack=True)
        check("админ-API: локально без токена по-прежнему можно",
              main._admin_allowed("127.0.0.1"), attack=True)
    finally:
        main.ADMIN_TOKEN = saved


def main_run():
    print("=" * 66)
    print("СТРЕСС-ТЕСТЫ ПРОТОКОЛА (атаки на блоки/транзакции)")
    print("=" * 66)
    test_valid_chain_sanity()
    test_coinbase_overpay()
    test_negative_output_inflation()
    test_nan_output()
    test_coinbase_maturity_policy()
    test_double_spend_same_block()
    test_wrong_key_signature()
    test_tampered_tx_after_signature()
    test_rewrite_history_needs_work()
    test_blob_path_traversal()
    test_duplicate_seq_different_n()
    test_xyz_comment_injection()
    test_worker_registration_nonce()
    test_admin_api_without_token()

    broken = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 66)
    print(f"ИТОГО: {len(RESULTS) - len(broken)}/{len(RESULTS)} проверок пройдено")
    if broken:
        print("\nПРОБИЛИ (протокол принял атаку или не отбил):")
        for name, _, is_attack, extra in broken:
            print(f"  ✗ {name} {extra}")
    print("=" * 66)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main_run())
