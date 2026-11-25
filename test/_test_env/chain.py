#!/usr/bin/env python3
"""
Блокчейн-блоки принятых результатов + UTXO-транзакции.

Блок: results/<element>/blocks/000001_Cu_2.zip
Имя: <номер>_<элемент>_<атомов>.zip

Внутри:
  header.json                 — version, seq, element, n_atoms, prev_sha256,
                                created_at (ISO 8601 UTC), state_sha256, state_path
  tx-0000-coinbase.json       — 50 монет на адрес мастера (последний ключ)
  tx-0001-reward.json         — 25 монет воркеру (если у мастера есть
                                подтверждённый баланс)

Файл состояния (.wfn) в блок НЕ кладётся: он хранится один раз в
results/<element>/blobs/<sha256>.wfn, а в заголовке — ссылка.

Транзакции — в духе Bitcoin UTXO: входы ссылаются на (txid, vout) предыдущих
выходов, coinbase-вход специальный. Выход становится доступен спустя
`confirmations` блоков (по умолчанию 1; меняется в master-cli).
"""

import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from datetime import datetime

import crypto

BLOCK_VERSION = 1
ZERO_HASH = "0" * 64
BLOCK_RE = re.compile(r"^(\d{6})_([A-Za-z]{1,3})_(\d+)\.zip$")
HEADER_NAME = "header.json"
DFT_NAME = "dft.json"

COINBASE_VALUE = 50.0
REWARD_VALUE = 25.0
TX_FEE = 0.1                     # комиссия за перевод (как в Bitcoin: входы − выходы)
DEFAULT_CONFIRMATIONS = 1


# ---------- общее ----------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    """ISO 8601 UTC: '2026-11-25T10:45:00Z' — парсится JS/Python/C++ из коробки."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def blocks_dir(element_dir: str) -> str:
    return os.path.join(element_dir, "blocks")


def blobs_dir(element_dir: str) -> str:
    return os.path.join(element_dir, "blobs")


def store_blob(element_dir: str, wfn_path: str) -> tuple[str, str]:
    """
    Кладёт файл состояния в blobs/<sha256>.wfn (дедупликация по содержимому).

    :return: (sha256, относительный путь от каталога элемента)
    """
    sha = sha256_file(wfn_path)
    rel = f"blobs/{sha}.wfn"
    dst = os.path.join(element_dir, "blobs", f"{sha}.wfn")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copyfile(wfn_path, dst)
    return sha, rel


# ---------- транзакции ----------
def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def txid_of(tx: dict) -> str:
    """txid = sha256(тело без txid и без подписей входов) — как в Bitcoin."""
    body = {k: v for k, v in tx.items() if k not in ("txid", "_file")}
    if "inputs" in body:
        body = dict(body)
        body["inputs"] = [
            {k: v for k, v in i.items() if k not in ("signature", "public_key")}
            for i in body["inputs"]
        ]
    return hashlib.sha256(canonical_json(body)).hexdigest()


def sign_transaction(tx: dict, signer, utxo: dict) -> dict:
    """
    Подписывает входы Ed25519. `signer` — объект с методами
    public_for(address) и sign(address, message).
    """
    tid = txid_of(tx)
    msg = bytes.fromhex(tid)
    for i in tx.get("inputs", []):
        if "coinbase" in i:
            continue
        info = utxo[(i["txid"], int(i["vout"]))]
        pub = signer.public_for(info["address"])
        i["public_key"] = pub.hex()
        i["signature"] = signer.sign(info["address"], msg).hex()
    return tx


def tx_fee(tx: dict, utxo: dict) -> float:
    """Комиссия = сумма входов − сумма выходов (для не-coinbase)."""
    if tx.get("type") == "coinbase":
        return 0.0
    total_in = sum(utxo[(i["txid"], int(i["vout"]))]["value"]
                   for i in tx.get("inputs", []))
    total_out = sum(float(o["value"]) for o in tx.get("outputs", []))
    return round(total_in - total_out, 8)


def make_coinbase(height: int, address: str, value: float = COINBASE_VALUE) -> dict:
    return {
        "version": 1,
        "type": "coinbase",
        "height": int(height),
        "inputs": [{"coinbase": f"block-{int(height):06d}"}],
        "outputs": [{"value": float(value), "address": address}],
    }


def make_transfer(inputs: list[dict], outputs: list[dict]) -> dict:
    return {
        "version": 1,
        "type": "transfer",
        "inputs": [{"txid": i["txid"], "vout": int(i["vout"])} for i in inputs],
        "outputs": [{"value": float(o["value"]), "address": o["address"]} for o in outputs],
    }


def list_blocks(element_dir: str) -> list[dict]:
    d = blocks_dir(element_dir)
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        m = BLOCK_RE.match(name)
        if not m:
            continue
        path = os.path.join(d, name)
        out.append({"seq": int(m.group(1)), "element": m.group(2),
                    "n_atoms": int(m.group(3)), "path": path, "name": name,
                    "sha256": sha256_file(path)})
    out.sort(key=lambda b: b["seq"])
    return out


def next_seq(element_dir: str) -> int:
    blocks = list_blocks(element_dir)
    return (blocks[-1]["seq"] + 1) if blocks else 1


def previous_hash(element_dir: str, seq: int) -> str:
    if seq <= 1:
        return ZERO_HASH
    for b in list_blocks(element_dir):
        if b["seq"] == seq - 1:
            return b["sha256"]
    raise ValueError(f"Нет блока {seq - 1} — цепочка разорвана")


def block_name(seq: int, element: str, n_atoms: int) -> str:
    return f"{seq:06d}_{element}_{int(n_atoms)}.zip"


def read_block_txs(path: str) -> list[dict]:
    txs = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if name.startswith("tx-") and name.endswith(".json"):
                tx = json.loads(z.read(name).decode("utf-8"))
                tx["_file"] = name
                txs.append(tx)
    return txs


def utxo_set(element_dir: str) -> dict:
    """
    Текущее множество непотраченных выходов: (txid, vout) -> {value, address, height, coinbase}.
    """
    utxo = {}
    for b in list_blocks(element_dir):
        try:
            txs = read_block_txs(b["path"])
        except Exception:
            continue
        for tx in txs:
            txid = tx.get("txid") or txid_of(tx)
            if tx.get("type") != "coinbase":
                for i in tx.get("inputs", []):
                    utxo.pop((i.get("txid"), int(i.get("vout", -1))), None)
            for vout, out in enumerate(tx.get("outputs", [])):
                utxo[(txid, vout)] = {
                    "value": float(out["value"]),
                    "address": out["address"],
                    "height": b["seq"],
                    "coinbase": tx.get("type") == "coinbase",
                }
    return utxo


def available_utxos(element_dir: str, addresses: set[str],
                    confirmations: int = DEFAULT_CONFIRMATIONS,
                    height: int | None = None) -> list[dict]:
    """Выходы, доступные на высоте `height` (height + confirmations <= высота нового блока)."""
    height = height if height is not None else next_seq(element_dir)
    out = []
    for (txid, vout), info in utxo_set(element_dir).items():
        if info["address"] not in addresses:
            continue
        if info["height"] + max(0, confirmations) > height:
            continue
        out.append({"txid": txid, "vout": vout, **info})
    return out


def build_transfer(element_dir: str, from_addresses: set[str], to_address: str,
                   change_address: str, value: float = REWARD_VALUE,
                   confirmations: int = DEFAULT_CONFIRMATIONS,
                   height: int | None = None,
                   fee: float = TX_FEE) -> dict | None:
    """
    Собирает перевод `value` монет на `to_address` с подтверждённых выходов
    мастера. Комиссия `fee` уходит майнеру (в coinbase этого же блока),
    сдача возвращается на `change_address`. Если средств мало — None.
    """
    height = height if height is not None else next_seq(element_dir)
    available = available_utxos(element_dir, from_addresses, confirmations, height)
    if not available:
        return None

    chosen, total = [], 0.0
    for u in sorted(available, key=lambda x: -x["value"]):
        chosen.append(u)
        total += u["value"]
        if total + 1e-9 >= value + fee:
            break
    if total + 1e-9 < value + fee:
        return None

    outputs = [{"value": round(value, 8), "address": to_address}]
    change = round(total - value - fee, 8)
    if change > 1e-9:
        outputs.append({"value": change, "address": change_address})
    return make_transfer(chosen, outputs)


# ---------- запись блока ----------
def validate_tx(tx: dict, element_dir: str, height: int | None = None,
                confirmations: int = DEFAULT_CONFIRMATIONS,
                utxo: dict | None = None) -> tuple[bool, list[str]]:
    """Проверяет перевод: UTXO, подтверждения, подписи, суммы."""
    problems = []
    if tx.get("type") != "transfer":
        return False, ["транзакция не transfer"]
    height = height if height is not None else next_seq(element_dir)
    if utxo is None:
        utxo = utxo_set(element_dir)

    tid = tx.get("txid") or txid_of(tx)
    inputs = tx.get("inputs") or []
    if not inputs:
        problems.append("нет входов")

    total_in = 0.0
    seen = set()
    for i in inputs:
        key = (i.get("txid"), int(i.get("vout", -1)))
        if key in seen:
            problems.append(f"повторный вход {key}")
        seen.add(key)
        info = utxo.get(key)
        if info is None:
            problems.append(f"вход {key} не найден или уже потрачен")
            continue
        if info["height"] + max(0, confirmations) > height:
            problems.append(f"вход {key} не подтверждён")
        total_in += info["value"]

        pub_hex, sig_hex = i.get("public_key"), i.get("signature")
        if not pub_hex or not sig_hex:
            problems.append(f"вход {key} без подписи")
            continue
        try:
            pub = bytes.fromhex(pub_hex)
            sig = bytes.fromhex(sig_hex)
        except ValueError:
            problems.append(f"вход {key}: ключ/подпись не hex")
            continue
        if crypto.address_from_public(pub) != info["address"]:
            problems.append(f"вход {key}: ключ не соответствует адресу")
        elif not crypto.verify(pub, bytes.fromhex(tid), sig):
            problems.append(f"вход {key}: неверная подпись")

    outputs = tx.get("outputs") or []
    if not outputs:
        problems.append("нет выходов")
    total_out = sum(float(o.get("value", 0)) for o in outputs)
    if total_out <= 0:
        problems.append("нулевая сумма выходов")
    if total_in + 1e-8 < total_out:
        problems.append(f"сумма входов {total_in} < суммы выходов {total_out}")
    return (not problems), problems


def build_wallet_transfer(wallet, element_dir: str, to_address: str,
                          value: float, fee: float = TX_FEE,
                          confirmations: int = DEFAULT_CONFIRMATIONS,
                          height: int | None = None) -> dict | None:
    """Собирает и подписывает перевод с адресов кошелька (wallet.addresses())."""
    height = height if height is not None else next_seq(element_dir)
    change = wallet.latest_key()["address"]
    tx = build_transfer(element_dir, wallet.addresses(), to_address, change,
                        value=value, confirmations=confirmations,
                        height=height, fee=fee)
    if tx is None:
        return None
    sign_transaction(tx, wallet, utxo_set(element_dir))
    tx["txid"] = txid_of(tx)
    return tx


def height_of_hash(element_dir: str, sha: str) -> int | None:
    """Высота блока с данным sha256 (для поиска общего предка при форке)."""
    for b in list_blocks(element_dir):
        if b["sha256"] == sha:
            return b["seq"]
    return None


def truncate_chain(element_dir: str, keep_height: int) -> tuple[list[dict], list[dict]]:
    """
    Откатывает цепочку до keep_height: блоки выше уносятся в orphan/<ts>/.
    Возвращает (удалённые блоки, их transfer-транзакции для мемпула).
    """
    d = blocks_dir(element_dir)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    orphan = os.path.join(element_dir, "orphan", ts)
    removed, txs = [], []
    for b in list_blocks(element_dir):
        if b["seq"] <= keep_height:
            continue
        try:
            for tx in read_block_txs(b["path"]):
                if tx.get("type") != "coinbase":
                    txs.append(tx)
        except Exception:  # noqa: BLE001
            pass
        os.makedirs(orphan, exist_ok=True)
        dst = os.path.join(orphan, b["name"])
        shutil.move(b["path"], dst)
        removed.append({**b, "orphan_path": dst})
    return removed, txs


def restore_blocks(removed: list[dict]):
    """Возвращает блоки из orphan обратно в blocks/ (откат неудачного reorg)."""
    for b in removed:
        path = b["orphan_path"]
        if not os.path.exists(path):
            continue
        # path = <element>/orphan/<ts>/<name>
        element_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        dst = os.path.join(blocks_dir(element_dir), b["name"])
        shutil.move(path, dst)


def write_block(element_dir: str, element: str, n_atoms: int,
                state_sha256: str, state_path: str,
                transactions: list[dict],
                cp2k_version: str = "",
                cp2k_params: dict | None = None,
                dft: dict | None = None) -> str:
    """
    Создаёт следующий блок цепочки элемента.

    :param state_sha256: sha256 файла состояния (лежит в blobs/)
    :param state_path: относительный путь 'blobs/<sha>.wfn'
    :param transactions: список транзакций (coinbase первой)
    :param cp2k_version: версия CP2K, которой считал мастер
    :param cp2k_params: CP2K_PARAMS (CUTOFF, EPS_SCF, ячейка и т.д.)
    :param dft: числа мастера (energy_ev, forces_ev_ang, geometry, ...) —
                по ним другой мастер сможет пересчитать блок
    :return: путь к zip
    """
    d = blocks_dir(element_dir)
    os.makedirs(d, exist_ok=True)
    seq = next_seq(element_dir)
    prev = previous_hash(element_dir, seq)
    path = os.path.join(d, block_name(seq, element, n_atoms))

    header = {
        "version": BLOCK_VERSION,
        "seq": seq,
        "element": element,
        "n_atoms": int(n_atoms),
        "prev_sha256": prev,
        "created_at": now_iso(),
        "state_sha256": state_sha256,
        "state_path": state_path,
        "cp2k_version": cp2k_version,
        "cp2k_params": cp2k_params or {},
    }

    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(HEADER_NAME, json.dumps(header, ensure_ascii=False, indent=2))
        if dft is not None:
            z.writestr(DFT_NAME, json.dumps(dft, ensure_ascii=False, indent=2))
        for idx, tx in enumerate(transactions):
            tx = dict(tx)
            tx["txid"] = txid_of(tx)
            fname = f"tx-{idx:04d}-{tx.get('type', 'tx')}.json"
            z.writestr(fname, json.dumps(tx, ensure_ascii=False, indent=2))
    os.replace(tmp, path)
    return path


# ---------- проверка цепочки ----------
def _parse_iso(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except (TypeError, ValueError):
        return False


def read_block(path: str) -> tuple[dict, dict, list[dict], set]:
    """Возвращает (header, dft, txs, имена файлов) из zip-блока."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        header = json.loads(z.read(HEADER_NAME).decode("utf-8")) if HEADER_NAME in names else {}
        dft = json.loads(z.read(DFT_NAME).decode("utf-8")) if DFT_NAME in names else {}
        txs = []
        for name in sorted(names):
            if name.startswith("tx-") and name.endswith(".json"):
                tx = json.loads(z.read(name).decode("utf-8"))
                tx["_file"] = name
                txs.append(tx)
    return header, dft, txs, names


def _check_txs(txs: list[dict], utxo: dict, label: str,
               add_outputs: bool = False) -> tuple[float, list[str]]:
    problems = []
    coinbases = [t for t in txs if t.get("type") == "coinbase"]
    if len(coinbases) != 1:
        problems.append(f"{label}: coinbase-транзакций {len(coinbases)}, ожидается 1")

    block_fee = 0.0
    for tx in txs:
        tid = tx.get("txid") or txid_of(tx)
        if tid != txid_of(tx):
            problems.append(f"{label}: txid транзакции не совпадает с содержимым")
        if tx.get("type") == "coinbase":
            continue

        total_in = 0.0
        for i in tx.get("inputs", []):
            key = (i.get("txid"), int(i.get("vout", -1)))
            if key not in utxo:
                problems.append(f"{label}: вход {key} ссылается на неизвестный/потраченный выход")
                continue
            info = utxo.pop(key)
            total_in += info["value"]

            pub_hex, sig_hex = i.get("public_key"), i.get("signature")
            if not pub_hex or not sig_hex:
                problems.append(f"{label}: вход {key} без подписи")
                continue
            try:
                pub = bytes.fromhex(pub_hex)
                sig = bytes.fromhex(sig_hex)
            except ValueError:
                problems.append(f"{label}: вход {key}: публичный ключ/подпись не hex")
                continue
            if crypto.address_from_public(pub) != info["address"]:
                problems.append(f"{label}: вход {key}: ключ не соответствует адресу")
            elif not crypto.verify(pub, bytes.fromhex(tid), sig):
                problems.append(f"{label}: вход {key}: неверная подпись")

        total_out = sum(o["value"] for o in tx.get("outputs", []))
        if total_in + 1e-8 < total_out:
            problems.append(f"{label}: сумма входов {total_in} < суммы выходов {total_out}")
        block_fee += max(0.0, total_in - total_out)

    if coinbases:
        cb_value = sum(o["value"] for o in coinbases[0].get("outputs", []))
        if cb_value > COINBASE_VALUE + block_fee + 1e-8:
            problems.append(
                f"{label}: coinbase {cb_value} > {COINBASE_VALUE} + комиссии {block_fee:.8f}")

    if add_outputs:
        for tx in txs:
            tid = tx.get("txid") or txid_of(tx)
            for vout, out in enumerate(tx.get("outputs", [])):
                utxo[(tid, vout)] = {"value": float(out["value"]),
                                     "address": out["address"], "coinbase": tx.get("type") == "coinbase"}
    return block_fee, problems


def verify_block(element_dir: str, block_path: str,
                 expected_seq: int | None = None,
                 expected_prev: str | None = None) -> dict:
    """
    Проверяет ОДИН скачанный блок против локальной цепочки, ничего не записывая.
    Возвращает {ok, problems, header, dft, txs}.
    """
    problems = []
    try:
        header, dft, txs, names = read_block(block_path)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "problems": [f"блок не читается: {e}"],
                "header": {}, "dft": {}, "txs": []}

    element = os.path.basename(os.path.normpath(element_dir))
    seq = int(header.get("seq", -1))
    if expected_seq is not None and seq != expected_seq:
        problems.append(f"seq {seq} != ожидаемого {expected_seq}")
    if header.get("element") != element:
        problems.append(f"элемент блока {header.get('element')} != {element}")
    if expected_prev is not None and header.get("prev_sha256") != expected_prev:
        problems.append("prev_sha256 не совпадает с головой локальной цепочки")
    if header.get("version") != BLOCK_VERSION:
        problems.append(f"version {header.get('version')} != {BLOCK_VERSION}")
    if not _parse_iso(header.get("created_at", "")):
        problems.append("created_at не ISO 8601")
    if not header.get("cp2k_version") or not isinstance(header.get("cp2k_params"), dict):
        problems.append("нет cp2k_version/cp2k_params")

    n_atoms = int(header.get("n_atoms", -1))
    sha = header.get("state_sha256")
    rel = header.get("state_path")
    blob = os.path.join(element_dir, rel) if rel else ""
    if not sha or not rel:
        problems.append("нет state_sha256/state_path")
    elif not os.path.exists(blob):
        problems.append(f"нет файла состояния {rel} (нужен blob с пира)")
    elif sha256_file(blob) != sha:
        problems.append("sha256 файла состояния не совпадает")

    if not dft:
        problems.append("нет dft.json — блок нельзя пересчитать")
    else:
        geom = dft.get("geometry")
        forces = dft.get("forces_ev_ang")
        if not isinstance(geom, list) or len(geom) != n_atoms:
            problems.append("dft.json: geometry не соответствует n_atoms")
        if not isinstance(forces, list) or len(forces) != n_atoms:
            problems.append("dft.json: forces не соответствуют n_atoms")
        if dft.get("energy_ev") is None:
            problems.append("dft.json: нет energy_ev")

    utxo = utxo_set(element_dir)
    for key in list(utxo):
        info = utxo[key]
        utxo[key] = {"value": info["value"], "address": info["address"],
                     "height": info["height"], "coinbase": info["coinbase"]}
    _, tx_problems = _check_txs(txs, utxo, os.path.basename(block_path))
    problems.extend(tx_problems)

    return {"ok": not problems, "problems": problems,
            "header": header, "dft": dft, "txs": txs}


def verify_chain(element_dir: str, element: str | None = None) -> dict:
    """
    Проверяет: версию/поля заголовка, время, prev_sha256, наличие blob с нужным
    хэшем, транзакции (один coinbase на 50, ссылки входов на UTXO, отсутствие
    двойных трат, сумма входов >= суммы выходов).
    """
    blocks = list_blocks(element_dir)
    problems = []
    utxo: dict = {}
    expected_prev = ZERO_HASH
    expected_seq = 1

    for b in blocks:
        if b["seq"] != expected_seq:
            problems.append(f"пропуск в нумерации: ожидался {expected_seq}, найден {b['seq']}")
        if element and b["element"] != element:
            problems.append(f"{b['name']}: элемент {b['element']} != {element}")

        try:
            with zipfile.ZipFile(b["path"]) as z:
                names = set(z.namelist())
                if HEADER_NAME not in names:
                    problems.append(f"{b['name']}: нет {HEADER_NAME}")
                    header = {}
                else:
                    header = json.loads(z.read(HEADER_NAME).decode("utf-8"))
                dft = (json.loads(z.read(DFT_NAME).decode("utf-8"))
                       if DFT_NAME in names else {})
        except Exception as e:  # noqa: BLE001
            problems.append(f"{b['name']}: не читается как zip ({e})")
            expected_prev = b["sha256"]
            expected_seq += 1
            continue

        if header:
            if header.get("version") != BLOCK_VERSION:
                problems.append(f"{b['name']}: version {header.get('version')} != {BLOCK_VERSION}")
            if header.get("prev_sha256") != expected_prev:
                problems.append(f"{b['name']}: prev_sha256 не совпадает с предыдущим блоком")
            if int(header.get("seq", -1)) != b["seq"]:
                problems.append(f"{b['name']}: seq в заголовке != имени файла")
            if header.get("element") != b["element"]:
                problems.append(f"{b['name']}: элемент в заголовке != имени файла")
            if int(header.get("n_atoms", -1)) != b["n_atoms"]:
                problems.append(f"{b['name']}: число атомов в заголовке != имени файла")
            if not _parse_iso(header.get("created_at", "")):
                problems.append(f"{b['name']}: created_at не ISO 8601")
            if not header.get("cp2k_version"):
                problems.append(f"{b['name']}: нет cp2k_version")
            if not isinstance(header.get("cp2k_params"), dict) or not header.get("cp2k_params"):
                problems.append(f"{b['name']}: нет cp2k_params")

            if not dft or dft.get("energy_ev") is None:
                problems.append(f"{b['name']}: нет dft.json/energy_ev")
            else:
                geom = dft.get("geometry")
                if not isinstance(geom, list) or len(geom) != b["n_atoms"]:
                    problems.append(f"{b['name']}: geometry в dft.json не соответствует n_atoms")

            sha = header.get("state_sha256")
            rel = header.get("state_path")
            if not sha or not rel:
                problems.append(f"{b['name']}: нет state_sha256/state_path")
            else:
                blob = os.path.join(element_dir, rel)
                if not os.path.exists(blob):
                    problems.append(f"{b['name']}: нет файла состояния {rel}")
                elif sha256_file(blob) != sha:
                    problems.append(f"{b['name']}: sha256 файла состояния не совпадает")

        try:
            txs = read_block_txs(b["path"])
        except Exception as e:  # noqa: BLE001
            problems.append(f"{b['name']}: транзакции не читаются ({e})")
            txs = []

        coinbases = [t for t in txs if t.get("type") == "coinbase"]
        if len(coinbases) != 1:
            problems.append(f"{b['name']}: coinbase-транзакций {len(coinbases)}, ожидается 1")

        # обычные транзакции: ссылки на UTXO, подписи Ed25519, комиссии
        block_fee = 0.0
        for tx in txs:
            tid = tx.get("txid") or txid_of(tx)
            if tid != txid_of(tx):
                problems.append(f"{b['name']}: txid транзакции не совпадает с содержимым")
            if tx.get("type") == "coinbase":
                continue

            total_in = 0.0
            for i in tx.get("inputs", []):
                key = (i.get("txid"), int(i.get("vout", -1)))
                if key not in utxo:
                    problems.append(f"{b['name']}: вход {key} ссылается на неизвестный/потраченный выход")
                    continue
                info = utxo.pop(key)
                total_in += info["value"]

                pub_hex, sig_hex = i.get("public_key"), i.get("signature")
                if not pub_hex or not sig_hex:
                    problems.append(f"{b['name']}: вход {key} без подписи")
                    continue
                try:
                    pub = bytes.fromhex(pub_hex)
                    sig = bytes.fromhex(sig_hex)
                except ValueError:
                    problems.append(f"{b['name']}: вход {key}: публичный ключ/подпись не hex")
                    continue
                if crypto.address_from_public(pub) != info["address"]:
                    problems.append(f"{b['name']}: вход {key}: ключ не соответствует адресу")
                elif not crypto.verify(pub, bytes.fromhex(tid), sig):
                    problems.append(f"{b['name']}: вход {key}: неверная подпись")

            total_out = sum(o["value"] for o in tx.get("outputs", []))
            if total_in + 1e-8 < total_out:
                problems.append(f"{b['name']}: сумма входов {total_in} < суммы выходов {total_out}")
            block_fee += max(0.0, total_in - total_out)

        # coinbase может забрать субсидию + комиссии блока (как в Bitcoin)
        if coinbases:
            cb_value = sum(o["value"] for o in coinbases[0].get("outputs", []))
            if cb_value > COINBASE_VALUE + block_fee + 1e-8:
                problems.append(
                    f"{b['name']}: coinbase {cb_value} > {COINBASE_VALUE} + комиссии {block_fee:.8f}")

        # добавляем все выходы блока в UTXO
        for tx in txs:
            tid = tx.get("txid") or txid_of(tx)
            for vout, out in enumerate(tx.get("outputs", [])):
                utxo[(tid, vout)] = {"value": float(out["value"]), "address": out["address"],
                                     "height": b["seq"], "coinbase": tx.get("type") == "coinbase"}

        expected_prev = b["sha256"]
        expected_seq += 1

    return {"ok": not problems, "blocks": len(blocks), "problems": problems}


def balances(element_dir: str, addresses: set[str] | None = None,
             confirmations: int = DEFAULT_CONFIRMATIONS) -> dict:
    """Балансы по адресам: неподтверждённые и доступные (>= confirmations)."""
    height = next_seq(element_dir)
    result: dict[str, dict] = {}
    for info in utxo_set(element_dir).values():
        addr = info["address"]
        if addresses is not None and addr not in addresses:
            continue
        rec = result.setdefault(addr, {"available": 0.0, "pending": 0.0, "total": 0.0})
        rec["total"] = round(rec["total"] + info["value"], 8)
        if info["height"] + max(0, confirmations) <= height:
            rec["available"] = round(rec["available"] + info["value"], 8)
        else:
            rec["pending"] = round(rec["pending"] + info["value"], 8)
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Проверка цепочки блоков элемента")
    ap.add_argument("element_dir", help="каталог элемента, напр. results/Cu")
    args = ap.parse_args()
    element = os.path.basename(os.path.normpath(args.element_dir))
    print(json.dumps(verify_chain(args.element_dir, element), ensure_ascii=False, indent=2))
