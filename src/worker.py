#!/usr/bin/env python3
"""
Полноценный воркер для расчета DFT.

Мастер выдаёт готовую геометрию задания: координаты и сорт атомов приходят
в задании. Воркер ничего не генерирует и не меняет (это важно: иначе близость
к шаблону/seed была бы недоказуема).

Воркер:
  1. считает ВЫДАННУЮ структуру в CP2K до сходимости SCF;
  2. отправляет мастеру файл состояния (-RESTART.wfn), геометрию и телеметрию
     (E/gap/mag) — числа воркера мастер не использует в качестве истины.

Мастер сверяет присланную геометрию с выданной (дёшево, без CP2K), затем сам
считает E/силы/gap/mag от присланной плотности (см. src/main.py,
verify_state_payload) и принимает результат, только если SCF сходится за
RESTART_VERIFY_SCF шагов.
"""

import hashlib
import os
import re
import sys
import time
import json
import base64
import logging
import argparse
import threading

import numpy as np
import requests

import cp2k_runner
import identity
import wallet_store


RETRY_DELAY = 5  # секунд между попытками подключения к серверу

# Шифрованный клиент мастера. None — обычный requests (оффлайн-тесты).
CLIENT = None


def _http():
    """Клиент для запросов к мастеру: зашифрованный, если он настроен."""
    return CLIENT if CLIENT is not None else requests
PEER_REFRESH_SEC = 60    # как часто спрашивать у мастеров адреса других мастеров
MAX_MASTERS = 32         # сколько адресов мастеров воркер готов держать
HEARTBEAT_SEC = 30       # как часто сообщать мастеру «я жив» (в т.ч. во время расчёта)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config_defaults() -> dict:
    """Значения из src/config.json (формат {"value": ..., "help": ...})."""
    try:
        with open(os.path.join(PROJECT_ROOT, "src", "config.json"), encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    return {k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
            for k, v in raw.items()}


def setup_logger(worker_id: str, log_dir: str):
    """Настройка логирования в консоль и файл."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{worker_id}.log")

    logger = logging.getLogger(worker_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# Что мастер НЕ имеет права задавать воркеру: исполняемый файл, таймауты и
# каталоги. Иначе мастер (или подменённый адрес мастера) присылает
# cp2k_cmd = "sh -c id" и получает выполнение команд на машине воркера.
LOCAL_CP2K_CMD = os.environ.get("CP2K_CMD") or "cp2k"
LOCAL_CP2K_TIMEOUT = int(os.environ.get("WORKER_CP2K_TIMEOUT") or 86400)
_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$")


def _local_cfg(task_cfg: dict, project_root: str, results_root: str,
               verbose: bool | None = None) -> dict:
    """
    Конфиг задания для CP2K: физика — от мастера, исполнение — своё.

    От мастера берём только то, что определяет расчёт: элемент/систему,
    псевдопотенциал (сверенный со справочником), состав, заряд,
    мультиплетность, профиль сети и E_iso. Всё, что касается запуска
    процессов и путей, задаётся локально.
    """
    import cp2k_runner

    cfg = dict(task_cfg or {})
    element = str(cfg.get("element") or "")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{0,63}", element):
        raise ValueError(f"недопустимое имя системы: {element!r}")

    # псевдопотенциал: только из справочника (строка уходит в CP2K-ввод)
    potential = str(cfg.get("potential") or "")
    if potential not in set(cp2k_runner.PSEUDOPOTENTIALS.values()):
        raise ValueError(f"неизвестный псевдопотенциал: {potential!r}")

    # виды атомов: только химические символы из справочника
    species = cfg.get("species")
    if isinstance(species, list):
        for sym in species:
            if not isinstance(sym, str) or not _SYMBOL_RE.match(sym) \
                    or sym not in cp2k_runner.PSEUDOPOTENTIALS:
                raise ValueError(f"недопустимый вид атома в задании: {sym!r}")

    comp = cfg.get("composition")
    if isinstance(comp, dict):
        for sym, cnt in comp.items():
            if not isinstance(sym, str) or not _SYMBOL_RE.match(sym) \
                    or sym not in cp2k_runner.PSEUDOPOTENTIALS:
                raise ValueError(f"недопустимый элемент состава: {sym!r}")
            if type(cnt) is not int or cnt < 1:
                raise ValueError(f"неверное количество {sym}: {cnt!r}")

    # локальное исполнение: команда, таймаут, каталоги — свои
    out_dir = os.path.join(results_root, element)
    cfg["cp2k_cmd"] = LOCAL_CP2K_CMD
    cfg["cp2k_timeout_sec"] = LOCAL_CP2K_TIMEOUT
    cfg["output_dir"] = out_dir
    cfg["failed_ids_file"] = os.path.join(out_dir, f"{element}_failed_ids.txt")
    if verbose is not None:
        cfg["verbose_cp2k"] = bool(verbose)
    return cfg


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _upload_state(base_url: str, session: str, worker_id: str,
                  struct_id: str, path: str | None, logger,
                  timeout: int = 1800) -> str | None:
    """
    Заливает -RESTART.wfn на мастер ПОТОКОМ (requests читает файл кусками).

    В JSON состояние не кладём: на кластере это гигабайты base64 в памяти у
    обеих сторон. Мастер вернёт sha256, его и передаём в /submit-result.
    """
    if not path or not os.path.exists(path):
        logger.warning("нет файла состояния — заливать нечего")
        return None
    try:
        with open(path, "rb") as f:
            if CLIENT is not None:
                # сессия/тикет идут зашифрованным запросом, файл — открытым
                r = CLIENT.upload_state(
                    base_url + "/submit-state",
                    {"worker_id": worker_id, "session": session,
                     "struct_id": struct_id},
                    f, timeout=timeout)
            else:
                r = requests.post(
                    base_url + "/submit-state",
                    params={"worker_id": worker_id, "session": session,
                            "struct_id": struct_id},
                    data=f,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=timeout)
        if r.status_code != 200:
            logger.warning(f"мастер не принял состояние: HTTP {r.status_code} "
                           f"{r.text[:200]}")
            return None
        sha = str((r.json() or {}).get("sha256") or "")
        return sha or None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"ошибка заливки состояния: {e}")
        return None


def _register_once(reg_url: str, name: str, pubkey_hash: str, public_key: str,
                   wallet=None) -> dict | None:
    """
    Одна попытка регистрации с доказательством ключа.

    Мастер выдаёт одноразовый nonce, воркер подписывает его приватным ключом
    и получает worker_id + session. Session — единственное удостоверение:
    сам worker_id публичен (/queue-status), по нему задачи не выдают.
    Возвращает {"worker_id", "session", ...} или None.
    """
    payload = {"name": name, "pubkey_hash": pubkey_hash, "public_key": public_key}
    http = _http()
    if wallet is not None:
        # одноразовый nonce тоже через шифрованный канал (он и есть доказательство)
        challenge_url = reg_url.replace("/register-worker", "/register-challenge")
        ch = http.post(challenge_url, timeout=10) if CLIENT is not None \
            else http.get(challenge_url, timeout=10)
        ch.raise_for_status()
        nonce = str((ch.json() or {}).get("nonce") or "")
        if not nonce:
            return None
        payload["nonce"] = nonce
        payload["signature"] = wallet.sign(
            pubkey_hash, b"cp2k-mace-register:" + bytes.fromhex(nonce)).hex()
    reg_resp = http.post(reg_url, json=payload, timeout=10)
    reg_resp.raise_for_status()
    reg_data = reg_resp.json() or {}
    if reg_data.get("worker_id") and reg_data.get("session"):
        return reg_data
    return None


def register_with_retry(reg_url: str, name: str, pubkey_hash: str, public_key: str,
                        wallet=None, retry_delay: int = RETRY_DELAY) -> dict:
    """Регистрируется, пока мастер не выдаст worker_id и session."""
    while True:
        print(f"Регистрируюсь на сервере {reg_url} "
              f"(name={name}, pubkey={pubkey_hash[:12]}...)")
        try:
            reg = _register_once(reg_url, name, pubkey_hash, public_key, wallet)
            if reg:
                return reg
            print(f"⚠ Сервер не выдал worker_id/session. Повтор через {retry_delay} с...")
        except KeyboardInterrupt:
            print("\nВоркер остановлен пользователем до регистрации.")
            sys.exit(0)
        except Exception as e:  # noqa: BLE001
            print(f"✗ Регистрация не удалась: {e}. Повтор через {retry_delay} с...")

        try:
            time.sleep(retry_delay)
        except KeyboardInterrupt:
            print("\nВоркер остановлен пользователем до регистрации.")
            sys.exit(0)


def _worker_password(arg: str | None) -> str | None:
    if arg:
        return arg
    if os.environ.get("WORKER_PASSWORD"):
        return os.environ["WORKER_PASSWORD"]
    try:
        if sys.stdin.isatty():
            import getpass
            return getpass.getpass("Пароль воркера: ")
    except Exception:  # noqa: BLE001
        pass
    return None


def run_task(task: dict, cfg: dict, logger, verbose: bool | None = None):
    """
    Считает ВЫДАННУЮ мастером геометрию и собирает payload.

    Воркер не создаёт и не меняет структуру: координаты и сорт атомов приходят
    в задании. Мастер сверит присланное с выданным (дёшево, без CP2K) и только
    после этого запустит свои шаги DFT от присланного -RESTART.wfn.

    Возвращает (success, data). При ошибке data содержит "message".
    """
    struct_id = task.get("struct_id", "?")
    geometry = task.get("geometry")
    if not geometry:
        return False, {"error": True,
                       "message": "мастер не передал геометрию в задании"}

    coords = np.asarray(geometry, dtype=np.float64)
    n_atoms = int(task.get("n_atoms") or len(coords))
    dim = int(task.get("dim", 3))
    used_seed = task.get("seed")

    species = [str(s) for s in (task.get("species") or [])]
    if len(species) != n_atoms:
        species = [cfg["element"]] * n_atoms

    cfg = dict(cfg)
    # Флаг панели переопределяет verbose из задания (это настройка конкретного
    # запуска воркера, а не мастера).
    if verbose is not None:
        cfg["verbose_cp2k"] = bool(verbose)
    cfg["progress"] = logger.info
    cfg["species"] = species

    # Энергию изолированного атома считает ТОЛЬКО мастер и передаёт её в задании.
    e_iso = cfg.get("e_isolated_atom")
    if e_iso is None:
        logger.warning("Мастер не передал E_iso — E_binding/atom в логе будет отсутствовать")

    logger.info(f"Считаю выданную мастером структуру {struct_id}: n={n_atoms}, "
                f"dim={dim}, seed={used_seed}, элемент {cfg['element']}")
    t0 = time.time()
    try:
        res = cp2k_runner.run_dft_for_single_structure(
            coords=coords,
            n_atoms=n_atoms,
            dim=dim,
            struct_id=struct_id,
            cfg=cfg,
            save_wfn=True,
        )
    except Exception as ex:  # noqa: BLE001
        logger.exception(f"Ошибка при выполнении {struct_id}")
        return False, {"error": True, "message": f"{type(ex).__name__}: {ex}"}
    elapsed = time.time() - t0

    if res.get("error"):
        logger.error(f"Расчёт {struct_id} не удался: rc={res.get('rc')}")
        return False, {"error": True,
                       "message": f"CP2K error (rc={res.get('rc')})"}

    if not res.get("scf_converged", False):
        msg = (f"SCF не сошёлся ({res.get('scf_steps')} шагов)"
               if res.get("scf_steps") else "SCF не сошёлся")
        logger.warning(f"Расчёт {struct_id}: {msg} — мастер выдаст новую геометрию")
        return False, {"error": True, "message": msg}

    wfn_path = res.get("wfn_path")
    if not wfn_path or not os.path.exists(wfn_path):
        logger.error("CP2K не создал файл состояния (-RESTART.wfn)")
        return False, {"error": True,
                       "message": "CP2K не создал файл состояния (-RESTART.wfn)"}
    # Файл не читаем в память: размер и хэш считаем потоком, а сам файл
    # воркер зальёт на мастера отдельным запросом (/submit-state).
    wfn_size = os.path.getsize(wfn_path)
    wfn_sha256 = _sha256_file(wfn_path)

    data = {
        "geometry": coords.tolist(),
        "species": species,
        "n_atoms": n_atoms,
        "dim": dim,
        "seed": used_seed,
        "cell_size": cp2k_runner.resolve_cell_size(coords, cfg),
        "element": cfg["element"],
        "potential": cfg.get("potential"),
        "scf_converged": True,
        "scf_steps": res.get("scf_steps"),
        "elapsed": res.get("elapsed", elapsed),
        "hash": res.get("hash"),
        "wfn_name": os.path.basename(wfn_path),
        "wfn_path": wfn_path,          # локальный путь: уйдёт потоком, не в JSON
        "wfn_sha256": wfn_sha256,
        "wfn_size": wfn_size,
        # ниже — только телеметрия, мастер её перепроверит сам
        "energy_ev": res["energy_ev"],
        "gap": res.get("gap"),
        "mag": res.get("mag"),
        "e_isolated_atom_ev": e_iso,
        "e_binding_per_atom": (
            (e_iso * n_atoms - res["energy_ev"]) / n_atoms
            if e_iso is not None else None
        ),
    }

    e_bind = data["e_binding_per_atom"]
    parts = [f"E = {res['energy_ev']:.6f} eV"]
    if e_bind is not None:
        parts.append(f"E_binding/atom = {e_bind:.6f} eV")
    else:
        parts.append(f"E/atom = {res['energy_ev'] / n_atoms:.6f} eV")
    if res.get("gap") is not None:
        parts.append(f"Gap = {res['gap']:.4f} eV")
    if res.get("mag") is not None:
        parts.append(f"MagMom = {res['mag']:.4f} μB")
    if res.get("ip_koopmans_ev") is not None:
        parts.append(f"IP(Купманс) = {res['ip_koopmans_ev']:.3f} eV")
    if res.get("dipole_total_debye") is not None:
        parts.append(f"|μ| = {res['dipole_total_debye']:.2e} Д")
    logger.info(
        f"Расчёт {struct_id} СОШЁЛСЯ за {elapsed:.1f} с: " + ", ".join(parts)
        + f", состояние {wfn_size} байт — отправляю мастеру"
    )
    return True, data


def discover_peers(masters: list[str], logger) -> list[str]:
    """
    Обмен адресами с мастерами (как в p2p): спрашиваем у каждого известного
    мастера его список пиров и добавляем неизвестные адреса — чтобы уметь
    переключиться, если основной мастер занят или недоступен.
    """
    added = []
    for url in list(masters):
        try:
            r = requests.get(url.rstrip("/") + "/peers", timeout=5)
            if r.status_code != 200:
                # старый мастер: список пиров был только в админ-API
                r = requests.get(url.rstrip("/") + "/admin/peers", timeout=5)
            if r.status_code != 200:
                continue
            for peer in (r.json().get("peers") or []):
                u = str(peer).strip().rstrip("/")
                if u and u not in masters and len(masters) < MAX_MASTERS:
                    masters.append(u)
                    added.append(u)
        except Exception:  # noqa: BLE001
            continue
    if added:
        logger.info(f"Узнал новые адреса мастеров: {', '.join(added)}")
    return added


def _masters_list(base_url: str, extra: list[str] | None) -> list[str]:
    """Список мастеров: WORKER_MASTERS + --master, иначе основной."""
    raw = []
    env = os.environ.get("WORKER_MASTERS", "")
    raw += [x.strip() for x in env.split(",") if x.strip()]
    raw += [x.strip() for x in (extra or []) if x.strip()]
    raw = [u.rstrip("/") for u in raw if u]
    if not raw:
        raw = [base_url.rstrip("/")]
    out = []
    for u in raw:
        if u not in out:
            out.append(u)
    return out


def deliver_result(masters: list[str], primary: str, element: str, name: str,
                   pubkey_hash: str, struct_id: str, data: dict, logger,
                   attempts: int = 30, delay: int = 10,
                   wallet=None, public_key: str | None = None,
                   task: dict | None = None, state_path: str | None = None):
    """
    Ищет мастера, который возьмёт готовый результат (ready):
    GET /work/height (поддерживает ли элемент) → POST /register-worker с
    подписью nonce (worker_id + session) → POST /submit-result ready.

    В ready-запросе уходят prev_id и nonce из задания: работа привязана к
    голове цепочки, и только мастер с этой же головой может проверить кадр
    (geometry == Generate(seed(prev_id, nonce))) и положить его в блок.
    Регистрация у каждого мастера своя: worker_id другого мастера ничего не
    доказывает, а без подписи ключа готовый результат не примет никто.
    """
    order = [primary] + [m for m in masters if m != primary]
    for attempt in range(1, attempts + 1):
        for url in order:
            try:
                r = _http().get(url + "/work/height",
                                params={"element": element}, timeout=10)
                if r.status_code != 200:
                    continue
                info = r.json()
                if info.get("status") != "ok":
                    logger.info(f"{url}: {info.get('status')} — пропускаю")
                    continue
                reg = _register_once(url + "/register-worker", name, pubkey_hash,
                                     public_key, wallet)
                if not reg:
                    logger.warning(f"{url}: регистрация с подписью не удалась")
                    continue
                ready_data = dict(data or {})
                ready_data.pop("wfn_path", None)
                if state_path:
                    # другому мастеру — своя заливка: состояние у него своё
                    sha = _upload_state(url, reg["session"], reg["worker_id"],
                                        struct_id, state_path, logger)
                    if not sha:
                        logger.warning(f"{url}: состояние не залито")
                        continue
                    ready_data["state_sha256"] = sha
                if task:
                    # Привязка к голове: (prev_id, nonce) — это та самая работа,
                    # которую выдал первый мастер. Без неё новый мастер не может
                    # ни проверить кадр, ни положить его в блок.
                    for field in ("prev_id", "nonce"):
                        if task.get(field) is not None:
                            ready_data[field] = task[field]
                sub = _http().post(url + "/submit-result", json={
                    "worker_id": reg["worker_id"],
                    "session": reg["session"],
                    "struct_id": struct_id,
                    "success": True,
                    "data": ready_data,
                    "ready": True,
                }, timeout=180)
                if sub.status_code != 200:
                    logger.warning(f"{url}: HTTP {sub.status_code}")
                    continue
                res = sub.json()
                if res.get("status") == "accepted":
                    return url, res
                logger.warning(f"{url}: {res.get('status')} {res.get('reason', '')}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{url}: {e}")
        if attempt < attempts:
            logger.info(f"Все мастера заняты (попытка {attempt}/{attempts}), "
                        f"жду {delay} с и пробую снова")
            time.sleep(delay)
    return None, None


def start_heartbeat(base_url: str, holder: dict, logger,
                    interval: int = HEARTBEAT_SEC) -> threading.Thread:
    """Фоновый heartbeat: мастер должен знать, что воркер жив, пока считает CP2K.

    Без этого мастер не отличает «воркер упал» от «воркер считает 40 минут»,
    и задача висит за мёртвым воркером до task_timeout_sec (часы/сутки).
    """
    def loop():
        while not holder.get("stop"):
            wid = holder.get("worker_id")
            if wid:
                try:
                    r = _http().get(base_url + "/heartbeat",
                                    params={"worker_id": wid,
                                            "session": holder.get("session")},
                                    timeout=5)
                    if r.status_code == 200 and r.json().get("status") == "unknown":
                        # Перерегистрацию делает основной цикл (400 от /get-task) —
                        # heartbeat не должен гонки ради сам создавать воркера
                        logger.warning("Мастер не знает этот worker_id (перезапуск мастера?)")
                except Exception:  # noqa: BLE001
                    pass          # сеть пропала — молчим, задача не должна умереть из-за этого
            holder["stop_event"].wait(interval)

    holder.setdefault("stop_event", threading.Event())
    th = threading.Thread(target=loop, daemon=True, name="heartbeat")
    th.start()
    return th


def main():
    # Автоматически считываем переменные окружения, переданные через Docker (или fallback на дефолты)
    cfg_defaults = _config_defaults()
    default_server = os.environ.get("MASTER_HOST", "host.docker.internal")
    # Воркер — клиент и порт не слушает; это порт МАСТЕРА по умолчанию.
    # Для CLI без --master: MASTER_PORT или 8000. В панели адрес мастера
    # задаётся целиком (host:port), поэтому отдельный ключ не нужен.
    default_port = int(os.environ.get("MASTER_PORT") or 8000)

    parser = argparse.ArgumentParser(description="Запуск воркера CP2K")
    parser.add_argument("--port", type=int, default=default_port, help="Порт сервера main.py")
    parser.add_argument("--server", type=str, default=default_server, help="Хост сервера main.py")
    parser.add_argument("--retry-delay", type=int, default=RETRY_DELAY, help="Пауза между попытками подключения (сек)")
    parser.add_argument("--name", type=str, default=None,
                        help="ярлык процесса для логов и панели")
    parser.add_argument("--wallet", type=str, default=None,
                        help="имя кошелька (иначе WORKER_WALLET); кошельки создаются "
                             "явно: wallet.py create --name NAME")
    parser.add_argument("--password", type=str, default=None,
                        help="пароль кошелька воркера (иначе WORKER_PASSWORD или запрос)")
    parser.add_argument("--master", action="append", default=None,
                        help="URL мастера (можно несколько); первый — основной")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=None,
                        help="подробный лог CP2K (по умолчанию — как передал мастер)")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false",
                        help="не печатать шаги SCF")
    args = parser.parse_args()

    # Отпечаток сборки CP2K: считать чужим билдом бессмысленно — мастер всё
    # равно отбросит состояние (другая арифметика/формат), а часы уже потрачены.
    build = cp2k_runner.cp2k_build(cfg_defaults)
    if cp2k_runner.cp2k_build_allowed(build):
        print(f"CP2K build: {build}")
    elif cp2k_runner.cp2k_build_identified(build):
        print(f"✗ CP2K {build} не входит в сборки сети:")
        for allowed in cp2k_runner.CP2K_BUILD_EXPECTED:
            print(f"  • {allowed}")
        print("  Пересоберите образ из актуального Dockerfile (pin по digest).")
        sys.exit(4)
    else:
        print("⚠ CP2K_BUILD не задан — сборка не проверена (старый образ или запуск мимо образа)")

    base_url = f"http://{args.server}:{args.port}"
    masters = _masters_list(base_url, args.master)
    base_url = masters[0]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_root = os.path.join(project_root, "results")
    log_dir = os.path.join(results_root, "worker_logs")

    # Кошелёк — самостоятельная сущность: он не «воркерский» и не создаётся
    # на лету. Воркер открывает кошелёк по имени и паролю (та же база
    # results/keys.db). Имя процесса ниже — только ярлык для логов и панели.
    worker_name = args.name or identity.new_worker_name()
    conn = wallet_store.open_db(os.path.join(results_root, "keys.db"))
    try:
        wallet = wallet_store.open_wallet(
            conn, args.wallet or os.environ.get("WORKER_WALLET"),
            _worker_password(args.password))
    except wallet_store.WalletError as e:
        print(f"✗ {e}")
        sys.exit(2)

    key = wallet.latest_key()
    pubkey_hash = key["address"]
    public_key = key["public_key"]

    # Транспортное шифрование: X25519-ключ выводится из seed кошелька.
    # Ключ мастера запоминается при первом обращении (results/worker_trust.json).
    global CLIENT
    try:
        from secure_client import SecureClient
        trust_path = os.path.join(results_root, "worker_trust.json")
        CLIENT = SecureClient(wallet.private_seed(key["address"]), trust_path)
    except Exception as e:  # noqa: BLE001
        print(f"⚠ шифрование транспорта недоступно ({e}); мастер ответит 426 из сети")
        CLIENT = None

    # Регистрация воркера (POST с name + адресом кошелька)
    reg_url = f"{base_url}/register-worker"
    reg = register_with_retry(reg_url, worker_name, pubkey_hash,
                              public_key, wallet=wallet,
                              retry_delay=args.retry_delay)
    worker_id, session = reg["worker_id"], reg["session"]

    logger = setup_logger(worker_id, log_dir)
    logger.info(f"Воркер {worker_id} зарегистрирован (name={worker_name}, "
                f"адрес={pubkey_hash[:16]}...). Логи: {log_dir}/{worker_id}.log")
    logger.info(f"Кошелёк «{wallet.name}»: адресов {len(wallet.keys())}, "
                f"награды идут на {key['address'][:16]}…")

    task_url = f"{base_url}/get-task"
    submit_url = f"{base_url}/submit-result"

    # heartbeat стартует ДО цикла задач: пока идёт CP2K (минуты-часы),
    # мастер видит, что воркер жив, и не отбирает у него задачу
    holder = {"worker_id": worker_id, "session": session, "stop": False,
              "stop_event": threading.Event()}
    start_heartbeat(base_url, holder, logger)
    logger.info(f"Heartbeat каждые {HEARTBEAT_SEC} с — мастер увидит пропажу воркера "
                f"за несколько минут, а не за task_timeout_sec")

    logger.info("Начинаем опрос очереди задач...")
    last_peer_refresh = 0.0

    try:
        while True:
            try:
                if time.time() - last_peer_refresh > PEER_REFRESH_SEC:
                    last_peer_refresh = time.time()
                    discover_peers(masters, logger)

                response = _http().get(
                    task_url,
                    params={"worker_id": worker_id, "session": session},
                    timeout=10,
                )
                response.raise_for_status()
                res_data = response.json()

                status = res_data.get("status")
                if status == "empty":
                    time.sleep(3)
                    continue
                if status == "paused":
                    logger.info("Мастер на паузе — жду 5 с")
                    time.sleep(5)
                    continue
                if status == "busy":
                    logger.info(f"Задача {res_data.get('struct_id')} ещё числится за мной "
                                f"({res_data.get('age_sec')} с); жду до таймаута")
                    time.sleep(10)
                    continue

                task = res_data.get("task")
                struct_id = task["struct_id"]
                n_atoms = int(task["n_atoms"])
                dim = int(task.get("dim", 3))
                # Физика — из задания, исполнение — локальное (см. _local_cfg):
                # мастер не должен выбирать бинарник и каталоги воркера.
                cfg = _local_cfg(task["cfg"], project_root, results_root,
                                 verbose=args.verbose)
                os.makedirs(cfg["output_dir"], exist_ok=True)

                logger.info(f"Получено задание {struct_id}: элемент {cfg['element']} "
                            f"({cfg['potential']}), n={n_atoms}, dim={dim}")

                success, data = run_task(task, cfg, logger, verbose=args.verbose)

                if success:
                    logger.info(f"Результат {struct_id} готов: отправляю -RESTART.wfn и геометрию мастеру")
                else:
                    logger.error(f"Задание {struct_id} не выполнено: {data.get('message')}")

                send_data = {k: v for k, v in data.items() if k != "wfn_path"}
                if success:
                    sha = _upload_state(base_url, session, worker_id,
                                        struct_id, data.get("wfn_path"), logger)
                    if sha:
                        send_data["state_sha256"] = sha
                    else:
                        logger.warning("состояние не залито — мастер отклонит сдачу")

                payload = {
                    "worker_id": worker_id,
                    "session": session,
                    "struct_id": struct_id,
                    "success": success,
                    "data": send_data,
                }

                sub_resp = _http().post(submit_url, json=payload, timeout=60)
                sub_resp.raise_for_status()
                sub_data = sub_resp.json()
                status = sub_data.get("status")
                if status in ("busy", "not_found"):
                    logger.warning(f"Мастер {base_url} занят/снял задачу ({status}: "
                                   f"{sub_data.get('reason', '')}) — ищу другой мастер")
                    if success:
                        target, res = deliver_result(
                            masters, base_url, cfg["element"], worker_name,
                            pubkey_hash, struct_id, data, logger,
                            wallet=wallet, public_key=public_key, task=task,
                            state_path=data.get("wfn_path"))
                        if target:
                            logger.info(f"Готовый результат {struct_id} принят мастером "
                                        f"{target} ({res.get('kept')})")
                        else:
                            logger.error("Ни один мастер не принял результат")
                    else:
                        logger.warning("Расчёт не удался — результат никому не отправляю")
                else:
                    logger.info(f"Результат по {struct_id} отправлен. Ответ сервера: {status}"
                                + (f" (причина: {sub_data.get('reason')})"
                                   if sub_data.get("reason") else ""))

            except requests.exceptions.HTTPError as http_err:
                # 400/403 от /get-task — мастер перезапустился (нет нашего worker_id)
                # или не признал сессию: проходим регистрацию заново
                code = (http_err.response.status_code
                        if http_err.response is not None else None)
                if code in (400, 403):
                    logger.warning("Мастер не знает воркера/сессию — прохожу регистрацию заново")
                    reg = register_with_retry(reg_url, worker_name, pubkey_hash,
                                              public_key, wallet=wallet,
                                              retry_delay=args.retry_delay)
                    worker_id, session = reg["worker_id"], reg["session"]
                    holder["worker_id"] = worker_id
                    holder["session"] = session
                    logger.info(f"Новый worker_id: {worker_id}")
                else:
                    logger.warning(f"HTTP-ошибка связи с сервером: {http_err}. "
                                   f"Повтор через {args.retry_delay} сек...")
                    time.sleep(args.retry_delay)
            except requests.exceptions.RequestException as re_err:
                logger.warning(f"Сетевая ошибка связи с сервером: {re_err}. Повтор через {args.retry_delay} сек...")
                time.sleep(args.retry_delay)
            except Exception as ex:
                logger.exception(f"Непредвиденная критическая ошибка в цикле воркера: {ex}")
                time.sleep(3)

    except KeyboardInterrupt:
        logger.info("Воркер остановлен пользователем.")


if __name__ == "__main__":
    main()
