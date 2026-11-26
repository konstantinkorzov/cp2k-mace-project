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

import os
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


RETRY_DELAY = 5          # секунд между попытками подключения к серверу
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


def register_with_retry(reg_url: str, name: str, pubkey_hash: str, public_key: str,
                        wallet=None, retry_delay: int = RETRY_DELAY) -> str:
    """Регистрируется на сервере и доказывает владение ключом.

    Мастер выдаёт одноразовый nonce, воркер подписывает его своим приватным
    ключом. Без этого чужой адрес можно было бы «занять» и снимать задачи.
    """
    while True:
        payload = {"name": name, "pubkey_hash": pubkey_hash, "public_key": public_key}
        if wallet is not None:
            try:
                ch = requests.get(
                    reg_url.replace("/register-worker", "/register-challenge"),
                    timeout=10)
                if ch.status_code == 200:
                    nonce = str(ch.json().get("nonce") or "")
                    if nonce:
                        payload["nonce"] = nonce
                        payload["signature"] = wallet.sign(
                            pubkey_hash,
                            b"cp2k-mace-register:" + bytes.fromhex(nonce)).hex()
            except Exception:  # noqa: BLE001
                pass      # старый мастер: регистрируемся без подписи (unverified)
        try:
            print(f"Регистрируюсь на сервере {reg_url} "
                  f"(name={name}, pubkey={pubkey_hash[:12]}...)")
            reg_resp = requests.post(reg_url, json=payload, timeout=10)
            reg_resp.raise_for_status()
            reg_data = reg_resp.json()
            worker_id = reg_data.get("worker_id")
            if worker_id:
                return worker_id
            print(f"⚠ Сервер вернул ответ без worker_id: {reg_data}. Повтор через {retry_delay} с...")
        except KeyboardInterrupt:
            print("\nВоркер остановлен пользователем до регистрации.")
            sys.exit(0)
        except Exception as e:
            print(f"✗ Сервер недоступен при регистрации: {e}. Повтор через {retry_delay} с...")

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
    with open(wfn_path, "rb") as f:
        wfn_bytes = f.read()

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
        "wfn_b64": base64.b64encode(wfn_bytes).decode("ascii"),
        "wfn_size": len(wfn_bytes),
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
        + f", состояние {len(wfn_bytes)} байт — отправляю мастеру"
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
                   attempts: int = 30, delay: int = 10):
    """
    Ищет мастера, который возьмёт готовый результат (ready):
    GET /work/height (регистрация + высота) → POST /submit-result ready.
    """
    order = [primary] + [m for m in masters if m != primary]
    for attempt in range(1, attempts + 1):
        for url in order:
            try:
                r = requests.get(url + "/work/height",
                                 params={"element": element, "name": name,
                                         "pubkey_hash": pubkey_hash},
                                 timeout=10)
                if r.status_code != 200:
                    continue
                info = r.json()
                if info.get("status") != "ok":
                    logger.info(f"{url}: {info.get('status')} — пропускаю")
                    continue
                sub = requests.post(url + "/submit-result", json={
                    "worker_id": info["worker_id"],
                    "struct_id": struct_id,
                    "success": True,
                    "data": data,
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
                    r = requests.get(base_url + "/heartbeat",
                                     params={"worker_id": wid}, timeout=5)
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
                        help="имя воркера; при запуске генерируется новая пара ключей")
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

    # Кошелёк воркера в ОБЩЕЙ базе (results/keys.db), но со своей солью пароля:
    # даже при одинаковом пароле с мастером хэш и ключи разные.
    worker_name = args.name or identity.new_worker_name()
    conn = wallet_store.open_db(os.path.join(results_root, "keys.db"))
    try:
        wallet, created = wallet_store.ensure_worker(conn, worker_name,
                                                     _worker_password(args.password))
    except wallet_store.WalletError as e:
        print(f"✗ {e}")
        sys.exit(2)

    key = wallet.latest_key()
    pubkey_hash = key["address"]
    public_key = key["public_key"]
    worker_name = wallet.name

    # Регистрация воркера (POST с name + адресом кошелька)
    reg_url = f"{base_url}/register-worker"
    worker_id = register_with_retry(reg_url, worker_name, pubkey_hash,
                                    public_key, wallet=wallet,
                                    retry_delay=args.retry_delay)

    logger = setup_logger(worker_id, log_dir)
    logger.info(f"Воркер {worker_id} зарегистрирован (name={worker_name}, "
                f"адрес={pubkey_hash[:16]}...). Логи: {log_dir}/{worker_id}.log")
    if created:
        logger.info("Создан кошелёк воркера с паролем, заданным при запуске "
                    "(--password / WORKER_PASSWORD; для веб-панели — поле «Пароль кошелька»)")

    task_url = f"{base_url}/get-task"
    submit_url = f"{base_url}/submit-result"

    # heartbeat стартует ДО цикла задач: пока идёт CP2K (минуты-часы),
    # мастер видит, что воркер жив, и не отбирает у него задачу
    holder = {"worker_id": worker_id, "stop": False,
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

                response = requests.get(
                    task_url,
                    params={"worker_id": worker_id, "pubkey_hash": pubkey_hash},
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
                cfg = task["cfg"]

                # Рабочий каталог воркера (кэш CP2K: inp/out/json).
                # В Docker контейнеры монтируют одну и ту же папку проекта,
                # поэтому путь совпадает с results/<element> мастера.
                os.makedirs(cfg["output_dir"], exist_ok=True)

                logger.info(f"Получено задание {struct_id}: элемент {cfg['element']} "
                            f"({cfg['potential']}), n={n_atoms}, dim={dim}")

                success, data = run_task(task, cfg, logger, verbose=args.verbose)

                if success:
                    logger.info(f"Результат {struct_id} готов: отправляю -RESTART.wfn и геометрию мастеру")
                else:
                    logger.error(f"Задание {struct_id} не выполнено: {data.get('message')}")

                payload = {
                    "worker_id": worker_id,
                    "struct_id": struct_id,
                    "success": success,
                    "data": data,
                }

                sub_resp = requests.post(submit_url, json=payload, timeout=60)
                sub_resp.raise_for_status()
                sub_data = sub_resp.json()
                status = sub_data.get("status")
                if status in ("busy", "not_found"):
                    logger.warning(f"Мастер {base_url} занят/снял задачу ({status}: "
                                   f"{sub_data.get('reason', '')}) — ищу другой мастер")
                    if success:
                        target, res = deliver_result(
                            masters, base_url, cfg["element"], worker_name,
                            pubkey_hash, struct_id, data, logger)
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
                # 400 от /get-task — мастер перезапустился и не знает наш worker_id
                if http_err.response is not None and http_err.response.status_code == 400:
                    logger.warning("Мастер не знает воркера — прохожу регистрацию заново")
                    worker_id = register_with_retry(reg_url, worker_name, pubkey_hash,
                                                    public_key, wallet=wallet,
                                                    retry_delay=args.retry_delay)
                    holder["worker_id"] = worker_id
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
