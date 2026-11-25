#!/usr/bin/env python3
"""
Полноценный воркер для расчета DFT.

Мастер присылает воркеру только «желаемое число атомов» (n_atoms) и размерность (dim).

Воркер:
  1. сам генерирует случайную структуру нужного размера (structure_generator);
  2. делает DFT-расчёт в CP2K до сходимости SCF;
  3. отправляет мастеру файл состояния (-RESTART.wfn), геометрию и телеметрию
     (E/gap/mag) — числа воркера мастер не использует в качестве истины.

Мастер сам считает E/силы/gap/mag от присланной плотности (см. src/main.py,
verify_state_payload) и принимает результат, только если SCF сходится за
RESTART_VERIFY_SCF шагов.
"""

import os
import sys
import time
import base64
import random
import logging
import argparse

import requests

import cp2k_runner
import identity
import wallet_store
import structure_generator


RETRY_DELAY = 5          # секунд между попытками подключения к серверу
LOCAL_RETRIES = 3        # сколько раз воркер локально повторяет задачу (новая структура)
PEER_REFRESH_SEC = 60    # как часто спрашивать у мастеров адреса других мастеров
MAX_MASTERS = 32         # сколько адресов мастеров воркер готов держать


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
                        retry_delay: int = RETRY_DELAY) -> str:
    """Регистрируется на сервере с именем и хэшем публичного ключа."""
    payload = {"name": name, "pubkey_hash": pubkey_hash, "public_key": public_key}
    while True:
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


def generate_structure(n_atoms: int, dim: int, cfg: dict, struct_id: str, logger):
    """
    Генерирует ОДНУ случайную структуру нужного размера.
    Seed случайный на каждую задачу: разные воркеры (и разные попытки) не должны
    выдавать одинаковые кластеры.
    """
    seed = random.SystemRandom().randrange(10 ** 9)
    logger.info(f"Генерирую случайную структуру: n={n_atoms}, dim={dim}, seed={seed}, "
                f"элемент {cfg['element']}, min_dist={structure_generator.min_dist(cfg):.2f}, "
                f"max_dist={structure_generator.max_dist(cfg):.2f}")
    coords = structure_generator.generate_random_cluster(
        n_atoms=n_atoms,
        dim=dim,
        cfg=cfg,
        seed=seed,
        max_attempts=5,
    )
    logger.info(f"Структура готова: {coords.shape[0]} атомов (задание {struct_id})")
    return coords


def run_task(task: dict, cfg: dict, logger):
    """
    Выполняет задание: генерация структуры → DFT до сходимости → сборка payload'а.
    Возвращает (success, data). При ошибке data содержит "message".

    Воркер всегда шлёт файл состояния (-RESTART.wfn): мастер сам считает из него.
    """
    struct_id = task["struct_id"]
    n_atoms = int(task["n_atoms"])
    dim = int(task.get("dim", 3))
    # Прогресс CP2K (шаги SCF, сходимость) — в тот же лог, что и остальные сообщения
    cfg = dict(cfg)
    cfg["progress"] = logger.info

    # Энергию изолированного атома считает ТОЛЬКО мастер и передаёт её в задании.
    # Воркер сам CP2K для E_iso не запускает (мастеру он доверяет).
    e_iso = cfg.get("e_isolated_atom")
    if e_iso is None:
        logger.warning("Мастер не передал E_iso — E_binding/atom в логе будет отсутствовать")

    last_error = "неизвестная ошибка"

    for attempt in range(1, LOCAL_RETRIES + 1):
        try:
            coords = generate_structure(n_atoms, dim, cfg, struct_id, logger)

            t0 = time.time()
            res = cp2k_runner.run_dft_for_single_structure(
                coords=coords,
                n_atoms=n_atoms,
                dim=dim,
                struct_id=f"{struct_id}_try{attempt}",
                cfg=cfg,
                save_wfn=True,
            )
            elapsed = time.time() - t0

            if res.get("error"):
                last_error = f"CP2K error (rc={res.get('rc')})"
                logger.error(f"Попытка {attempt}/{LOCAL_RETRIES}: расчёт {struct_id} не удался: {last_error}")
                continue

            # Главное требование задания из README: структура должна СОЙТИСЬ.
            # Не сошлось — генерируем новую структуру и считаем заново.
            if not res.get("scf_converged", False):
                last_error = (f"SCF не сошёлся ({res.get('scf_steps')} шагов)"
                              if res.get("scf_steps") else "SCF не сошёлся")
                logger.warning(f"Попытка {attempt}/{LOCAL_RETRIES}: {last_error} — беру новую структуру")
                continue

            data = {
                "geometry": coords.tolist(),
                "n_atoms": n_atoms,
                "dim": dim,
                "cell_size": cp2k_runner.resolve_cell_size(coords, cfg),
                "element": cfg["element"],
                "potential": cfg.get("potential") or cp2k_runner.PSEUDOPOTENTIALS.get(cfg["element"]),
                "scf_converged": True,
                "scf_steps": res.get("scf_steps"),
                "elapsed": res.get("elapsed", elapsed),
                "hash": res.get("hash"),
            }

            wfn_path = res.get("wfn_path")
            if not wfn_path or not os.path.exists(wfn_path):
                last_error = "CP2K не создал файл состояния (-RESTART.wfn)"
                logger.error(f"Попытка {attempt}/{LOCAL_RETRIES}: {last_error}")
                continue
            with open(wfn_path, "rb") as f:
                wfn_bytes = f.read()
            data.update({
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
            })

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
            # Физические наблюдаемые (телеметрия воркера; канон — числа мастера)
            if res.get("ip_koopmans_ev") is not None:
                parts.append(f"IP(Купманс) = {res['ip_koopmans_ev']:.3f} eV")
            if res.get("dipole_total_debye") is not None:
                parts.append(f"|μ| = {res['dipole_total_debye']:.2e} Д")
            logger.info(
                f"Расчёт {struct_id} СОШЁЛСЯ за {elapsed:.1f} с: " + ", ".join(parts)
                + f", состояние {len(wfn_bytes)} байт — отправляю мастеру"
            )
            return True, data

        except Exception as ex:
            last_error = f"{type(ex).__name__}: {ex}"
            logger.exception(f"Попытка {attempt}/{LOCAL_RETRIES}: ошибка при выполнении {struct_id}")

    return False, {"error": True, "message": last_error}


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


def main():
    # Автоматически считываем переменные окружения, переданные через Docker (или fallback на дефолты)
    default_server = os.environ.get("MASTER_HOST", "host.docker.internal")
    default_port = int(os.environ.get("MASTER_PORT", "8000"))

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
    args = parser.parse_args()

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
                                    public_key, retry_delay=args.retry_delay)

    logger = setup_logger(worker_id, log_dir)
    logger.info(f"Воркер {worker_id} зарегистрирован (name={worker_name}, "
                f"адрес={pubkey_hash[:16]}...). Логи: {log_dir}/{worker_id}.log")
    if created:
        logger.info("Создан кошелёк воркера с паролем, заданным при запуске "
                    "(--password / WORKER_PASSWORD; для веб-панели — поле «Пароль кошелька»)")

    task_url = f"{base_url}/get-task"
    submit_url = f"{base_url}/submit-result"

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

                success, data = run_task(task, cfg, logger)

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
