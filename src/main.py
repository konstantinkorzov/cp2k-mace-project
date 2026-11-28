#!/usr/bin/env python3
"""
Главный модуль (Master/Server). Управляет очередью, регистрирует воркеры,
раздает задачи и отслеживает статус их выполнения.

Архитектура нулевого доверия:
  * мастер НЕ генерирует структуры: он раздаёт воркеру только желаемое число
    атомов в кластере (n_atoms) и размерность (dim);
  * воркер сам генерирует случайную структуру, считает DFT в CP2K до сходимости
    и присылает мастеру файл состояния (-RESTART.wfn) + геометрию;
  * мастер НЕ доверяет числам воркера: сам пишет вход CP2K (свой элемент, базис,
    псевдопотенциал, ячейка, функционал, геометрия), стартует от присланной
    плотности и берёт E/силы/gap/mag только из своего прогона;
  * сертификат — SCF сходится за RESTART_VERIFY_SCF (3) шагов от присланного
    состояния; иначе состояние не стационарно и результат отклоняется;
  * в файлы (XYZ + frames/*) пишутся ТОЛЬКО результаты, прошедшие проверку;
    отклонённая задача возвращается в очередь.
"""
TOL_ENERGY_EV = 0.02       # только для печати телеметрии E_воркера - E_мастера

import os
import sys
import json
import time
import base64
import signal
import secrets
import subprocess
import re
import shutil
import glob
import hashlib
import hmac
import tempfile
import argparse
import asyncio
import threading
import traceback
import numpy as np

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

import warnings

# e3nn 0.4.x при импорте читает свой constants.pt через torch.load(weights_only=False),
# а torch>=2.4 ругается на это FutureWarning. Файл лежит внутри самого пакета e3nn
# (не пользовательский ввод), поэтому шум гасим точечно, не трогая другие warnings.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.load.*weights_only=False.*",
    category=FutureWarning,
    module=r"e3nn\..*",
)

# MACE/ASE/BFGS живут в отдельном каталоге mace_pipeline/ — мастер их не импортирует.

# generate_structures больше не используется: структуры генерирует воркер
# (см. src/worker.py + src/structure_generator.py)
import cp2k_runner
import consensus
import crypto
import chain as chain_mod
import wallet_store
import structure_generator
import archive_maker

import requests

from config_loader import Config


# ---------- КОНСТАНТЫ ----------
TASK_TIMEOUT = 3600        # резервный предел удержания задачи, секунды

# Живость воркера. Воркер шлёт heartbeat раз в 30 с (src/worker.py), в том числе
# пока считает CP2K. Если признаков жизни нет дольше WORKER_LEASE_SEC, его
# задача возвращается в очередь сразу — ждать task_timeout_sec (часы/сутки) не надо.
WORKER_LEASE_SEC = 180
# Через сколько забываем мёртвого воркера целиком (освобождает worker_id и pubkey)
WORKER_FORGET_SEC = 1800

# --- удостоверение воркера ---
# worker_id — публичный номер: он есть в /queue-status и в блоках, поэтому
# как пропуск он не годится. Право говорить за воркера даёт только session,
# который мастер выдаёт при регистрации с подписью nonce (/register-challenge).
WORKER_SESSION_MAX = 4        # сколько живых сессий держим (мастеров у воркера несколько)

# --- предохранители регистрации ---
MAX_REGISTERED_WORKERS = 200  # потолок памяти мастера
REG_LIMIT_PER_IP = 60         # challenge+register с одного IP в минуту (~30 регистраций)
REG_WINDOW_SEC = 60.0

# --- ready-приёмка (результат принесён мастеру, который задание не выдавал) ---
# Каждая приёмка — прогон CP2K за счёт мастера, поэтому квота на воркера и
# запрет повторной проверки той же пары (prev_id, nonce).
READY_QUOTA_PER_HOUR = 12


def _task_timeout() -> int:
    """Жёсткий предел удержания задачи: config.json → task_timeout_sec.

    Это резервный предохранитель на случай, если сам мастер не заметил пропажу
    воркера (например, завис поток watchdog). Обычный путь — WORKER_LEASE_SEC:
    задача возвращается через 3 минуты молчания воркера.
    """
    try:
        return int(CONFIG.get("task_timeout_sec") or TASK_TIMEOUT)
    except Exception:  # noqa: BLE001
        return TASK_TIMEOUT
MAX_TASK_ATTEMPTS = 5      # сколько раз задача возвращается в очередь (ошибка/провал проверки)

# Допуски дешёвых проверок геометрии.
# atom_diameter — параметр ГЕНЕРАЦИИ (как координаты) и в приёмку не входит:
# иначе мастера с разными диаметрами принимали бы разные структуры — это форк.
# Консенсусный минимум — физический: ближе 0.5 Å атомы не бывают ни в одной
# реальной системе (короче всего H–H ≈ 0.74 Å). Константа — в consensus.py.
MIN_ATOM_DISTANCE_ANG = consensus.MIN_ATOM_DISTANCE_ANG
MIN_DIST_TOLERANCE = 0.9   # только для проверки шаблона при старте (генерация)

# Потолок размера задачи для приёмки: n_atoms приходит из запроса, поэтому
# всё, что больше, отбивается ДО аллокаций numpy и запуска CP2K. Живёт в
# consensus.py вместе с потолком расписания — иначе расписание выдавало бы
# задачи, которые мастер тут же отказывается принимать.
MAX_TASK_ATOMS = consensus.MAX_TASK_ATOMS



# ---------- Пути ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

LOG_FILE = os.path.join(RESULTS_ROOT, "main.log")
PID_FILE = os.path.join(RESULTS_ROOT, "main.pid")
STATUS_FILE = os.path.join(RESULTS_ROOT, "main.status.json")
ARCHIVE_FILE = os.path.join(RESULTS_ROOT, "artifacts.zip")

HARTREE_TO_EV = 27.211386
HARTREE_BOHR_TO_EV_ANG = 51.422067


# ---------- Tee-логгер ----------
class Tee:
    def __init__(self, path: str, stream):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", encoding="utf-8", buffering=1)
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


_original_stdout = sys.stdout
_original_stderr = sys.stderr


def install_tee_logger():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    sys.stdout = Tee(LOG_FILE, _original_stdout)
    sys.stderr = Tee(LOG_FILE, _original_stderr)


def restore_std_logger():
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr


# ---------- Статус процесса ----------
_START_TS = time.time()


def write_status(state: str, exit_code=None, extra=None):
    data = {
        "state": state,
        "pid": os.getpid(),
        "started_at": _START_TS,
        "updated_at": time.time(),
        "exit_code": exit_code,
    }
    if extra:
        data.update(extra)
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[status] не удалось записать {STATUS_FILE}: {e}")


def write_pid():
    try:
        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception as e:
        print(f"[pid] не удалось записать {PID_FILE}: {e}")


def remove_pid():
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass


# ---------- Обработка сигналов ----------
class _Terminated(Exception):
    pass


def _sig_handler(signum, frame):
    raise _Terminated(f"Получен сигнал {signum}")


def install_signal_handlers():
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)


# ---------- Загрузка конфига ----------
def _require(cfg, keys):
    missing = [k for k in keys if k not in cfg or cfg[k] is None]
    if missing:
        raise KeyError(f"В config.json отсутствуют обязательные параметры: {missing}")


def load_config(path: str, results_root: str | None = None) -> Config:
    global RESULTS_ROOT
    if results_root:
        RESULTS_ROOT = os.path.abspath(results_root)
    cfg = Config.load(path)

    # Генерацию делает воркер; мастер передаёт только диаметр атома (он же
    # определяет min/max расстояние в _check_geometry — параметр консенсуса,
    # поэтому живёт в system, а не в общих настройках).
    cfg["cell_size"] = float(cp2k_runner.CP2K_PARAMS["min_cell_size"])

    cfg.setdefault("hartree_to_ev", HARTREE_TO_EV)
    cfg.setdefault("hartree_bohr_to_ev_ang", HARTREE_BOHR_TO_EV_ANG)
    cfg.setdefault("cp2k_cmd", "cp2k")
    cfg.setdefault("verbose_cp2k", True)
    # Приёмка всегда одна: по -RESTART.wfn. Число шагов SCF на приёмку, требование
    # подтверждения чтения restart — константы в cp2k_runner.py (RESTART_VERIFY_SCF).
    cfg.setdefault("restart_max_mb", 50)                  # лимит размера принятого wfn

    # Мастеру обязательно нужно описание системы
    _require(cfg, ["system"])

    system = cfg.get("system") or {}
    cfg["atom_diameter"] = float(
        system.get("atom_diameter") or cfg.get("atom_diameter") or 2.0)
    name = str(system.get("name") or cfg.get("element") or "Cu")
    if not chain_mod.SYSTEM_RE.match(name):
        raise SystemExit(
            f"system.name {name!r} — слаг: латиница, цифры, . _ - (например cu3, al2o3). "
            "Формула задаётся в system.composition, а не в имени")
    comp = system.get("composition") or cfg.get("composition") or {}
    comp = {str(k): int(v) for k, v in comp.items()}
    if not comp:
        if not chain_mod.SYMBOL_RE.match(name):
            raise SystemExit(
                f"system.composition не задан, а имя {name!r} не химический символ — "
                "задайте состав, например {'Al': 2, 'O': 3}")
        comp = {name: 1}
    cfg["element"] = name                 # slug системы: каталог results/<slug> и header
    cfg["composition"] = comp
    cfg["formula_size"] = sum(comp.values())
    cfg.setdefault("charge", 0)
    cfg.setdefault("multiplicity", 0)
    cfg["potential"] = (cp2k_runner.get_potential(next(iter(comp)))
                        if len(comp) == 1 else
                        ", ".join(f"{sym}:{cp2k_runner.get_potential(sym)}" for sym in comp))
    cfg["n_valence"] = sum(
        cp2k_runner.valence_electrons(cp2k_runner.get_potential(sym)) * cnt
        for sym, cnt in comp.items())

    element_dir = os.path.join(RESULTS_ROOT, cfg["element"])
    cfg["output_dir"] = element_dir
    cfg["output_xyz"] = os.path.join(element_dir, f"{cfg['element']}_generated_structures.xyz")
    cfg["output_all_xyz_name"] = f"{cfg['element']}_all_data.xyz"
    cfg["frames_dir"] = os.path.join(element_dir, "frames")
    cfg["wallet_store_db"] = os.path.join(RESULTS_ROOT, "keys.db")
    cfg.setdefault("peers", [])
    if not isinstance(cfg["peers"], list):
        cfg["peers"] = [str(cfg["peers"])]
    # Пиры, добавленные через CLI, живут в своём results-каталоге
    cfg["peers_file"] = os.path.join(RESULTS_ROOT, "peers.json")
    if os.path.exists(cfg["peers_file"]):
        try:
            with open(cfg["peers_file"], "r", encoding="utf-8") as f:
                extra = json.load(f)
            if isinstance(extra, list):
                cfg["peers"] = list(cfg["peers"]) + [p for p in extra if p not in cfg["peers"]]
        except Exception as e:  # noqa: BLE001
            print(f"[Master] ⚠ не прочитан {cfg['peers_file']}: {e}")
    cfg.setdefault("coinbase_confirmations", chain_mod.DEFAULT_CONFIRMATIONS)
    cfg["failed_ids_file"] = os.path.join(element_dir, f"{cfg['element']}_failed_ids.txt")
    cfg["verified_ids_file"] = os.path.join(
        element_dir, f"{cfg['element']}_verified_ids.txt"
    )
    cfg["isolated_atom_energy_file"] = os.path.join(
        element_dir, f"{cfg['element']}_isolated_atom_energy.txt"
    )
    # На каждый элемент — свой кэш E_iso (для соединений вроде Al2O3)
    cfg["isolated_atom_energy_files"] = {
        sym: os.path.join(element_dir, f"{sym}_isolated_atom_energy.txt")
        for sym in comp
    }
    cfg["lattice_str"] = (
        f"{cfg['cell_size']} 0.0 0.0 0.0 "
        f"{cfg['cell_size']} 0.0 0.0 0.0 "
        f"{cfg['cell_size']}"
    )

    cfg["results_root"] = RESULTS_ROOT
    cfg["log_file"] = os.path.join(RESULTS_ROOT, "main.log")
    cfg["pid_file"] = os.path.join(RESULTS_ROOT, "main.pid")
    cfg["status_file"] = os.path.join(RESULTS_ROOT, "main.status.json")
    cfg["archive_file"] = os.path.join(RESULTS_ROOT, "artifacts.zip")

    return cfg


CONFIG = load_config(CONFIG_PATH)



# ---------- CLI / Порт ----------
def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=None,
                        help="порт FastAPI-сервера (иначе port из config.json или 8000)")
    parser.add_argument("--password", type=str, default=None,
                        help="пароль кошелька мастера (иначе MASTER_PASSWORD или запрос)")
    parser.add_argument("--wallet", type=str, default=None,
                        help="имя кошелька мастера (иначе MASTER_WALLET); кошельки "
                             "создаются явно: wallet.py create --name NAME")
    parser.add_argument("--peer", action="append", default=None,
                        help="URL другого мастера для синхронизации (можно несколько)")
    parser.add_argument("--results-dir", default=None,
                        help="каталог результатов (для второго мастера на той же машине)")
    args, _ = parser.parse_known_args()
    return args


def _master_password() -> str | None:
    """Пароль мастера: --password → MASTER_PASSWORD → интерактивный запрос."""
    if getattr(_master_password, "override", None):
        return _master_password.override
    if os.environ.get("MASTER_PASSWORD"):
        return os.environ["MASTER_PASSWORD"]
    try:
        if sys.stdin.isatty():
            import getpass
            return getpass.getpass("Пароль мастера: ")
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------- Очередь заданий: мастер раздаёт только число атомов ----------
# ---------- Расписание размеров по высоте цепочки ----------
# =====================================================================
# РАСПИСАНИЕ РАЗМЕРОВ — в consensus.py (один источник правды с chain.py)
# =====================================================================
# Эти числа определяют, какой размер кластера считается на каждой высоте
# цепочки. Если у двух мастеров они разные — цепочки разойдутся, поэтому
# меняются только вместе с версией протокола. Пиры проверяют то же самое
# при приёме блока (chain.validate_header), а не только мастер.
ATOM_SCHEDULE = consensus.ATOM_SCHEDULE

# Размерности чередуются: объёмная (3) и плоская (2) структуры через одну.
DIM_CYCLE = consensus.DIM_CYCLE

# Тег запуска мастера. Без него после перезапуска снова выдавался n2_t00000,
# и в цепочке могли оказаться два блока с ОДИНАКОВЫМ struct_id (структуры
# при этом разные — воркер генерирует их сам, — но идентификатор задания
# должен быть уникален, иначе датасет и логи нельзя однозначно разобрать).
_RUN_TAG = f"{int(time.time()):x}{os.urandom(2).hex()}"

# Сколько заданий добавлять, когда очередь опустела (этап ещё не закрыт).
# Небольшой избыток нужен, чтобы воркеры не простаивали на пересборке; держать
# в памяти весь этап (тысячи заданий) смысла нет — при смене этапа они снимаются.
TASK_REFILL_MIN = 8


def _task_seed(task_index: int) -> int:
    """Опорное число генератора структуры: время запуска + номер задания.

    Не константа: структуры одного размера не должны повторяться между
    запусками мастера. Значение уходит воркеру в задании и пишется в блок,
    поэтому кадр воспроизводим: тот же seed даёт ту же геометрию.
    """
    return int((time.time_ns() ^ (task_index * 0x9E3779B1)) % (2 ** 31))


def schedule_enabled(cfg=None) -> bool:
    """Расписание — единственный режим, поэтому всегда True (оставлено для совместимости)."""
    return True


def stage_for_height(height: int, cfg) -> dict:
    """
    Этап = размер кластера. Расчёт — в consensus.stage_for_height, чтобы
    мастер и проверка блока считали по одной и той же функции.

    Высота height — сколько блоков уже принято; этап s покрывает
    [C(s-1), C(s)), где C — сумма blocks по предыдущим этапам.
    """
    return consensus.stage_for_height(height, int(cfg.get("formula_size") or 1))


GEOMETRY_RETRIES = 5          # сколько разных seed-ов перебрать при неудаче генерации
GEOMETRY_SHIFT = 7919         # сдвиг seed при повторной выдаче того же задания
GEOMETRY_MATCH_TOL = 1e-6     # Å: допуск сравнения присланной и выданной геометрии


def _issue_geometry(task: dict) -> dict:
    """Геометрия задания, привязанная к голове цепочки (протокол v2).

    seed = H(prev_id ‖ nonce). Значит, любая правка предыдущего блока меняет
    геометрию этого и требует нового DFT. Переписать K блоков = K работ.
    """
    n_atoms = int(task["n_atoms"])
    dim = int(task["dim"])
    element_dir = CONFIG["output_dir"]
    prev_id = chain_mod.previous_hash(element_dir, chain_mod.next_seq(element_dir))

    last_error = None
    nonce = 0
    fixed_nonce = task.get("nonce")
    for _ in range(GEOMETRY_RETRIES):
        nonce = int(fixed_nonce) if fixed_nonce is not None else secrets.randbits(64)
        seed = chain_mod.geometry_seed(prev_id, nonce)
        try:
            coords = structure_generator.generate_for_seed(seed, n_atoms, dim, CONFIG)
            species = structure_generator.assign_species(n_atoms, CONFIG, seed=seed)
            return {
                "coords": np.asarray(coords, dtype=float).tolist(),
                "species": [str(x) for x in species],
                "seed": seed,
                "nonce": nonce,
                "prev_id": prev_id,
            }
        except Exception as e:  # noqa: BLE001
            last_error = e

    raise RuntimeError(f"не удалось построить геометрию n={n_atoms}, dim={dim}: {last_error}")


def schedule_tasks_for_stage(stage: dict, cfg, start_index: int = 0,
                             count: int | None = None) -> list[dict]:
    """Задания этапа. Размерность чередуется: 3D, 2D, 3D, 2D…"""
    n_atoms = int(stage["n_atoms"])
    count = int(count or stage["blocks"])
    return [{
        "struct_id": f"n{n_atoms}_t{start_index + i:05d}_{_RUN_TAG}",
        "task_index": start_index + i,
        "n_atoms": n_atoms,
        "dim": int(DIM_CYCLE[(start_index + i) % len(DIM_CYCLE)]),
        # seed генерации: по нему мастер строит геометрию задания
        "seed": _task_seed(start_index + i),
        "attempts": 0,
    } for i in range(count)]


def _sync_stage(reason: str = ""):
    """Пересобирает очередь, когда высота перешла на новый этап (размер)."""
    if not schedule_enabled(CONFIG):
        return
    global pending_tasks
    height = chain_mod.next_seq(CONFIG["output_dir"]) - 1
    stage = stage_for_height(height, CONFIG)
    if CONFIG.get("_stage") == stage["stage"]:
        CONFIG["_stage_info"] = stage
        return
    CONFIG["_stage"] = stage["stage"]
    CONFIG["_stage_info"] = stage
    with task_queue_lock:
        CONFIG["_task_counter"] = 0
        pending_tasks = schedule_tasks_for_stage(stage, CONFIG)
        CONFIG["_task_counter"] = stage["blocks"]
    suffix = f" ({reason})" if reason else ""
    print(f"[Master] 🧬 этап {stage['stage']}{suffix}: кластер n={stage['n_atoms']}, "
          f"блоков на этап {stage['blocks']}; старые размеры больше не принимаются")


def task_cfg_payload(cfg) -> dict:
    """Часть конфига, которая передаётся воркеру вместе с заданием."""
    return {
        "element": cfg["element"],
        "potential": cfg["potential"],
        "cp2k_cmd": cfg["cp2k_cmd"],
        "cp2k_timeout_sec": int(cfg.get("cp2k_timeout_sec") or 3600),
        "verbose_cp2k": cfg["verbose_cp2k"],
        "hartree_to_ev": cfg["hartree_to_ev"],
        "hartree_bohr_to_ev_ang": cfg["hartree_bohr_to_ev_ang"],
        "output_dir": cfg["output_dir"],
        "failed_ids_file": cfg["failed_ids_file"],
        # единственный параметр генерации: диаметр атома (min/max_dist — константы
        # structure_generator: min_dist = diameter, max_dist = 2*diameter)
        "atom_diameter": cfg["atom_diameter"],
        # состав системы: воркер раскладывает формулу по атомам сам
        "composition": cfg.get("composition") or {},
        "formula_size": cfg.get("formula_size") or 1,
        "charge": cfg.get("charge") or 0,
        "multiplicity": cfg.get("multiplicity") or 0,
        # энергия изолированного атома считается ТОЛЬКО мастером (при старте)
        # и передаётся воркеру для лога E_binding/atom
        "e_isolated_atom": cfg.get("e_isolated_atom"),
    }


def _load_id_set(path):
    ids = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(line)
    return ids


# ---------- Вспомогательная функция для записи сбойных ID ----------
def _mark_as_failed(struct_id, filepath):
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"{struct_id}\n")


def _check_geometry(coords: np.ndarray, n_atoms: int, cfg) -> tuple[bool, str]:
    """Дешёвые проверки геометрии: размер, конечность, отсутствие слипшихся атомов."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        return False, f"geometry shape {coords.shape}"
    if coords.shape[0] != n_atoms:
        return False, f"atom count {coords.shape[0]} != {n_atoms}"
    if not np.isfinite(coords).all():
        return False, "non-finite coordinates"
    if n_atoms >= 2:
        # Блочно: матрица N×N×3 для n=10⁴ — это гигабайты, а нам нужен только
        # минимум. Память O(block²), время то же O(N²).
        step = 512
        d_min = float("inf")
        for i0 in range(0, n_atoms, step):
            a = coords[i0:i0 + step]
            for j0 in range(i0, n_atoms, step):
                b = coords[j0:j0 + step]
                d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
                if i0 == j0:
                    np.fill_diagonal(d, np.inf)
                m = float(d.min()) if d.size else float("inf")
                if m < d_min:
                    d_min = m
        if d_min < MIN_ATOM_DISTANCE_ANG:
            return False, (f"atoms too close: d_min={d_min:.4f} < "
                           f"{MIN_ATOM_DISTANCE_ANG} Å")
    return True, ""


# =====================================================================
# ПРИЁМКА ПО ФАЙЛУ СОСТОЯНИЯ (-RESTART.wfn): мастер считает всё сам
# =====================================================================
_NAO_TOTAL: dict[tuple[str, int], int] = {}


def _decode_wfn_b64(payload: dict):
    """Раскодирует присланный файл состояния из payload (base64)."""
    raw = payload.get("wfn_b64")
    if not raw:
        return None
    try:
        return base64.b64decode(raw, validate=False)
    except Exception:
        return None


def _geometry_key(coords, cfg) -> str:
    """Ключ геометрии для дедупликации/keep-best (координаты с точностью 1e-3 Å)."""
    payload = {
        "coords": [[round(float(x), 3) for x in atom] for atom in coords],
        "element": cfg["element"],
        "n_atoms": len(coords),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _rebuild_aggregate_xyz() -> int:
    """
    Пересобирает единый <element>_all_data.xyz из кадров frames/*.xyz
    (нужно, потому что лучший кадр для геометрии может быть ПЕРЕЗАПИСАН).
    """
    frames = CONFIG["frames_dir"]
    aggregate = os.path.join(CONFIG["output_dir"], CONFIG["output_all_xyz_name"])
    if not os.path.isdir(frames):
        return 0
    files = sorted(f for f in os.listdir(frames) if f.endswith(".xyz"))
    tmp = aggregate + ".tmp"
    with open(tmp, "w", encoding="utf-8") as out:
        for f in files:
            with open(os.path.join(frames, f), "r", encoding="utf-8") as src:
                out.write(src.read())
    os.replace(tmp, aggregate)
    return len(files)


def _effective_multiplicity(n_atoms: int, cfg) -> int:
    """Мультиплетность, которая реально ушла в CP2K.

    Если в конфиге 0 — CP2K берёт её по чётности числа электронов (1 или 2).
    В блок пишется фактическое значение, чтобы по датасету было видно, какое
    спиновое состояние посчитано: у одного и того же элемента оно зависит от n.
    """
    mult = int(cfg.get("multiplicity") or 0)
    if mult > 0:
        return mult
    unit = max(1, int(cfg.get("formula_size") or 1))
    electrons = (int(cfg.get("n_valence") or 0) * int(n_atoms) // unit
                 - int(cfg.get("charge") or 0))
    return 1 if electrons % 2 == 0 else 2


def _stale_reason(n_atoms: int) -> str | None:
    """В режиме расписания работа старого размера не принимается."""
    if not schedule_enabled(CONFIG):
        return None
    stage = stage_for_height(chain_mod.next_seq(CONFIG["output_dir"]) - 1, CONFIG)
    if int(n_atoms) != int(stage["n_atoms"]):
        return (f"этап {stage['stage']}: n={stage['n_atoms']} — работа для n={n_atoms} "
                f"больше не принимается")
    return None


def _same_geometry(coords: np.ndarray, species, issued: dict) -> tuple[bool, str]:
    """Сверка присланной геометрии с той, что выдал мастер (без CP2K).

    Воркер не должен двигать атомы: он получает готовые координаты и обязан
    вернуть их же. Допуск — только на округление JSON (~1e-6 Å).
    """
    ref = np.asarray(issued.get("coords") or issued.get("geometry") or [], dtype=np.float64)
    if ref.shape != coords.shape:
        return False, f"форма {coords.shape} != выданной {ref.shape}"
    if not np.allclose(coords, ref, atol=GEOMETRY_MATCH_TOL, rtol=0.0):
        shift = float(np.max(np.abs(coords - ref)))
        return False, f"атомы смещены на {shift:.3g} Å"
    want = issued.get("species")
    if want and species is not None:
        if [str(s) for s in species] != [str(s) for s in want]:
            return False, "другой сорт или порядок атомов"
    return True, ""


def verify_state_payload(cfg, task: dict, payload: dict, issued: dict | None = None) -> dict:
    """
    Приёмка по файлу состояния (архитектура A). Мастер НЕ верит числом воркера:

      1. геометрия — от воркера (это единственное, чем он управляет: число атомов,
         конечность, минимальная дистанция);
      2. синтаксис состояния: размер, структура Fortran-записей, nao/nspin/столбцы
         (формат зафиксирован зондом test/probe_wfn.py);
      3. один прогон CP2K ОТ этого состояния в задаче МАСТЕРА: элемент,
         псевдопотенциал, базис, ячейка, функционал и геометрия — из конфига мастера;
      4. сверка: restart прочитан (без предупреждений о несовместимости),
         nao файла == nao системы мастера, столбцы == nspin×n_occ,
         состояние доводится до SCF-сходимости за RESTART_VERIFY_SCF шагов;
      5. E, силы, gap, mag — из прогона мастера (числа воркера — только телеметрия).
    """
    t0 = time.perf_counter()
    struct_id = task.get("struct_id", "?")
    n_atoms = int(task["n_atoms"])

    def fail(reason, **extra):
        res = {"valid": False, "reason": reason, "elapsed": time.perf_counter() - t0}
        res.update(extra)
        print(f"  [Master] ✗ {struct_id}: {reason}")
        return res

    # Потолок и этап — до геометрии, numpy и CP2K: размер приходит из запроса.
    if not 1 <= n_atoms <= MAX_TASK_ATOMS:
        return fail(f"n_atoms {n_atoms} вне диапазона 1..{MAX_TASK_ATOMS}")
    stale = _stale_reason(n_atoms)
    if stale:
        return fail(stale)

    print(f"\n[Master] Приёмка состояния '{struct_id}' (n_atoms={n_atoms})...")

    if not payload.get("geometry"):
        return fail("нет геометрии в payload")
    coords = np.asarray(payload["geometry"], dtype=np.float64)
    ok, reason = _check_geometry(coords, n_atoms, cfg)
    if not ok:
        return fail(f"geometry rejected: {reason}")

    # 0. Мастер выдал геометрию сам (seed → координаты), значит воркер обязан
    #    посчитать ровно её. Сверка дешёвая и идёт до DFT. Готовый путь
    #    (результат принесён мастеру, который задание не выдавал) эталона не имеет.
    if issued:
        same, why = _same_geometry(coords, payload.get("species"), issued)
        if not same:
            return fail(f"геометрия не та, что выдана в задании: {why}")

    # Состав. При обычной выдаче эталон — species от мастера; в ready-пути
    # (результат принесён другому мастеру) проверяем только количество.
    issued_species = (issued or {}).get("species")
    species = payload.get("species")
    if issued_species:
        if species is not None and [str(s) for s in species] != [str(s) for s in issued_species]:
            return fail("состав/порядок атомов не совпадает с выданным заданием")
        cfg_species = [str(s) for s in issued_species]
    elif species:
        if not isinstance(species, list) or len(species) != n_atoms:
            return fail("species не соответствует числу атомов")
        mult = n_atoms // int(cfg.get("formula_size") or 1)
        expected = {sym: cnt * mult for sym, cnt in cfg["composition"].items()}
        got: dict[str, int] = {}
        for sym in species:
            got[str(sym)] = got.get(str(sym), 0) + 1
        if got != expected:
            return fail(f"состав {got} != ожидаемого {expected}")
        cfg_species = [str(s) for s in species]
    else:
        cfg_species = [cfg["element"]] * n_atoms
    run_cfg = dict(cfg)
    run_cfg["species"] = cfg_species

    raw = _decode_wfn_b64(payload)
    if raw is None:
        return fail("нет файла состояния (wfn_b64)")
    max_bytes = float(cfg.get("restart_max_mb", 50)) * 1024 * 1024
    if len(raw) > max_bytes:
        return fail(f"файл состояния {len(raw)} байт > лимита {max_bytes:.0f}")

    workdir = tempfile.mkdtemp(prefix="wfn_recv_")
    try:
        wfn_path = os.path.join(workdir, "submitted-RESTART.wfn")
        with open(wfn_path, "wb") as f:
            f.write(raw)

        head = cp2k_runner.parse_wfn_header(wfn_path)
        if not head["ok"]:
            return fail(f"синтаксис состояния: {head['reason']}")

        # Обученное ожидание nao системы — ловит другой элемент/размер до запуска CP2K
        expected_nao = _NAO_TOTAL.get((cfg["element"], n_atoms))
        if expected_nao and head["nao"] != expected_nao:
            return fail(f"nao={head['nao']} не соответствует {cfg['element']}×{n_atoms}"
                        f" (ожидалось {expected_nao})")

        res = cp2k_runner.verify_state_from_wfn(
            coords, wfn_path, run_cfg,
            project=f"verify_{struct_id}",
        )
        if not res.get("ok"):
            return fail(res.get("reason", "состояние не принято"),
                        state={"submitted": head, "sysinfo": res.get("sysinfo"),
                               "scf": res.get("scf"), "restart": res.get("restart")})

        info = res.get("sysinfo") or {}
        nspin = info.get("nspin") or 2
        if info.get("nao") and head["nao"] != info["nao"]:
            return fail(f"nao файла ({head['nao']}) != nao системы мастера ({info['nao']})")
        n_occ = info.get("n_occ")
        if n_occ and head["columns"] not in (n_occ, n_occ * nspin):
            return fail(f"столбцов {head['columns']}, ожидалось {n_occ} или {n_occ * nspin}")

        canon = res.get("canonical") or {}
        if canon.get("ok") and canon.get("size") and canon["size"] != head["size"]:
            print(f"  [Master] ! размер канонического wfn мастера ({canon['size']}) "
                  f"!= присланного ({head['size']})")

        if n_atoms:
            _NAO_TOTAL[(cfg["element"], n_atoms)] = head["nao"]

        first = res.get("first_step") or {}
        e_claim = payload.get("energy_ev")
        if e_claim is not None:
            try:
                d = abs(float(e_claim) - float(res["energy_ev"]))
                print(f"  [Master] телеметрия: E_воркера - E_мастера = {d:.6f} eV")
                if d > TOL_ENERGY_EV:
                    print(f"  [Master] ! воркер заявил другую энергию, но данные берём от мастера")
            except (TypeError, ValueError):
                pass

        print(f"  [Master] 1. состояние прочитано: nao={head['nao']}, столбцов={head['columns']}, "
              f"размер={head['size']}")
        print(f"  [Master] 2. SCF сошёлся за {res['scf'].get('steps')} шагов "
              f"(лимит {cp2k_runner.RESTART_VERIFY_SCF}), градиент первой строки={first.get('grad')}")
        print(f"  [Master] 3. E = {res['energy_ev']:.6f} eV (посчитал мастер), n_elec={info.get('nelec')}")
        e_iso = CONFIG.get("e_isolated_atom")
        e_bind = ((e_iso * n_atoms - float(res["energy_ev"])) / n_atoms
                  if e_iso is not None else None)
        gap = res.get("gap")
        mag = res.get("mag")
        gap_s = f"{gap:.4f}" if gap is not None else "н/д"
        mag_s = f"{mag:.4f}" if mag is not None else "н/д"
        bind_s = f"{e_bind:.6f}" if e_bind is not None else "н/д"
        print(f"  [Master] 4. gap = {gap_s} eV | MagMom = {mag_s} μB | "
              f"E_binding/atom = {bind_s} eV")
        print(f"  [Master] Вердикт: ✅ PASS")

        return {
            "valid": True,
            "reason": "",
            "elapsed": time.perf_counter() - t0,
            # true — геометрия сверена с выданной мастером;
            # geometry_policy: seed (выдал мастер) | worker (ready-путь)
            "geometry_verified": bool(issued),
            "geometry_policy": ("worker" if not issued else "seed"),
            "energy_ev": float(res["energy_ev"]),
            "energy_hartree": float(res.get("energy_hartree", float("nan"))),
            "forces_ev_ang": res.get("forces_ev_ang"),
            "gap": gap,
            "mag": mag,
            "props": res.get("props") or {},
            "e_isolated_atom": e_iso,
            "e_binding_per_atom": e_bind,
            "state": {
                "submitted": head,
                "sysinfo": info,
                "scf": res.get("scf"),
                "props": res.get("props") or {},
                "restart": {"mentioned": (res.get("restart") or {}).get("mentioned"),
                            "warnings": (res.get("restart") or {}).get("warnings", [])},
                "first_step": first,
                "canonical": {k: v for k, v in (res.get("canonical") or {}).items()
                              if k in ("ok", "size", "nao", "columns", "nspin")},
            },
            "s_master": None,
            "deep_checked": True,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _drain_mempool(utxo: dict, height: int, confirmations: int):
    """
    Отбирает транзакции для блока: по убыванию комиссии (как в Bitcoin),
    не больше MAX_TXS_PER_BLOCK. Остальные остаются ждать следующего блока.
    Невалидные/конфликтующие (вход уже потрачен) удаляются.
    """
    included, drop = [], []
    fee_total = 0
    with MEMPOOL_LOCK:
        items = list(MEMPOOL.items())

    def fee_of(item):
        fee = chain_mod.tx_fee_safe(item[1], utxo)
        return -1 if fee is None else fee

    for tid, tx in sorted(items, key=fee_of, reverse=True):
        if len(included) >= chain_mod.MAX_TXS_PER_BLOCK:
            break                     # остаток ждёт следующего блока
        ok, _problems = chain_mod.validate_tx(
            tx, CONFIG["output_dir"], height, confirmations, utxo=utxo)
        if not ok:
            drop.append(tid)
            continue
        total_in = 0
        for i in tx["inputs"]:
            total_in += int(utxo.pop((i["txid"], int(i["vout"])))["value"])
        for vout, o in enumerate(tx["outputs"]):
            utxo[(tid, vout)] = {"value": int(o["value"]),
                                 "address": o["address"], "height": height}
        fee_total += total_in - sum(int(o["value"]) for o in tx["outputs"])
        included.append(tx)
        drop.append(tid)
    if drop:
        with MEMPOOL_LOCK:
            for tid in drop:
                MEMPOOL.pop(tid, None)
        _save_mempool()
    return included, int(fee_total)


def _write_chain_block(task_data: dict, payload: dict, verdict: dict,
                       info: dict | None = None) -> str | None:
    """
    Кладёт проверенный результат в блокчейн-блок: blob + подписанные UTXO-транзакции.

    info (необязательно) заполняется итогом: block — путь к блоку,
    reward_paid — ушла ли награда воркеру, reason — почему блока нет.
    """
    def _fail(reason: str) -> None:
        if info is not None:
            info["reason"] = reason

    raw = payload.get("wfn_b64")
    if not raw:
        print("[Master] ! в payload нет wfn — блок не создан")
        _fail("в payload нет wfn")
        return None
    try:
        wfn_bytes = base64.b64decode(raw)
    except Exception as e:  # noqa: BLE001
        print(f"[Master] ! wfn_b64 не декодируется: {e}")
        _fail(f"wfn_b64 не декодируется: {e}")
        return None

    struct_id = task_data["struct_id"]
    n_atoms = int(task_data["n_atoms"])
    w_id = payload.get("worker_id")

    with tempfile.TemporaryDirectory(prefix="chain_block_") as td:
        wfn_path = os.path.join(td, f"{struct_id}.wfn")
        with open(wfn_path, "wb") as f:
            f.write(wfn_bytes)
        state_sha, state_rel = chain_mod.store_blob(CONFIG["output_dir"], wfn_path)

    wallet = CONFIG.get("wallet")
    if wallet is None:
        print("[Master] ! кошелёк мастера не разблокирован — блок не создан")
        _fail("кошелёк мастера не разблокирован")
        return None
    latest = wallet.latest_key()
    if latest is None:
        print("[Master] ! нет ключей мастера — блок не создан")
        _fail("у мастера нет ключей")
        return None
    master_addresses = wallet.addresses()

    # Самая длинная цепочка: сначала догоняем пиров; свой блок — только если никто не выше
    cu = catch_up_chain()
    if not cu["ok"]:
        print("[Master] ✗ не удалось догнать самую длинную цепочку — свой блок не создаю")
        _fail("не удалось догнать цепочку")
        return None
    if cu["accepted"]:
        print(f"[Master] догнал сеть: высота {cu['height']} (принято {cu['accepted']} блоков)")

    seq = chain_mod.next_seq(CONFIG["output_dir"])
    worker_address = (registered_workers.get(w_id) or {}).get("pubkey_hash")
    confirmations = int(CONFIG.get("coinbase_confirmations",
                                   chain_mod.DEFAULT_CONFIRMATIONS))

    reward = None
    fee = 0          # целое: суммы в консенсусе — только int (и ':d' в логе)
    # Награда выплачивается только на адрес вида sha256-hex (64 символа).
    # Иначе монеты уходят в никуда: такой выход нельзя потратить ни одним кошельком.
    if worker_address and not re.fullmatch(r"[0-9a-f]{64}", str(worker_address)):
        print(f"[Master] ! адрес воркера {worker_address!r} не похож на адрес — "
              "награда не выплачивается")
        worker_address = None
    utxo = chain_mod.utxo_set(CONFIG["output_dir"])
    if worker_address:
        reward = chain_mod.build_transfer(
            CONFIG["output_dir"], master_addresses, worker_address,
            latest["address"], value=chain_mod.REWARD_VALUE,
            confirmations=confirmations, height=seq,
        )
        if reward is not None:
            fee = int(chain_mod.tx_fee(reward, utxo))
            # подписываем входы Ed25519 ключами мастера (txid от подписей не зависит)
            chain_mod.sign_transaction(reward, wallet, utxo)
            # учитываем reward в локальном UTXO, чтобы мемпул не конфликтовал
            for i in reward["inputs"]:
                utxo.pop((i["txid"], int(i["vout"])), None)
            rtid = reward.get("txid") or chain_mod.txid_of(reward)
            for vout, o in enumerate(reward["outputs"]):
                utxo[(rtid, vout)] = {"value": int(o["value"]),
                                      "address": o["address"], "height": seq}

    mem_txs, mem_fee = _drain_mempool(utxo, seq, confirmations)

    # coinbase = субсидия + комиссии блока (как в Bitcoin)
    coinbase = chain_mod.make_coinbase(seq, latest["address"],
                                       chain_mod.COINBASE_VALUE + fee + mem_fee)
    txs = [coinbase] + ([reward] if reward else []) + mem_txs

    dft = {
        "energy_ev": verdict.get("energy_ev"),
        "energy_hartree": verdict.get("energy_hartree"),
        "forces_ev_ang": verdict.get("forces_ev_ang"),
        "geometry": payload.get("geometry"),
        # виды атомов: для многокомпонентных систем без них блок не проверить
        "species": payload.get("species"),
        "homo_lumo_gap_ev": verdict.get("gap"),
        "magnetic_moment": verdict.get("mag"),
        "e_isolated_atom_ev": verdict.get("e_isolated_atom"),
        "e_binding_per_atom_ev": verdict.get("e_binding_per_atom"),
        # была ли геометрия сверена с выданной мастером (только шаблонный режим)
        "geometry_verified": bool(verdict.get("geometry_verified")),
        "geometry_policy": verdict.get("geometry_policy"),
        "geometry_seed": payload.get("seed"),
        # параметр генерации (не консенсус): нужен для воспроизведения выборки
        "atom_diameter": CONFIG.get("atom_diameter"),
        # фактическое электронное состояние: у одного элемента мультиплетность
        # зависит от n и геометрии, поэтому она обязана быть в датасете
        "charge": int(CONFIG.get("charge") or 0),
        "multiplicity": _effective_multiplicity(n_atoms, CONFIG),
        "scf": (verdict.get("state") or {}).get("scf"),
    }
    # Физические наблюдаемые (мастер посчитал в том же прогоне, что и E):
    # диполь, заряды/спины по атомам, разложение энергии, занятый спектр,
    # IP/EA по Купмансу, <S²>, max/RMS силы. Сотни байт на блок.
    props = verdict.get("props") or (verdict.get("state") or {}).get("props") or {}
    dft.update(props)
    dft["props_version"] = cp2k_runner.PROPS_VERSION

    utxo_before = chain_mod.utxo_set(CONFIG["output_dir"])
    path = chain_mod.write_block(
        CONFIG["output_dir"], CONFIG["element"], n_atoms, state_sha, state_rel, txs,
        cp2k_version=CONFIG.get("cp2k_version", "unknown"),
        cp2k_build=CONFIG.get("cp2k_build", ""),
        cp2k_params=cp2k_runner.CP2K_PARAMS,
        dft=dft,
        nonce=int(task_data.get("_nonce", 0)),
        dim=int(task_data.get("dim", 3)),
        atom_diameter=float(CONFIG.get("atom_diameter") or 2.0),
        composition=CONFIG.get("composition"),
    )

    v = chain_mod.verify_block(CONFIG["output_dir"], path, expected_seq=seq,
                               utxo=utxo_before)
    if not v["ok"]:
        bad = path + ".bad"
        try:
            os.replace(path, bad)
        except OSError:
            pass
        print("[Master] ✗ собранный блок не прошёл проверку и не опубликован: "
              + "; ".join(v["problems"][:3]))
        _fail("блок не прошёл проверку: " + "; ".join(v["problems"][:2]))
        return None

    print(f"[Master] ⛓ блок {os.path.basename(path)}: blob {state_sha[:12]}…, "
          f"coinbase {chain_mod.COINBASE_VALUE + fee + mem_fee:d} → {latest['address'][:12]}…")
    _announce_to_peers(seq)
    _sync_stage("свой блок")
    if mem_txs:
        print(f"[Master] 📦 в блок включено транзакций из мемпула: {len(mem_txs)} "
              f"(комиссия {mem_fee:g})")
    if reward:
        print(f"[Master] 💰 награда {chain_mod.REWARD_VALUE:d} → {worker_address[:12]}…, "
              f"комиссия {fee:g} (входов: {len(reward['inputs'])})")
    else:
        print(f"[Master] ⚠ награда {chain_mod.REWARD_VALUE:d} не выплачена: "
              f"нет подтверждённых средств (conf={confirmations})")
    if info is not None:
        info["block"] = path
        info["reward_paid"] = bool(reward)
    return path


def _persist_verified_state(task_data: dict, payload: dict, verdict: dict,
                            block_info: dict | None = None) -> str:
    """
    Записывает кадр в набор для MACE: E/F/gap/mag — из прогона мастера.
    Дубликаты геометрии не плодятся: для одной геометрии хранится только
    кадр с наименьшей энергией (keep-best).

    :return: "accepted" | "kept_old"
    """
    struct_id = task_data["struct_id"]
    n_atoms = int(task_data["n_atoms"])
    dim = int(task_data["dim"])
    coords = np.asarray(payload["geometry"], dtype=np.float64)
    energy_ev = float(verdict["energy_ev"])
    forces = verdict.get("forces_ev_ang") or [[0.0, 0.0, 0.0]] * n_atoms
    gap = verdict.get("gap")
    mag = verdict.get("mag")
    e_iso = verdict.get("e_isolated_atom")
    e_bind = verdict.get("e_binding_per_atom")
    key = _geometry_key(coords, CONFIG)

    frames = CONFIG["frames_dir"]
    os.makedirs(frames, exist_ok=True)
    meta_path = os.path.join(frames, f"{key}.json")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if old.get("energy_ev") is not None and energy_ev >= float(old["energy_ev"]) - 1e-6:
                print(f"[Master] = {struct_id}: геометрия {key} уже есть с E={old['energy_ev']:.6f} eV, "
                      f"новая не лучше — оставляем старый кадр")
                return "kept_old"
        except Exception:
            old = None

    props = verdict.get("props") or {}
    extra_comment = ""
    for field, fmt in (("ip_koopmans_ev", "%.4f"),
                       ("ea_koopmans_ev", "%.4f"),
                       ("dipole_total_debye", "%.4f")):
        value = props.get(field)
        if value is not None:
            extra_comment += f"{field}={fmt % value} "

    record = cp2k_runner.build_xyz_record(
        coords, energy_ev, forces, n_atoms, dim, struct_id, CONFIG, gap, mag,
        e_binding_per_atom=e_bind, e_isolated_atom=e_iso,
        extra_comment=extra_comment,
    )
    tmp = os.path.join(frames, f"{key}.xyz.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(record)
    os.replace(tmp, os.path.join(frames, f"{key}.xyz"))

    state = verdict.get("state") or {}
    meta = {
        "geometry_key": key,
        "struct_id": struct_id,
        "n_atoms": n_atoms,
        "dim": dim,
        "element": CONFIG["element"],
        "potential": CONFIG["potential"],
        "cell_size": cp2k_runner.resolve_cell_size(coords, CONFIG),
        "geometry": coords.tolist(),
        "energy_ev": energy_ev,
        "energy_hartree": verdict.get("energy_hartree"),
        "homo_lumo_gap_ev": gap,
        "magnetic_moment": mag,
        "e_isolated_atom_ev": e_iso,
        "e_binding_per_atom_ev": e_bind,
        "charge": int(CONFIG.get("charge") or 0),
        "multiplicity": _effective_multiplicity(n_atoms, CONFIG),
        "physical": props,
        "checks": {
            "verified_by": "master_one_step_from_restart",
            "nao": (state.get("submitted") or {}).get("nao"),
            "nspin": (state.get("submitted") or {}).get("nspin"),
            "columns": (state.get("submitted") or {}).get("columns"),
            "sysinfo": state.get("sysinfo"),
            "scf": state.get("scf"),
            "restart_read_confirmed": (state.get("restart") or {}).get("mentioned"),
            "first_step_grad": (state.get("first_step") or {}).get("grad"),
            "restart_verify_scf_limit": cp2k_runner.RESTART_VERIFY_SCF,
            "energy_verified": True,
        },
        "provenance": {
            "worker_id": payload.get("worker_id"),
            "attempts": task_data.get("attempts", 0),
            "worker_elapsed_sec": payload.get("elapsed"),
            "worker_energy_claim_ev": payload.get("energy_ev"),
            "worker_wfn_size": payload.get("wfn_size"),
            # seed геометрии: тот же seed + та же версия генератора = тот же кадр
            "seed": payload.get("seed"),
            "atom_diameter": CONFIG.get("atom_diameter"),
            # true — присланная геометрия совпала с выданной (шаблонный режим)
            "geometry_verified": bool(verdict.get("geometry_verified")),
            "geometry_policy": verdict.get("geometry_policy"),
            "verified_at": time.time(),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    n_frames = _rebuild_aggregate_xyz()
    with open(CONFIG["verified_ids_file"], "a", encoding="utf-8") as f:
        f.write(f"{struct_id}\n")

    print(f"[Master] ✓ {struct_id}: кадр принят для MACE (E={energy_ev:.6f} eV, "
          f"кадров всего {n_frames})")
    _write_chain_block(task_data, payload, verdict, info=block_info)
    return "accepted"


# =====================================================================
# СЕРВЕР ОЧЕРЕДИ С УЧЕТОМ НОМЕРОВ ВОРКЕРОВ
# =====================================================================
app = FastAPI()


def _body_limit_bytes() -> int:
    """Лимит тела = base64 от максимально допустимого wfn + запас.

    Легитимный wfn ограничен restart_max_mb, base64 даёт 1.33×. Один knob:
    подняли restart_max_mb для 55+ атомов — поднялся и лимит.
    """
    mb = max(1, int(CONFIG.get("restart_max_mb", 50)))
    return int(mb * 1024 * 1024 * 1.4) + 2 * 1024 * 1024


@app.middleware("http")
async def _body_guard(request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        limit = _body_limit_bytes()
        declared = request.headers.get("content-length")
        from fastapi.responses import JSONResponse
        if declared and declared.isdigit() and int(declared) > limit:
            return JSONResponse({"detail": f"тело запроса больше {limit} байт"},
                                status_code=413)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                return JSONResponse({"detail": f"тело запроса больше {limit} байт"},
                                    status_code=413)
        request._body = bytes(body)
    return await call_next(request)


@app.middleware("http")
async def _reg_guard(request, call_next):
    """
    Лимит регистраций по IP: регистрация даёт право на DFT-прогоны за счёт
    мастера, поэтому бесплатно набивать воркеров нельзя ни ключами, ни
    скриптом. Счётчик по IP (ключи бесплатны), окно — REG_WINDOW_SEC.
    """
    try:
        path = (request.url.path or "").rstrip("/") or "/"
        ip = request.client.host if getattr(request, "client", None) else "?"
    except Exception:  # noqa: BLE001
        return await call_next(request)
    if path in ("/register-worker", "/register-challenge") and not _reg_rate_ok(ip):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"detail": f"слишком много регистраций с {ip}: не больше "
                       f"{REG_LIMIT_PER_IP} запросов в {int(REG_WINDOW_SEC)} с"},
            status_code=429)
    return await call_next(request)


def _admin_allowed(host: str, token: str | None = None) -> bool:
    """
    Админ-API: локально — всегда, снаружи — только по MASTER_ADMIN_TOKEN.

    Локальный доступ нужен панели и master-cli на той же машине. Удалённое
    управление (--url http://host:8000) требует токена: без него любой, кто
    видит порт мастера, менял бы config.json, паузу и пиров.
    """
    if (host or "") in ("127.0.0.1", "::1", "localhost"):
        return True
    if not ADMIN_TOKEN or not token:
        return False
    return hmac.compare_digest(str(token), ADMIN_TOKEN)


@app.middleware("http")
async def _admin_guard(request, call_next):
    if request.url.path.startswith("/admin"):
        host = (request.client.host if request.client else "") or ""
        token = (request.headers.get("x-admin-token")
                 or request.query_params.get("token"))
        if not _admin_allowed(host, token):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                {"detail": "admin API: нужен MASTER_ADMIN_TOKEN (заголовок "
                           "X-Admin-Token или ?token=)"},
                status_code=401)
    return await call_next(request)

task_queue_lock = threading.RLock()   # RLock: блок пишется под этим же lock, а внутри зовётся _sync_stage()
pending_tasks = []      # Список нерешенных структур
active_tasks = {}       # struct_id -> {"task": dict, "time_sent": float, "worker_id": str}
verifying_tasks = set() # struct_id, результаты которых проверяются прямо сейчас
completed_tasks = set() # Завершенные struct_id
failed_tasks = set()    # Окончательно забракованные struct_id

next_worker_id = 1
# worker_id -> {name, pubkey_hash, public_key, registered_at, last_seen}
registered_workers: dict[str, dict] = {}
# pubkey_hash -> worker_id (уникальность воркера по хэшу публичного ключа)
worker_by_pubkey: dict[str, str] = {}
# Пауза выдачи задач (управляется master-cli.py)
PAUSED = False
# Мастер занят сборкой/догоном цепочки: новые работы не принимаются
BUSY = threading.Event()

# Мемпул: txid -> транзакция, ожидающая включения в блок.
# Политика как в Bitcoin: ограниченный размер, вытеснение по комиссии,
# замена конфликта только более дорогой транзакцией.
MEMPOOL: dict[str, dict] = {}
MEMPOOL_LOCK = threading.Lock()
MEMPOOL_MAX_TX = 10_000          # ~5 МБ тел; больше не держим
MEMPOOL_SAVE_INTERVAL = 2.0      # дебаунс записи (раньше файл писался на каждый submit)
_MEMPOOL_DIRTY = False
_MEMPOOL_SAVED_AT = 0.0


def _mempool_path() -> str:
    return os.path.join(CONFIG["output_dir"], "mempool.json")


def _save_mempool(force: bool = False):
    """Персист мемпула с дебаунсом: O(n²) записи на флуд больше нет."""
    global _MEMPOOL_DIRTY, _MEMPOOL_SAVED_AT
    now = time.time()
    with MEMPOOL_LOCK:
        if not force and now - _MEMPOOL_SAVED_AT < MEMPOOL_SAVE_INTERVAL:
            _MEMPOOL_DIRTY = True
            return
        data = list(MEMPOOL.values())
        _MEMPOOL_DIRTY = False
        _MEMPOOL_SAVED_AT = now
    try:
        tmp = _mempool_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _mempool_path())
    except Exception as e:  # noqa: BLE001
        print(f"[Master] ! не удалось сохранить мемпул: {e}")


def _mempool_flush_loop(interval: float = MEMPOOL_SAVE_INTERVAL):
    """Фоновая запись отложенных изменений мемпула."""
    while True:
        time.sleep(interval)
        if _MEMPOOL_DIRTY:
            _save_mempool(force=True)


def _tx_input_keys(tx: dict) -> set:
    return {(i.get("txid"), int(i.get("vout", -1)))
            for i in tx.get("inputs", []) if not i.get("coinbase")}


def _conflicting_txs(tx: dict) -> list[str]:
    """txid-ы мемпула, тратящие те же входы (взаимоисключающие)."""
    keys = _tx_input_keys(tx)
    if not keys:
        return []
    with MEMPOOL_LOCK:
        return [tid for tid, other in MEMPOOL.items()
                if other is not tx and (_tx_input_keys(other) & keys)]


def _drop_conflicts(tx: dict) -> list[str]:
    """Убирает конфликты без сравнения комиссий (загрузка/реорг)."""
    clash = _conflicting_txs(tx)
    if clash:
        with MEMPOOL_LOCK:
            for tid in clash:
                MEMPOOL.pop(tid, None)
    return clash


def _load_mempool():
    """Читает мемпул с диска; лишнее сверх лимита отсекается по комиссии."""
    path = _mempool_path()
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[Master] ! мемпул не прочитан ({e}) — начинаю с пустого")
        return 0
    loaded = 0
    for tx in data if isinstance(data, list) else []:
        tid = tx.get("txid") or chain_mod.txid_of(tx)
        tx["txid"] = tid
        _drop_conflicts(tx)
        with MEMPOOL_LOCK:
            MEMPOOL[tid] = tx
        loaded += 1

    if len(MEMPOOL) > MEMPOOL_MAX_TX:
        utxo = chain_mod.utxo_set(CONFIG["output_dir"])
        ranked = []
        with MEMPOOL_LOCK:
            items = list(MEMPOOL.items())
        for tid, tx in items:
            fee = chain_mod.tx_fee_safe(tx, utxo)
            ranked.append(((-1 if fee is None else fee), tid))
        ranked.sort()
        victims = [tid for _, tid in ranked[:len(ranked) - MEMPOOL_MAX_TX]]
        with MEMPOOL_LOCK:
            for tid in victims:
                MEMPOOL.pop(tid, None)
        print(f"[Master] мемпул усечён до {MEMPOOL_MAX_TX} tx (вытеснено {len(victims)})")

    if loaded:
        print(f"[Master] Мемпул восстановлен: {loaded} tx")
    return loaded


def _prune_mempool(reason: str = "") -> int:
    """
    Чистит мемпул после появления нового блока (своего или из сети):
    если вход уже потрачен (tx попала в блок) или конфликтует — убираем.
    """
    with MEMPOOL_LOCK:
        items = list(MEMPOOL.items())
    if not items:
        return 0

    utxo = chain_mod.utxo_set(CONFIG["output_dir"])
    height = chain_mod.next_seq(CONFIG["output_dir"])
    conf = int(CONFIG.get("coinbase_confirmations", chain_mod.DEFAULT_CONFIRMATIONS))
    drop = []
    for tid, tx in items:
        ok, _problems = chain_mod.validate_tx(
            tx, CONFIG["output_dir"], height, conf, utxo=utxo)
        if not ok:
            drop.append(tid)
    if not drop:
        return 0

    with MEMPOOL_LOCK:
        for tid in drop:
            MEMPOOL.pop(tid, None)
    _save_mempool()
    suffix = f" ({reason})" if reason else ""
    print(f"[Master] мемпул обновлён{suffix}: убрано {len(drop)} tx "
          f"(попали в блок или конфликтуют)")
    return len(drop)


@app.get("/queue-status")
def queue_status():
    with task_queue_lock:
        return {
            "pending": len(pending_tasks),
            "active": len(active_tasks),
            "verifying": len(verifying_tasks),
            "completed": len(completed_tasks),
            "failed": len(failed_tasks),
            "paused": PAUSED,
            "stage": CONFIG.get("_stage_info"),
            "workers": sorted(registered_workers.keys()),
            # Задача возвращается в очередь, если воркер молчит дольше этого
            "worker_lease_sec": WORKER_LEASE_SEC,
            "active_details": {
                sid: {
                    "worker_id": info["worker_id"],
                    "age_sec": round(time.time() - info["time_sent"], 1),
                    "worker_silent_sec": round(time.time() - _worker_last_seen(info), 1),
                    "attempts": info["task"].get("attempts", 0),
                }
                for sid, info in active_tasks.items()
            },
        }


def _worker_last_seen(info: dict) -> float:
    """Когда воркер, держащий задачу, последний раз подавал признаки жизни."""
    rec = registered_workers.get(info["worker_id"]) or {}
    return max(float(rec.get("last_seen") or 0.0), float(info["time_sent"]))


def _requeue_worker_tasks(worker_id: str, reason: str) -> list[str]:
    """Снимает все задачи воркера и возвращает их в очередь (под task_queue_lock)."""
    moved = [sid for sid, info in active_tasks.items() if info["worker_id"] == worker_id]
    for sid in moved:
        info = active_tasks.pop(sid)
        print(f"[Master] ↺ {sid} снята с {worker_id}: {reason}")
        _requeue_or_fail(info["task"], reason)
    return moved


def _requeue_or_fail(task_data: dict, reason: str, *, requeue_always: bool = False):
    """
    Возвращает задание в очередь либо окончательно бракует его.
    Вызывать только под task_queue_lock.
    """
    sid = task_data["struct_id"]
    task_data["attempts"] = int(task_data.get("attempts", 0)) + 1

    if not requeue_always and task_data["attempts"] >= MAX_TASK_ATTEMPTS:
        failed_tasks.add(sid)
        _mark_as_failed(sid, CONFIG["failed_ids_file"])
        print(f"[Master] ✗ {sid} отклонена окончательно после "
              f"{task_data['attempts']} попыток: {reason}")
        return

    pending_tasks.append(task_data)
    print(f"[Master] ↺ {sid} возвращена в очередь "
          f"(попытка {task_data['attempts']}/{MAX_TASK_ATTEMPTS}): {reason}")


class WorkerRegistration(BaseModel):
    name: str | None = None
    pubkey_hash: str | None = None
    public_key: str | None = None
    nonce: str | None = None
    signature: str | None = None


# ---------- сессия воркера: право говорить за worker_id ----------
_REG_IP_LOCK = threading.Lock()
_REG_IP_HITS: dict[str, list[float]] = {}
_REG_IP_MAX_KEYS = 4096


def _reset_reg_rate():
    """Сбросить счётчики регистраций по IP (тесты, диагностика)."""
    with _REG_IP_LOCK:
        _REG_IP_HITS.clear()


def _reg_rate_ok(ip: str) -> bool:
    """
    Скользящее окно: не больше REG_LIMIT_PER_IP запросов регистрации в минуту
    с одного IP. Считаем по IP, а не по ключу: ключи бесплатны, а IP — нет.
    """
    now = time.time()
    key = str(ip or "?")
    with _REG_IP_LOCK:
        if len(_REG_IP_HITS) > _REG_IP_MAX_KEYS:
            for k, v in list(_REG_IP_HITS.items()):
                if not v or now - v[-1] > REG_WINDOW_SEC:
                    _REG_IP_HITS.pop(k, None)
        hits = [t for t in _REG_IP_HITS.get(key, []) if now - t < REG_WINDOW_SEC]
        if len(hits) >= REG_LIMIT_PER_IP:
            _REG_IP_HITS[key] = hits
            return False
        hits.append(now)
        _REG_IP_HITS[key] = hits
    return True


def _session_add_locked(worker_id: str) -> str:
    """
    Новый session-токен воркера. В памяти держим только sha256: дамп памяти
    мастера не даёт играть за воркера. Старые сессии живут, пока их не
    вытеснят новые — воркер одновременно ходит к нескольким мастерам.
    """
    token = secrets.token_urlsafe(32)
    rec = registered_workers[worker_id]
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    sessions = rec.setdefault("sessions", [])
    sessions.append(digest)
    del sessions[:-WORKER_SESSION_MAX]
    return token


def _auth_worker(worker_id: str, session: str | None) -> dict:
    """
    Проверяет право говорить за worker_id.

    worker_id публичен (/queue-status, блоки), поэтому без session задача не
    выдаётся и результат не принимается: иначе знающий номер брал бы задачи
    жертвы, держал их heartbeat'ом и останавливал очередь.
    """
    rec = registered_workers.get(worker_id)
    if not rec:
        raise HTTPException(status_code=400,
                            detail="Неизвестный воркер. Сначала пройдите регистрацию.")
    given = (session or "").strip()
    if not given:
        raise HTTPException(
            status_code=403,
            detail="нужен session: он выдаётся при регистрации с подписью "
                   "(GET /register-challenge → POST /register-worker)")
    digest = hashlib.sha256(given.encode("utf-8", "ignore")).hexdigest()
    if not any(hmac.compare_digest(digest, s) for s in rec.get("sessions") or []):
        # Перебирать 256-битный токен бессмысленно, поэтому троттлинг не нужен:
        # проверка — сравнение двух хэшей, а отказ должен быть понятным.
        raise HTTPException(status_code=403, detail="неверный session")
    rec["last_seen"] = time.time()
    return rec


def _drop_worker_locked(worker_id: str, reason: str) -> None:
    """Снимает воркера с учёта; его задачи возвращаются в очередь."""
    rec = registered_workers.pop(worker_id, None)
    if not rec:
        return
    pk = rec.get("pubkey_hash")
    if pk and worker_by_pubkey.get(pk) == worker_id:
        worker_by_pubkey.pop(pk, None)
    _requeue_worker_tasks(worker_id, f"воркер снят с учёта: {reason}")


def _evict_workers_locked(now: float) -> int:
    """
    Освобождает место под новых воркеров: сначала молчащие дольше
    WORKER_FORGET_SEC, затем самые старые бесключевые.
    """
    dropped = 0
    for wid, rec in list(registered_workers.items()):
        if len(registered_workers) < MAX_REGISTERED_WORKERS:
            break
        if now - float(rec.get("last_seen") or 0.0) > WORKER_FORGET_SEC:
            _drop_worker_locked(wid, "нет связи")
            dropped += 1
    if len(registered_workers) >= MAX_REGISTERED_WORKERS:
        for wid, rec in sorted(registered_workers.items(),
                               key=lambda kv: float(kv[1].get("registered_at") or 0.0)):
            if len(registered_workers) < MAX_REGISTERED_WORKERS:
                break
            if not rec.get("pubkey_hash"):
                _drop_worker_locked(wid, "нет ключа, мастер перегружен")
                dropped += 1
    return dropped


def _register_worker(name: str | None, pubkey_hash: str | None,
                     public_key: str | None, verified: bool = False,
                     requeue: bool = True) -> dict:
    """
    Регистрация воркера. Возвращает worker_id и session.

    verified=True — воркер подписал одноразовый nonce. HTTP-слой принимает
    только такие регистрации: без подписи worker_id выдавался бы бесплатно,
    а с ним — право на DFT-прогоны за счёт мастера.
    """
    global next_worker_id
    with task_queue_lock:
        now = time.time()
        if verified and pubkey_hash and pubkey_hash in worker_by_pubkey:
            worker_id = worker_by_pubkey[pubkey_hash]
            rec = registered_workers[worker_id]
            rec.update({"last_seen": now,
                        "name": name or rec.get("name"),
                        "public_key": public_key or rec.get("public_key")})
            session = _session_add_locked(worker_id)
            print(f"[Master] Воркер {worker_id} переподключился (pubkey {pubkey_hash[:12]}…)")
            # Воркер перезапустился: задачу, которую он держал, он уже не посчитает
            moved = (_requeue_worker_tasks(worker_id, "воркер перезапустился")
                     if requeue else [])
            return {"status": "ok", "worker_id": worker_id, "known": True,
                    "requeued": moved, "session": session}

        if len(registered_workers) >= MAX_REGISTERED_WORKERS:
            _evict_workers_locked(now)
            if len(registered_workers) >= MAX_REGISTERED_WORKERS:
                print(f"[Master] ✗ отказ в регистрации: {len(registered_workers)} "
                      f"воркеров, лимит {MAX_REGISTERED_WORKERS}")
                raise HTTPException(
                    status_code=503,
                    detail="мастер перегружен воркерами: повторите позже")

        worker_id = f"worker_{next_worker_id}"
        next_worker_id += 1
        registered_workers[worker_id] = {
            "name": name,
            "pubkey_hash": pubkey_hash,
            "public_key": public_key,
            "registered_at": now,
            "last_seen": now,
            "sessions": [],
        }
        if verified and pubkey_hash:
            worker_by_pubkey[pubkey_hash] = worker_id
        session = _session_add_locked(worker_id)

        print(f"[Master] Зарегистрирован воркер {worker_id}"
              + (f" (name={name}, pubkey={pubkey_hash[:12]}…)" if pubkey_hash else " (без ключа)"))
        return {"status": "ok", "worker_id": worker_id, "known": False,
                "session": session}


@app.get("/register-challenge")
def register_challenge():
    """Одноразовый nonce: воркер подписывает его и доказывает владение ключом."""
    return {"nonce": _issue_registration_nonce(), "ttl": REG_NONCE_TTL,
            "sign_prefix": REG_SIGN_PREFIX.decode("ascii")}


@app.post("/register-worker")
def register_worker(reg: WorkerRegistration):
    """
    Регистрация только с доказательством ключа: nonce с /register-challenge,
    подписанный приватным ключом адреса pubkey_hash.

    Без подписи регистрация бесплатна: скрипт набивает тысячи воркеров
    (память мастера), занимает задачи и не считает их. Поэтому 403.
    """
    if not (reg.nonce and reg.signature and reg.public_key and reg.pubkey_hash):
        raise HTTPException(
            status_code=403,
            detail="регистрация требует подписи nonce: GET /register-challenge → "
                   "подписать «cp2k-mace-register:»+nonce → POST сюда")
    reason = _verify_registration(reg.nonce, reg.public_key,
                                  reg.pubkey_hash, reg.signature)
    if reason:
        raise HTTPException(status_code=403,
                            detail=f"регистрация не подтверждена: {reason}")
    return _register_worker(reg.name, reg.pubkey_hash, reg.public_key,
                            verified=True)


@app.get("/register-worker")
def register_worker_get():
    """Старый путь протокола v1 (worker_id без ключа) закрыт."""
    raise HTTPException(
        status_code=410,
        detail="GET /register-worker больше не выдаёт worker_id: используйте "
               "POST с подписью nonce (GET /register-challenge)")


def _expire_tasks(now: float | None = None) -> list[str]:
    """
    Возвращает в очередь задачи, за которые больше некому отвечать:

      * воркер молчит дольше WORKER_LEASE_SEC (нет heartbeat) — обычный путь;
        задача возвращается через ~3 минуты, а не через сутки;
      * задача висит дольше _task_timeout() — резервный предохранитель.

    Заодно забывает мёртвых воркеров, чтобы не копились worker_1…worker_N.
    """
    expired: list[str] = []
    now = now if now is not None else time.time()
    limit = _task_timeout()
    with task_queue_lock:
        for sid, info in list(active_tasks.items()):
            silent = now - _worker_last_seen(info)
            hard = now - info["time_sent"] > limit
            if not hard and silent <= WORKER_LEASE_SEC:
                continue
            active_tasks.pop(sid, None)
            reason = (f"таймаут {limit} с" if hard
                      else f"воркер {info['worker_id']} молчит {int(silent)} с")
            print(f"[Master Watchdog] Задача {sid} (воркер {info['worker_id']}): "
                  f"{reason} — возвращаю в очередь")
            _requeue_or_fail(info["task"], reason, requeue_always=hard)
            expired.append(sid)
    return expired


def _forget_dead_workers(now: float | None = None) -> list[str]:
    """Забывает воркеров, молчащих дольше WORKER_FORGET_SEC (список освобождается)."""
    now = now if now is not None else time.time()
    forgotten: list[str] = []
    with task_queue_lock:
        for wid, rec in list(registered_workers.items()):
            last_seen = float(rec.get("last_seen") or 0.0)
            if now - last_seen <= WORKER_FORGET_SEC:
                continue
            registered_workers.pop(wid, None)
            for pk, mapped in list(worker_by_pubkey.items()):
                if mapped == wid:
                    worker_by_pubkey.pop(pk, None)
            forgotten.append(wid)
            print(f"[Master] Воркер {wid} забыт: нет связи {int(now - last_seen)} с")
    return forgotten


def _watchdog_loop(interval: int = 30):
    """Фоновый watchdog: снимает задачи с пропавших воркеров каждые 30 с."""
    while True:
        time.sleep(interval)
        try:
            _expire_tasks()
            _forget_dead_workers()
        except Exception as e:  # noqa: BLE001
            print(f"[Master Watchdog] ошибка: {e}")


@app.get("/get-task")
def get_task(worker_id: str, session: str | None = None):
    global pending_tasks, active_tasks, PAUSED
    with task_queue_lock:
        # Знать worker_id недостаточно: номер публичен, сессия — нет
        _auth_worker(worker_id, session)

        if PAUSED:
            return {"status": "paused", "message": "Мастер на паузе: задачи не выдаются"}
        if BUSY.is_set():
            return {"status": "busy", "retry_after_sec": 5,
                    "message": "Мастер собирает цепочку — задачи не выдаются"}

        # Проверка зависших задач (Watchdog): пропавшие воркеры и жёсткий таймаут
        now = time.time()
        _expire_tasks(now)
        limit = _task_timeout()

        # Одна задача на воркера: пока не сдал/не истёк таймаут — новую не даём
        for sid, info in active_tasks.items():
            if info["worker_id"] == worker_id:
                age = round(time.time() - info["time_sent"], 1)
                return {
                    "status": "busy",
                    "struct_id": sid,
                    "age_sec": age,
                    "retry_after_sec": max(0, int(limit - age)),
                }

        if not pending_tasks and schedule_enabled(CONFIG):
            # Этап ещё не закончен (были отказы/таймауты) — добираем задания того же размера
            stage = stage_for_height(chain_mod.next_seq(CONFIG["output_dir"]) - 1, CONFIG)
            if stage["done"] < stage["blocks"]:
                start = int(CONFIG.get("_task_counter", 0))
                # Избыток задач — буфер против отвалов воркеров, но не весь этап:
                # на этапе n=55 это 3968 заданий, которые всё равно снимутся
                # при смене размера.
                buffer = max(TASK_REFILL_MIN, 2 * len(registered_workers))
                pending_tasks.extend(
                    schedule_tasks_for_stage(stage, CONFIG, start, count=buffer))
                CONFIG["_task_counter"] = start + buffer

        if not pending_tasks:
            return {"status": "empty", "message": "Нет доступных задач"}

        _rate_limit("task", worker_id)
        task = pending_tasks.pop(0)
        struct_id = task["struct_id"]

        # Геометрию делает МАСТЕР: воркеру уезжают готовые координаты и сорт
        # атомов. Так близость к шаблону/seed проверяема (см. verify_state_payload),
        # а не держится на доверии к воркеру.
        try:
            issued = _issue_geometry(task)
        except Exception as e:  # noqa: BLE001
            pending_tasks.insert(0, task)
            print(f"[Master] ✗ не удалось построить геометрию для {struct_id}: {e}")
            raise HTTPException(status_code=503,
                                detail=f"геометрия задания не построена: {e}")

        task_payload = {
            "struct_id": struct_id,
            "n_atoms": task["n_atoms"],
            "dim": task["dim"],
            "attempts": task.get("attempts", 0),
            # seed, которым мастер построил геометрию (для воспроизводимости)
            "seed": issued["seed"],
            "nonce": issued["nonce"],
            "prev_id": issued["prev_id"],
            "geometry": issued["coords"],
            "species": issued["species"],
            "cfg": task_cfg_payload(CONFIG),
        }

        active_tasks[struct_id] = {
            "task": task,
            "time_sent": now,
            "worker_id": worker_id,
            # эталон для сверки: если вернут другое — это не наше задание
            "geometry": issued["coords"],
            "species": issued["species"],
            "seed": issued["seed"],
            "nonce": issued["nonce"],
            "prev_id": issued["prev_id"],
        }

        print(f"Воркер [{worker_id}] получил задание {struct_id}: "
              f"n={task['n_atoms']}, dim={task['dim']}, seed={issued['seed']}, "
              f"элемент {CONFIG['element']} ({CONFIG['potential']})"
              + f" | В очереди: {len(pending_tasks)} | У воркеров: {len(active_tasks)}")
        return {"status": "ok", "task": task_payload}

@app.get("/heartbeat")
def heartbeat(worker_id: str, session: str | None = None):
    """Признак жизни воркера (раз в 30 с, в том числе во время расчёта CP2K).

    Без heartbeat мастер считает воркера пропавшим через WORKER_LEASE_SEC и
    возвращает его задачу в очередь. Сессия обязательна: чужой heartbeat
    продлевал бы удержание чужой задачи.
    """
    with task_queue_lock:
        if worker_id not in registered_workers:
            return {"status": "unknown", "message": "воркер не зарегистрирован"}
        try:
            _auth_worker(worker_id, session)
        except HTTPException as e:  # noqa: PERF203
            return {"status": "unauthorized", "message": str(e.detail)}
        return {
            "status": "ok",
            "lease_sec": WORKER_LEASE_SEC,
            "active": [sid for sid, info in active_tasks.items()
                       if info["worker_id"] == worker_id],
        }


# Rate limit: одну задачу воркеру и один результат от воркера — не чаще
# раза в 5 секунд. Защищает от флуда без потери смысла (задача считается
# минутами, поэтому честный воркер этот лимит не замечает).
RATE_LIMIT_SEC = 5
_RATE_LOCK = threading.Lock()
_RATE: dict[tuple[str, str], float] = {}


def _rate_limit(kind: str, key: str, min_interval: float | None = None):
    min_interval = RATE_LIMIT_SEC if min_interval is None else float(min_interval)
    if min_interval <= 0:
        return
    now = time.time()
    with _RATE_LOCK:
        prev = _RATE.get((kind, key), 0.0)
        wait = min_interval - (now - prev)
        if wait > 0:
            raise HTTPException(
                status_code=429,
                detail=f"{kind}: не чаще раза в {min_interval:g} с",
                headers={"Retry-After": str(int(wait) + 1)})
        _RATE[(kind, key)] = now


def reset_rate_limits():
    """Сбросить лимиты (тесты и ручная диагностика)."""
    with _RATE_LOCK:
        _RATE.clear()


STRUCT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")

# ---------- proof-of-key при регистрации воркера ----------
# Без него чужой адрес можно «занять»: мастер вернёт тот же worker_id и снимет
# задачу жертвы (DoS), а также позволит валить её задачи. Nonce одноразовый.
REG_NONCE_TTL = 120
REG_NONCE_MAX = 5000
_REG_NONCES: dict[str, float] = {}
_REG_LOCK = threading.Lock()
REG_SIGN_PREFIX = b"cp2k-mace-register:"


def _issue_registration_nonce() -> str:
    nonce = os.urandom(32).hex()
    now = time.time()
    with _REG_LOCK:
        if len(_REG_NONCES) >= REG_NONCE_MAX:
            for key, ts in list(_REG_NONCES.items()):
                if now - ts > REG_NONCE_TTL:
                    _REG_NONCES.pop(key, None)
        _REG_NONCES[nonce] = now
    return nonce


def _verify_registration(nonce, public_key, pubkey_hash, signature) -> str | None:
    """None — доказано; иначе причина отказа."""
    if not nonce or not public_key or not signature:
        return "нет nonce/подписи"
    now = time.time()
    with _REG_LOCK:
        issued = _REG_NONCES.pop(str(nonce), None)   # одноразовый
    if issued is None:
        return "nonce неизвестен или уже использован"
    if now - issued > REG_NONCE_TTL:
        return "nonce истёк"
    try:
        pub = bytes.fromhex(public_key)
        sig = bytes.fromhex(signature)
        nonce_bytes = bytes.fromhex(str(nonce))
    except ValueError:
        return "ключ/подпись/nonce не hex"
    if len(pub) != 32 or len(sig) != 64 or len(nonce_bytes) != 32:
        return "неверная длина ключа/подписи/nonce"
    if crypto.address_from_public(pub) != str(pubkey_hash):
        return "ключ не соответствует адресу"
    if not crypto.verify(pub, REG_SIGN_PREFIX + nonce_bytes, sig):
        return "неверная подпись nonce"
    return None


# ---------- ready-приёмка: квота, дедуп, привязка к голове ----------
_READY_LOCK = threading.Lock()
_READY_HITS: dict[str, list[float]] = {}
_READY_SEEN: dict[str, float] = {}
_READY_SEEN_MAX = 20000


def _reset_ready_quota():
    """Сбросить квоты и дедуп ready-приёмок (тесты, диагностика)."""
    with _READY_LOCK:
        _READY_HITS.clear()
        _READY_SEEN.clear()


def _ready_quota_ok(worker_id: str) -> bool:
    """Скользящее окно: не больше READY_QUOTA_PER_HOUR приёмок на воркера."""
    now = time.time()
    with _READY_LOCK:
        hits = [t for t in _READY_HITS.get(str(worker_id), []) if now - t < 3600]
        if len(hits) >= READY_QUOTA_PER_HOUR:
            _READY_HITS[str(worker_id)] = hits
            return False
        hits.append(now)
        _READY_HITS[str(worker_id)] = hits
    return True


def _ready_key(prev_id: str, nonce: int) -> str:
    return f"{prev_id}:{int(nonce)}"


def _ready_seen(key: str) -> bool:
    now = time.time()
    with _READY_LOCK:
        if len(_READY_SEEN) > _READY_SEEN_MAX:
            for k, t in list(_READY_SEEN.items()):
                if now - t > 7200:
                    _READY_SEEN.pop(k, None)
        return key in _READY_SEEN


def _ready_mark(key: str) -> None:
    with _READY_LOCK:
        _READY_SEEN[key] = time.time()


def _ready_unmark(key: str) -> None:
    """Неудачная приёмка не должна выглядеть принятой при повторе."""
    with _READY_LOCK:
        _READY_SEEN.pop(key, None)


def _ready_binding(payload: dict, sid: str):
    """
    Привязка ready-результата к голове цепочки.

    Работа в этой сети — пара (prev_id, nonce): она однозначно задаёт
    геометрию через seed(prev_id, nonce). Поэтому готовый кадр проверяем
    теми же правилами, что и свой блок: prev_id == голова мастера,
    n_atoms по расписанию, geometry == Generate(seed(prev_id, nonce)).

    Возвращает (issued, task_data, причина). issued — эталон для
    verify_state_payload: дальше проверка идёт тем же кодом, что у обычного пути.
    """
    prev_id = str(payload.get("prev_id") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", prev_id):
        return None, None, "нет prev_id (64 hex) в ready-результате"
    try:
        nonce = int(payload.get("nonce"))
    except (TypeError, ValueError):
        return None, None, "нет nonce в ready-результате"
    if not 0 <= nonce < (1 << 64):
        return None, None, "nonce вне диапазона uint64"
    try:
        n_atoms = int(payload.get("n_atoms") or 0)
        dim = int(payload.get("dim") or 3)
    except (TypeError, ValueError):
        return None, None, "n_atoms/dim не числа"
    if not 1 <= n_atoms <= MAX_TASK_ATOMS:
        return None, None, f"n_atoms {n_atoms} вне диапазона 1..{MAX_TASK_ATOMS}"
    if dim not in (1, 2, 3):
        return None, None, f"dim {dim} — не 1/2/3"
    geometry = payload.get("geometry")
    if not geometry:
        return None, None, "в ready-результате нет geometry"
    if len(geometry) != n_atoms:
        return None, None, "geometry не соответствует n_atoms"
    stale = _stale_reason(n_atoms)
    if stale:
        return None, None, stale

    tip = chain_mod.previous_hash(CONFIG["output_dir"],
                                  chain_mod.next_seq(CONFIG["output_dir"]))
    if prev_id != tip:
        return None, None, ("prev_id не совпадает с головой мастера: работа для "
                            "старой головы — ищите мастера с этой головой")

    try:
        issued = _issue_geometry({"n_atoms": n_atoms, "dim": dim, "nonce": nonce})
    except Exception as e:  # noqa: BLE001
        return None, None, f"геометрия для (prev_id, nonce) не строится: {e}"

    try:
        coords = np.asarray(geometry, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None, "geometry не приводится к числам"
    same, why = _same_geometry(coords, payload.get("species"), issued)
    if not same:
        return None, None, f"готовый кадр не из seed(prev_id, nonce): {why}"

    task_data = {"struct_id": sid, "task_index": -1, "n_atoms": n_atoms,
                 "dim": dim, "attempts": 0, "_nonce": nonce}
    return issued, task_data, None


class TaskResult(BaseModel):
    worker_id: str
    session: str | None = None   # выдаётся при регистрации с подписью ключа
    struct_id: str
    success: bool
    data: dict = None
    ready: bool = False     # результат готов, задача выдана другим мастером


@app.post("/submit-result")
def submit_result(result: TaskResult):
    """
    Мастер не доверяет результату: снимает задачу с учёта, затем верифицирует
    payload и только после этого пишет файлы.

    Если мастер занят сборкой цепочки — работа НЕ ставится в очередь и НЕ
    принимается: воркер получает busy и уносит результат другому мастеру.

    ready=True — воркер уже посчитал работу по заданию другого мастера и
    принёс готовый результат сюда (после /work/height).
    """
    sid = result.struct_id
    w_id = result.worker_id
    if not isinstance(sid, str) or not STRUCT_ID_RE.match(sid):
        return {"status": "rejected", "reason": "недопустимый struct_id"}
    _auth_worker(w_id, result.session)
    _rate_limit("submit", str(w_id))

    # ---------- готовый результат от воркера (задание было у другого мастера) ----------
    if result.ready:
        if w_id not in registered_workers:
            raise HTTPException(status_code=400,
                                detail="неизвестный воркер — сначала /register-worker")
        if BUSY.is_set():
            return {"status": "busy", "reason": "master is syncing"}

        payload = dict(result.data or {})
        payload.setdefault("worker_id", w_id)
        if sid in completed_tasks:
            return {"status": "accepted", "kept": "already", "energy_ev": None}

        # Кадр обязан быть работой для ГОЛОВЫ этого мастера: пара (prev_id, nonce)
        # задаёт геометрию, и без этой привязки ready — любой кадр за счёт мастера,
        # который ещё и нельзя положить в блок.
        issued, task_data, why = _ready_binding(payload, sid)
        if why:
            print(f"[Master] ✗ ready-результат {sid} от {w_id}: {why}")
            return {"status": "rejected", "reason": why}

        key = _ready_key(issued["prev_id"], issued["nonce"])
        if _ready_seen(key):
            return {"status": "accepted", "kept": "already", "energy_ev": None}
        if not _ready_quota_ok(w_id):
            return {"status": "rejected",
                    "reason": f"квота ready-приёмок: {READY_QUOTA_PER_HOUR} в час"}
        _ready_mark(key)   # до прогона CP2K: неудачная попытка тоже тратит квоту

        payload["seed"] = issued["seed"]
        payload.setdefault("species", issued["species"])
        block_info: dict = {}
        verifying_tasks.add(sid)
        try:
            verdict = verify_state_payload(CONFIG, task_data, payload, issued=issued)
            verifying_tasks.discard(sid)
            if not verdict["valid"]:
                _ready_unmark(key)
                print(f"[Master] ✗ ready-результат {sid} от {w_id} отклонён: "
                      f"{verdict['reason']}")
                return {"status": "rejected", "reason": verdict["reason"]}

            # Блок пишем под тем же lock, что и обычный путь: голова не должна
            # уйти, пока шла проверка (прогон CP2K — минуты), и два потока не
            # должны собрать два блока на одной высоте.
            with task_queue_lock:
                tip = chain_mod.previous_hash(CONFIG["output_dir"],
                                              chain_mod.next_seq(CONFIG["output_dir"]))
                if tip != issued["prev_id"]:
                    _ready_unmark(key)
                    print(f"[Master] ✗ ready-результат {sid}: голова ушла, "
                          f"блок не собран")
                    return {"status": "stale",
                            "reason": "голова мастера изменилась во время проверки"}
                kept = _persist_verified_state(task_data, payload, verdict,
                                               block_info=block_info)
                active_tasks.pop(sid, None)
                completed_tasks.add(sid)

            print(f"[Master] ✓ ready-результат {sid} от {w_id} принят ({kept}), "
                  f"блок {block_info.get('block')}")
            resp = {"status": "accepted", "kept": kept,
                    "energy_ev": verdict.get("energy_ev"),
                    "scf": (verdict.get("state") or {}).get("scf"),
                    "block": (os.path.basename(block_info["block"])
                              if block_info.get("block") else None),
                    "reward_paid": bool(block_info.get("reward_paid"))}
            if not resp["block"]:
                resp["reason"] = block_info.get("reason") or "блок не собран"
            return resp
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            verifying_tasks.discard(sid)
            if not block_info.get("block"):
                _ready_unmark(key)     # блока нет — пусть воркер повторит
            return {"status": "error", "reason": str(e)}

    # ---------- обычный путь: задача выдана этим мастером ----------
    with task_queue_lock:
        if sid not in active_tasks:
            print(f"[Master] Получен результат для неизвестной или просроченной задачи {sid} от воркера {w_id}")
            return {"status": "not_found"}
        info = active_tasks[sid]
        task_data = info["task"]

        # Мастер занят сборкой цепочки: работу не принимаем, но задачу НЕ снимаем —
        # она останется за воркером и вернётся в очередь по таймауту _task_timeout().
        if BUSY.is_set():
            print(f"[Master] занят синхронизацией: {sid} остаётся за воркером {w_id}, "
                  f"ответ busy; вернётся по таймауту {_task_timeout()} с")
            return {"status": "busy", "reason": "master is syncing"}

        active_tasks.pop(sid)
        verifying_tasks.add(sid)
        if info["worker_id"] != w_id:
            # Задачу выдавали другому воркеру — результат не принимаем
            verifying_tasks.discard(sid)
            _requeue_or_fail(task_data, f"результат от чужого воркера {w_id}")
            print(f"[Master] ✗ {w_id} прислал результат задачи {sid}, "
                  f"выданной воркеру {info['worker_id']} — отклонено")
            return {"status": "rejected", "reason": "worker mismatch"}

    try:
        # --- Голова цепочки изменилась: результат всё равно не примут ---
        # Проверка дешёвая и делается ДО прогона CP2K.
        if info.get("prev_id"):
            tip = chain_mod.previous_hash(CONFIG["output_dir"],
                                          chain_mod.next_seq(CONFIG["output_dir"]))
            if info["prev_id"] != tip:
                with task_queue_lock:
                    verifying_tasks.discard(sid)
                    _requeue_or_fail(task_data, "голова цепочки изменилась")
                return {"status": "rejected", "reason": "stale head"}

        # --- Неуспех расчёта: возвращаем задачу в очередь (или бракуем) ---
        if not result.success:
            reason = (result.data or {}).get("message") or (result.data or {}).get("error_reason") \
                or "воркер сообщил об ошибке расчета"
            with task_queue_lock:
                verifying_tasks.discard(sid)
                _requeue_or_fail(task_data, f"worker failure: {reason}")
            return {"status": "rejected", "reason": reason}

        # --- Приёмка: только по файлу состояния (-RESTART.wfn) ---
        payload = dict(result.data or {})
        payload.setdefault("worker_id", w_id)

        # Эталон — та геометрия, которую мастер выдал (info), а не присланная.
        # Сверка дешёвая; только после неё запускаются последние шаги DFT.
        task_data["_nonce"] = info.get("nonce", 0)
        verdict = verify_state_payload(CONFIG, task_data, payload, issued=info)
        persist = _persist_verified_state

        with task_queue_lock:
            verifying_tasks.discard(sid)
            if verdict["valid"]:
                # Голова могла измениться, пока шла проверка (минуты-часы):
                # блок на старом prev_id уже не собрать — возвращаем задачу.
                tip = chain_mod.previous_hash(CONFIG["output_dir"],
                                              chain_mod.next_seq(CONFIG["output_dir"]))
                if info.get("prev_id") and info["prev_id"] != tip:
                    _requeue_or_fail(task_data, "голова изменилась во время проверки")
                    return {"status": "rejected", "reason": "stale head"}
                kept = persist(task_data, payload, verdict)
                completed_tasks.add(sid)
                print(f"[Master] ✓ Воркер [{w_id}] сдал результат {sid}: проверка пройдена "
                      f"({kept})")
                return {
                    "status": "accepted",
                    "kept": kept,
                    "energy_ev": verdict.get("energy_ev"),
                    "scf": (verdict.get("state") or {}).get("scf"),
                }

            # Проверка не пройдена — результат НЕ пишем
            print(f"[Master] ✗ Воркер [{w_id}] прислал результат для {sid}, не прошедший проверку. "
                  f"В файлы НЕ записано.")
            _requeue_or_fail(task_data, verdict["reason"])
            return {"status": "rejected", "reason": verdict["reason"]}

    except Exception as e:
        traceback.print_exc()
        with task_queue_lock:
            verifying_tasks.discard(sid)
            _requeue_or_fail(task_data, f"master verification error: {e}")
        return {"status": "error", "reason": str(e)}


# =====================================================================
# СЕТЬ МАСТЕРОВ: синхронизация цепочки
# =====================================================================
PEER_TIMEOUT = 10
PEER_TOL_ENERGY_EV = 0.02
MAX_PEERS = 64                 # сколько адресов мастер готов держать в списке пиров
PEER_GOSSIP_INTERVAL = 120     # как часто обмениваться адресами с пирами
# Наблюдаемые в dft.json должны воспроизводиться пиром: пересчёт идёт от ТОЙ ЖЕ
# плотности тем же CP2K, поэтому допуски на порядок жёстче, чем для энергии.
PEER_TOL_GAP_EV = 0.01
PEER_TOL_MAG_UB = 0.05
PEER_TOL_DIPOLE_DEBYE = 0.01
MAX_CATCHUP_ROUNDS = 10
BAD_PEER_COOLDOWN = 300
_BAD_PEERS: dict[str, float] = {}


def _local_head() -> tuple[int, str]:
    blocks = chain_mod.list_blocks(CONFIG["output_dir"])
    if not blocks:
        return 0, chain_mod.ZERO_HASH
    return blocks[-1]["seq"], blocks[-1]["block_id"]


def _normalize_peer(url: str) -> str:
    return str(url).strip().rstrip("/")


def _collect_peers(args) -> list[str]:
    raw = list(CONFIG.get("peers") or [])
    raw += list(getattr(args, "peer", None) or [])
    raw += [p.strip() for p in os.environ.get("MASTER_PEERS", "").split(",") if p.strip()]

    port = getattr(args, "port", 8000)
    self_urls = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    out = []
    for p in raw:
        p = _normalize_peer(p)
        if p and p not in out and p not in self_urls:
            out.append(p)
    return out


def _fetch_blob(peer: str, sha: str) -> bytes | None:
    """Скачивает файл состояния с лимитом размера.

    sha обязан быть 64 hex: из него строится имя blobs/<sha>.wfn, иначе
    "../../x" писал бы файл вне каталога элемента.
    """
    import re as _re
    if not _re.fullmatch(r"[0-9a-f]{64}", str(sha or "")):
        return None
    limit = max(1, int(CONFIG.get("restart_max_mb", 50))) * 1024 * 1024
    try:
        with requests.get(f"{peer}/chain/blob/{sha}", timeout=60, stream=True) as r:
            if r.status_code != 200:
                return None
            declared = r.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > limit:
                return None
            chunks, total = [], 0
            for chunk in r.iter_content(1 << 20):
                total += len(chunk)
                if total > limit:
                    return None
                chunks.append(chunk)
            return b"".join(chunks)
    except Exception:  # noqa: BLE001
        return None


BLOCK_MAX_MB = 64          # легитимный блок — десятки КБ; больше — мусор


def _fetch_block(peer: str, seq: int) -> bytes | None:
    """Скачивает блок с ограничением размера (как _fetch_blob для wfn)."""
    limit = BLOCK_MAX_MB * 1024 * 1024
    try:
        with requests.get(f"{peer}/chain/block/{seq}", timeout=120,
                          stream=True) as r:
            if r.status_code != 200:
                return None
            declared = r.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > limit:
                return None
            chunks, total = [], 0
            for chunk in r.iter_content(1 << 20):
                total += len(chunk)
                if total > limit:
                    return None
                chunks.append(chunk)
            return b"".join(chunks)
    except Exception:  # noqa: BLE001
        return None


def _expected_n_atoms(seq: int) -> int:
    """Размер кластера для высоты seq по расписанию (консенсус)."""
    return int(stage_for_height(max(0, int(seq) - 1), CONFIG)["n_atoms"])


def _accept_peer_block(peer: str, seq: int, head: str,
                       block_bytes: bytes) -> tuple[bool, list[str]]:
    """Проверяет чужой блок локально: структура/подписи, затем DFT-пересчёт.

    Blob пишется через .part и удаляется, если блок не принят: иначе пир
    забивает диск мусорными файлами состояния (до restart_max_mb каждый).
    """
    element_dir = CONFIG["output_dir"]
    blocks_dir = chain_mod.blocks_dir(element_dir)
    os.makedirs(blocks_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="peer_block_", dir=element_dir) as td:
        path = os.path.join(td, f"peer_{seq}.zip")
        with open(path, "wb") as f:
            f.write(block_bytes)

        try:
            header, dft, _txs, _names = chain_mod.read_block(path)
        except Exception as e:  # noqa: BLE001
            return False, [f"блок не читается: {e}"]

        sha = header.get("state_sha256")
        if not sha or not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
            return False, ["нет/некорректный state_sha256"]

        # Имя блока строим только из КОНСЕНСУСНОГО n_atoms для этой высоты:
        # чужой n_atoms давал второе имя с тем же seq и ломал цепочку.
        expected_n = _expected_n_atoms(seq)
        if int(header.get("n_atoms", -1)) != expected_n:
            return False, [f"n_atoms {header.get('n_atoms')} != {expected_n} "
                           f"(этап для высоты {seq})"]
        local_comp = CONFIG.get("composition") or {str(CONFIG.get("element")): 1}
        if header.get("composition") != local_comp:
            return False, [f"состав блока {header.get('composition')} != {local_comp}"]
        if any(b["seq"] == seq for b in chain_mod.list_blocks(element_dir)):
            return False, [f"блок с seq {seq} уже есть"]

        blob = _fetch_blob(peer, sha)
        if blob is None:
            return False, [f"пир не отдал blob {sha[:12]}…"]

        blob_path = os.path.join(chain_mod.blobs_dir(element_dir), f"{sha}.wfn")
        os.makedirs(os.path.dirname(blob_path), exist_ok=True)
        created_blob = False
        accepted = False
        try:
            if not os.path.exists(blob_path):
                # .part + os.replace: никто не увидит недописанный файл
                part = blob_path + ".part"
                with open(part, "wb") as f:
                    f.write(blob)
                os.replace(part, blob_path)
                created_blob = True

            v = chain_mod.verify_block(element_dir, path,
                                       expected_seq=seq, expected_prev=head)
            if not v["ok"]:
                return False, v["problems"][:4]

            coords = dft.get("geometry")
            res = cp2k_runner.verify_state_from_wfn(
                coords, blob_path, CONFIG, project=f"peer_{seq}")
            if not res.get("ok"):
                return False, [f"DFT-проверка: {res.get('reason')}"]

            e_block = float(dft.get("energy_ev"))
            if abs(res["energy_ev"] - e_block) > PEER_TOL_ENERGY_EV:
                return False, [f"энергия блока {e_block:.6f} != "
                               f"пересчёт {res['energy_ev']:.6f} eV"]

            # Наблюдаемые: блок обязан совпасть с пересчётом. Иначе блок
            # «подкрашен»: wfn настоящий, а числа в dft.json — нет.
            props = res.get("props") or {}
            claimed_checks = (
                ("homo_lumo_gap_ev", res.get("gap"), PEER_TOL_GAP_EV, "eV"),
                ("magnetic_moment", res.get("mag"), PEER_TOL_MAG_UB, "μB"),
                ("dipole_total_debye", props.get("dipole_total_debye"),
                 PEER_TOL_DIPOLE_DEBYE, "Д"),
            )
            for key, mine, tol, unit in claimed_checks:
                claimed = dft.get(key)
                if claimed is None or mine is None:
                    continue  # старые блоки без этих полей — не повод отбраковывать
                if abs(float(claimed) - float(mine)) > tol:
                    return False, [f"{key}: в блоке {float(claimed):.6f}, "
                                   f"пересчёт {float(mine):.6f} {unit}"]

            dst = os.path.join(blocks_dir,
                               chain_mod.block_name(seq, CONFIG["element"], expected_n))
            shutil.move(path, dst)
            accepted = True
            print(f"[Peer] ✓ блок {seq} от {peer}: DFT-пересчёт сошёлся "
                  f"(E={res['energy_ev']:.6f} eV), блок записан")
            return True, []
        finally:
            # Мусор от непринятого блока не должен оставаться на диске.
            if created_blob and not accepted:
                try:
                    os.remove(blob_path)
                except OSError:
                    pass


def _find_fork_ancestor(peer: str, local_height: int) -> int | None:
    """
    Ищет высоту общего предка с пиром, спускаясь от local_height вниз:
    скачивает блок пира с этим seq и сравнивает sha256 с нашим.
    """
    mine = {b["seq"]: b["block_id"] for b in chain_mod.list_blocks(CONFIG["output_dir"])}
    for seq in range(local_height, 0, -1):
        data = _fetch_block(peer, seq)
        if data is None:
            continue
        try:
            header, _dft, _txs, _names = chain_mod.read_block_bytes(data)
        except Exception:  # noqa: BLE001
            continue
        if header.get("block_id") and header["block_id"] == mine.get(seq):
            return seq
    return None


def sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def reorg_from_peer(peer: str, ancestor: int, remote: int) -> dict:
    """
    Переключает локальную цепочку на ветку пира: откат до ancestor,
    возврат transfer-транзакций короткой ветки в мемпул, затем догон
    блоков ancestor+1..remote с DFT-проверкой. При ошибке — откат.
    """
    element_dir = CONFIG["output_dir"]
    print(f"[Reorg] ветка {peer} длиннее: общий предок {ancestor}, "
          f"догоняю до {remote}")

    removed, txs = chain_mod.truncate_chain(element_dir, ancestor)
    if removed:
        print(f"[Reorg] откатил {len(removed)} блок(ов) короткой ветки")
    if txs:
        with MEMPOOL_LOCK:
            for tx in txs:
                tid = tx.get("txid") or chain_mod.txid_of(tx)
                tx["txid"] = tid
                MEMPOOL[tid] = tx
        print(f"[Reorg] в мемпул возвращено транзакций: {len(txs)}")

    accepted = 0
    try:
        for seq in range(ancestor + 1, remote + 1):
            local, head = _local_head()
            if local >= seq:
                continue
            data = _fetch_block(peer, seq)
            if data is None:
                raise RuntimeError(f"блок {seq}: не скачан или слишком большой")
            ok, why = _accept_peer_block(peer, seq, head, data)
            if not ok:
                raise RuntimeError(f"блок {seq}: {'; '.join(why)}")
            accepted += 1
        print(f"[Reorg] ✓ переключился на ветку {peer} (высота {remote})")
        _prune_mempool("reorg")
        _sync_stage("reorg")
        return {"accepted": accepted, "errors": []}
    except Exception as e:  # noqa: BLE001
        # Откат: удалить новые блоки и вернуть старые из orphan
        for b in chain_mod.list_blocks(element_dir):
            if b["seq"] > ancestor:
                try:
                    os.remove(b["path"])
                except OSError:
                    pass
        chain_mod.restore_blocks(removed)
        print(f"[Reorg] ✗ ошибка ({e}); восстановил прежнюю ветку")
        return {"accepted": 0, "errors": [str(e)]}


def _is_fork_conflict(text: str) -> bool:
    """Ошибка = расхождение ветвей (v5: prev_id; старые: prev_sha256)."""
    return "prev_id не совпадает" in text or "prev_sha256" in text


def sync_from_peer(peer: str) -> dict:
    peer = _normalize_peer(peer)
    try:
        r = requests.get(f"{peer}/chain/height", timeout=PEER_TIMEOUT)
        r.raise_for_status()
        info = r.json()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "accepted": 0, "errors": [f"{peer}: {e}"]}

    if info.get("element") != CONFIG["element"]:
        return {"ok": False, "accepted": 0,
                "errors": [f"{peer}: элемент {info.get('element')} != {CONFIG['element']}"]}

    remote = int(info.get("height", 0))
    accepted, errors = 0, []
    while True:
        local, head = _local_head()
        if local >= remote:
            break
        seq = local + 1
        data = _fetch_block(peer, seq)
        if data is None:
            errors.append(f"блок {seq}: не скачан (HTTP или размер)")
            break
        try:
            ok, why = _accept_peer_block(peer, seq, head, data)
        except Exception as e:  # noqa: BLE001
            # битый блок не должен ронять приём работы: исключение = вина пира
            errors.append(f"блок {seq}: {type(e).__name__}: {e}")
            _BAD_PEERS[peer] = time.time() + BAD_PEER_COOLDOWN
            break
        if not ok:
            text = "; ".join(why)
            if _is_fork_conflict(text) and remote > local:
                ancestor = _find_fork_ancestor(peer, local)
                if ancestor is not None:
                    print(f"[Fork] {peer} предлагает другую ветку, общий предок {ancestor}")
                    res = reorg_from_peer(peer, ancestor, remote)
                    accepted += res.get("accepted", 0)
                    errors.extend(res.get("errors", []))
                    break
            errors.append(f"блок {seq}: {text}")
            break
        accepted += 1

    if accepted:
        _prune_mempool("блок из сети")
        _sync_stage("блок из сети")
    return {"ok": not errors, "accepted": accepted, "errors": errors}


def _gossip_peers() -> list[str]:
    """
    Обмен адресами с пирами (как в p2p): спрашиваем у каждого известного мастера
    его список пиров и добавляем неизвестные адреса к себе.
    """
    known = [p.rstrip("/") for p in (CONFIG.get("peers") or [])]
    added = []
    for peer in list(known):
        if len(known) >= MAX_PEERS:
            break
        try:
            r = requests.get(f"{peer}/admin/peers", timeout=PEER_TIMEOUT)
            if r.status_code != 200:
                continue
            remote = r.json().get("peers") or []
        except Exception:  # noqa: BLE001
            continue
        for url in remote:
            normalized = _normalize_peer(url)
            if not normalized or normalized in known or len(known) >= MAX_PEERS:
                continue
            known.append(normalized)
            added.append(normalized)
    if added:
        CONFIG["peers"] = known
        _persist_peers(known)
        print(f"[Peer] узнал новые адреса мастеров: {', '.join(added)}")
    return added


def _gossip_loop(interval: int = PEER_GOSSIP_INTERVAL):
    """Фон: обмен адресами + догон сети, даже если анонс потерялся."""
    while True:
        time.sleep(interval)
        try:
            _gossip_peers()
            if CONFIG.get("peers"):
                for res in sync_all_peers():
                    if res.get("accepted"):
                        print(f"[Peer] {res['peer']}: догнал {res['accepted']} блок(ов)")
        except Exception as e:  # noqa: BLE001
            print(f"[Peer] gossip ошибка: {e}")


def sync_all_peers() -> list[dict]:
    out = []
    for peer in list(CONFIG.get("peers") or []):
        res = sync_from_peer(peer)
        out.append({"peer": peer, **res})
        if res["accepted"]:
            print(f"[Peer] {peer}: принято блоков {res['accepted']}")
        for e in res.get("errors", []):
            print(f"[Peer] {peer}: {e}")
    return out


def _peer_heights() -> list[tuple[str, int]]:
    """Высоты здоровых пиров (недоступные/плохие временно исключены)."""
    now = time.time()
    out = []
    for peer in list(CONFIG.get("peers") or []):
        if _BAD_PEERS.get(peer, 0) > now:
            continue
        try:
            r = requests.get(f"{peer}/chain/height", timeout=PEER_TIMEOUT)
            r.raise_for_status()
            info = r.json()
        except Exception:  # noqa: BLE001
            continue
        if info.get("element") != CONFIG["element"]:
            continue
        out.append((peer, int(info.get("height", 0))))
    return out


def catch_up_chain(max_rounds: int = MAX_CATCHUP_ROUNDS) -> dict:
    """
    Догоняет самую длинную цепочку у пиров и повторяет опрос, пока кто-то
    опережает. Пока идёт догон, мастер «занят» (BUSY): новые работы не
    принимаются, воркеры уносят результаты другим мастерам.
    """
    accepted = 0
    rounds = 0
    BUSY.set()
    try:
        while rounds < max_rounds:
            local, _ = _local_head()
            heights = _peer_heights()
            if not heights:
                return {"ok": True, "height": local, "rounds": rounds,
                        "accepted": accepted}
            peer, remote = max(heights, key=lambda x: x[1])
            if remote <= local:
                return {"ok": True, "height": local, "rounds": rounds,
                        "accepted": accepted}

            print(f"[CatchUp] у {peer} высота {remote} > локальной {local} — догоняю")
            res = sync_from_peer(peer)
            rounds += 1
            accepted += res.get("accepted", 0)
            if not res.get("ok") or res.get("accepted", 0) == 0:
                _BAD_PEERS[peer] = time.time() + BAD_PEER_COOLDOWN
                print(f"[CatchUp] {peer} не догнан ({res.get('errors')}), "
                      f"исключаю на {BAD_PEER_COOLDOWN} с")

        local, _ = _local_head()
        return {"ok": False, "height": local, "rounds": rounds,
                "accepted": accepted}
    finally:
        BUSY.clear()


def _announce_to_peers(height: int):
    for peer in list(CONFIG.get("peers") or []):
        try:
            requests.post(f"{_normalize_peer(peer)}/chain/announce",
                          json={"height": height, "element": CONFIG["element"]},
                          timeout=5)
        except Exception:  # noqa: BLE001
            pass


class TxSubmit(BaseModel):
    tx: dict


@app.post("/tx/submit")
def tx_submit(req: TxSubmit):
    if BUSY.is_set():
        return {"status": "busy", "reason": "master is syncing"}
    tx = dict(req.tx or {})
    conf = int(CONFIG.get("coinbase_confirmations", chain_mod.DEFAULT_CONFIRMATIONS))
    utxo = chain_mod.utxo_set(CONFIG["output_dir"])
    ok, problems = chain_mod.validate_tx(tx, CONFIG["output_dir"],
                                         confirmations=conf, utxo=utxo)
    if not ok:
        return {"status": "rejected", "reason": "; ".join(problems[:5])}

    canonical = chain_mod.txid_of(tx)
    if tx.get("txid") and tx["txid"] != canonical:
        return {"status": "rejected", "reason": "txid не совпадает с телом"}
    tx["txid"] = canonical
    tid = canonical
    fee = chain_mod.tx_fee_safe(tx, utxo)
    fee = 0 if fee is None else fee
    keys = _tx_input_keys(tx)

    evicted = None
    with MEMPOOL_LOCK:
        # 1) Конфликт (тот же вход): заменить можно только более дорогой tx.
        clash = [t for t, other in MEMPOOL.items()
                 if other is not tx and (_tx_input_keys(other) & keys)]
        if clash:
            old_fee = 0
            for t in clash:
                f = chain_mod.tx_fee_safe(MEMPOOL[t], utxo)
                old_fee += max(0, f or 0)
            if fee <= old_fee:
                return {"status": "rejected",
                        "reason": f"замена только дороже: конфликтующая "
                                  f"комиссия {old_fee}, ваша {fee}"}

        # 2) Лимит размера: вытесняем самую дешёвую, если новая дороже.
        #    Конфликты уйдут сами — их не считаем и не выбираем жертвой.
        remaining = len(MEMPOOL) - len(clash)
        if remaining >= MEMPOOL_MAX_TX and tid not in MEMPOOL:
            cheapest_tid, cheapest_fee = None, None
            for t, other in MEMPOOL.items():
                if t in clash:
                    continue
                f = chain_mod.tx_fee_safe(other, utxo)
                f = -1 if f is None else f
                if cheapest_fee is None or f < cheapest_fee:
                    cheapest_tid, cheapest_fee = t, f
            if cheapest_tid is None or fee <= cheapest_fee:
                return {"status": "rejected",
                        "reason": f"мемпул полон: минимальная комиссия "
                                  f"{cheapest_fee}, ваша {fee}"}
            MEMPOOL.pop(cheapest_tid, None)
            evicted = cheapest_tid

        for t in clash:
            MEMPOOL.pop(t, None)
        MEMPOOL[tid] = tx

    _save_mempool()
    if clash:
        print(f"[Master] мемпул: {len(clash)} конфликтующих tx заменено "
              f"более дорогой {tid[:12]}…")
    if evicted:
        print(f"[Master] мемпул: вытеснена дешёвая tx {evicted[:12]}… "
              f"(комиссия новой {fee})")
    print(f"[Master] 📥 tx {tid[:12]}… принята в мемпул (комиссия {fee})")
    return {"status": "accepted", "txid": tid}


@app.get("/tx/mempool")
def tx_mempool():
    """Транзакции в мемпуле. `txs` — полные тела, чтобы кошелёк мог их показать."""
    with MEMPOOL_LOCK:
        txs = [dict(tx) for tx in MEMPOOL.values()]
    txs.sort(key=lambda t: t.get("txid") or "")
    return {"count": len(txs), "txids": [t.get("txid") for t in txs], "txs": txs}


@app.get("/work/height")
def work_height(element: str):
    """
    Публичная справка: «поддерживаешь элемент, какая высота цепочки?».

    Воркера здесь НЕ регистрируем: worker_id без подписи — это бесплатное
    право на DFT-прогоны за счёт мастера. Доставка готового результата
    другому мастеру: POST /register-worker (с подписью) → /submit-result ready.
    """
    if element != CONFIG["element"]:
        raise HTTPException(status_code=404,
                            detail=f"элемент {element} не поддерживается")
    if BUSY.is_set():
        return {"status": "busy", "reason": "master is syncing"}
    h, sha = _local_head()
    return {"status": "ok", "element": element, "height": h,
            "head_sha256": sha}


@app.get("/peers")
def peers_public():
    """
    Публичный список адресов мастеров — то, что нужно воркерам для gossip.
    Секрета нет: ровно эти URL мастер и так сообщает воркерам.
    """
    height, _sha = _local_head()
    peers = [str(p) for p in (CONFIG.get("peers") or []) if str(p).strip()]
    return {"element": CONFIG["element"], "height": height, "peers": peers}


@app.get("/chain/height")
def chain_height():
    h, sha = _local_head()
    return {"element": CONFIG["element"], "height": h, "head_sha256": sha}


@app.get("/chain/block/{seq}")
def chain_block(seq: int):
    from fastapi.responses import FileResponse
    for b in chain_mod.list_blocks(CONFIG["output_dir"]):
        if b["seq"] == seq:
            return FileResponse(b["path"], media_type="application/zip", filename=b["name"])
    raise HTTPException(status_code=404, detail="block not found")


@app.get("/chain/blob/{sha}")
def chain_blob(sha: str):
    from fastapi.responses import FileResponse
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise HTTPException(status_code=400, detail="bad sha256")
    path = os.path.join(chain_mod.blobs_dir(CONFIG["output_dir"]), f"{sha}.wfn")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="blob not found")
    return FileResponse(path, media_type="application/octet-stream")


class Announce(BaseModel):
    height: int
    element: str | None = None


@app.post("/chain/announce")
def chain_announce(a: Announce):
    if a.element and a.element != CONFIG["element"]:
        return {"ok": False, "reason": "element mismatch"}
    local, _ = _local_head()
    if a.height > local:
        threading.Thread(target=sync_all_peers, daemon=True).start()
    return {"ok": True, "local": local}


# =====================================================================
# АДМИН-API ДЛЯ master-cli.py (управление работающим мастером)
# =====================================================================
ADMIN_TOKEN = os.environ.get("MASTER_ADMIN_TOKEN", "").strip()
ADMIN_SETTINGS = {
    "verbose_cp2k": bool,
    "restart_max_mb": int,
    "atom_diameter": float,
    # coinbase_confirmations — настройка кошелька (wallet.py --set-confirmations),
    # не мастера: из админ-API убрана.
}


def _check_admin(token: str | None = None):
    """Совместимость: доступ уже ограничен middleware (только localhost)."""
    return None


def _persist_config_value(key: str, value):
    """Сохраняет значение в src/config.json, не ломая формат {value, help}."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entry = raw.get(key)
    if isinstance(entry, dict) and "value" in entry:
        entry["value"] = value
    else:
        raw[key] = {"value": value, "help": "изменено через master-cli"}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


class ConfigUpdate(BaseModel):
    key: str
    value: object


@app.get("/admin/status")
def admin_status(token: str | None = None):
    _check_admin(token)
    with task_queue_lock:
        active_by_worker = {info["worker_id"]: sid for sid, info in active_tasks.items()}
        workers = []
        for wid, rec in sorted(registered_workers.items()):
            sid = active_by_worker.get(wid)
            workers.append({
                "worker_id": wid,
                "name": rec.get("name"),
                "pubkey": (rec.get("pubkey_hash") or "")[:16],
                "active_task": sid,
                "age_sec": (round(time.time() - active_tasks[sid]["time_sent"], 1)
                            if sid else None),
                "last_seen_sec": (round(time.time() - rec["last_seen"], 1)
                                  if rec.get("last_seen") else None),
            })
        return {
            "paused": PAUSED,
            "element": CONFIG["element"],
            "potential": CONFIG["potential"],
            "pending": len(pending_tasks),
            "active": len(active_tasks),
            "verifying": len(verifying_tasks),
            "completed": len(completed_tasks),
            "failed": len(failed_tasks),
            "workers": workers,
            "config": {k: CONFIG.get(k) for k in ADMIN_SETTINGS},
        }


@app.get("/admin/config")
def admin_config_list(token: str | None = None):
    _check_admin(token)
    return {k: {"value": CONFIG.get(k), "type": t.__name__} for k, t in ADMIN_SETTINGS.items()}


@app.post("/admin/config")
def admin_config_set(update: ConfigUpdate, token: str | None = None):
    _check_admin(token)
    key = update.key
    if key not in ADMIN_SETTINGS:
        raise HTTPException(status_code=400,
                            detail=f"Нельзя менять {key!r}; доступно: {sorted(ADMIN_SETTINGS)}")
    try:
        if ADMIN_SETTINGS[key] is bool:
            if isinstance(update.value, str):
                value = update.value.strip().lower() in ("1", "true", "yes", "on")
            else:
                value = bool(update.value)
        else:
            value = ADMIN_SETTINGS[key](update.value)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Неверное значение: {e}")

    CONFIG[key] = value
    _persist_config_value(key, value)
    print(f"[Master] ⚙ настройка {key} = {value!r} (изменена через admin API)")
    return {"status": "ok", "key": key, "value": value}


@app.post("/admin/pause")
def admin_pause(token: str | None = None):
    global PAUSED
    _check_admin(token)
    PAUSED = True
    print("[Master] ⏸ выдача задач приостановлена (master-cli pause)")
    return {"status": "ok", "paused": True}


@app.post("/admin/resume")
def admin_resume(token: str | None = None):
    global PAUSED
    _check_admin(token)
    PAUSED = False
    print("[Master] ▶ выдача задач возобновлена (master-cli resume)")
    return {"status": "ok", "paused": False}


@app.get("/admin/chain")
def admin_chain(verify: bool = False, token: str | None = None):
    _check_admin(token)
    element = CONFIG["element"]
    blocks = chain_mod.list_blocks(CONFIG["output_dir"])
    result = {
        "element": element,
        "blocks": [{k: b[k] for k in ("seq", "element", "n_atoms", "name")}
                   for b in blocks],
    }
    if verify:
        result["chain"] = chain_mod.verify_chain(CONFIG["output_dir"], element)
    return result


class PeerAdd(BaseModel):
    url: str


def _persist_peers(peers: list[str]):
    with open(CONFIG["peers_file"], "w", encoding="utf-8") as f:
        json.dump(peers, f, ensure_ascii=False, indent=2)


@app.get("/admin/peers")
def admin_peers(token: str | None = None):
    _check_admin(token)
    return {"peers": list(CONFIG.get("peers") or []), "height": _local_head()[0]}


@app.post("/admin/peers")
def admin_peer_add(req: PeerAdd, token: str | None = None):
    _check_admin(token)
    url = _normalize_peer(req.url)
    if not url:
        raise HTTPException(status_code=400, detail="пустой url")
    peers = list(CONFIG.get("peers") or [])
    if url not in peers:
        peers.append(url)
        CONFIG["peers"] = peers
        _persist_peers(peers)
    print(f"[Master] + пир {url}")
    return {"status": "ok", "peers": peers}


@app.delete("/admin/peers")
def admin_peer_del(url: str, token: str | None = None):
    _check_admin(token)
    target = _normalize_peer(url)
    peers = [p for p in (CONFIG.get("peers") or [])
             if _normalize_peer(p) != target]
    CONFIG["peers"] = peers
    _persist_peers(peers)
    print(f"[Master] − пир {target}")
    return {"status": "ok", "peers": peers}


@app.post("/admin/sync")
def admin_sync(token: str | None = None):
    _check_admin(token)
    return {"catch_up": catch_up_chain()}


@app.get("/admin/keys")
def admin_keys(token: str | None = None):
    _check_admin(token)
    wallet = CONFIG["wallet"]
    keys = wallet.keys()
    current = wallet.latest_key()
    return {"current": current["id"] if current else None,
            "count": len(keys), "keys": keys}


@app.post("/admin/keys/new")
def admin_keys_new(token: str | None = None):
    _check_admin(token)
    key = CONFIG["wallet"].add_key()
    print(f"[Master] 🔑 новый ключ #{key['id']}: {key['address']}")
    return {"status": "ok", "key": key}


class PasswordChange(BaseModel):
    old: str
    new: str


@app.post("/admin/passwd")
def admin_passwd(req: PasswordChange, token: str | None = None):
    _check_admin(token)
    try:
        CONFIG["wallet"].change_password(req.old, req.new)
    except wallet_store.WalletError as e:
        raise HTTPException(status_code=400, detail=str(e))
    print("[Master] 🔐 пароль мастера изменён")
    return {"status": "ok"}


@app.get("/admin/balances")
def admin_balances(token: str | None = None):
    _check_admin(token)
    master_addresses = CONFIG["wallet"].addresses()
    conf = int(CONFIG.get("coinbase_confirmations", chain_mod.DEFAULT_CONFIRMATIONS))
    all_bal = chain_mod.balances(CONFIG["output_dir"], None, conf)
    return {
        "confirmations": conf,
        "master": {a: b for a, b in all_bal.items() if a in master_addresses},
        "workers": {a: b for a, b in all_bal.items() if a not in master_addresses},
    }


def run_fastapi_server(port: int):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

# ---------- main ----------
def main():
    install_tee_logger()
    install_signal_handlers()
    write_pid()
    write_status("running")

    exit_code = 0
    final_state = "finished"
    try:
        _main_impl()
    except _Terminated as e:
        print(f"\n[main] Остановлено: {e}")
        exit_code = 130
        final_state = "stopped"
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        exit_code = code
        final_state = "failed" if code != 0 else "finished"
        raise
    except Exception as e:
        print(f"\n[main] Ошибка: {e}")
        traceback.print_exc()
        exit_code = 1
        final_state = "failed"
    finally:
        try:
            print("\n[main] Формирую архив артефактов...")
            archive_maker.make_artifacts_archive(CONFIG)
        except Exception as e:
            print(f"[main] Ошибка при упаковке: {e}")
        try:
            _save_mempool(force=True)
        except Exception:  # noqa: BLE001
            pass
        remove_pid()
        write_status(final_state, exit_code=exit_code)
        restore_std_logger()


def _main_impl():
    global pending_tasks

    args = get_args()
    # Порт: --port важнее, иначе из config.json, иначе 8000
    port = args.port
    _master_password.override = args.password

    global CONFIG, pending_tasks, LOG_FILE, PID_FILE, STATUS_FILE, ARCHIVE_FILE
    results_dir = args.results_dir or os.environ.get("MASTER_RESULTS_DIR")
    if results_dir:
        CONFIG = load_config(CONFIG_PATH, results_root=results_dir)
        LOG_FILE = CONFIG["log_file"]
        PID_FILE = CONFIG["pid_file"]
        STATUS_FILE = CONFIG["status_file"]
        ARCHIVE_FILE = CONFIG["archive_file"]
    CONFIG["peers"] = _collect_peers(args)

    if port is None:
        port = int(CONFIG.get("port") or 8000)

    print("=" * 60)
    print(f"ЗАПУСК В РЕЖИМЕ СЕРВЕРА (ПОРТ: {port})")
    print(f"Элемент: {CONFIG['element']} | Псевдопотенциал: {CONFIG['potential']} (q={CONFIG['n_valence']})")
    print(f"Мастер раздаёт ТОЛЬКО число атомов — структуру генерирует воркер")
    print(f"Приёмка: один прогон CP2K от -RESTART.wfn, SCF за "
          f"{cp2k_runner.RESTART_VERIFY_SCF} шагов; E/силы/gap/mag — от мастера")
    print(f"Рабочий каталог: {CONFIG['output_dir']}")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    _load_mempool()

    # Кошелёк мастера: --wallet NAME + --password / MASTER_PASSWORD.
    # Кошелёк — самостоятельная сущность и НЕ создаётся на лету: только явно
    # (wallet.py create / кнопка «Создать» в панели). Опечатка в имени не должна
    # молча заводить пустой кошелёк и уводить туда эмиссию.
    conn = wallet_store.open_db(CONFIG["wallet_store_db"])
    CONFIG["wallet_conn"] = conn
    wallet_name = args.wallet or os.environ.get("MASTER_WALLET")
    try:
        wallet = wallet_store.open_wallet(conn, wallet_name, _master_password())
    except wallet_store.WalletError as e:
        print(f"[Master] ✗ {e}")
        sys.exit(2)
    CONFIG["wallet"] = wallet
    mkey = wallet.latest_key()
    print(f"[Master] Кошелёк «{wallet.name}»: адресов {len(wallet.keys())}, "
          f"ключ #{mkey['id']}: {mkey['address']}")
    print(f"[Master] Подтверждений до доступности баланса: "
          f"{CONFIG.get('coinbase_confirmations')}")

    # Проверка цепочки при старте — работаем только с целой цепочкой
    chk = chain_mod.verify_chain(CONFIG["output_dir"], CONFIG["element"])
    if not chk["ok"]:
        print("[Master] ✗ цепочка блоков повреждена:")
        for p in chk["problems"][:10]:
            print(f"  - {p}")
        sys.exit(3)
    print(f"[Master] Цепочка блоков: {chk['blocks']} блок(ов), целая")

    CONFIG["cp2k_version"] = cp2k_runner.cp2k_version(CONFIG)
    CONFIG["cp2k_build"] = cp2k_runner.cp2k_build(CONFIG)
    print(f"[Master] CP2K {CONFIG['cp2k_version']}")
    if cp2k_runner.cp2k_build_allowed(CONFIG["cp2k_build"]):
        print(f"[Master] CP2K build: {CONFIG['cp2k_build']}")
    elif cp2k_runner.cp2k_build_identified(CONFIG["cp2k_build"]):
        # Сборка опознана, но не та, которой считает сеть: другой MPI/флаги —
        # другая арифметика и другой формат лога, блоки такого мастера пиры
        # не примут. Лучше не стартовать, чем считать мусор.
        print(f"[Master] ✗ CP2K {CONFIG['cp2k_build']} не входит в сборки сети:")
        for allowed in cp2k_runner.CP2K_BUILD_EXPECTED:
            print(f"  • {allowed}")
        print("[Master]   пересоберите образ из актуального Dockerfile (pin по digest)")
        sys.exit(4)
    else:
        print("[Master] ⚠ CP2K_BUILD не задан (старый образ или запуск мимо образа): "
              "в блоке будет «build не указан»")
    if CONFIG["peers"]:
        print(f"[Master] Пиры: {', '.join(CONFIG['peers'])}")
        threading.Thread(target=catch_up_chain, daemon=True).start()

    threading.Thread(target=_watchdog_loop, daemon=True).start()
    threading.Thread(target=_mempool_flush_loop, daemon=True).start()
    threading.Thread(target=_gossip_loop, daemon=True).start()

    # Энергия изолированного атома — один раз, для энергии связи в кадрах
    print("\n" + "=" * 60)
    print("ЭНЕРГИЯ ИЗОЛИРОВАННОГО АТОМА (для энергии связи)")
    print("=" * 60)
    CONFIG["e_isolated_atom"] = None
    per_element = {}
    for sym, cnt in CONFIG["composition"].items():
        piece = dict(CONFIG)
        piece["element"] = sym
        piece["potential"] = cp2k_runner.get_potential(sym)
        piece["isolated_atom_energy_file"] = CONFIG["isolated_atom_energy_files"][sym]
        per_element[sym] = cp2k_runner.calculate_isolated_atom_energy(piece)
    if all(v is not None for v in per_element.values()):
        total = sum(per_element[s] * c for s, c in CONFIG["composition"].items())
        CONFIG["e_isolated_atom"] = total / CONFIG["formula_size"]
        print(f"[Master] E_iso по элементам: {per_element}")
        print(f"[Master] Средняя E_iso на атом: {CONFIG['e_isolated_atom']:.6f} eV")
    else:
        print("[Master] E_iso не посчитана — E_binding в кадрах будет отсутствовать")

    # 1) Постановка задач: мастер НЕ генерирует структуры, только число атомов
    print("\n" + "=" * 60)
    print("ПОСТАНОВКА ЗАДАЧ (без генерации структур мастером)")
    print("=" * 60)

    failed_ids = _load_id_set(CONFIG["failed_ids_file"])
    with task_queue_lock:
        failed_tasks.update(failed_ids)

    _sync_stage("старт")
    stage = CONFIG["_stage_info"]
    print(f"[Master] Расписание: этап {stage['stage']}, кластер n={stage['n_atoms']}, "
          f"{stage['blocks']} блоков на этап (уже сделано {stage['done']})")
    with task_queue_lock:
        print(f"[Server] Очередь заполнена. Всего задач: {len(pending_tasks)}")
    print(f"[Server] Поднимаю FastAPI сервер на http://0.0.0.0:{port} ...")

    server_thread = threading.Thread(target=run_fastapi_server, args=(port,), daemon=True)
    server_thread.start()

    print("\n[Шаг с воркерами] Сервер готов принимать регистрацию и выдавать задачи. Нажмите Ctrl+C для выхода.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Server] Остановка сервера.")


if __name__ == "__main__":
    main()