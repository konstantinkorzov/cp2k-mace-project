#!/usr/bin/env python3
"""
Оффлайн-тесты архитектуры нулевого доверия (CP2K НЕ требуется).

Модули проекта импортируются в изолированной песочнице `_test_env/`, куда
подкладываются заглушки тяжёлых зависимостей (fastapi, uvicorn, pydantic, mace,
ase, find_minimum, xyz_to_html). Матрица S_master эмулируется аналитической
синтетической функцией fake_overlap(), поэтому проверки можно гонять без CP2K.

Запуск (из корня проекта):
    python3 test/test_zero_trust.py

Проверяется:
  * парсинг AO-матриц из лога CP2K;
  * восстановление (λ, C) из H и S (Löwdin);
  * правила верификации мастера (что ловят, а что нет);
  * «глубокая» выборочная проверка энергии;
  * генерация структуры воркером и сборка payload'а с матрицами;
  * запись результатов только после успешной проверки;
  * построение очереди заданий из config.json.
"""

import os
import sys
import json
import time
import types
import base64
import shutil
import logging
import tempfile

import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
SRC = os.path.join(PROJECT_ROOT, "src")
ENV = os.path.join(TEST_DIR, "_test_env")

RESULTS = []


def check(name, ok, extra=""):
    RESULTS.append((name, bool(ok), extra))
    print(f"[{'OK ' if ok else 'FAIL'}] {name} {extra}")


# ---------- изолированное окружение ----------
STUB_FILES = {
    "fastapi.py": '''
class HTTPException(Exception):
    def __init__(self, status_code=500, detail="", headers=None):
        self.status_code = status_code
        self.detail = detail
        self.headers = headers or {}
        super().__init__(detail)


class FastAPI:
    def __init__(self, *a, **k):
        pass

    def _deco(self, *a, **k):
        def wrap(f):
            return f
        return wrap

    get = _deco
    post = _deco
    delete = _deco
    middleware = _deco
''',
    "uvicorn.py": "def run(*a, **k):\n    pass\n",
    "pydantic.py": '''
class BaseModel:
    def __init__(self, **data):
        ann = {}
        for klass in reversed(type(self).__mro__):
            ann.update(getattr(klass, "__annotations__", {}) or {})
        for name in ann:
            setattr(self, name, data.get(name, None))
        for k, v in data.items():
            setattr(self, k, v)
''',
    "find_minimum.py": "def run_find_minimum(*a, **k):\n    return None\n",
    "xyz_to_html.py": "def structure_to_html(*a, **k):\n    return ''\n",
}


def prepare_env():
    if os.path.exists(ENV):
        shutil.rmtree(ENV)
    os.makedirs(ENV)

    # реальные модули проекта
    for f in ("main.py", "cp2k_runner.py", "structure_generator.py", "worker.py",
              "config_loader.py", "archive_maker.py", "identity.py", "chain.py",
              "consensus.py", "crypto.py", "wallet_store.py", "wallet.py",
              "webwallet.py"):
        shutil.copy(os.path.join(SRC, f), os.path.join(ENV, f))

    # реальный config.json проекта (формат {"value":..., "help":...})
    shutil.copy(os.path.join(SRC, "config.json"), os.path.join(ENV, "config.json"))

    # заглушки тяжёлых зависимостей
    for fname, content in STUB_FILES.items():
        with open(os.path.join(ENV, fname), "w", encoding="utf-8") as f:
            f.write(content)

    os.makedirs(os.path.join(ENV, "mace"), exist_ok=True)
    with open(os.path.join(ENV, "mace", "__init__.py"), "w", encoding="utf-8") as f:
        f.write("from . import calculators\n")
    with open(os.path.join(ENV, "mace", "calculators.py"), "w", encoding="utf-8") as f:
        f.write("class MACECalculator:\n    def __init__(self, *a, **k):\n        pass\n")

    os.makedirs(os.path.join(ENV, "ase"), exist_ok=True)
    with open(os.path.join(ENV, "ase", "__init__.py"), "w", encoding="utf-8") as f:
        f.write("from . import io\n")
    with open(os.path.join(ENV, "ase", "io.py"), "w", encoding="utf-8") as f:
        f.write("def read(*a, **k):\n    return None\n\n\ndef write(*a, **k):\n    return None\n")


# ---------- синтетическая матрица перекрывания ----------
B = 6   # «число базисных функций» на атом


def worker_tests():
    print("\n--- 12. Воркер: генерация структуры + сборка payload'а ---")
    logger = logging.getLogger("test_worker")
    logger.addHandler(logging.NullHandler())

    original_runner = cp2k_runner.run_dft_for_single_structure

    def fake_runner(coords, n_atoms, dim, struct_id, cfg, force_recompute=False,
                    save_wfn=True):
        res = {
            "energy_ev": -42.0, "energy_hartree": -1.5,
            "forces_ev_ang": [[0.0, 0.0, 0.0]] * n_atoms,
            "gap": 0.8, "mag": 0.0, "cached": False, "elapsed": 1.0, "hash": "abc",
            "scf_converged": True, "scf_steps": 7,
        }
        if save_wfn:
            wfn = os.path.join(cfg["output_dir"], "fake-RESTART.wfn")
            make_wfn(wfn, nspin=2, nao=50, n_occ=11)
            res["wfn_path"] = wfn
        return res

    cp2k_runner.run_dft_for_single_structure = fake_runner
    main.CONFIG["e_isolated_atom"] = -1000.0  # как будто её посчитал мастер
    try:
        cfg = main.task_cfg_payload(main.CONFIG)
        cfg["output_dir"] = tempfile.mkdtemp(prefix="worker_scratch_")
        task = {"struct_id": "n2_d3_0000", "n_atoms": 2, "dim": 3, "seed": 12345,
                "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 2.7]],
                "species": ["Cu", "Cu"]}

        ok, data = worker.run_task(task, cfg, logger)
        check("задача выполнена", ok, data.get("message", ""))
        check("payload содержит геометрию нужного размера", ok and len(data["geometry"]) == 2)
        check("payload содержит файл состояния (wfn_b64)",
              ok and bool(data.get("wfn_b64")) and data.get("wfn_size", 0) > 1000,
              f"size={data.get('wfn_size')}")
        check("матрицы H/lam/C не шлются", "H" not in data and "C" not in data)
        check("E_iso и E_binding/atom посчитаны",
              data.get("e_isolated_atom_ev") == -1000.0
              and abs((data.get("e_binding_per_atom") or 0) - (-1000.0 * 2 - (-42.0)) / 2) < 1e-9,
              f"iso={data.get('e_isolated_atom_ev')} bind={data.get('e_binding_per_atom')}")
        check("payload содержит энергии и условия расчёта",
              ok and data["energy_ev"] is not None
              and data["cell_size"] == cp2k_runner.resolve_cell_size(
                  np.asarray(data["geometry"]), cfg)
              and data["potential"] == cfg["potential"])

        g = np.asarray(data["geometry"])
        d = float(np.linalg.norm(g[0] - g[1]))
        check("межатомное расстояние >= min_dist",
              d >= structure_generator.min_dist(cfg) - 1e-9, f"d={d:.3f}")
        check("все атомы внутри ячейки",
              float(np.abs(g - data["cell_size"] / 2.0).max()) <= data["cell_size"] / 2.0)

        decoded = base64.b64decode(data["wfn_b64"])
        check("состояние раскодируется и разбирается мастером",
              cp2k_runner.parse_wfn_header(
                  os.path.join(cfg["output_dir"], "fake-RESTART.wfn"))["ok"]
              and len(decoded) == data["wfn_size"])

        # --- несходящийся расчёт: воркер обязан взять новую структуру ---
        bad = dict(
            energy_ev=-42.0, energy_hartree=-1.5, forces_ev_ang=[[0.0, 0.0, 0.0]] * 2,
            gap=None, mag=None, cached=False, elapsed=1.0, hash="x",
            scf_converged=False, scf_steps=None, wfn_path=None,
        )
        cp2k_runner.run_dft_for_single_structure = lambda **kw: dict(bad)
        ok3, data3 = worker.run_task(task, cfg, logger)
        check("несходящийся расчёт не отправляется мастеру",
              (not ok3) and "сош" in data3.get("message", "").lower(), data3.get("message", ""))

        # --- провал CP2K ---
        cp2k_runner.run_dft_for_single_structure = lambda **kw: {"error": True, "rc": 1, "elapsed": 0.1}
        ok4, data4 = worker.run_task(task, cfg, logger)
        check("провал CP2K сообщается мастеру как неуспех", (not ok4) and "message" in data4)
    finally:
        cp2k_runner.run_dft_for_single_structure = original_runner
        main.CONFIG.pop("e_isolated_atom", None)
        shutil.rmtree(cfg["output_dir"], ignore_errors=True)


def queue_tests():
    print("\n--- 13. Очередь заданий и конфиг ---")
    cfg = main.CONFIG

    st1 = main.stage_for_height(0, cfg)
    check("этап 1: 2 атома, 10 блоков",
          st1["n_atoms"] == 2 and st1["blocks"] == 10, str(st1))
    st2 = main.stage_for_height(10, cfg)
    check("высота 10 → 3 атома (димер больше не принимается)",
          st2["n_atoms"] == 3 and st2["stage"] == 2, str(st2))
    check("размер растёт с высотой",
          main.stage_for_height(5000, cfg)["n_atoms"] > st2["n_atoms"])

    tasks = main.schedule_tasks_for_stage(st1, cfg)
    ids = [t["struct_id"] for t in tasks]
    check("заданий на этап = blocks", len(tasks) == st1["blocks"], f"{len(tasks)}")
    check("ids уникальны", len(ids) == len(set(ids)))
    check("размерности чередуются 2D/3D",
          sorted({t["dim"] for t in tasks}) == [2, 3],
          str([t["dim"] for t in tasks[:6]]))
    check("мастер не передаёт готовые coords", all("coords" not in t for t in tasks))

    check("расписание — константы, а не config.json",
          "schedule" not in cfg and "max_atoms" in main.ATOM_SCHEDULE)
    check("заряд/мультиплетность в конфиге",
          "charge" in cfg and "multiplicity" in cfg)
    check("конфиг: псевдопотенциал и валентность определены",
          "GTH-PBE-q11" in str(cfg["potential"]) and cfg["n_valence"] == 11)
    check("конфиг: реестры заданы",
          all(main.CONFIG.get(k) for k in ("verified_ids_file", "failed_ids_file")))
    payload_cfg = main.task_cfg_payload(main.CONFIG)
    check("воркеру передаётся диаметр атома", "atom_diameter" in payload_cfg)
    check("воркеру передаётся состав", "composition" in payload_cfg)
    # v6: профиль сети уходит воркеру целиком и ровно с нашими значениями —
    # это не «ручка выбора», а требование считать теми же числами.
    check("воркеру передаётся профиль сети (не выбор физики)",
          all(payload_cfg.get(k) == v
              for k, v in main.consensus.CP2K_PROFILE.items())
          and not any(k in payload_cfg for k in (
              "verify_mode", "box_size_margin", "min_dist", "max_dist",
              "random_seed", "max_attempts_per_atom")),
          {k: payload_cfg.get(k) for k in ("cutoff", "eps_scf", "mpi_ranks")})


# ---------- макет файла состояния (-RESTART.wfn) по РЕАЛЬНОМУ формату ----------
# Формат снят зондом test/probe_wfn.py на CP2K 2024.3:
#   [20B: 5×int32] [8B] [8B] [56B: 14×int32] [16B: 4×int32] [176B: nspin*n_occ×f8]
#   затем nspin*n_occ блоков по 8*nao байт (столбцы коэффициентов) и хвост 200 байт.
def make_wfn(path, nspin=2, nao=50, n_occ=11, nset=7):
    import struct

    def rec(data):
        return struct.pack("<i", len(data)) + data + struct.pack("<i", len(data))

    def f8(vals):
        return struct.pack(f"<{len(vals)}d", *vals)

    parts = [
        struct.pack("<5i", nspin, 2, nao, 1, nset),
        struct.pack("<2i", 1, 1),
        struct.pack("<2i", nset, nset),
        struct.pack("<14i", 1, 1, 3, 3, 5, 5, 7, 1, 1, 3, 3, 5, 5, 7),
        struct.pack("<4i", n_occ, n_occ, n_occ + 1, n_occ),
        f8([-0.5 - 0.01 * i for i in range(nspin * n_occ)]),
    ]
    for _spin in range(nspin):
        for _col in range(n_occ):
            parts.append(f8([0.1] * nao))
    parts.append(f8([0.0] * 25))

    blob = b"".join(rec(p) for p in parts)
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


STATE_CONTROL = {"ok": True, "nao": 50, "n_occ": 11, "nspin": 2, "converged": True,
                 "energy_ev": -2614.38, "warnings": [], "grad": 1e-7}


def fake_verify_state(coords, wfn_path, cfg, project="master_state", max_scf=None,
                      timeout=None, verbose=False):
    """Заглушка master-side прогона CP2K: эмулирует разбор лога."""
    head = cp2k_runner.parse_wfn_header(wfn_path)
    c = STATE_CONTROL
    res = {
        "ok": False, "reason": "", "submitted": head,
        "sysinfo": {"nao": c["nao"], "n_occ": c["n_occ"], "nmo": c["n_occ"] * 2,
                    "nelec": c["n_occ"] * 2, "nspin": c["nspin"]},
        "scf": {"converged": c["converged"], "steps": 1 if c["converged"] else None,
                "not_converged": not c["converged"]},
        "restart": {"mentioned": True, "warnings": list(c["warnings"]), "lines": ["read"]},
        "first_step": {"grad": c["grad"], "energy_hartree": None, "line": "1 OT DIIS ..."},
        "canonical": {"ok": True, "size": head["size"], "nao": c["nao"],
                      "columns": head["columns"], "nspin": c["nspin"]},
        "elapsed": 0.05, "log_path": "<stub>",
    }
    if c["warnings"]:
        res["reason"] = "CP2K предупредил о несовместимом restart: " + c["warnings"][0]
        return res
    if not c["converged"]:
        res["reason"] = f"состояние не доводится до сходимости за {max_scf} шагов"
        return res
    res.update({"ok": True, "energy_ev": c["energy_ev"],
                "energy_hartree": c["energy_ev"] / 27.211386,
                "forces_ev_ang": [[0.01, -0.02, 0.03]] * len(coords),
                "gap": 0.8, "mag": 0.0})
    return res


def restart_tests():
    print("\n--- 15. Формат -RESTART.wfn и приёмка по состоянию ---")
    tmp = tempfile.mkdtemp(prefix="wfn_test_")
    try:
        # 1) макет Cu2 (nao=50, nspin=2, n_occ=11) — точная копия структуры реального файла
        p = os.path.join(tmp, "cu2-RESTART.wfn")
        size = make_wfn(p, nspin=2, nao=50, n_occ=11)
        check("макет воспроизводит размер реального Cu2-wfn (9516 байт)", size == 9516, f"{size} байт")
        head = cp2k_runner.parse_wfn_header(p)
        check("разбор заголовка: ok/nao/nspin/столбцы",
              head["ok"] and head["nao"] == 50 and head["nspin"] == 2 and head["columns"] == 22,
              f"nao={head['nao']}, nspin={head['nspin']}, cols={head['columns']}")

        # 2) файл «другого элемента»: другое nao
        p_other = os.path.join(tmp, "fe2-RESTART.wfn")
        make_wfn(p_other, nspin=2, nao=78, n_occ=16)
        head_other = cp2k_runner.parse_wfn_header(p_other)
        check("другой элемент/базис виден по nao", head_other["nao"] == 78)

        # 3) мусор вместо файла
        p_bad = os.path.join(tmp, "bad.wfn")
        with open(p_bad, "wb") as f:
            f.write(os.urandom(4096))
        check("мусорный файл не проходит синтаксис", not cp2k_runner.parse_wfn_header(p_bad)["ok"])

        # 4) приёмка: честное состояние
        original_verify = cp2k_runner.verify_state_from_wfn
        cp2k_runner.verify_state_from_wfn = fake_verify_state
        cfg = main.CONFIG
        main._NAO_TOTAL.clear()
        task = {"struct_id": "n2_d3_0000", "task_index": 0, "n_atoms": 2, "dim": 3, "attempts": 0}
        A = [[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]]
        payload = {"geometry": A, "energy_ev": -2614.38,
                   "wfn_b64": base64.b64encode(open(p, "rb").read()).decode("ascii")}
        v = main.verify_state_payload(cfg, task, payload)
        check("честное состояние принято", v["valid"], v.get("reason", ""))
        check("энергия/силы от мастера", v.get("energy_ev") == -2614.38 and v.get("forces_ev_ang"))
        check("обучен nao системы (50 для Cu₂)", main._NAO_TOTAL.get(("Cu", 2)) == 50,
              str(main._NAO_TOTAL))

        # 5) состояние «другого элемента» ловится ещё до CP2K (по обученному nao)
        payload_other = {"geometry": A, "wfn_b64": base64.b64encode(open(p_other, "rb").read()).decode("ascii")}
        v2 = main.verify_state_payload(cfg, task, payload_other)
        check("nao не от этой задачи отклонён", not v2["valid"], v2.get("reason", ""))

        # 6) несходящееся состояние
        STATE_CONTROL["converged"] = False
        v3 = main.verify_state_payload(cfg, task, payload)
        check("нестационарное состояние отклонено", not v3["valid"], v3.get("reason", ""))
        STATE_CONTROL["converged"] = True

        # 7) CP2K предупредил о несовместимом restart
        STATE_CONTROL["warnings"] = ["*** WARNING restart file not compatible ***"]
        v4 = main.verify_state_payload(cfg, task, payload)
        check("предупреждение о несовместимом restart = отказ", not v4["valid"])
        STATE_CONTROL["warnings"] = []

        # 8) лимит размера файла
        cfg["restart_max_mb"] = 0.0001
        v5 = main.verify_state_payload(cfg, task, payload)
        check("слишком большой файл отклонён", not v5["valid"], v5.get("reason", ""))
        cfg["restart_max_mb"] = 50

        print("\n--- 16. Keep-best по геометрии (набор для MACE) ---")
        v = main.verify_state_payload(cfg, task, payload)
        v["forces_ev_ang"] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        check("первый кадр принят", main._persist_verified_state(task, payload, v) == "accepted")
        frame_files = os.listdir(main.CONFIG["frames_dir"])
        check("создан ровно один кадр", len([f for f in frame_files if f.endswith('.xyz')]) == 1, str(frame_files))

        worse = dict(v, energy_ev=v["energy_ev"] + 1.0)
        check("худшая энергия той же геометрии не перезаписывает кадр",
              main._persist_verified_state(task, payload, worse) == "kept_old")
        better = dict(v, energy_ev=v["energy_ev"] - 1.0)
        check("лучшая энергия заменяет кадр",
              main._persist_verified_state(task, payload, better) == "accepted")

        agg = os.path.join(main.CONFIG["output_dir"], main.CONFIG["output_all_xyz_name"])
        with open(agg, encoding="utf-8") as f:
            text = f.read()
        check("агрегатный XYZ содержит ровно один кадр с лучшей энергией",
              text.count("\n") > 3 and text.count("Properties=species") == 1
              and f"energy={better['energy_ev']:.8f}" in text)

        print("\n--- 17. Цикл мастера в режиме restart ---")
        main.pending_tasks.clear(); main.active_tasks.clear()
        main.completed_tasks.clear(); main.failed_tasks.clear(); main.registered_workers.clear()
        reg = main._register_worker("test", None, None)
        wid, sess = reg["worker_id"], reg["session"]
        # В v4 сверка геометрии включается для блоков с зарегистрированным
        # шаблоном (это проверяется отдельным тестом); здесь — сам механизм.
        t = {**task, "struct_id": "n2_d3_0007", "attempts": 0}
        main.pending_tasks.append(t)
        got = main.get_task(wid, session=sess)
        check("мастер не передаёт режим проверки (он всегда restart)",
              "verify_mode" not in got["task"]["cfg"])
        issued = got["task"]
        wrong = {"geometry": [[0.0, 0.0, 0.0], [3.4, 0.0, 0.0]],
                 "species": issued["species"],
                 "wfn_b64": base64.b64encode(open(p, "rb").read()).decode("ascii")}
        bad = main.submit_result(main.TaskResult(
            worker_id=wid, session=sess, struct_id=issued["struct_id"],
            success=True, data=wrong))
        check("подменённая геометрия отклоняется до CP2K",
              bad.get("status") == "rejected" and "не та" in str(bad.get("reason")), str(bad))

        # задание вернулось в очередь — выдаём снова (новый seed → новая геометрия)
        got = main.get_task(wid, session=sess)
        issued = got["task"]
        good = {"geometry": issued["geometry"],
                "species": issued["species"],
                "energy_ev": -2614.38,
                "wfn_b64": base64.b64encode(open(p, "rb").read()).decode("ascii")}
        resp = main.submit_result(main.TaskResult(worker_id=wid, session=sess,
                                                  struct_id=issued["struct_id"],
                                                  success=True, data=good))
        check("submit-result принял состояние", resp["status"] == "accepted", str(resp))

        unknown = main.submit_result(main.TaskResult(
            worker_id=wid, session=sess, struct_id="n2_d3_0008",
            success=True, data=good))
        check("результат для неизвестной задачи не принимается",
              unknown.get("status") == "not_found", str(unknown))

        # шаблонов больше нет, но геометрия всё равно сверяется с выданной:
        # мастер выдал координаты сам, воркер обязан считать ровно их
        main.pending_tasks.clear(); main.active_tasks.clear()
        main.pending_tasks.append({**task, "struct_id": "n2_d3_0010", "attempts": 0})
        issued2 = main.get_task(wid, session=sess)["task"]
        check("без шаблона задание тоже содержит геометрию",
              len(issued2.get("geometry") or []) == 2)
        free = {"geometry": [[0.0, 0.0, 0.0], [3.4, 0.0, 0.0]],
                "species": issued2["species"],
                "wfn_b64": base64.b64encode(open(p, "rb").read()).decode("ascii")}
        free_resp = main.submit_result(main.TaskResult(
            worker_id=wid, session=sess, struct_id=issued2["struct_id"],
            success=True, data=free))
        check("подменённая геометрия отклоняется без всяких шаблонов",
              free_resp.get("status") == "rejected", str(free_resp))

        shutil.rmtree(main.CONFIG["frames_dir"], ignore_errors=True)
        for f in (agg, main.CONFIG["verified_ids_file"], main.CONFIG["failed_ids_file"]):
            if os.path.exists(f):
                os.remove(f)
    finally:
        cp2k_runner.verify_state_from_wfn = original_verify
        shutil.rmtree(tmp, ignore_errors=True)


FAKE_CP2K = r'''
import os, re, sys
mode = os.environ.get("FAKE_CP2K_MODE", "ok")
args = sys.argv[1:]
inp = args[args.index("-i") + 1] if "-i" in args else "in.inp"
project = "master_state"
natoms = 2
try:
    text = open(inp, encoding="utf-8").read()
    m = re.search(r"PROJECT\s+(\S+)", text)
    if m:
        project = m.group(1)
    natoms = len(re.findall(r"^\s+[A-Z][a-z]?\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+", text, re.M)) or 2
except Exception:
    pass
print(" CP2K| version 2024.3 (fake)")
print(" SCF_GUESS RESTART")
print(" Reading wavefunction from file %s-RESTART.wfn" % project)
if mode == "warning":
    print(" *** WARNING *** restart file is not compatible: different number of basis functions")
print(" Number of orbital functions: 50")
print(" Number of molecular orbitals: 50")
print(" Number of occupied orbitals: 11")
print(" Number of electrons: 22")
print(" Number of spin channels: 2")
print("  SCF WAVEFUNCTION OPTIMIZATION")
print("   Step     Update method      Time    Convergence         Total energy    Change")
print("   ----------------------------------------------------------------")
print("      1 OT DIIS     0.15E+00    0.5     0.44442328     -2609.4767334953 -2.61E+03")
print("      2 OT DIIS     0.15E+00    0.6     0.00000012     -2609.4767334953 -1.00E-09")
print("  *** SCF run NOT converged ***" if mode == "notconverged" else
      "  *** SCF run converged in     2 steps ***")
print(" ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:   -96.0987654321")
print(" HOMO - LUMO gap [eV] :    1.234500")
with open(project + ".forces", "w", encoding="utf-8") as f:
    f.write(" ATOMIC FORCES in [a.u.]\n")
    for i in range(natoms):
        f.write(" %6d      1   Cu     0.001000      -0.002000       0.003000\n" % (i + 1))
    f.write(" SUM OF ATOMIC FORCES           0.000000       0.000000       0.000000\n")
'''


def master_state_tests():
    """Проверка самого мастера через «фальшивый cp2k» — без настоящего CP2K."""
    print("\n--- 18. Мастер: реальный прогон приёмки (фальшивый cp2k) ---")
    tmp = tempfile.mkdtemp(prefix="fakecp2k_")
    try:
        fake = os.path.join(tmp, "fake_cp2k.py")
        with open(fake, "w", encoding="utf-8") as f:
            f.write(FAKE_CP2K)
        wfn = os.path.join(tmp, "submitted-RESTART.wfn")
        make_wfn(wfn, nspin=2, nao=50, n_occ=11)

        cfg = dict(main.CONFIG)
        cfg["cp2k_cmd"] = f"{sys.executable} {fake}"
        cfg["hartree_to_ev"] = 27.211386
        cfg["hartree_bohr_to_ev_ang"] = 51.422067
        cfg["require_restart_read_confirmation"] = True
        coords = [[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]]

        os.environ.pop("FAKE_CP2K_MODE", None)
        res = cp2k_runner.verify_state_from_wfn(coords, wfn, cfg, project="master_state")
        check("приёмка состояния: ok", res["ok"], res.get("reason", ""))
        check("энергия разобрана из лога",
              abs(res["energy_ev"] - (-96.0987654321 * 27.211386)) < 1e-6,
              f"E={res.get('energy_ev')}")
        check("силы разобраны (2 атома)", len(res.get("forces_ev_ang") or []) == 2)
        check("сходимость: 2 шага", res["scf"]["converged"] and res["scf"]["steps"] == 2)
        check("чтение restart подтверждено логом", res["restart"]["mentioned"])
        check("system info: nao/занятые/электроны",
              res["sysinfo"].get("nao") == 50 and res["sysinfo"].get("n_occ") == 11
              and res["sysinfo"].get("nelec") == 22)
        check("невязка первой строки OT разобрана",
              res["first_step"].get("grad") is not None
              and abs(res["first_step"]["grad"] - 0.44442328) < 1e-9,
              str(res["first_step"]))
        check("HOMO-LUMO gap разобран", abs((res.get("gap") or 0) - 1.2345) < 1e-9)

        os.environ["FAKE_CP2K_MODE"] = "notconverged"
        res2 = cp2k_runner.verify_state_from_wfn(coords, wfn, cfg, project="master_state")
        check("несходящееся состояние отклонено",
              (not res2["ok"]) and "сходим" in res2.get("reason", ""), res2.get("reason", ""))

        os.environ["FAKE_CP2K_MODE"] = "warning"
        res3 = cp2k_runner.verify_state_from_wfn(coords, wfn, cfg, project="master_state")
        check("предупреждение о несовместимом restart = отказ",
              (not res3["ok"]) and "restart" in res3.get("reason", "").lower(),
              res3.get("reason", ""))
        os.environ.pop("FAKE_CP2K_MODE", None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def identity_tests():
    print("\n--- 19. Идентичность воркера ---")
    a = identity.make_identity("alpha")
    b = identity.make_identity("beta")
    check("приватный ключ — 256 бит", len(a["private_key"]) == 64, a["private_key"][:16] + "...")
    check("публичный ключ — 32 байта", len(bytes.fromhex(a["public_key"])) == 32)
    check("pubkey_hash — 64 hex", len(a["pubkey_hash"]) == 64)
    check("разные имена — разные id", a["pubkey_hash"] != b["pubkey_hash"])
    check("каждый запуск — новая пара",
          identity.make_identity("alpha")["pubkey_hash"] != a["pubkey_hash"])


def block_dft(element_dir, n_atoms=2, dim=None, nonce=0, system="Cu",
              composition=None):
    """dft.json, геометрия которого следует из (prev_id, nonce) — правило v3.

    dim=None — размерность по высоте (протокол v6), как её строит мастер.
    """
    import structure_generator

    comp = composition or {system: 1}
    seq = chain.next_seq(element_dir)
    prev = chain.previous_hash(element_dir, seq)
    if dim is None:
        dim = chain.consensus.dim_for_height(seq - 1)
    header = {"prev_id": prev, "nonce": nonce, "n_atoms": n_atoms, "dim": dim,
              "atom_diameter": 2.0, "system": system, "composition": comp}
    geom = chain.expected_geometry(header)
    species = structure_generator.assign_species(
        n_atoms, {"composition": comp},
        seed=chain.geometry_seed(prev, nonce))
    return {"energy_ev": -42.0,
            "forces_ev_ang": [[0.0, 0.0, 0.0]] * n_atoms,
            "geometry": np.asarray(geom).tolist(),
            "species": species}


def chain_tests():
    print("\n--- 20. Блокчейн-блоки, подписи и комиссии ---")
    import zipfile
    from datetime import datetime

    tmp = tempfile.mkdtemp(prefix="chain_test_")
    tmp2 = tempfile.mkdtemp(prefix="chain_nosig_")
    try:
        wfn = os.path.join(tmp, "state.wfn")
        make_wfn(wfn)
        sha1, rel1 = chain.store_blob(tmp, wfn)
        sha2, rel2 = chain.store_blob(tmp, wfn)
        check("blob дедуплицируется по sha256",
              sha1 == sha2 and rel1 == rel2 == f"blobs/{sha1}.wfn")

        conn = wallet_store.open_db(os.path.join(tmp, "keys.db"))
        wallet = wallet_store.create_wallet(conn, "master", "root")
        master = wallet.latest_key()["address"]
        worker = "b" * 64
        cp2k = {"cp2k_version": "2024.3",
                "cp2k_build": cp2k_runner.CP2K_BUILD_CPU,
                "cp2k_params": dict(cp2k_runner.CP2K_PARAMS)}

        p1 = chain.write_block(tmp, "Cu", 2, sha1, rel1,
                               [chain.make_coinbase(1, master)],
                               dft=block_dft(tmp), **cp2k)
        check("имя блока 000001_Cu_2.zip",
              os.path.basename(p1) == "000001_Cu_2.zip", os.path.basename(p1))

        with zipfile.ZipFile(p1) as z:
            names = set(z.namelist())
            h1 = json.loads(z.read("header.json"))
            tx1 = json.loads(z.read("tx-0000-coinbase.json"))

        expected = {"version", "seq", "system", "composition", "n_atoms", "dim",
                    "atom_diameter", "prev_id", "created_at", "state_sha256",
                    "state_path", "cp2k_version", "cp2k_build", "cp2k_params",
                    "tx_root", "nonce", "dft_sha256", "geom_version",
                    "block_id"}
        check("в заголовке только нужные поля", set(h1) == expected, sorted(h1))
        check("отпечаток сборки CP2K попал в заголовок",
              h1.get("cp2k_build") == cp2k_runner.CP2K_BUILD_CPU, h1.get("cp2k_build"))
        check("version = 6", h1["version"] == 6)
        check("created_at — ISO 8601 UTC",
              datetime.strptime(h1["created_at"], "%Y-%m-%dT%H:%M:%SZ") is not None)
        check("CP2K version и params в блоке",
              h1["cp2k_version"] == "2024.3" and h1["cp2k_params"]["cutoff"] == 400)
        check("первый блок: prev = нули", h1["prev_id"] == chain.ZERO_HASH)

        # --- расписание размеров проверяется и пиром, а не только мастером ---
        off = dict(h1, n_atoms=3)
        p_off = chain.validate_header(off)
        check("n_atoms не по расписанию отвергается",
              any("расписания" in p for p in p_off), p_off[:2])

        ok_n = chain.validate_header(dict(h1, block_id=h1["block_id"]))
        check("свой n_atoms расписанию соответствует",
              not any("расписания" in p for p in ok_n), ok_n[:2])

        p_dim = chain.validate_header(dict(h1, dim=1))
        check("dim вне расписания (1) отвергается",
              any("не из расписания" in p for p in p_dim), p_dim[:2])

        # v6: размерность задаётся высотой (seq 2 → 2D)
        p_h = chain.validate_header(dict(h1, seq=2, dim=3))
        check("dim не по высоте отвергается",
              any("для высоты" in p for p in p_h), p_h[:2])
        check("dim по высоте принимается",
              not any("dim" in p for p in chain.validate_header(dict(h1, seq=2, dim=2))))

        # v6: профиль вычислений — часть консенсуса
        check("профиль сети в блоке",
              not chain.consensus.profile_mismatch(h1["cp2k_params"]))
        p_prof = chain.validate_header(
            dict(h1, cp2k_params=dict(h1["cp2k_params"], cutoff=200)))
        check("чужой профиль (cutoff=200) отвергается",
              any("не совпадают" in p for p in p_prof), p_prof[:2])
        p_rank = chain.validate_header(
            dict(h1, cp2k_params=dict(h1["cp2k_params"], mpi_ranks=8)))
        check("чужое число рангов отвергается",
              any("не совпадают" in p for p in p_rank), p_rank[:2])
        check("расписание: высота 10 → n=3, высота 0 → n=2",
              main.stage_for_height(0, {"formula_size": 1})["n_atoms"] == 2
              and main.stage_for_height(10, {"formula_size": 1})["n_atoms"] == 3)

        # --- молекула: слаг системы + состав в заголовке ---
        tmpm = tempfile.mkdtemp(prefix="chain_mol_")
        try:
            wfnm = os.path.join(tmpm, "state.wfn")
            make_wfn(wfnm)
            sham, relm = chain.store_blob(tmpm, wfnm)
            comp = {"Al": 2, "O": 3}
            dftm = block_dft(tmpm, n_atoms=5, system="al2o3", composition=comp)
            pm = chain.write_block(tmpm, "al2o3", 5, sham, relm,
                                   [chain.make_coinbase(1, master)], dft=dftm,
                                   composition=comp, **cp2k)
            check("молекула: имя блока с слагом",
                  os.path.basename(pm) == "000001_al2o3_5.zip", os.path.basename(pm))
            with zipfile.ZipFile(pm) as z:
                hm = json.loads(z.read("header.json"))
            check("молекула: состав в заголовке", hm["composition"] == comp)
            check("молекула: цепочка проверяется",
                  chain.verify_chain(tmpm, "al2o3")["ok"],
                  chain.verify_chain(tmpm, "al2o3")["problems"][:2])
            bad = chain.validate_header(dict(hm, system="Al2O3!"))
            check("молекула: неверный слаг отвергается",
                  any("system" in p for p in bad), bad[:2])

        finally:
            shutil.rmtree(tmpm, ignore_errors=True)
        check("wfn в блок не кладётся, только ссылка",
              not any(n.endswith(".wfn") for n in names) and h1["state_path"] == rel1)
        check("coinbase — 50 000 000 монет",
              sum(o["value"] for o in tx1["outputs"]) == chain.COINBASE_VALUE)

        no_reward = chain.build_transfer(tmp, {master}, worker, master,
                                         chain.REWARD_VALUE, 1, height=1)
        check("в блоке 1 награда невозможна (нет подтверждений)", no_reward is None)

        reward = chain.build_transfer(tmp, {master}, worker, master,
                                      chain.REWARD_VALUE, 1, height=2,
                                      fee=chain.TX_FEE)
        check("перевод 25 воркеру собирается", reward is not None
              and any(o["address"] == worker and o["value"] == chain.REWARD_VALUE
                      for o in reward["outputs"]))

        utxo = chain.utxo_set(tmp)
        fee = chain.tx_fee(reward, utxo)
        check("комиссия = входы − выходы", fee == chain.TX_FEE, fee)
        chain.sign_transaction(reward, wallet, utxo)
        check("входы подписаны Ed25519",
              all(i.get("signature") and i.get("public_key") for i in reward["inputs"]))

        cb2 = chain.make_coinbase(2, master, chain.COINBASE_VALUE + fee)
        p2 = chain.write_block(tmp, "Cu", 2, sha1, rel1, [cb2, reward],
                               dft=block_dft(tmp), **cp2k)

        with zipfile.ZipFile(p2) as z:
            h2 = json.loads(z.read("header.json"))
            cb_tx = json.loads(z.read("tx-0000-coinbase.json"))
        check("второй блок: prev_id = block_id(первого)",
              h2["prev_id"] == h1["block_id"])
        check("coinbase = 50 000 000 + комиссия",
              sum(o["value"] for o in cb_tx["outputs"]) == chain.COINBASE_VALUE + fee)

        v = chain.verify_chain(tmp, "Cu")
        check("цепочка целая", v["ok"], v["problems"])

        bal = chain.balances(tmp, None, 1)
        check("воркеру доступно 25 000 000", bal[worker]["available"] == chain.REWARD_VALUE,
              bal.get(worker))
        check("у мастера 75 000 000", bal[master]["total"] == chain.COINBASE_VALUE + chain.REWARD_VALUE,
              bal.get(master))

        # --- подпись обязательна ---
        wfn2 = os.path.join(tmp2, "state.wfn")
        make_wfn(wfn2)
        sha3, rel3 = chain.store_blob(tmp2, wfn2)
        chain.write_block(tmp2, "Cu", 2, sha3, rel3,
                          [chain.make_coinbase(1, master)],
                          dft=block_dft(tmp2), **cp2k)
        bad_reward = chain.build_transfer(tmp2, {master}, worker, master,
                                          chain.REWARD_VALUE, 1, height=2)
        chain.write_block(tmp2, "Cu", 2, sha3, rel3,
                          [chain.make_coinbase(2, master, 50.0), bad_reward],
                          dft=block_dft(tmp2), **cp2k)
        v2 = chain.verify_chain(tmp2, "Cu")
        check("блок без подписи входа отклоняется",
              (not v2["ok"]) and any("без подписи" in p for p in v2["problems"]),
              v2["problems"][:2])

        # --- мемпул: сборка и проверка перевода ---
        to_addr = "c" * 64
        tx = chain.build_wallet_transfer(wallet, tmp, to_addr, 10.0,
                                         fee=chain.TX_FEE, confirmations=1, height=3)
        check("кошелёк собрал перевод",
              tx is not None and any(o["address"] == to_addr for o in tx["outputs"]))
        ok_tx, probs = chain.validate_tx(tx, tmp, height=3, confirmations=1)
        check("валидная tx проходит проверку", ok_tx, probs)

        bad_tx = json.loads(json.dumps(tx))
        bad_tx["inputs"][0]["signature"] = "00" * 64
        check("подпись входа проверяется",
              not chain.validate_tx(bad_tx, tmp, height=3, confirmations=1)[0])

        utxo = chain.utxo_set(tmp)
        ok1, _ = chain.validate_tx(tx, tmp, height=3, confirmations=1, utxo=utxo)
        for i in tx["inputs"]:
            utxo.pop((i["txid"], int(i["vout"])), None)
        ok2, _ = chain.validate_tx(tx, tmp, height=3, confirmations=1, utxo=utxo)
        check("повторная трата того же входа отклоняется", ok1 and not ok2)

        # --- мемпул: включение в блок и комиссия ---
        saved_out = main.CONFIG.get("output_dir")
        try:
            main.CONFIG["output_dir"] = tmp
            main.MEMPOOL.clear()
            main.MEMPOOL[tx["txid"]] = tx
            inc, mem_fee = main._drain_mempool(chain.utxo_set(tmp), 3, 1)
            check("мемпул включён в блок",
                  len(inc) == 1 and abs(mem_fee - chain.TX_FEE) < 1e-9
                  and not main.MEMPOOL, mem_fee)
        finally:
            main.CONFIG["output_dir"] = saved_out
            main.MEMPOOL.clear()

        # --- форк: откат в orphan, возврат tx, восстановление ---
        check("height_of_hash находит высоту",
              chain.height_of_hash(tmp, chain.sha256_file(p1)) == 1)
        removed, txs = chain.truncate_chain(tmp, 1)
        check("откат уносит блок в orphan",
              len(removed) == 1 and not os.path.exists(p2), len(removed))
        check("tx короткой ветки возвращаются",
              any(t.get("type") == "transfer" for t in txs))
        chain.restore_blocks(removed)
        check("restore возвращает блок",
              os.path.exists(p2) and chain.verify_chain(tmp, "Cu")["ok"])

        # --- проверка ОДНОГО блока (как при синхронизации с пиром) ---
        tmp3 = tempfile.mkdtemp(prefix="chain_one_")
        edir = os.path.join(tmp3, "Cu")
        os.makedirs(edir, exist_ok=True)
        try:
            wfn3 = os.path.join(edir, "state.wfn")
            make_wfn(wfn3)
            sha4, rel4 = chain.store_blob(edir, wfn3)
            p3 = chain.write_block(edir, "Cu", 2, sha4, rel4,
                                   [chain.make_coinbase(1, master)],
                                   dft=block_dft(edir), **cp2k)
            vb = chain.verify_block(edir, p3, expected_seq=1,
                                    expected_prev=chain.ZERO_HASH)
            check("verify_block принимает корректный блок", vb["ok"], vb["problems"])

            vb2 = chain.verify_block(edir, p3, expected_seq=2,
                                     expected_prev=chain.ZERO_HASH)
            check("verify_block отвергает не тот seq", not vb2["ok"])

            # блок с размером не по расписанию (высота 1 → 3 атома нельзя)
            p_off3 = chain.write_block(edir, "Cu", 3, sha4, rel4,
                                       [chain.make_coinbase(2, master)],
                                       dft=block_dft(edir, n_atoms=3), **cp2k)
            vb3 = chain.verify_block(edir, p_off3, expected_seq=2,
                                     expected_prev=h1["block_id"])
            check("verify_block отвергает блок не по расписанию",
                  (not vb3["ok"])
                  and any("расписания" in p for p in vb3["problems"]),
                  vb3["problems"][:2])
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)

        # --- порча: меняем dft.json внутри блока ---
        with zipfile.ZipFile(p1) as z:
            entries = {n: z.read(n) for n in z.namelist()}
        tampered = json.loads(entries["dft.json"])
        tampered["energy_ev"] = float(tampered["energy_ev"]) + 1.0
        entries["dft.json"] = json.dumps(tampered, ensure_ascii=False).encode("utf-8")
        with zipfile.ZipFile(p1, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, data in entries.items():
                z.writestr(name, data)
        check("подмена dft внутри блока ловится", not chain.verify_chain(tmp, "Cu")["ok"])

        # --- адрес получателя: 64 hex, иначе перевод в никуда ---
        check("адрес-опечатка отклоняется",
              bool(chain.invalid_outputs({"outputs": [{"value": 1, "address": "abc"}]})))
        check("адрес в верхнем регистре отклоняется",
              bool(chain.invalid_outputs({"outputs": [{"value": 1, "address": "A" * 64}]})))
        check("корректный адрес принимается",
              not chain.invalid_outputs({"outputs": [{"value": 1, "address": "a" * 64}]}))

        # --- аудит: txid — обязательство на тело транзакции ---
        tx_bad = {"version": 1, "type": "transfer",
                  "inputs": [{"txid": "a" * 64, "vout": 0}],
                  "outputs": [{"value": 100, "address": "b" * 64}],
                  "txid": "c" * 64}
        ok_bad, probs = chain.validate_tx(tx_bad, tmp, utxo={})
        check("txid, не совпадающий с телом, отклоняется",
              not ok_bad and any("txid не совпадает" in x for x in probs), probs[:2])

        # --- аудит: мусор в coinbase не роняет проверку блока ---
        tmpv = tempfile.mkdtemp(prefix="chain_txid_")
        try:
            wfnv = os.path.join(tmpv, "state.wfn")
            make_wfn(wfnv)
            shav, relv = chain.store_blob(tmpv, wfnv)
            cb_bad = {"version": 1, "type": "coinbase", "height": 1,
                      "inputs": [{"coinbase": "block-000001"}],
                      "outputs": [{"value": None, "address": master}],
                      "txid": "c" * 64}
            pv = chain.write_block(tmpv, "Cu", 2, shav, relv, [cb_bad],
                                   dft=block_dft(tmpv, n_atoms=2), **cp2k)
            vb = chain.verify_block(tmpv, pv)
            check("coinbase с нечисловой суммой не роняет проверку",
                  vb["ok"] is False
                  and any("не целое" in x for x in vb["problems"]),
                  vb["problems"][:2])
            # записанный txid канонический, а tx_root считается по нему же
            stored = chain.read_block_txs(pv)[0]
            check("write_block пересчитывает txid, tx_root согласован",
                  stored["txid"] == chain.txid_of(stored)
                  and stored["txid"] != cb_bad["txid"]
                  and chain.tx_root_of([stored]) == chain.read_block(pv)[0]["tx_root"])
        finally:
            shutil.rmtree(tmpv, ignore_errors=True)

        # --- аудит: zip-бомба по числу записей ---
        import io as _io
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zb:
            for i in range(chain.MAX_BLOCK_ENTRIES + 1):
                zb.writestr(f"tx-{i:04d}-coinbase.json", "{}")
        too_many = False
        try:
            chain.read_block_bytes(buf.getvalue())
        except ValueError:
            too_many = True
        check("архив с лишними записями отбивается", too_many)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)


def wallet_tests():
    print("\n--- 22. Кошельки (имя + пароль; создание только явное) ---")
    tmp = tempfile.mkdtemp(prefix="wallet_test_")
    try:
        def err(fn):
            try:
                fn()
                return None
            except Exception as e:  # noqa: BLE001
                return e

        conn = wallet_store.open_db(os.path.join(tmp, "keys.db"))
        import stat as _stat
        _mode = _stat.S_IMODE(os.stat(os.path.join(tmp, "keys.db")).st_mode)
        check("keys.db доступен только владельцу", _mode == 0o600, oct(_mode))
        check("свежая база — ни одного кошелька", wallet_store.list_wallets(conn) == [])
        check("несуществующий кошелёк не открывается молча",
              err(lambda: wallet_store.open_wallet(conn, "master", "root")) is not None)

        m = wallet_store.create_wallet(conn, "master", "root")
        check("кошелёк создаётся только явной командой",
              m.name == "master" and len(m.keys()) == 1)
        check("занятое имя — ошибка",
              err(lambda: wallet_store.create_wallet(conn, "master", "other")) is not None)
        check("короткий пароль не принимается",
              err(lambda: wallet_store.create_wallet(conn, "short", "123")) is not None)
        check("кривое имя не принимается",
              err(lambda: wallet_store.create_wallet(conn, "bad name", "root")) is not None)

        w1 = wallet_store.create_wallet(conn, "worker1", "root")
        w2 = wallet_store.create_wallet(conn, "worker2", "root")
        check("один пароль открывает несколько кошельков",
              wallet_store.unlock(conn, "worker1", "root") is not None
              and wallet_store.unlock(conn, "worker2", "root") is not None)
        check("у кошельков с общим паролем разные адреса",
              w1.latest_key()["address"] != w2.latest_key()["address"])
        check("чужой пароль не открывает",
              wallet_store.unlock(conn, "worker1", "nope") is None)
        e = err(lambda: wallet_store.open_wallet(conn, "worker1", "nope"))
        check("ошибка пароля говорит, что пароль неверный",
              e is not None and "неверный пароль" in str(e))

        addr = m.latest_key()["address"]
        sig = m.sign(addr, b"hello")
        check("Ed25519-подпись кошелька проверяется",
              crypto.verify(m.public_for(addr), b"hello", sig))

        m.change_password("root", "newpass")
        check("старый пароль мастера после смены не работает",
              wallet_store.unlock(conn, "master", "root") is None)
        m2 = wallet_store.unlock(conn, "master", "newpass")
        check("новый пароль работает, ключи на месте",
              m2 is not None and m2.latest_key()["address"] == addr)
        sig2 = m2.sign(addr, b"x")
        check("ключи перешифрованы и подпись валидна",
              crypto.verify(m2.public_for(addr), b"x", sig2))
        check("смена пароля одного кошелька не трогает другие",
              wallet_store.unlock(conn, "worker1", "root") is not None
              and wallet_store.unlock(conn, "worker2", "root") is not None)

        names = {w["name"] for w in wallet_store.list_wallets(conn)}
        check("список кошельков показывает имена", names == {"master", "worker1", "worker2"})

        element_dir = os.path.join(tmp, "element")
        os.makedirs(element_dir, exist_ok=True)
        check("wallet.py показывает кошелёк по имени и паролю",
              wallet.show(m2, element_dir, 1, "element") == 0)

        # --- элемент = цепочка, панель: список/создание/выбор ---
        import webwallet

        os.makedirs(os.path.join(tmp, "Cu", "blocks"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "Fe", "blocks"), exist_ok=True)
        listed = webwallet.list_wallets(tmp)
        check("панель получает список кошельков без пароля",
              listed.get("ok") and {w["name"] for w in listed["wallets"]}
              == {"master", "worker1", "worker2"})
        made = webwallet.create_wallet(tmp, "panelw", "pass1234")
        check("панель создаёт кошелёк явно",
              made.get("ok") and any(w["name"] == "panelw" for w in made["wallets"]))
        check("панель не даёт создать дубль имени",
              not webwallet.create_wallet(tmp, "panelw", "pass1234").get("ok"))

        info = webwallet.collect(tmp, "worker1", "root", 1, element="Cu")
        check("кошелёк открывается по имени и паролю",
              info.get("ok") and info["wallet"]["name"] == "worker1")
        check("неверное имя — ошибка, без автосоздания",
              not webwallet.collect(tmp, "nosuch", "root", 1, element="Cu").get("ok")
              and "nosuch" not in {w["name"] for w in webwallet.list_wallets(tmp)["wallets"]})
        check("панель читает выбранную цепочку (элемент)",
              info.get("element") == "Cu")
        check("в каталоге видны обе цепочки (Cu и Fe)",
              {c["name"] for c in info["chains"]} >= {"Cu", "Fe"})
        check("чужой элемент (выход из каталога) отвергается",
              not webwallet.collect(tmp, "worker1", "root", 1, element="../etc").get("ok"))
        got = webwallet.new_address(tmp, "worker1", "root")
        check("новый адрес добавляется в выбранный кошелёк",
              got.get("ok") and len(got.get("addresses") or []) == 2)
        changed = webwallet.change_password(tmp, "worker2", "root", "new2")
        check("смена пароля из панели меняет только свой кошелёк",
              changed.get("ok")
              and wallet_store.unlock(conn, "worker2", "new2") is not None
              and wallet_store.unlock(conn, "worker1", "root") is not None)

        # --- команды кошелька: адрес, переименование, копия, подпись ---
        w2x = wallet_store.unlock(conn, "worker2", "new2")
        n_before = len(w2x.keys())
        key = w2x.add_key()
        check("--new-address добавляет адрес",
              bool(key["address"]) and len(w2x.keys()) == n_before + 1)
        renamed = wallet_store.rename_wallet(conn, "worker2", "worker2x")
        check("--rename меняет только имя",
              renamed["name"] == "worker2x"
              and wallet_store.unlock(conn, "worker2x", "new2") is not None
              and wallet_store.unlock(conn, "worker2", "new2") is None)
        check("--rename в занятое имя — ошибка",
              err(lambda: wallet_store.rename_wallet(conn, "worker2x", "master")) is not None)

        dest = os.path.join(tmp, "backup.db")
        made_copy = wallet.backup_db(os.path.join(tmp, "keys.db"), dest)
        conn2 = wallet_store.open_db(made_copy)
        check("--backup: копия открывается и содержит те же кошельки",
              {w["name"] for w in wallet_store.list_wallets(conn2)}
              == {w["name"] for w in wallet_store.list_wallets(conn)})
        check("--backup не перезаписывает существующий файл",
              err(lambda: wallet.backup_db(os.path.join(tmp, "keys.db"), dest)) is not None)

        w2y = wallet_store.unlock(conn, "worker2x", "new2")
        addr2 = w2y.latest_key()["address"]
        sig = wallet.sign_text(w2y, addr2, "cp2k-mace")
        check("--sign-message/--verify-message: верная подпись",
              wallet.verify_text(w2y, addr2, "cp2k-mace", sig))
        check("--verify-message: другой текст не проходит",
              not wallet.verify_text(w2y, addr2, "cp2k-mace!", sig))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def busy_tests():
    print("\n--- 24. Занятый мастер и передача работы другому ---")
    import logging

    tmp = tempfile.mkdtemp(prefix="busy_")
    saved = {k: main.CONFIG.get(k) for k in ("output_dir", "element")}
    old_verify, old_persist = main.verify_state_payload, main._persist_verified_state
    old_get, old_post = worker.requests.get, worker.requests.post
    try:
        main.CONFIG.update({"output_dir": tmp, "element": "Cu"})
        main.pending_tasks.clear()
        main.active_tasks.clear()
        main.BUSY.clear()
        r = main._register_worker("busy_t", "hash_busy", None)
        wid, sess = r["worker_id"], r["session"]

        main.BUSY.set()
        check("busy: задача не выдаётся",
              main.get_task(wid, session=sess).get("status") == "busy")

        main.BUSY.clear()
        main.pending_tasks.append({"struct_id": "b1", "task_index": 0,
                                   "n_atoms": 2, "dim": 3, "attempts": 0})
        got = main.get_task(wid, session=sess)
        check("задача выдана до занятости", got.get("status") == "ok")
        sid = got["task"]["struct_id"]

        main.BUSY.set()
        payload = {"geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
                   "n_atoms": 2, "wfn_b64": "AA=="}
        busy = main.submit_result(main.TaskResult(worker_id=wid, session=sess,
                                                  struct_id=sid,
                                                  success=True, data=payload))
        check("submit на занятом мастере → busy", busy.get("status") == "busy", busy)
        check("задача НЕ вернулась в очередь сразу",
              all(t["struct_id"] != sid for t in main.pending_tasks))
        check("задача остаётся за воркером до таймаута",
              sid in main.active_tasks)
        main._expire_tasks(now=main.active_tasks[sid]["time_sent"]
                               + main._task_timeout() + 1)
        check("по таймауту задача вернулась в очередь",
              any(t["struct_id"] == sid for t in main.pending_tasks))

        main.verify_state_payload = lambda cfg, task, payload, issued=None: {
            "valid": True, "energy_ev": -1.0, "gap": 0.0, "mag": 0.0,
            "state": {"scf": {"steps": 1}}}
        main._persist_verified_state = lambda t, p, v, block_info=None: "accepted"
        main._reset_ready_quota()

        # ready-результат — работа для головы мастера: (prev_id, nonce)
        issued_r = main._issue_geometry({"n_atoms": 2, "dim": 3, "nonce": 31337})
        ready_payload = {"geometry": issued_r["coords"],
                         "species": issued_r["species"],
                         "n_atoms": 2, "dim": 3,
                         "prev_id": issued_r["prev_id"], "nonce": 31337,
                         "wfn_b64": "AA=="}
        main.BUSY.clear()
        ready = main.submit_result(main.TaskResult(
            worker_id=wid, session=sess, struct_id="b_ready", success=True,
            data=ready_payload, ready=True))
        check("ready-результат принимается без задачи",
              ready.get("status") == "accepted", ready)

        main.BUSY.set()
        ready_busy = main.submit_result(main.TaskResult(
            worker_id=wid, session=sess, struct_id="b_ready2", success=True,
            data=ready_payload, ready=True))
        check("ready на занятом мастере → busy",
              ready_busy.get("status") == "busy", ready_busy)
        main.BUSY.clear()
        main._reset_ready_quota()

        try:
            main.work_height(element="Fe")
            check("work/height: чужой элемент отклонён", False)
        except Exception:
            check("work/height: чужой элемент отклонён", True)
        before = len(main.registered_workers)
        wh = main.work_height(element="Cu")
        check("work/height — справка, воркера не регистрирует",
              wh.get("status") == "ok" and "worker_id" not in wh
              and len(main.registered_workers) == before, wh)

        # --- воркер: занятый мастер → другой мастер ---
        class R:
            def __init__(self, code, data):
                self.status_code = code
                self._d = data

            def json(self):
                return self._d

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

        seen = []

        def fake_get(url, params=None, timeout=None):
            seen.append(("get", url))
            if "p2" in url:
                if url.endswith("/register-challenge"):
                    return R(200, {"nonce": "11" * 32})
                return R(200, {"status": "ok", "height": 0, "element": "Cu"})
            return R(200, {"status": "busy"})

        def fake_post(url, json=None, timeout=None):
            seen.append(("post", url))
            if "p2" in url:
                if url.endswith("/register-worker"):
                    return R(200, {"status": "ok", "worker_id": "w2",
                                   "session": "sess_w2"})
                assert (json or {}).get("session") == "sess_w2", json
                return R(200, {"status": "accepted", "kept": "accepted"})
            return R(200, {"status": "busy"})

        worker.requests.get, worker.requests.post = fake_get, fake_post
        target, res = worker.deliver_result(
            ["http://p1", "http://p2"], "http://p1", "Cu", "w", "h", "sid", {},
            logging.getLogger("busy_test"), attempts=1)
        check("воркер уносит работу другому мастеру",
              target == "http://p2" and res.get("status") == "accepted", target)
    finally:
        main.verify_state_payload, main._persist_verified_state = old_verify, old_persist
        main._reset_ready_quota()
        worker.requests.get, worker.requests.post = old_get, old_post
        main.BUSY.clear()
        main.active_tasks.clear()
        main.pending_tasks.clear()
        main.CONFIG.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)


def lease_tests():
    print("\n--- 26. Lease: пропавший воркер не держит задачу ---")
    tmp = tempfile.mkdtemp(prefix="lease_")
    saved = {k: main.CONFIG.get(k)
             for k in ("output_dir", "element", "failed_ids_file")}
    try:
        main.CONFIG.update({"output_dir": tmp, "element": "Cu",
                            "failed_ids_file": os.path.join(tmp, "failed.txt")})
        main.pending_tasks.clear()
        main.active_tasks.clear()
        main.registered_workers.clear()
        main.worker_by_pubkey.clear()
        reg_l = main._register_worker("lease_t", "hash_lease", None, verified=True)
        wid, sess = reg_l["worker_id"], reg_l["session"]

        main.pending_tasks.append({"struct_id": "L1", "task_index": 0,
                                   "n_atoms": 2, "dim": 3, "attempts": 0})
        main.get_task(wid, session=sess)
        t0 = main.active_tasks["L1"]["time_sent"]

        # воркер шлёт heartbeat → задача не снимается
        main.registered_workers[wid]["last_seen"] = t0 + 100
        main._expire_tasks(now=t0 + 100 + main.WORKER_LEASE_SEC - 1)
        check("lease: живой воркер держит задачу", "L1" in main.active_tasks)

        # воркер замолчал → задача вернулась в очередь через 3 минуты, а не через task_timeout
        main._expire_tasks(now=t0 + 100 + main.WORKER_LEASE_SEC + 1)
        check("lease: задача пропавшего воркера вернулась в очередь",
              "L1" not in main.active_tasks
              and any(t["struct_id"] == "L1" for t in main.pending_tasks))

        # жёсткий таймаут срабатывает даже при свежем heartbeat
        main.pending_tasks.clear()
        main.pending_tasks.append({"struct_id": "L2", "task_index": 0,
                                   "n_atoms": 2, "dim": 3, "attempts": 0})
        main.get_task(wid, session=sess)
        t1 = main.active_tasks["L2"]["time_sent"]
        late = t1 + main._task_timeout() + 1
        main.registered_workers[wid]["last_seen"] = late
        main._expire_tasks(now=late)
        check("lease: жёсткий таймаут тоже возвращает задачу", "L2" not in main.active_tasks)

        # перезапуск воркера (тот же pubkey) возвращает его текущую задачу
        main.pending_tasks.clear()
        main.pending_tasks.append({"struct_id": "L3", "task_index": 0,
                                   "n_atoms": 2, "dim": 3, "attempts": 0})
        main.get_task(wid, session=sess)
        main._register_worker("lease_t", "hash_lease", None, verified=True)
        check("lease: перерегистрация воркера возвращает его задачу",
              "L3" not in main.active_tasks
              and any(t["struct_id"] == "L3" for t in main.pending_tasks))

        # забывание мёртвого воркера не трогает живого
        main.registered_workers[wid]["last_seen"] = time.time() - main.WORKER_FORGET_SEC - 1
        forgotten = main._forget_dead_workers()
        check("lease: мёртвый воркер забыт", wid in forgotten
              and wid not in main.registered_workers)
    finally:
        main.pending_tasks.clear()
        main.active_tasks.clear()
        main.registered_workers.clear()
        main.worker_by_pubkey.clear()
        main.CONFIG.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)


def master_geometry_tests():
    print("\n--- 27. Мастер — источник геометрии ---")
    import tempfile as _tempfile

    tmp = _tempfile.mkdtemp(prefix="geom_")
    saved = {k: main.CONFIG.get(k) for k in ("output_dir", "element")}
    try:
        main.CONFIG.update({"output_dir": tmp, "element": "Cu"})
        main.pending_tasks.clear()
        main.active_tasks.clear()
        main.registered_workers.clear()
        main.worker_by_pubkey.clear()
        reg_g = main._register_worker("geom_t", "hash_geom", None)
        wid, sess = reg_g["worker_id"], reg_g["session"]

        task = {"struct_id": "n2_g1", "task_index": 0, "n_atoms": 2, "dim": 3,
                "seed": 777, "attempts": 0}
        main.pending_tasks.append(dict(task))
        got = main.get_task(wid, session=sess)
        payload = got.get("task") or {}
        check("задание содержит готовую геометрию и сорт атомов",
              len(payload.get("geometry") or []) == 2
              and len(payload.get("species") or []) == 2, str(payload.get("struct_id")))
        check("мастер хранит эталон выданной геометрии",
              payload.get("struct_id") in main.active_tasks
              and len(main.active_tasks[payload["struct_id"]]["geometry"]) == 2)

        # детерминизм: (prev_id, nonce) → та же геометрия
        fixed = dict(task, nonce=12345)
        a = main._issue_geometry(dict(fixed))
        b = main._issue_geometry(dict(fixed))
        check("геометрия детерминирована по (prev_id, nonce)",
              a["coords"] == b["coords"] and a["species"] == b["species"])

        g0 = main._issue_geometry(dict(task, nonce=4242))
        check("геометрия задания без шаблонов",
              "template_sha" not in g0 and "template_obj" not in g0)

        # --- админ-API: локально или по токену ---
        saved_token = main.ADMIN_TOKEN
        try:
            main.ADMIN_TOKEN = ""
            check("админка: локально без токена",
                  main._admin_allowed("127.0.0.1"))
            check("админка: снаружи без токена закрыта",
                  not main._admin_allowed("203.0.113.5"))
            main.ADMIN_TOKEN = "s3cret-token"
            check("админка: снаружи по верному токену",
                  main._admin_allowed("203.0.113.5", "s3cret-token"))
            check("админка: неверный токен отклонён",
                  not main._admin_allowed("203.0.113.5", "bad"))
        finally:
            main.ADMIN_TOKEN = saved_token
        check("форк: распознаётся сообщение v5",
              main._is_fork_conflict(
                  "prev_id не совпадает с головой локальной цепочки"))
        check("форк: распознаётся старое сообщение v1",
              main._is_fork_conflict("prev_sha256 не совпадает"))
        check("публичный /peers отдаёт список",
              isinstance(main.peers_public().get("peers"), list))

        # --- H3: потолок n_atoms до аллокаций и блочная проверка расстояний ---
        over = main.verify_state_payload(
            main.CONFIG,
            {"struct_id": "big", "n_atoms": main.MAX_TASK_ATOMS + 1, "dim": 3},
            {"geometry": [[0.0, 0.0, 0.0]]})
        check("n_atoms сверх потолка отклоняется до аллокаций",
              not over["valid"] and "диапазона" in over["reason"],
              over["reason"])
        line = np.zeros((3000, 3))
        line[:, 0] = np.arange(3000) * 2.0
        ok_line, why_line = main._check_geometry(line, 3000, main.CONFIG)
        check("геометрия 3000 атомов проверяется блочно", ok_line, why_line)
    finally:
        main.pending_tasks.clear()
        main.active_tasks.clear()
        main.registered_workers.clear()
        main.worker_by_pubkey.clear()
        main.CONFIG.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)


def ready_binding_tests():
    print("\n--- 31. ready: привязка к (prev_id, nonce), блок и квота ---")
    tmp = tempfile.mkdtemp(prefix="ready_")
    saved = {k: main.CONFIG.get(k) for k in
             ("output_dir", "element", "frames_dir", "verified_ids_file")}
    old_write, old_verify = main._write_chain_block, main.verify_state_payload
    old_quota = main.READY_QUOTA_PER_HOUR
    blocks = []
    try:
        main.CONFIG.update({"output_dir": tmp, "element": "Cu",
                            "frames_dir": os.path.join(tmp, "frames"),
                            "verified_ids_file": os.path.join(tmp, "verified.txt")})
        main.registered_workers.clear()
        main.worker_by_pubkey.clear()
        main.completed_tasks.clear()
        main.active_tasks.clear()
        main.pending_tasks.clear()
        main.BUSY.clear()
        main._reset_ready_quota()
        reg = main._register_worker("ready_t", "hash_ready", None)
        wid, sess = reg["worker_id"], reg["session"]

        issued = main._issue_geometry({"n_atoms": 2, "dim": 3, "nonce": 424242})
        good = {"geometry": issued["coords"], "species": issued["species"],
                "n_atoms": 2, "dim": 3, "prev_id": issued["prev_id"],
                "nonce": 424242, "wfn_b64": "AA=="}

        def submit(sid, data):
            return main.submit_result(main.TaskResult(
                worker_id=wid, session=sess, struct_id=sid, success=True,
                data=data, ready=True))

        r = submit("r1", {"geometry": issued["coords"], "n_atoms": 2})
        check("ready без prev_id/nonce отклонён", r.get("status") == "rejected", r)

        r = submit("r2", dict(good, prev_id="11" * 32))
        check("ready для чужой головы отклонён",
              r.get("status") == "rejected" and "голов" in str(r.get("reason")), r)

        r = submit("r3", dict(good, geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 9.0]]))
        check("ready с геометрией не из seed отклонён",
              r.get("status") == "rejected" and "seed" in str(r.get("reason")), r)

        r = submit("r4", dict(good, n_atoms=55))
        check("ready с n_atoms не по расписанию отклонён",
              r.get("status") == "rejected", r)

        # правильный кадр: прогон CP2K заглушён, запись блока — под запись
        main.verify_state_payload = lambda cfg, task, payload, issued=None: {
            "valid": True, "energy_ev": -1.0, "gap": 0.0, "mag": 0.0,
            "state": {"scf": {"steps": 1}}}

        def fake_write(task_data, payload, verdict, info=None):
            blocks.append((task_data.get("_nonce"), task_data.get("dim"),
                           payload.get("seed")))
            if info is not None:
                info["block"] = os.path.join(tmp, "000001_Cu_2.zip")
                info["reward_paid"] = True
            return info["block"] if info is not None else None

        main._write_chain_block = fake_write
        main.READY_QUOTA_PER_HOUR = 5

        r = submit("r5", good)
        check("ready с привязкой к голове принят",
              r.get("status") == "accepted", r)
        check("в блок ушёл nonce из ready-результата",
              bool(blocks) and blocks[-1][0] == 424242, str(blocks[-1:]))
        check("эталон геометрии — из seed(prev_id, nonce)",
              bool(blocks) and blocks[-1][2] == issued["seed"])
        check("ответ сообщает о блоке и награде",
              bool(r.get("block")) and r.get("reward_paid") is True, r)

        n_blocks = len(blocks)
        r = submit("r6", good)
        check("повтор (prev_id, nonce) не тратит прогон",
              r.get("status") == "accepted" and len(blocks) == n_blocks, r)

        main.READY_QUOTA_PER_HOUR = 2
        main._reset_ready_quota()
        main.completed_tasks.clear()
        last = None
        for i, nonce in enumerate((5, 6, 7), start=1):
            iss = main._issue_geometry({"n_atoms": 2, "dim": 3, "nonce": nonce})
            last = submit(f"q{i}", {"geometry": iss["coords"],
                                    "species": iss["species"], "n_atoms": 2,
                                    "dim": 3, "prev_id": iss["prev_id"],
                                    "nonce": nonce, "wfn_b64": "AA=="})
        check("квота ready-приёмок соблюдается",
              last.get("status") == "rejected" and "квота" in str(last.get("reason")),
              last)
    finally:
        main._write_chain_block = old_write
        main.verify_state_payload = old_verify
        main.READY_QUOTA_PER_HOUR = old_quota
        main._reset_ready_quota()
        main.completed_tasks.clear()
        main.active_tasks.clear()
        main.pending_tasks.clear()
        main.registered_workers.clear()
        main.worker_by_pubkey.clear()
        main.CONFIG.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)


def money_int_tests():
    print("\n--- 33. Деньги: только целые, без эпсилонов ---")
    tmp = tempfile.mkdtemp(prefix="money_")
    try:
        owner = "b" * 64
        # UTXO с огромной суммой: во float 10**30 и 10**30+1 — одно и то же число
        huge = 10 ** 30
        utxo = {("a" * 64, 0): {"value": huge, "address": owner,
                                "height": 1, "coinbase": False}}

        def make(to_value):
            tx = {"version": 1, "type": "transfer",
                  "inputs": [{"txid": "a" * 64, "vout": 0}],
                  "outputs": [{"value": to_value, "address": "c" * 64}]}
            tx["txid"] = chain.txid_of(tx)
            return tx

        ok, problems = chain.validate_tx(make(huge + 1), tmp, height=10,
                                         confirmations=0, utxo=dict(utxo))
        check("вход 10^30 < выход 10^30+1 отклонён (без эпсилона)",
              not ok and any("сумма входов" in p for p in problems), problems)

        ok, problems = chain.validate_tx(make(huge), tmp, height=10,
                                         confirmations=0, utxo=dict(utxo))
        check("равные огромные суммы не ловятся как расхождение",
              not any("сумма входов" in p for p in problems), problems)

        ok, problems = chain.validate_tx(make(1.5), tmp, height=10,
                                         confirmations=0, utxo=dict(utxo))
        check("дробная сумма в выходе отклонена",
              not ok and any("не целое" in p for p in problems), problems)

        fee_cb = chain.tx_fee(chain.make_coinbase(1, owner), {})
        check("tx_fee(coinbase) — int 0", type(fee_cb) is int and fee_cb == 0, repr(fee_cb))

        fee, _ = chain._check_txs([], {}, "t")
        check("_check_txs возвращает int", type(fee) is int, repr(fee))

        # v6: эмиссия пропорциональна числу атомов, делится в coinbase 50/50
        check("субсидия пропорциональна n",
              chain.subsidy_for(2) == 50_000_000 and chain.subsidy_for(5) == 125_000_000,
              (chain.subsidy_for(2), chain.subsidy_for(5)))
        check("coinbase делится 50/50",
              chain.coinbase_split(2) == (25_000_000, 25_000_000),
              chain.coinbase_split(2))
        check("допуск энергии растёт с n",
              chain.consensus.peer_tol_energy_ev(2) == 0.02
              and chain.consensus.peer_tol_energy_ev(2000) > 0.02,
              chain.consensus.peer_tol_energy_ev(2000))

        # build_transfer: ровно хватает на сумму+комиссию, без 1e-9
        conn = wallet_store.open_db(os.path.join(tmp, "keys.db"))
        wallet = wallet_store.create_wallet(conn, "m", "root")
        addr = wallet.latest_key()["address"]
        check("баланс пустого кошелька — int",
              all(type(v) is int for r in chain.balances(tmp, {addr}).values()
                  for v in r.values()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def reward_format_tests():
    print("\n--- 32. Блок: эмиссия делится в coinbase (мастер + воркер) ---")
    root = tempfile.mkdtemp(prefix="ready_wb_")
    tmp = os.path.join(root, "Cu")      # система = имя каталога цепочки
    os.makedirs(tmp, exist_ok=True)
    saved = {k: main.CONFIG.get(k) for k in ("output_dir", "element", "wallet")}
    try:
        main.CONFIG.update({"output_dir": tmp, "element": "Cu"})
        wfn = os.path.join(root, "state.wfn")
        make_wfn(wfn)
        with open(wfn, "rb") as f:
            wfn_b64 = base64.b64encode(f.read()).decode("ascii")

        issued = main._issue_geometry({"n_atoms": 2, "dim": 3, "nonce": 7})
        addr = "c" * 64
        main.registered_workers["worker_x"] = {"pubkey_hash": addr, "name": "x",
                                               "sessions": []}

        class W:
            def latest_key(self):
                return {"address": addr}

            def addresses(self):
                return [addr]

        main.CONFIG["wallet"] = W()
        payload = {"wfn_b64": wfn_b64, "geometry": issued["coords"],
                   "species": issued["species"], "worker_id": "worker_x",
                   "seed": issued["seed"]}
        verdict = {"energy_ev": -1.0,
                   "forces_ev_ang": [[0.0, 0.0, 0.0]] * 2,
                   "gap": 0.1, "mag": 0.0}
        task_data = {"struct_id": "wb1", "n_atoms": 2, "dim": 3,
                     "_nonce": 7, "attempts": 0}
        info = {}
        path = main._REAL_WRITE_CHAIN_BLOCK(task_data, payload, verdict, info=info)
        # v6: награда больше не зависит от того, есть ли у мастера монеты —
        # она часть эмиссии этого блока (второй выход coinbase).
        check("эмиссия блока делится в coinbase (мастер + воркер)",
              bool(path) and info.get("emission") == chain.subsidy_for(2)
              and info.get("worker_share") == chain.coinbase_split(2)[1]
              and info.get("reward_paid") is True, info)
    finally:
        main.registered_workers.pop("worker_x", None)
        main.CONFIG.update(saved)
        shutil.rmtree(root, ignore_errors=True)


def build_tests():
    print("\n--- 28. Пины сборки CP2K (Dockerfile/Makefile ↔ константы) ---")
    with open(os.path.join(PROJECT_ROOT, "Dockerfile"), encoding="utf-8") as f:
        dockerfile = f.read()
    with open(os.path.join(PROJECT_ROOT, "Makefile"), encoding="utf-8") as f:
        makefile = f.read()
    check("CPU-сборка закреплена в Dockerfile",
          cp2k_runner.CP2K_BUILD_CPU in dockerfile)
    check("GPU-сборка закреплена в Makefile",
          cp2k_runner.CP2K_BUILD_GPU in makefile)
    check("обе сборки содержат digest",
          all("@sha256:" in b for b in cp2k_runner.CP2K_BUILD_EXPECTED))
    check("отпечаток неизвестной сборки не пропускается",
          not cp2k_runner.cp2k_build_allowed("cp2k/cp2k:2024.3_other@sha256:deadbeef"))


def rfc8032_tests():
    print("\n--- 29. Ed25519: векторы RFC 8032 и малые порядки ---")
    vectors = [
        ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
         "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
         "",
         "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
         "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
        ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
         "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
         "72",
         "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
         "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
        ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
         "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
         "af82",
         "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
         "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
    ]
    for i, (sk_hex, pk_hex, msg_hex, sig_hex) in enumerate(vectors, 1):
        seed = bytes.fromhex(sk_hex)
        msg = bytes.fromhex(msg_hex)
        pub = crypto.public_key_from_seed(seed)
        sig = crypto.sign(seed, msg)
        check(f"RFC 8032 вектор {i}: публичный ключ", pub.hex() == pk_hex, pub.hex()[:16])
        check(f"RFC 8032 вектор {i}: подпись", sig.hex() == sig_hex, sig.hex()[:16])
        check(f"RFC 8032 вектор {i}: проверка", crypto.verify(pub, msg, sig))
        check(f"RFC 8032 вектор {i}: другое сообщение отклонено",
              not crypto.verify(pub, msg + b"x", sig))

    # identity как публичный ключ: при A=identity уравнение S·B = R + h·A
    # сводится к S·B = R, поэтому подпись проходит арифметику — её обязан
    # отбить именно фильтр малых порядков (раньше он не срабатывал).
    small = bytes([1]) + b"\x00" * 31
    S = 1
    R = crypto._encode_point(crypto._scalarmult(crypto._B, S))
    forged = R + S.to_bytes(32, "little")
    check("identity распознаётся как точка малого порядка",
          crypto._is_small_order(small))
    check("identity как публичный ключ: подпись не проходит",
          not crypto.verify(small, b"msg", forged))
    check("identity с любым S не проходит",
          all(not crypto.verify(
                  small, b"msg",
                  crypto._encode_point(crypto._scalarmult(crypto._B, s))
                  + s.to_bytes(32, "little"))
              for s in (2, 5, 12345)))

    # неканоническая S: S + q при том же R не должна проходить
    seed = bytes.fromhex(vectors[0][0])
    msg = b""
    sig = crypto.sign(seed, msg)
    pub = crypto.public_key_from_seed(seed)
    S = int.from_bytes(sig[32:], "little")
    q = 2 ** 252 + 27742317777372353535851937790883648493
    if S + q < 2 ** 256:
        malleated = sig[:32] + (S + q).to_bytes(32, "little")
        check("неканоническая S (S+q) отклонена", not crypto.verify(pub, msg, malleated))


def mempool_policy_tests():
    print("\n--- 30. Мемпул: вытеснение по комиссии и отбор в блок ---")
    tmp = tempfile.mkdtemp(prefix="mempool_")
    old_dir = main.CONFIG.get("output_dir")
    old_max = main.MEMPOOL_MAX_TX
    old_block = chain.MAX_TXS_PER_BLOCK
    try:
        main.CONFIG["output_dir"] = tmp
        main.MEMPOOL.clear()
        main.MEMPOOL_MAX_TX = 3

        seed = os.urandom(32)
        pub = crypto.public_key_from_seed(seed)
        addr = crypto.address_from_public(pub)

        state = os.path.join(tmp, "state.wfn")
        with open(state, "wb") as f:
            f.write(b"mempool-test" * 50)
        sha, rel = chain.store_blob(tmp, state)
        dft = {"energy_ev": -42.0,
               "forces_ev_ang": [[0.0, 0.0, 0.0]] * 2,
               "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]}
        coinbases = []
        for seq in range(1, 7):
            cb = chain.make_coinbase(seq, addr, chain.COINBASE_VALUE)
            chain.write_block(tmp, "Cu", 2, sha, rel, [cb],
                              dft=block_dft(tmp),
                              cp2k_version="2024.3",
                              cp2k_build=cp2k_runner.CP2K_BUILD_CPU,
                              cp2k_params=dict(cp2k_runner.CP2K_PARAMS))
            coinbases.append(cb)

        def make_tx(cb, fee):
            tx = chain.make_transfer(
                [{"txid": chain.txid_of(cb), "vout": 0}],
                [{"value": chain.COINBASE_VALUE - fee, "address": addr}])
            tid = chain.txid_of(tx)
            for i in tx["inputs"]:
                i["public_key"] = pub.hex()
                i["signature"] = crypto.sign(seed, bytes.fromhex(tid)).hex()
            tx["txid"] = chain.txid_of(tx)
            return tx

        def submit(tx):
            return main.tx_submit(main.TxSubmit(tx=tx))

        t100 = make_tx(coinbases[0], 100_000)
        t200 = make_tx(coinbases[1], 200_000)
        t300 = make_tx(coinbases[2], 300_000)
        check("дешёвая tx принята", submit(t100).get("status") == "accepted")
        check("средняя tx принята", submit(t200).get("status") == "accepted")
        check("дорогая tx принята", submit(t300).get("status") == "accepted")

        too_cheap = submit(make_tx(coinbases[3], 50_000))
        check("полный мемпул не берёт tx с комиссией ниже минимальной",
              too_cheap.get("status") == "rejected" and "полон" in str(too_cheap.get("reason")),
              str(too_cheap))

        t400 = make_tx(coinbases[3], 400_000)
        check("дорогая tx вытесняет дешёвую", submit(t400).get("status") == "accepted")
        check("дешёвая (100k) вытеснена",
              t100["txid"] not in main.MEMPOOL and len(main.MEMPOOL) == 3,
              f"в мемпуле {len(main.MEMPOOL)}")

        low_replace = make_tx(coinbases[2], 250_000)
        r = submit(low_replace)
        check("конфликт дешевле существующего отклонён",
              r.get("status") == "rejected" and "дороже" in str(r.get("reason")), str(r))
        high_replace = make_tx(coinbases[2], 350_000)
        check("конфликт дороже существующего заменяет его",
              submit(high_replace).get("status") == "accepted"
              and t300["txid"] not in main.MEMPOOL)

        # отбор в блок: дорогие первыми, лимит переводов на блок
        chain.MAX_TXS_PER_BLOCK = 1
        utxo = chain.utxo_set(tmp)
        included, fee = main._drain_mempool(utxo, 7, 1)
        check("в блок попала самая дорогая tx", len(included) == 1
              and included[0]["txid"] == t400["txid"],
              f"включено {len(included)}")
        check("остальные остались в мемпуле для следующего блока",
              len(main.MEMPOOL) == 2, f"осталось {len(main.MEMPOOL)}")
        check("комиссия блока посчитана", fee == 400_000, fee)
    finally:
        chain.MAX_TXS_PER_BLOCK = old_block
        main.MEMPOOL_MAX_TX = old_max
        main.MEMPOOL.clear()
        if old_dir is not None:
            main.CONFIG["output_dir"] = old_dir
        shutil.rmtree(tmp, ignore_errors=True)


def peer_tests():
    print("\n--- 23. Самая длинная цепочка (catch-up перед майнингом) ---")
    import types

    tmp = tempfile.mkdtemp(prefix="catchup_")
    saved = {k: main.CONFIG.get(k) for k in ("output_dir", "peers", "element")}
    old_head, old_sync, old_get = main._local_head, main.sync_from_peer, main.requests.get
    old_bad = dict(main._BAD_PEERS)
    main._BAD_PEERS.clear()
    try:
        main.CONFIG.update({"output_dir": tmp, "peers": ["http://peer1"], "element": "Cu"})
        state = {"h": 0}
        main._local_head = lambda: (state["h"], "0" * 64)
        main.requests.get = lambda *a, **k: types.SimpleNamespace(
            status_code=200, raise_for_status=lambda: None,
            json=lambda: {"element": "Cu", "height": 2})

        def good_sync(peer):
            state["h"] = 2
            return {"ok": True, "accepted": 2, "errors": []}

        main.sync_from_peer = good_sync
        res = main.catch_up_chain()
        check("догоняет цепочку до высоты пира",
              res["ok"] and res["height"] == 2 and res["accepted"] == 2, res)

        state["h"] = 0
        main._BAD_PEERS.clear()
        main.sync_from_peer = lambda peer: {"ok": False, "accepted": 0, "errors": ["boom"]}
        res2 = main.catch_up_chain()
        check("плохой пир исключается, работа продолжается",
              res2["ok"] and "http://peer1" in main._BAD_PEERS, res2)

        check("без раундов догон не считается успешным",
              main.catch_up_chain(max_rounds=0)["ok"] is False)
    finally:
        main._local_head, main.sync_from_peer, main.requests.get = old_head, old_sync, old_get
        main._BAD_PEERS.clear()
        main._BAD_PEERS.update(old_bad)
        main.CONFIG.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)


def worker_registry_tests():
    print("\n--- 21. Регистрация по ключу и одна задача на воркера ---")
    main.registered_workers.clear()
    main.worker_by_pubkey.clear()
    main.active_tasks.clear()
    main.pending_tasks.clear()
    main.PAUSED = False

    r1 = main._register_worker("alpha", "hash_a", "pub_a", verified=True)
    r2 = main._register_worker("beta", "hash_b", "pub_b", verified=True)
    check("разные ключи — разные воркеры", r1["worker_id"] != r2["worker_id"])
    r1b = main._register_worker("alpha", "hash_a", "pub_a", verified=True)
    check("тот же ключ — тот же worker_id",
          r1b["worker_id"] == r1["worker_id"] and r1b["known"])
    check("регистрация выдаёт session",
          bool(r1.get("session")) and bool(r1b.get("session"))
          and r1["session"] != r1b["session"], "старая сессия остаётся живой")

    main.pending_tasks.append({"struct_id": "reg_0001", "task_index": 0,
                               "n_atoms": 2, "dim": 3, "attempts": 0})
    main.pending_tasks.append({"struct_id": "reg_0002", "task_index": 1,
                               "n_atoms": 2, "dim": 3, "attempts": 0})

    got1 = main.get_task(r1["worker_id"], session=r1["session"])
    check("первая задача выдана", got1.get("status") == "ok")
    busy = main.get_task(r1["worker_id"], session=r1["session"])
    check("вторую задачу воркер не получает", busy.get("status") == "busy",
          busy.get("struct_id"))
    got2 = main.get_task(r2["worker_id"], session=r2["session"])
    check("другой уникальный воркер получает задачу", got2.get("status") == "ok")

    # H2: worker_id публичен (/queue-status), поэтому пропуск — только session
    try:
        main.get_task(r2["worker_id"], session=r1["session"])
        check("чужой session отклонён", False)
    except Exception as e:
        check("чужой session отклонён", getattr(e, "status_code", None) == 403, str(e))
    try:
        main.get_task(r2["worker_id"])
        check("без session задача не выдаётся", False)
    except Exception as e:
        check("без session задача не выдаётся",
              getattr(e, "status_code", None) == 403, str(e))
    try:
        main.submit_result(main.TaskResult(worker_id=r2["worker_id"],
                                           struct_id="reg_0002",
                                           success=False, data={}))
        check("без session результат не принимается", False)
    except Exception as e:
        # повторная попытка без session может упереться в троттлинг (429) —
        # это тоже отказ, поэтому принимаем 403 и 429
        check("без session результат не принимается",
              getattr(e, "status_code", None) in (403, 429), str(e))

    hb = main.heartbeat(r2["worker_id"], session="подделка")
    check("чужой heartbeat не продлевает lease", hb.get("status") == "unauthorized", hb)

    main.PAUSED = True
    check("пауза блокирует выдачу",
          main.get_task(r1["worker_id"], session=r1["session"]).get("status") == "paused")
    main.PAUSED = False

    main.pending_tasks.clear()
    main.active_tasks.clear()
    main.registered_workers.clear()
    main.worker_by_pubkey.clear()

    # --- регистрация в HTTP-слое требует подписи ключа ---
    try:
        main.register_worker(main.WorkerRegistration(name="anon"))
        check("регистрация без подписи отклонена", False)
    except Exception as e:
        check("регистрация без подписи отклонена",
              getattr(e, "status_code", None) == 403, str(e))

    try:
        main.register_worker_get()
        check("GET /register-worker закрыт (410)", False)
    except Exception as e:
        check("GET /register-worker закрыт (410)",
              getattr(e, "status_code", None) == 410, str(e))

    seed = b"\x11" * 32
    pub = crypto.public_key_from_seed(seed)
    addr = crypto.address_from_public(pub)
    nonce = main._issue_registration_nonce()
    sig = crypto.sign(seed, b"cp2k-mace-register:" + bytes.fromhex(nonce))
    signed = main.register_worker(main.WorkerRegistration(
        name="signed", pubkey_hash=addr, public_key=pub.hex(),
        nonce=nonce, signature=sig.hex()))
    check("регистрация с подписью принята",
          signed.get("status") == "ok" and bool(signed.get("session")), str(signed))
    try:
        main.register_worker(main.WorkerRegistration(
            name="signed", pubkey_hash=addr, public_key=pub.hex(),
            nonce=nonce, signature=sig.hex()))
        check("повтор nonce (replay) отклонён", False)
    except Exception as e:
        check("повтор nonce (replay) отклонён",
              getattr(e, "status_code", None) == 403, str(e))

    # --- потолок числа воркеров ---
    saved_cap = main.MAX_REGISTERED_WORKERS
    main.registered_workers.clear()
    main.worker_by_pubkey.clear()
    try:
        main.MAX_REGISTERED_WORKERS = 2
        main._register_worker("c1", "cap_a", "p1", verified=True)
        main._register_worker("c2", "cap_b", "p2", verified=True)
        try:
            main._register_worker("c3", "cap_c", "p3", verified=True)
            check("потолок воркеров соблюдается", False)
        except Exception as e:
            check("потолок воркеров соблюдается",
                  getattr(e, "status_code", None) == 503, str(e))
        check("при отказе старые воркеры целы", len(main.registered_workers) == 2)
    finally:
        main.MAX_REGISTERED_WORKERS = saved_cap
        main.registered_workers.clear()
        main.worker_by_pubkey.clear()

    # --- лимит регистраций по IP ---
    main._reset_reg_rate()
    taken = sum(1 for _ in range(main.REG_LIMIT_PER_IP) if main._reg_rate_ok("10.0.0.1"))
    check("лимит регистраций по IP",
          taken == main.REG_LIMIT_PER_IP
          and not main._reg_rate_ok("10.0.0.1")
          and main._reg_rate_ok("10.0.0.2"),
          f"взято {taken} из {main.REG_LIMIT_PER_IP}")
    main._reset_reg_rate()


def run():
    prepare_env()
    sys.path.insert(0, ENV)

    global main, cp2k_runner, worker, structure_generator, identity, chain
    global crypto, wallet_store, wallet
    import cp2k_runner
    import structure_generator
    import main
    import worker
    import identity
    import chain
    import crypto
    import wallet_store
    import wallet

    # Цепочка блоков проверяется отдельно (chain_tests); в приёмочных тестах
    # блоки не пишем, чтобы не трогать боевые results/. Настоящую реализацию
    # сохраняем: её проверяет тест формата награды.
    main._REAL_WRITE_CHAIN_BLOCK = main._write_chain_block
    main._write_chain_block = lambda *a, **k: None
    # Rate limit (5 с на выдачу/сдачу) — политика HTTP-слоя; в юнит-тестах
    # он мешает, поэтому выключаем и проверяем отдельно.
    main.RATE_LIMIT_SEC = 0

    # Все файловые артефакты тестов — во временном каталоге, боевые results/ не трогаем
    scratch = tempfile.mkdtemp(prefix="test_results_")
    saved = {k: main.CONFIG.get(k) for k in (
        "output_dir", "frames_dir", "verified_ids_file",
        "failed_ids_file", "output_all_xyz_name")}
    main.CONFIG.update({
        "output_dir": scratch,
        "frames_dir": os.path.join(scratch, "frames"),
        "verified_ids_file": os.path.join(scratch, "verified_ids.txt"),
        "failed_ids_file": os.path.join(scratch, "failed_ids.txt"),
        "output_all_xyz_name": "all.xyz",
    })
    try:
        _run_tests()
    finally:
        main.CONFIG.update(saved)
        shutil.rmtree(scratch, ignore_errors=True)


def _run_tests():
    # CP2K не нужен: приёмку состояния проверяет отдельный тест master_state_tests
    worker_tests()
    queue_tests()
    restart_tests()
    master_state_tests()
    identity_tests()
    chain_tests()
    wallet_tests()
    worker_registry_tests()
    peer_tests()
    busy_tests()
    lease_tests()
    master_geometry_tests()
    ready_binding_tests()
    money_int_tests()
    reward_format_tests()
    build_tests()
    rfc8032_tests()
    mempool_policy_tests()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 60)
    print(f"ИТОГО: {passed}/{len(RESULTS)} проверок пройдено")
    failed = [n for n, ok, _ in RESULTS if not ok]
    if failed:
        print("НЕ ПРОШЛИ: " + "; ".join(failed))
    print("=" * 60)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(run())
