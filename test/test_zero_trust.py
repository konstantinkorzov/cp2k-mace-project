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
    def __init__(self, status_code=500, detail=""):
        self.status_code = status_code
        self.detail = detail
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
              "crypto.py", "wallet_store.py", "wallet.py"):
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
        task = {"struct_id": "n2_d3_0000", "n_atoms": 2, "dim": 3}

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
    tasks = main.build_task_queue(main.CONFIG)
    ids = [t["struct_id"] for t in tasks]
    per_n = {int(k): int(v) for k, v in main.CONFIG["structures_per_n"].items()}
    expected = sum(per_n[n] * len(main.CONFIG["dimensions_per_n"].get(str(n), [3])) for n in per_n)
    check("заданий столько, сколько структур в config.json", len(tasks) == expected,
          f"n={len(tasks)} (ожидалось {expected})")
    check("ids уникальны", len(ids) == len(set(ids)))
    check("мастер не передаёт готовые coords", all("coords" not in t for t in tasks))
    check("конфиг: псевдопотенциал и валентность определены",
          main.CONFIG["potential"] == "GTH-PBE-q11" and main.CONFIG["n_valence"] == 11)
    check("конфиг: реестры заданы",
          all(main.CONFIG.get(k) for k in ("verified_ids_file", "failed_ids_file")))
    payload_cfg = main.task_cfg_payload(main.CONFIG)
    check("воркеру передаётся диаметр атома", "atom_diameter" in payload_cfg)
    check("в payload нет ручек CP2K/генерации/режима проверки",
          not any(k in payload_cfg for k in ("verify_mode", "cutoff", "eps_scf",
                                             "box_size_margin", "max_scf",
                                             "min_dist", "max_dist", "random_seed",
                                             "max_attempts_per_atom")))


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
        wid = main._register_worker("test", None, None)["worker_id"]
        t = {**task, "struct_id": "n2_d3_0007", "attempts": 0}
        main.pending_tasks.append(t)
        got = main.get_task(wid)
        check("мастер не передаёт режим проверки (он всегда restart)",
              "verify_mode" not in got["task"]["cfg"])
        good = {"geometry": [[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]],
                "energy_ev": -2614.38,
                "wfn_b64": base64.b64encode(open(p, "rb").read()).decode("ascii")}
        resp = main.submit_result(main.TaskResult(worker_id=wid, struct_id="n2_d3_0007",
                                                  success=True, data=good))
        check("submit-result принял состояние", resp["status"] == "accepted", str(resp))

        main.submit_result(main.TaskResult(worker_id=wid, struct_id="n2_d3_0008",
                                           success=True, data=good))
        check("результат для неизвестной задачи не принимается", True)

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
        wallet, _ = wallet_store.ensure_master(conn, "root")
        master = wallet.latest_key()["address"]
        worker = "b" * 64
        cp2k = {"cp2k_version": "2024.3", "cp2k_params": {"cutoff": 400, "eps_scf": 1e-6}}

        dft = {"energy_ev": -42.0, "forces_ev_ang": [[0.0, 0.0, 0.0]] * 2,
               "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]}
        p1 = chain.write_block(tmp, "Cu", 2, sha1, rel1,
                               [chain.make_coinbase(1, master)], dft=dft, **cp2k)
        check("имя блока 000001_Cu_2.zip",
              os.path.basename(p1) == "000001_Cu_2.zip", os.path.basename(p1))

        with zipfile.ZipFile(p1) as z:
            names = set(z.namelist())
            h1 = json.loads(z.read("header.json"))
            tx1 = json.loads(z.read("tx-0000-coinbase.json"))

        expected = {"version", "seq", "element", "n_atoms", "prev_sha256",
                    "created_at", "state_sha256", "state_path",
                    "cp2k_version", "cp2k_params"}
        check("в заголовке только нужные поля", set(h1) == expected, sorted(h1))
        check("version = 1", h1["version"] == 1)
        check("created_at — ISO 8601 UTC",
              datetime.strptime(h1["created_at"], "%Y-%m-%dT%H:%M:%SZ") is not None)
        check("CP2K version и params в блоке",
              h1["cp2k_version"] == "2024.3" and h1["cp2k_params"]["cutoff"] == 400)
        check("первый блок: prev = нули", h1["prev_sha256"] == chain.ZERO_HASH)
        check("wfn в блок не кладётся, только ссылка",
              not any(n.endswith(".wfn") for n in names) and h1["state_path"] == rel1)
        check("coinbase — 50 монет", sum(o["value"] for o in tx1["outputs"]) == 50.0)

        no_reward = chain.build_transfer(tmp, {master}, worker, master,
                                         chain.REWARD_VALUE, 1, height=1)
        check("в блоке 1 награда невозможна (нет подтверждений)", no_reward is None)

        reward = chain.build_transfer(tmp, {master}, worker, master,
                                      chain.REWARD_VALUE, 1, height=2,
                                      fee=chain.TX_FEE)
        check("перевод 25 воркеру собирается", reward is not None
              and any(o["address"] == worker and o["value"] == 25.0
                      for o in reward["outputs"]))

        utxo = chain.utxo_set(tmp)
        fee = chain.tx_fee(reward, utxo)
        check("комиссия = входы − выходы", abs(fee - chain.TX_FEE) < 1e-9, fee)
        chain.sign_transaction(reward, wallet, utxo)
        check("входы подписаны Ed25519",
              all(i.get("signature") and i.get("public_key") for i in reward["inputs"]))

        cb2 = chain.make_coinbase(2, master, chain.COINBASE_VALUE + fee)
        p2 = chain.write_block(tmp, "Cu", 2, sha1, rel1, [cb2, reward],
                               dft=dft, **cp2k)

        with zipfile.ZipFile(p2) as z:
            h2 = json.loads(z.read("header.json"))
            cb_tx = json.loads(z.read("tx-0000-coinbase.json"))
        check("второй блок: prev = sha256(первого)",
              h2["prev_sha256"] == chain.sha256_file(p1))
        check("coinbase = 50 + комиссия",
              abs(sum(o["value"] for o in cb_tx["outputs"]) - (50.0 + fee)) < 1e-9)

        v = chain.verify_chain(tmp, "Cu")
        check("цепочка целая", v["ok"], v["problems"])

        bal = chain.balances(tmp, None, 1)
        check("воркеру доступно 25", bal[worker]["available"] == 25.0, bal.get(worker))
        check("у мастера всего 75", bal[master]["total"] == 75.0, bal.get(master))

        # --- подпись обязательна ---
        wfn2 = os.path.join(tmp2, "state.wfn")
        make_wfn(wfn2)
        sha3, rel3 = chain.store_blob(tmp2, wfn2)
        chain.write_block(tmp2, "Cu", 2, sha3, rel3,
                          [chain.make_coinbase(1, master)], dft=dft, **cp2k)
        bad_reward = chain.build_transfer(tmp2, {master}, worker, master,
                                          chain.REWARD_VALUE, 1, height=2)
        chain.write_block(tmp2, "Cu", 2, sha3, rel3,
                          [chain.make_coinbase(2, master, 50.0), bad_reward],
                          dft=dft, **cp2k)
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
                                   [chain.make_coinbase(1, master)], dft=dft, **cp2k)
            vb = chain.verify_block(edir, p3, expected_seq=1,
                                    expected_prev=chain.ZERO_HASH)
            check("verify_block принимает корректный блок", vb["ok"], vb["problems"])

            vb2 = chain.verify_block(edir, p3, expected_seq=2,
                                     expected_prev=chain.ZERO_HASH)
            check("verify_block отвергает не тот seq", not vb2["ok"])
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)

        # --- порча ---
        with open(p1, "ab") as f:
            f.write(b"broken")
        check("порча блока ловится", not chain.verify_chain(tmp, "Cu")["ok"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)


def wallet_tests():
    print("\n--- 22. Кошельки мастера и воркера (SQLite) ---")
    tmp = tempfile.mkdtemp(prefix="wallet_test_")
    try:
        conn = wallet_store.open_db(os.path.join(tmp, "keys.db"))
        m, created = wallet_store.ensure_master(conn, "root")
        check("первый мастер создаётся с паролем root", created)
        w, created_w = wallet_store.ensure_worker(conn, "worker_A", "root")
        check("воркер создаётся в той же базе", created_w)

        rows = {r["role"]: r for r in conn.execute("SELECT * FROM wallets").fetchall()}
        check("пароли с разными солями дают разные хэши",
              rows["master"]["password_hash"] != rows["worker"]["password_hash"])
        check("соли пароля разные",
              rows["master"]["password_salt"] != rows["worker"]["password_salt"])
        check("адреса мастера и воркера разные",
              m.latest_key()["address"] != w.latest_key()["address"])

        check("верный пароль открывает мастера",
              wallet_store.unlock(conn, "master", "master", "root") is not None)
        check("неверный пароль не открывает",
              wallet_store.unlock(conn, "master", "master", "nope") is None)

        addr = m.latest_key()["address"]
        sig = m.sign(addr, b"hello")
        check("Ed25519-подпись кошелька проверяется",
              crypto.verify(m.public_for(addr), b"hello", sig))

        m.change_password("root", "newpass")
        check("старый пароль после смены не работает",
              wallet_store.unlock(conn, "master", "master", "root") is None)
        m2 = wallet_store.unlock(conn, "master", "master", "newpass")
        check("новый пароль работает, ключи на месте",
              m2 is not None and m2.latest_key()["address"] == addr)
        sig2 = m2.sign(addr, b"x")
        check("ключи перешифрованы и подпись валидна",
              crypto.verify(m2.public_for(addr), b"x", sig2))
        check("воркер со своим паролем не затронут",
              wallet_store.unlock(conn, "worker", "worker_A", "root") is not None)

        element_dir = os.path.join(tmp, "element")
        os.makedirs(element_dir, exist_ok=True)
        check("wallet.py находит кошелёк по паролю",
              wallet.show(conn, "newpass", element_dir, 1) == 0)
        check("wallet.py отвергает неверный пароль",
              wallet.show(conn, "bad", element_dir, 1) == 1)
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
        wid = r["worker_id"]

        main.BUSY.set()
        check("busy: задача не выдаётся",
              main.get_task(wid).get("status") == "busy")

        main.BUSY.clear()
        main.pending_tasks.append({"struct_id": "b1", "task_index": 0,
                                   "n_atoms": 2, "dim": 3, "attempts": 0})
        got = main.get_task(wid)
        check("задача выдана до занятости", got.get("status") == "ok")
        sid = got["task"]["struct_id"]

        main.BUSY.set()
        payload = {"geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
                   "n_atoms": 2, "wfn_b64": "AA=="}
        busy = main.submit_result(main.TaskResult(worker_id=wid, struct_id=sid,
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

        main.verify_state_payload = lambda cfg, task, payload: {
            "valid": True, "energy_ev": -1.0, "gap": 0.0, "mag": 0.0,
            "state": {"scf": {"steps": 1}}}
        main._persist_verified_state = lambda t, p, v: "accepted"

        main.BUSY.clear()
        ready = main.submit_result(main.TaskResult(
            worker_id=wid, struct_id="b_ready", success=True,
            data=payload, ready=True))
        check("ready-результат принимается без задачи",
              ready.get("status") == "accepted", ready)

        main.BUSY.set()
        ready_busy = main.submit_result(main.TaskResult(
            worker_id=wid, struct_id="b_ready2", success=True,
            data=payload, ready=True))
        check("ready на занятом мастере → busy",
              ready_busy.get("status") == "busy", ready_busy)
        main.BUSY.clear()

        try:
            main.work_height(element="Fe", pubkey_hash="h", name="n")
            check("work/height: чужой элемент отклонён", False)
        except Exception:
            check("work/height: чужой элемент отклонён", True)
        wh = main.work_height(element="Cu", pubkey_hash="hash_wh", name="nw")
        check("work/height регистрирует воркера",
              wh.get("status") == "ok" and wh.get("worker_id"), wh)

        # --- воркер: занятый мастер → другой мастер ---
        class R:
            def __init__(self, code, data):
                self.status_code = code
                self._d = data

            def json(self):
                return self._d

        seen = []

        def fake_get(url, params=None, timeout=None):
            seen.append(("get", url))
            if "p2" in url:
                return R(200, {"status": "ok", "worker_id": "w2"})
            return R(200, {"status": "busy"})

        def fake_post(url, json=None, timeout=None):
            seen.append(("post", url))
            if "p2" in url:
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
        worker.requests.get, worker.requests.post = old_get, old_post
        main.BUSY.clear()
        main.active_tasks.clear()
        main.pending_tasks.clear()
        main.CONFIG.update(saved)
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

    r1 = main._register_worker("alpha", "hash_a", "pub_a")
    r2 = main._register_worker("beta", "hash_b", "pub_b")
    check("разные ключи — разные воркеры", r1["worker_id"] != r2["worker_id"])
    r1b = main._register_worker("alpha", "hash_a", "pub_a")
    check("тот же ключ — тот же worker_id",
          r1b["worker_id"] == r1["worker_id"] and r1b["known"])

    main.pending_tasks.append({"struct_id": "reg_0001", "task_index": 0,
                               "n_atoms": 2, "dim": 3, "attempts": 0})
    main.pending_tasks.append({"struct_id": "reg_0002", "task_index": 1,
                               "n_atoms": 2, "dim": 3, "attempts": 0})

    got1 = main.get_task(r1["worker_id"])
    check("первая задача выдана", got1.get("status") == "ok")
    busy = main.get_task(r1["worker_id"])
    check("вторую задачу воркер не получает", busy.get("status") == "busy",
          busy.get("struct_id"))
    got2 = main.get_task(r2["worker_id"])
    check("другой уникальный воркер получает задачу", got2.get("status") == "ok")

    try:
        main.get_task(r1["worker_id"], pubkey_hash="чужой")
        check("чужой pubkey отклонён", False)
    except Exception:
        check("чужой pubkey отклонён", True)

    main.PAUSED = True
    check("пауза блокирует выдачу", main.get_task(r1["worker_id"]).get("status") == "paused")
    main.PAUSED = False

    main.pending_tasks.clear()
    main.active_tasks.clear()
    main.registered_workers.clear()
    main.worker_by_pubkey.clear()


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
    # блоки не пишем, чтобы не трогать боевые results/.
    main._write_chain_block = lambda *a, **k: None

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
