#!/usr/bin/env python3
"""
Быстрая проверка работоспособности проекта CP2K + MACE (zero-trust).

Запускать ПЕРВЫМ делом. Скрипт ничего не меняет в проекте, но пишет логи
в results/check_project.log и дублирует их в консоль.

Что проверяется:
  1. окружение: python, пакеты, наличие CP2K, версия;
  2. модули проекта: config.json, config_loader, cp2k_runner, structure_generator;
  3. оффлайн-тесты логики: test/test_zero_trust.py (CP2K не нужен);
  4. для элемента из config.json (по умолчанию Cu) — РЕАЛЬНЫЙ мини-расчёт:
     воркер: генерация Cu2 -> SCF до сходимости -> файл состояния (-RESTART.wfn);
     мастер: приёмка состояния (SCF_GUESS RESTART, один шаг) -> энергия/силы от мастера.
     Fe считается очень долго (медленная сходимость OT), включается --elements Cu,Fe.

Использование (запускать из корня проекта):
    python3 test/check_project.py                 # всё, включая мини-расчёт Cu2
    python3 test/check_project.py --no-cp2k       # только Python-часть (без CP2K)
    python3 test/check_project.py --elements Cu   # только медь (быстро)
    python3 test/check_project.py --elements Cu,Fe --keep-files   # + железо (долго)

В конце печатается блок «ВСТАВЬ ЭТО В ЧАТ» — его (и путь к полному логу) можно
просто скопировать для диагностики.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
TEST = os.path.join(ROOT, "test")
RESULTS = os.path.join(ROOT, "results")
LOG_PATH = os.path.join(RESULTS, "check_project.log")

STEPS = []       # (шаг, статус, комментарий)
PROBLEMS = []    # проблемные строки для итогового блока
START_TS = time.time()


# ---------- логирование: консоль + файл ----------
class Tee:
    def __init__(self, path, stream):
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


_orig_out, _orig_err = sys.stdout, sys.stderr


def setup_logging():
    os.makedirs(RESULTS, exist_ok=True)
    sys.stdout = Tee(LOG_PATH, _orig_out)
    sys.stderr = Tee(LOG_PATH, _orig_err)


def hr(title=""):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def step(name, status, note=""):
    """status: PASS / WARN / FAIL / SKIP"""
    STEPS.append((name, status, note))
    mark = {"PASS": "✅", "WARN": "⚠ ", "FAIL": "❌", "SKIP": "— "}[status]
    print(f"[{mark} {status}] {name}" + (f" :: {note}" if note else ""))
    if status in ("FAIL", "WARN"):
        PROBLEMS.append(f"{status}: {name} :: {note}")


def run_cmd(cmd, timeout=1800, cwd=None, env=None):
    """Запускает команду, возвращает (rc, combined_output)."""
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
        return proc.returncode, proc.stdout or ""
    except FileNotFoundError as e:
        return 127, f"команда не найдена: {e}"
    except subprocess.TimeoutExpired:
        return 124, f"таймаут {timeout} с"
    except Exception as e:
        return 1, f"ошибка запуска: {e}"


# ---------- шаги ----------
def step_env():
    hr("1. ОКРУЖЕНИЕ")
    print(f"root проекта : {ROOT}")
    print(f"python       : {sys.version.split()[0]}  ({sys.executable})")
    print(f"платформа    : {sys.platform}")
    print(f"cwd          : {os.getcwd()}")
    step("окружение: python", "PASS", sys.version.split()[0])

    for mod, hint in (("numpy", "pip install numpy"),
                      ("requests", "pip install requests"),
                      ("fastapi", "pip install fastapi"),
                      ("uvicorn", "pip install uvicorn"),
                      ("pydantic", "pip install pydantic")):
        try:
            __import__(mod)
            step(f"пакет {mod}", "PASS")
        except Exception as e:
            step(f"пакет {mod}", "FAIL", f"нет: {e} -> {hint}")

    optional = []
    for mod in ("mace", "ase", "torch"):
        try:
            __import__(mod)
            optional.append(mod)
        except Exception:
            pass
    step("опциональные MACE/ASE/torch", "PASS" if len(optional) == 3 else "WARN",
         (", ".join(optional) if optional else "нет — нужны только для MACE-этапов, сервер очереди работает без них"))


def find_cp2k(cfg_cmd=None):
    cands = [os.environ.get("CP2K_CMD"), cfg_cmd, "cp2k.psmp", "cp2k", "cp2k.ssmp"]
    for c in cands:
        if not c:
            continue
        exe = c.split()[0]
        if shutil.which(exe):
            return c
    return None


def step_cp2k(cp2k_cmd):
    hr("3. CP2K")
    if not cp2k_cmd:
        step("поиск CP2K", "SKIP", "не найден в PATH (см. инструкцию ниже)")
        print("""
CP2K не найден. Варианты:
  * Docker (как в README):  ./run.sh --cpu --name master --port 8443 --port2 8000 -d
                            (в образе cp2k уже есть, всё остальное тоже)
  * Ubuntu/WSL пакет:       sudo apt update && sudo apt install -y cp2k
  * conda-forge:            conda create -n dft -c conda-forge cp2k && conda activate dft
  * или указать вручную:    export CP2K_CMD=/путь/к/cp2k.psmp
Затем повторите: python3 test/check_project.py
""")
        return False

    print(f"команда CP2K : {cp2k_cmd}  ({shutil.which(cp2k_cmd.split()[0])})")
    rc, out = run_cmd(cp2k_cmd.split() + ["--version"], timeout=120)
    if rc != 0:
        rc, out = run_cmd(cp2k_cmd.split() + ["-v"], timeout=120)
    first = [ln for ln in out.splitlines() if ln.strip()][:4]
    for ln in first:
        print(f"  | {ln}")
    step("CP2K запускается", "PASS" if rc == 0 else "WARN",
         "версия получена" if rc == 0 else f"код {rc} (версия не прочитана, но бинарник найден)")
    return True


def step_project_modules():
    hr("2. МОДУЛИ ПРОЕКТА")
    sys.path.insert(0, SRC)
    try:
        import config_loader
        cfg = config_loader.Config.load(os.path.join(SRC, "config.json"))
        step("config.json читается", "PASS", f"ключей: {len(cfg)}")
    except Exception as e:
        step("config.json читается", "FAIL", f"{e}")
        return None, None, None

    for mod in ("cp2k_runner", "structure_generator"):
        try:
            __import__(mod)
            step(f"импорт {mod}", "PASS")
        except Exception as e:
            step(f"импорт {mod}", "FAIL", f"{e}")
            return cfg, None, None

    import cp2k_runner
    import structure_generator

    try:
        import worker as worker_mod
        step("импорт worker", "PASS")
    except Exception as e:
        worker_mod = None
        step("импорт worker", "WARN", f"{e} (нужен пакет requests)")

    system = cfg.get("system") or {}
    comp = system.get("composition") or cfg.get("composition") or {}
    el = system.get("name") or cfg.get("element") or next(iter(comp), "Cu")
    probe = next(iter(comp), el)
    try:
        pot = cp2k_runner.get_potential(probe)
        q = cp2k_runner.valence_electrons(pot)
        note = f"{el}: {comp or {probe: 1}} -> {probe} {pot} (q={q})"
        step("система/псевдопотенциал из config.json", "PASS", note)
    except Exception as e:
        pot, q = None, None
        step("система/псевдопотенциал из config.json", "FAIL", f"{e}")

    for key in ("schedule", "atom_diameter", "structures_per_n", "dimensions_per_n"):
        if key in cfg and cfg[key] is not None:
            print(f"  {key} = {cfg[key]!r}")
        else:
            print(f"  {key} = <нет>")

    print(f"  генерация: BOX_MARGIN {structure_generator.BOX_MARGIN}, "
          f"SEED {structure_generator.RANDOM_SEED}, "
          f"min_dist = diameter, max_dist = 2*diameter (константы structure_generator.py)")

    params = cp2k_runner.CP2K_PARAMS
    print(f"  CP2K: box >= {params['min_cell_size']} Å, padding {params['box_padding']} Å, "
          f"CUTOFF {params['cutoff']}, EPS_SCF {params['eps_scf']:g} "
          f"(константы CP2K_PARAMS в src/cp2k_runner.py)")

    print("  доступные элементы:", list(cp2k_runner.PSEUDOPOTENTIALS))
    return cfg, pot, worker_mod


def step_offline_tests():
    hr("4. ОФФЛАЙН-ТЕСТЫ ЛОГИКИ (без CP2K)")
    script = os.path.join(TEST, "test_zero_trust.py")
    if not os.path.exists(script):
        step("test_zero_trust.py", "SKIP", "файл не найден")
        return
    t0 = time.time()
    rc, out = run_cmd([sys.executable, script], timeout=600, cwd=ROOT)
    tail = [ln for ln in out.splitlines() if ln.strip()][-6:]
    for ln in tail:
        print(f"  | {ln}")
    step("test_zero_trust.py", "PASS" if rc == 0 else "FAIL",
         f"код {rc}, {time.time() - t0:.1f} с")


def step_props_parsers():
    """
    Парсеры физических наблюдаемых — на самом свежем готовом логе (CP2K не запускается).
    Проверяется главное: разложение энергии сходится, спектр согласован с gap, есть диполь.
    """
    hr("4a. ФИЗИЧЕСКИЕ НАБЛЮДАЕМЫЕ ИЗ ЛОГА (без CP2K)")
    try:
        import cp2k_runner
    except Exception as e:  # noqa: BLE001
        step("импорт cp2k_runner", "FAIL", f"{e}")
        return
    logs = sorted(glob.glob(os.path.join(RESULTS, "*", "**", "*.out"), recursive=True),
                  reverse=True)
    if not logs:
        step("парсеры наблюдаемых", "SKIP", "нет ни одного .out")
        return
    # Берём первый лог, в котором есть полный набор (а не просто самый новый файл:
    # рядом могут лежать оборванные прогоны без Малликена).
    log = props = None
    for candidate in logs[:20]:
        try:
            candidate_props = cp2k_runner.collect_physical_props(candidate)
        except Exception:  # noqa: BLE001
            continue
        if candidate_props.get("homo_ev") is not None and candidate_props.get("charges_mulliken"):
            log, props = candidate, candidate_props
            break
    if props is None:
        step("парсеры наблюдаемых", "SKIP", "нет .out с полным набором (нужен прогон с &MULLIKEN и &MO_CUBES)")
        return
    print(f"  лог: {log}")

    problems = []
    residual = props.get("energy_components_residual_hartree")
    if residual is None or residual > 1e-8:
        problems.append(f"разложение энергии не сходится (residual={residual})")
    charges = props.get("charges_mulliken") or []
    if not charges:
        problems.append("нет зарядов Малликена")
    homo, lumo = props.get("homo_ev"), props.get("lumo_ev")
    gap = cp2k_runner.parse_homo_lumo(log, 27.211386245988)
    if homo is None or lumo is None:
        problems.append("нет ε_HOMO/ε_LUMO")
    elif gap is not None and abs((lumo - homo) - gap) > 1e-2:
        problems.append(f"gap по спектру {lumo - homo:.4f} != {gap:.4f} eV")
    if props.get("dipole_total_debye") is None:
        problems.append("нет диполя")

    note = "; ".join(problems) if problems else (
        f"IP={props.get('ip_koopmans_ev'):.3f} eV, EA={props.get('ea_koopmans_ev', float('nan')):.3f} eV, "
        f"диполь={props.get('dipole_total_debye', 0):.2e} Д, зарядов {len(charges)}, "
        f"{len(json.dumps(props).encode())} байт")
    step("парсеры наблюдаемых", "PASS" if not problems else "WARN", note)


def build_cfgs(cfg, cfg_cmd):
    """Собирает cfg для воркера и мастера из config.json проекта."""
    return {
        "cp2k_cmd": cfg_cmd or cfg.get("cp2k_cmd", "cp2k"),
        "verbose_cp2k": False,
        "hartree_to_ev": cfg.get("hartree_to_ev", 27.211386),
        "hartree_bohr_to_ev_ang": cfg.get("hartree_bohr_to_ev_ang", 51.422067),
        "atom_diameter": cfg.get("atom_diameter", 2.0),
        "restart_max_mb": 50,
    }


def step_element_chain(cp2k_runner, structure_generator, cfg, element, cp2k_cmd, keep=False):
    """Реальная цепочка: воркер считает Cu2/Fe2 -> мастер принимает состояние."""
    hr(f"5. РЕАЛЬНЫЙ МИНИ-РАСЧЁТ: {element}2")
    base = build_cfgs(cfg, cp2k_cmd)
    try:
        potential = cp2k_runner.get_potential(element)
    except Exception as e:
        step(f"{element}: псевдопотенциал", "FAIL", f"{e}")
        return

    workdir = os.path.join(RESULTS, element, "checkup")
    os.makedirs(workdir, exist_ok=True)
    worker_cfg = dict(base, element=element, potential=potential, output_dir=workdir)
    master_cfg = dict(worker_cfg)

    try:
        coords = structure_generator.generate_random_cluster(2, 3, worker_cfg, seed=7, max_attempts=20)
        print(f"  геометрия: {coords.shape[0]} атомов, d = "
              f"{float(((coords[0] - coords[1]) ** 2).sum() ** 0.5):.3f} Å")
    except Exception as e:
        step(f"{element}: генерация структуры", "FAIL", f"{e}")
        return

    # --- мастер: энергия изолированного атома (один раз, кэш) ---
    print(f"  [мастер] энергия изолированного атома {element}...")
    iso_cfg = dict(worker_cfg)
    iso_cfg["isolated_atom_energy_file"] = os.path.join(
        workdir, f"{element}_isolated_atom_energy.txt")
    t0 = time.time()
    e_iso = cp2k_runner.calculate_isolated_atom_energy(iso_cfg)
    t_iso = time.time() - t0
    step(f"{element}: энергия изолированного атома",
         "PASS" if e_iso is not None else "WARN",
         f"{e_iso:.6f} eV за {t_iso:.1f} с" if e_iso is not None else "не посчитана")
    worker_cfg["e_isolated_atom"] = e_iso
    master_cfg["e_isolated_atom"] = e_iso

    # --- воркер: полный SCF ---
    print(f"  [воркер] полный SCF {element}2 (может занять минуты)...")
    t0 = time.time()
    res = cp2k_runner.run_dft_for_single_structure(
        coords=coords, n_atoms=2, dim=3, struct_id=f"checkup_{element}",
        cfg=worker_cfg, force_recompute=True, save_wfn=True,
    )
    t_worker = time.time() - t0
    if res.get("error"):
        step(f"{element}: расчёт воркера", "FAIL",
             f"rc={res.get('rc')} за {t_worker:.1f} с; лог: {workdir}")
        # диагностика: последний .out и файлы сил в каталоге
        logs = sorted(glob.glob(os.path.join(workdir, f"*{element}*.out")),
                      key=os.path.getmtime)
        if logs:
            print(f"  --- последние строки {os.path.basename(logs[-1])} ---")
            with open(logs[-1], "r", encoding="utf-8", errors="replace") as f:
                for ln in f.read().splitlines()[-15:]:
                    print(f"    | {ln[:180]}")
        forces = sorted(glob.glob(os.path.join(workdir, "*.forces")))
        print(f"  файлы сил в каталоге: {[os.path.basename(p) for p in forces] or 'нет'}")
        return
    print(f"  [воркер] E = {res['energy_ev']:.6f} eV, сходимость: "
          f"{res.get('scf_converged')} за {res.get('scf_steps')} шагов, {t_worker:.1f} с "
          f"({t_worker / max(1, res.get('scf_steps') or 1):.1f} с/шаг), "
          f"box = {res.get('cell_size', '?')} Å")
    if e_iso is not None:
        be = (e_iso * 2 - res["energy_ev"]) / 2
        print(f"  [воркер] E_binding/atom = {be:.6f} eV | gap = {res.get('gap')} | "
              f"mag = {res.get('mag')}")
    step(f"{element}: SCF воркера сошёлся", "PASS" if res.get("scf_converged") else "WARN",
         f"{t_worker:.1f} с, E={res['energy_ev']:.4f} eV")
    step(f"{element}: силы посчитаны",
         "PASS" if len(res.get("forces_ev_ang") or []) == 2 else "FAIL",
         f"{len(res.get('forces_ev_ang') or [])} строк")

    wfn = res.get("wfn_path")
    if not wfn or not os.path.exists(wfn):
        step(f"{element}: файл состояния (-RESTART.wfn)", "FAIL", f"не создан в {workdir}")
        return
    head = cp2k_runner.parse_wfn_header(wfn)
    print(f"  [состояние] размер {head['size']} байт, nao={head['nao']}, "
          f"nspin={head['nspin']}, столбцов={head['columns']}")
    step(f"{element}: разбор заголовка состояния", "PASS" if head["ok"] else "FAIL",
         head.get("reason") or f"nao={head['nao']} на атом {head['nao'] // 2}")

    # --- мастер: приёмка по состоянию ---
    print(f"  [мастер] приёмка состояния: один шаг SCF от присланного файла...")
    t0 = time.time()
    vres = cp2k_runner.verify_state_from_wfn(
        coords, wfn, master_cfg, project=f"checkup_verify_{element}",
    )
    t_master = time.time() - t0
    if not vres.get("ok"):
        reason = vres.get("reason", "?")
        print(f"  [мастер] отказ: {reason}")
        log = vres.get("log_path")
        if log and os.path.exists(log):
            print(f"  --- строки про restart из {log} ---")
            with open(log, "r", encoding="utf-8", errors="replace") as f:
                for ln in f:
                    if any(k in ln.upper() for k in ("RESTART", "WFN", "WAVEFUNCTION", "GUESS")):
                        print(f"    | {ln.rstrip()[:160]}")
        soft = ("подтвержд" in reason)
        step(f"{element}: приёмка состояния мастером", "WARN" if soft else "FAIL", reason)
        if soft:
            print("  (это может быть только расхождение формулировок лога CP2K — "
                  "пришлите строки выше)")
        return

    print(f"  [мастер] E = {vres['energy_ev']:.6f} eV (посчитал мастер), "
          f"сходимость за {vres['scf'].get('steps')} шагов, {t_master:.1f} с")
    print(f"  [мастер] nao системы = {(vres.get('sysinfo') or {}).get('nao')}, "
          f"занятых = {(vres.get('sysinfo') or {}).get('n_occ')}, "
          f"первая невязка OT = {(vres.get('first_step') or {}).get('grad')}")
    step(f"{element}: приёмка состояния мастером", "PASS",
         f"ΔE = {abs(vres['energy_ev'] - res['energy_ev']):.4f} eV, "
         f"воркер {t_worker:.1f} с / мастер {t_master:.1f} с "
         f"(×{t_worker / max(t_master, 1e-9):.1f} дешевле)")

    # --- наблюдаемые: их считает мастер в том же прогоне, что и E ---
    props = vres.get("props") or {}
    if props.get("homo_ev") is not None:
        step(f"{element}: наблюдаемые от мастера", "PASS",
             f"IP={-props['homo_ev']:.3f} eV (Купманс), "
             f"диполь={props.get('dipole_total_debye', 0):.2e} Д, "
             f"заряды: {props.get('charges_mulliken')}")
    else:
        step(f"{element}: наблюдаемые от мастера", "WARN",
             "спектр/диполь не разобраны — в блок уйдёт меньше полей")

    if not keep:
        print(f"  (логи CP2K остались в {workdir})")


def print_summary():
    hr("ИТОГИ")
    n_fail = sum(1 for _n, s, _c in STEPS if s == "FAIL")
    n_warn = sum(1 for _n, s, _c in STEPS if s == "WARN")
    n_pass = sum(1 for _n, s, _c in STEPS if s == "PASS")
    for name, status, note in STEPS:
        if status in ("FAIL", "WARN"):
            print(f"  {status:5s} {name} :: {note}")
    print(f"\n  PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}   "
          f"время {time.time() - START_TS:.1f} с")

    hr("ВСТАВЬ ЭТО В ЧАТ (для диагностики)")
    print(f"проект: {ROOT}")
    print(f"python: {sys.version.split()[0]}")
    if STEPS:
        print(f"PASS={n_pass} WARN={n_warn} FAIL={n_fail}")
    for p in PROBLEMS:
        print(f"  {p}")
    print(f"полный лог: {LOG_PATH}")
    print("--- последние 60 строк лога ---")
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        for ln in lines[-60:]:
            print(ln)
    except Exception as e:
        print(f"(лог не прочитан: {e})")

    return 1 if n_fail else 0


def main():
    ap = argparse.ArgumentParser(description="Health-check проекта CP2K+MACE (zero-trust)")
    ap.add_argument("--no-cp2k", action="store_true", help="не запускать расчёты CP2K")
    ap.add_argument("--elements", default="", help="список элементов через запятую (по умолчанию: element из config.json и Fe)")
    ap.add_argument("--keep-files", action="store_true", help="оставить логи CP2K (по умолчанию они тоже остаются)")
    args = ap.parse_args()

    setup_logging()
    hr("ПРОВЕРКА ПРОЕКТА CP2K + MACE (zero-trust)")
    print(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    rc = 0
    try:
        step_env()
        cfg, pot, _worker_mod = step_project_modules()
        cp2k_cmd = find_cp2k((cfg or {}).get("cp2k_cmd"))
        has_cp2k = step_cp2k(cp2k_cmd)
        step_offline_tests()
        step_props_parsers()

        if args.no_cp2k:
            step("реальные мини-расчёты", "SKIP", "--no-cp2k")
        elif not has_cp2k:
            step("реальные мини-расчёты", "SKIP", "CP2K не найден")
        elif cfg is None:
            step("реальные мини-расчёты", "SKIP", "config.json не прочитан")
        else:
            import cp2k_runner
            import structure_generator
            if args.elements:
                elements = [e.strip() for e in args.elements.split(",") if e.strip()]
            else:
                # По умолчанию — только система из config.json: Fe2 считается долго
                # (почти вырожденные d-состояния), его включает флаг --elements Cu,Fe.
                system = cfg.get("system") or {}
                comp = system.get("composition") or {}
                name = system.get("name") or cfg.get("element", "Cu")
                if len(comp) > 1:
                    step("реальные мини-расчёты", "SKIP",
                         f"система {name} из нескольких элементов — только оффлайн-проверки")
                    elements = []
                else:
                    elements = [name]
            for el in elements:
                try:
                    step_element_chain(cp2k_runner, structure_generator, cfg, el, cp2k_cmd,
                                       keep=args.keep_files)
                except Exception as e:
                    step(f"{el}: мини-расчёт", "FAIL", f"{type(e).__name__}: {e}")
                    traceback.print_exc()
    except KeyboardInterrupt:
        print("\nпрервано пользователем")
        rc = 130
    except Exception as e:
        step("health-check", "FAIL", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        rc = 1

    rc = max(rc, print_summary())
    sys.stdout = _orig_out
    sys.stderr = _orig_err
    return rc


if __name__ == "__main__":
    sys.exit(main())
