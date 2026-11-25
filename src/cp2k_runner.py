#!/usr/bin/env python3
"""
Модуль для работы с CP2K.

Две роли:

  * ВОРКЕР — генерирует .inp, запускает CP2K до сходимости SCF, парсит E/силы/gap/mag
             и отдаёт файл состояния (<project>-RESTART.wfn) для мастера.
  * МАСТЕР — пишет свой вход (элемент/базис/потенциал/ячейка/функционал/геометрия),
             стартует от присланной плотности и сам считает E/силы/gap/mag
             (verify_state_from_wfn). Числам воркера мастер не верит.

Шаблон входа CP2K у мастера и воркера ОДИН И ТОТ ЖЕ (render_cp2k_input),
поэтому состояние воркера применимо к задаче мастера без преобразований.
шаблон входа CP2K у мастера и у воркера ОДИН И ТОТ ЖЕ (render_cp2k_input).
Мастер лишь понижает MAX_SCF до 1 и печатает только OVERLAP — при этом ячейка,
PERIODIC, базис, псевдопотенциал, CUTOFF и REL_CUTOFF идентичны, иначе матрица
S_master перестала бы соответствовать матрицам воркера и честный расчёт
отвергался бы.
"""

import os
import re
import sys
import json
import glob
import time
import shutil
import struct
import tempfile
import hashlib
import subprocess

import numpy as np


# Словарь псевдопотенциалов для элементов (qN = число валентных электронов)
# Словарь псевдопотенциалов (GTH-PBE-qN = N валентных электронов).
# Добавить элемент — одна строка; база DZVP-MOLOPT-SR-GTH есть для H–Kr и части тяжёлых.
PSEUDOPOTENTIALS = {
    "H": "GTH-PBE-q1", "He": "GTH-PBE-q2",
    "Li": "GTH-PBE-q3", "Be": "GTH-PBE-q4", "B": "GTH-PBE-q3",
    "C": "GTH-PBE-q4", "N": "GTH-PBE-q5", "O": "GTH-PBE-q6",
    "F": "GTH-PBE-q7", "Ne": "GTH-PBE-q8",
    "Na": "GTH-PBE-q9", "Mg": "GTH-PBE-q10", "Al": "GTH-PBE-q3",
    "Si": "GTH-PBE-q4", "P": "GTH-PBE-q5", "S": "GTH-PBE-q6",
    "Cl": "GTH-PBE-q7", "Ar": "GTH-PBE-q8",
    "K": "GTH-PBE-q9", "Ca": "GTH-PBE-q10",
    "Ti": "GTH-PBE-q12", "V": "GTH-PBE-q13", "Cr": "GTH-PBE-q14",
    "Mn": "GTH-PBE-q15", "Fe": "GTH-PBE-q16", "Co": "GTH-PBE-q17",
    "Ni": "GTH-PBE-q18", "Cu": "GTH-PBE-q11", "Zn": "GTH-PBE-q12",
    "Br": "GTH-PBE-q7", "Kr": "GTH-PBE-q8",
    "Ag": "GTH-PBE-q11", "Cd": "GTH-PBE-q12", "I": "GTH-PBE-q7",
    "Au": "GTH-PBE-q11",
}


# ---------- Шаблон CP2K (общий для воркера и мастера) ----------
TEMPLATE = """&GLOBAL
  PROJECT {project}
  RUN_TYPE {run_type}
  PRINT_LEVEL {print_level}
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
{wfn_restart_line}    LSD
{charge_line}{multiplicity_line}
    &MGRID
      CUTOFF {cutoff}
      REL_CUTOFF {rel_cutoff}
    &END MGRID
    &QS
      EPS_DEFAULT {eps_default}
    &END QS
    &SCF
      SCF_GUESS {scf_guess}
      EPS_SCF {eps_scf}
      MAX_SCF {max_scf}
      {ignore_convergence}
      &OT
        MINIMIZER {ot_minimizer}
        PRECONDITIONER {ot_preconditioner}
{ot_extra}      &END OT
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
    &POISSON
      POISSON_SOLVER WAVELET
      PERIODIC NONE
    &END POISSON
    &PRINT
{print_blocks}    &END PRINT
  &END DFT

  &SUBSYS
    &CELL
      ABC {cell_size} {cell_size} {cell_size}
      PERIODIC NONE
    &END CELL
    &COORD
{coords_block}
    &END COORD
{kinds_block}
  &END SUBSYS
{force_print_block}&END FORCE_EVAL
"""

# ============================================================
# Вычислительные параметры CP2K — ЕДИНСТВЕННОЕ место, где их меняют.
# В config.json их специально нет: это редкие «железные» настройки,
# и они обязаны совпадать у воркера и мастера.
# При необходимости ключ можно переопределить в cfg (тесты/эксперименты).
# ============================================================
CP2K_PARAMS = {
    "cutoff": 400,
    "rel_cutoff": 50,
    "eps_default": 1.0e-10,
    "eps_scf": 1.0e-6,
    "max_scf": 200,
    "print_level": "LOW",
    # вакуум вокруг кластера: box = диаметр + 2*box_padding (Å)
    # 20 Å для Cu2 даёт dE = 0; 14 Å — уже +8.6 мэВ (периодические образы), поэтому не меньше 20.
    "box_padding": 6.0,
    "min_cell_size": 20.0,
    "max_cell_size": 30.0,
    "ot_minimizer": "DIIS",
    "ot_preconditioner": "FULL_SINGLE_INVERSE",
    "ot_energy_gap": 0.0,      # 0 — не печатать (дефолт CP2K 0.08 Ha)
    "ot_linesearch": "",       # пусто — не печатать (например, 3POINT)
    "print_mo_cubes": True,    # нужны, чтобы CP2K напечатал HOMO-LUMO gap
    "cleanup_cubes": True,     # удалять *.cube после разбора (экономия диска)
    "mpi_ranks": 1,            # 1 надёжнее: mpirun в контейнере иногда не завершается
    "omp_threads": 0,          # 0 — авто
}


# Блоки печати, специфичные для роли
MO_CUBES_BLOCK = """      &MO_CUBES
        NHOMO 1
        NLUMO 1
      &END MO_CUBES
"""

MULLIKEN_BLOCK = """      &MULLIKEN ON
      &END MULLIKEN
      &MOMENTS ON
      &END MOMENTS
"""


# ---------- Политика приёмки состояния (-RESTART.wfn) ----------
RESTART_VERIFY_SCF = 3                 # SCF от присланного состояния обязан сойтись за 3 шага
REQUIRE_RESTART_READ_CONFIRMATION = True
TOL_GRAD = 1e-4                        # только для метаданных кадра, решение не принимает
CP2K_TIMEOUT = 3600                    # таймаут одного прогона CP2K, секунды


def _num(cfg, key):
    """Значение параметра CP2K: из cfg, иначе константа из CP2K_PARAMS."""
    value = cfg.get(key)
    return CP2K_PARAMS[key] if value is None else value


def force_print_block(project: str) -> str:
    """
    Секция &PRINT &FORCES.

    FILENAME не задаём: CP2K всё равно добавляет префикс проекта и суффикс
    шага, а с FILENAME получается мусор вида '<project>-<FILENAME>-1_0.xyz'.
    Файл потом ищется по шаблону в locate_forces_file().
    """
    return """  &PRINT
    &FORCES ON
    &END FORCES
  &END PRINT

"""


def build_print_blocks(cfg: dict) -> str:
    """
    Собирает секцию &PRINT под задачу.

    Кубы МО нужны только ради строки HOMO-LUMO gap в логе (файлы удаляются).
    """
    parts = []
    if cfg.get("print_mo_cubes", CP2K_PARAMS["print_mo_cubes"]):
        parts.append(MO_CUBES_BLOCK)
    parts.append(MULLIKEN_BLOCK)
    return "".join(parts)


# ============================================================
# Псевдопотенциалы / число электронов
# ============================================================
def get_potential(element: str) -> str:
    potential = PSEUDOPOTENTIALS.get(element)
    if potential is None:
        raise KeyError(
            f"Для элемента {element} не задан псевдопотенциал. "
            f"Доступные: {list(PSEUDOPOTENTIALS.keys())}"
        )
    return potential


def valence_electrons(potential: str) -> int:
    """Число валентных электронов на атом из имени псевдопотенциала, напр. GTH-PBE-q11 -> 11."""
    m = re.search(r"-q(\d+)", potential or "")
    if not m:
        raise ValueError(f"Не удалось определить число валентных электронов из '{potential}'")
    return int(m.group(1))


# ---------- Хэш геометрии ----------
def _geometry_hash(coords, cfg):
    """
    Хэш от координат (с точностью 1e-4 Å) + ключевых параметров расчёта.
    Одинаковая геометрия + одинаковые параметры → одинаковый хэш → кэш валиден.
    """
    coords_rounded = [[round(float(x), 4) for x in atom] for atom in coords]
    payload = {
        "coords": coords_rounded,
        "cell_size": resolve_cell_size(coords, cfg),
        "element": cfg.get("element"),
        "potential": cfg.get("potential") or PSEUDOPOTENTIALS.get(cfg.get("element"), ""),
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ---------- Генерация .inp ----------
def _sci(value) -> str:
    """1e-06 -> '1.0E-06' (формат, который CP2K точно разберёт)."""
    return f"{float(value):.1E}"


def resolve_cell_size(coords, cfg: dict) -> float:
    """
    Размер кубической ячейки: диаметр кластера + 2*box_padding (Å).

    Раньше cfg["cell_size"]=20 стояло для всех кластеров: для Cu2 это давало
    сетку ~340³ и 23 с на шаг SCF (больше 13 минут на 33 шага). Box зависит
    ТОЛЬКО от геометрии и cfg — воркер и мастер вычисляют его одинаково.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.size == 0:
        return float(_num(cfg, "min_cell_size"))
    centered = arr - arr.mean(axis=0)
    diameter = 2.0 * float(np.linalg.norm(centered, axis=1).max())
    size = diameter + 2.0 * float(_num(cfg, "box_padding"))
    size = max(size, float(_num(cfg, "min_cell_size")))
    size = min(size, float(_num(cfg, "max_cell_size")))
    return float(np.ceil(size * 2.0) / 2.0)   # шаг 0.5 Å


def _species_list(n_atoms, cfg) -> list[str]:
    """Список символов по атомам: cfg['species'] или один элемент на всех."""
    species = cfg.get("species")
    if species and len(species) == n_atoms:
        return list(species)
    return [cfg["element"]] * n_atoms


def build_kinds_block(species: list[str]) -> str:
    """По одному &KIND на каждый химический элемент в системе."""
    seen, blocks = [], []
    for sym in species:
        if sym in seen:
            continue
        seen.append(sym)
        blocks.append(
            f"    &KIND {sym}\n"
            f"      BASIS_SET DZVP-MOLOPT-SR-GTH\n"
            f"      POTENTIAL {get_potential(sym)}\n"
            f"    &END KIND"
        )
    return "\n".join(blocks)


def _coords_block(coords, cfg, cell_size: float | None = None) -> str:
    """Текстовый блок &COORD с центрированием координат в ячейке."""
    element = cfg["element"]
    cell_size = float(cell_size if cell_size is not None else resolve_cell_size(coords, cfg))
    coords_arr = np.array(coords, dtype=np.float64)
    center_of_mass = coords_arr.mean(axis=0)
    cell_center = np.array([cell_size / 2.0] * 3)
    coords_arr = coords_arr - center_of_mass + cell_center
    species = _species_list(len(coords_arr), cfg)
    return "\n".join(
        f"      {symbol}  {x:.8f}  {y:.8f}  {z:.8f}"
        for symbol, (x, y, z) in zip(species, coords_arr)
    )


def render_cp2k_input(
    project: str,
    coords,
    cfg: dict,
    run_type: str = "ENERGY_FORCE",
    max_scf: int | None = None,
    ignore_convergence: bool = True,
    scf_guess: str = "ATOMIC",
    wfn_restart_file: str | None = None,
    print_blocks: str | None = None,
) -> str:
    """
    Единый шаблон входа CP2K (воркер и мастер).

    scf_guess/wfn_restart_file — стартовать с присланного состояния
                                 (SCF_GUESS RESTART + WFN_RESTART_FILE_NAME в &DFT).
    """
    element = cfg["element"]
    potential = cfg.get("potential") or get_potential(element)
    species = _species_list(len(coords), cfg)
    kinds_block = build_kinds_block(species)

    charge = int(cfg.get("charge") or 0)
    multiplicity = int(cfg.get("multiplicity") or 0)
    charge_line = f"    CHARGE {charge}\n" if charge else ""
    multiplicity_line = (f"    MULTIPLICITY {multiplicity}\n" if multiplicity > 0 else "")

    if print_blocks is None:
        print_blocks = build_print_blocks(cfg)

    wfn_line = ""
    if wfn_restart_file:
        # В CP2K 2024.3 это ключ &DFT, а не &SCF (в &SCF -> ABORT unknown keyword).
        wfn_line = f"    WFN_RESTART_FILE_NAME {wfn_restart_file}\n"

    ignore_kw = "IGNORE_CONVERGENCE_FAILURE T" if ignore_convergence else ""
    if max_scf is None:
        max_scf = int(_num(cfg, "max_scf"))

    ot_extra = ""
    gap = float(_num(cfg, "ot_energy_gap") or 0.0)
    if gap > 0:
        ot_extra += f"        ENERGY_GAP {gap:g}\n"
    linesearch = str(_num(cfg, "ot_linesearch") or "").strip()
    if linesearch:
        ot_extra += f"        LINESEARCH {linesearch}\n"

    cell_size = resolve_cell_size(coords, cfg)

    return TEMPLATE.format(
        project=project,
        run_type=run_type,
        print_level=_num(cfg, "print_level"),
        cutoff=_num(cfg, "cutoff"),
        rel_cutoff=_num(cfg, "rel_cutoff"),
        eps_default=_sci(_num(cfg, "eps_default")),
        eps_scf=_sci(_num(cfg, "eps_scf")),
        ot_minimizer=_num(cfg, "ot_minimizer"),
        ot_preconditioner=_num(cfg, "ot_preconditioner"),
        ot_extra=ot_extra,
        max_scf=max_scf,
        ignore_convergence=ignore_kw,
        scf_guess=scf_guess,
        wfn_restart_line=wfn_line,
        print_blocks=print_blocks,
        force_print_block=force_print_block(project),
        cell_size=cell_size,
        coords_block=_coords_block(coords, cfg, cell_size),
        kinds_block=kinds_block,
        charge_line=charge_line,
        multiplicity_line=multiplicity_line,
        element=element,
        potential=potential,
    )


def write_cp2k_input(inp_path: str, project: str, coords, cfg: dict):
    """Пишет .inp для расчёта воркера (ENERGY_FORCE + силы)."""
    with open(inp_path, "w", encoding="utf-8") as f:
        f.write(render_cp2k_input(project, coords, cfg))


# ============================================================
# Парсинг
# ============================================================
def parse_energy(outfile: str) -> float:
    pattern = re.compile(
        r"ENERGY\|\s+Total FORCE_EVAL.*?energy\s+\[(?:hartree|a\.u\.)\]\s*:\s*(-?\d+\.\d+(?:[Ee][-+]?\d+)?)"
    )
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    raise ValueError(f"Энергия не найдена в {outfile}")


def parse_forces(forcefile: str):
    forces = []
    with open(forcefile, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if "ATOMIC FORCES in [a.u.]" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"Секция ATOMIC FORCES не найдена в {forcefile}")
    j = start
    while j < len(lines):
        s = lines[j].strip()
        if s == "" or s.startswith("#"):
            j += 1
            continue
        break
    for line in lines[j:]:
        s = line.strip()
        if s == "" or s.startswith("SUM OF ATOMIC FORCES"):
            break
        parts = s.split()
        if len(parts) < 6:
            break
        try:
            fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
        except ValueError:
            break
        forces.append([fx, fy, fz])
    return forces


def locate_forces_file(workdir: str, project: str) -> str:
    """
    Ищет файл сил CP2K.

    Без FILENAME CP2K пишет '<project>-1_0_1.forces' (и иногда .xyz).
    Со старым FILENAME мог остаться буквальный '{project}.forces'.
    """
    direct = os.path.join(workdir, f"{project}.forces")
    if os.path.exists(direct) and os.path.getsize(direct) > 0:
        return direct

    patterns = [f"{project}*.forces", f"{project}*forces*", "*.forces", "*forces*"]
    for pattern in patterns:
        found = [p for p in sorted(glob.glob(os.path.join(workdir, pattern)))
                 if os.path.getsize(p) > 0]
        if found:
            return found[0]
    legacy = os.path.join(workdir, "{project}.forces")
    if os.path.exists(legacy):
        return legacy
    return direct


def read_forces(workdir: str, project: str, out_path: str):
    """
    Силы из файла, а если CP2K его не создал — из блока 'ATOMIC FORCES' в .out.

    CP2K при ENERGY_FORCE всегда печатает силы в лог, а отдельный файл
    &FORCES пишет непредсказуемо (добавляет префикс проекта и суффикс шага).
    """
    path = locate_forces_file(workdir, project)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            forces = parse_forces(path)
            if forces:
                return forces
        except Exception:
            pass
    return parse_forces(out_path)


def cleanup_cube_files(workdir: str, project: str | None = None):
    """Удаляет *.cube после разбора (CP2K пишет по 24 МБ на орбиталь)."""
    pattern = f"{project}*.cube" if project else "*.cube"
    for path in glob.glob(os.path.join(workdir, pattern)):
        try:
            os.remove(path)
        except OSError:
            pass


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def cp2k_argv(cfg: dict) -> list:
    """
    Команда запуска CP2K с учётом mpi_ranks.

    cp2k.psmp — MPI-бинарник: 1 ранг × N потоков плохо масштабируется на FFT,
    но mpirun в контейнере иногда не завершается после CP2K, поэтому по умолчанию 1.
    """
    cmd = str(cfg.get("cp2k_cmd") or "cp2k").split()
    if not cmd:
        cmd = ["cp2k"]
    ranks = int(_num(cfg, "mpi_ranks") or 0)
    if ranks <= 0:
        ranks = 1
    if ranks > 1 and not os.path.basename(cmd[0]).startswith("mpirun"):
        cmd = ["mpirun", "-np", str(ranks)] + cmd
    return cmd


def cp2k_env(cfg: dict) -> dict:
    env = os.environ.copy()
    ranks = int(_num(cfg, "mpi_ranks") or 0)
    if ranks <= 0:
        ranks = 1
    threads = int(_num(cfg, "omp_threads") or 0)
    if threads <= 0:
        threads = max(1, _cpu_count() // max(1, ranks))
    env["OMP_NUM_THREADS"] = str(threads)
    return env


def cp2k_version(cfg: dict) -> str:
    """Версия CP2K (для заголовка блока). Запускается один раз при старте."""
    base = str(cfg.get("cp2k_cmd") or "cp2k").split()
    if not base:
        return "unknown"
    for flag in ("--version", "-v"):
        try:
            out = subprocess.run(base + [flag], capture_output=True, text=True,
                                 timeout=60, env=cp2k_env(cfg)).stdout or ""
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"CP2K version\s+(\S+)", out)
        if m:
            return m.group(1)
    return "unknown"


def parse_homo_lumo(outfile: str, hartree_to_ev: float):
    """
    Ищет строки вида:
      HOMO - LUMO gap [eV] :    1.384688
    Возвращает минимальное значение в eV или None.
    """
    pattern = re.compile(
        r"HOMO\s*-\s*LUMO\s+gap\s*\[eV\]\s*:\s*(-?\d+\.\d+)",
        re.IGNORECASE,
    )
    gaps = []
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                gaps.append(float(m.group(1)))
    return min(gaps) if gaps else None


def parse_magnetic_moment(outfile: str):
    """
    Ищет магнитный момент в выводе CP2K.
    Основной источник — строка:
      # Total charge and spin       11.000000    11.000000    -0.000000     0.000000
    Последнее число — суммарный спиновый момент в μB.
    """
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Total charge and spin" in line:
                parts = line.split()
                try:
                    value = float(parts[-1])
                    return 0.0 if abs(value) < 1e-9 else value
                except (ValueError, IndexError):
                    continue

    # Fallback: старые паттерны
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    for pat in [
        r"Total\s+Magnetic\s+Moment\s*[:=]\s*(-?\d+\.\d+)",
        r"Integrated\s+absolute\s+spin\s+density\s*[:=]\s*(-?\d+\.\d+)",
        r"Magnetization\s*\(spin\)\s*[:=]\s*(-?\d+\.\d+)",
    ]:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


# ============================================================
# Физические наблюдаемые из лога CP2K
# ============================================================
# Всё ниже уже печатается штатным прогоном (PRINT_LEVEL LOW): нужны только
# &MO_CUBES (диагонализация занятого подпространства + OT-эйгенсолвер для LUMO),
# &MULLIKEN и &MOMENTS — они и так стоят в шаблоне.
#
# Отбор величин: (а) занимают десятки-сотни байт в блоке,
# (б) имеют экспериментальные аналоги в литературе:
#     * диполь — СВЧ/ИК-спектроскопия,
#     * IP/EA (по Купмансу) — фотоэлектронная спектроскопия (UPS/XPS),
#     * локальные спиновые моменты — магнетометрия, нейтронография, Мёссбауэр,
#     * заряд на атоме — сдвиги остова в XPS,
#     * разложение энергии — проверка целостности самого лога.
# Всё это НЕ добавляет ни одного шага SCF: числа берутся из того же прогона,
# которым мастер и так сертифицирует присланную плотность.

PROPS_VERSION = 1


def _clean(value: float) -> float:
    """−0.0 и 1e-17 → 0.0: в JSON не должно быть мнимых «отрицательных нулей»."""
    return 0.0 if abs(value) < 1e-12 else value


def _numbers(line: str) -> list:
    """Все токены строки, которые являются числами (экспонента E или D)."""
    out = []
    for token in line.replace("D", "E").split():
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


_ENERGY_COMPONENTS = (
    ("overlap", "Overlap energy of the core charge distribution"),
    ("self_core", "Self energy of the core charge distribution"),
    ("core_hamiltonian", "Core Hamiltonian energy"),
    ("hartree", "Hartree energy"),
    ("xc", "Exchange-correlation energy"),
    ("ewald", "Ewald energy"),
    ("total", "Total energy"),
)


def parse_energy_decomposition(outfile: str) -> dict:
    """
    Разложение полной энергии (Hartree) из блока после сходимости SCF:

      Self energy of the core charge distribution:   -128.80554643260098
      Core Hamiltonian energy:                         43.95346139776483
      Hartree energy:                                   4.26681591588035
      Exchange-correlation energy:                    -15.42364192648134
      Total energy:                                   -96.00891104543712

    Берётся последнее вхождение (для GEO_OPT/MD — финальная геометрия).
    """
    found = {}
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.lstrip()
            for key, label in _ENERGY_COMPONENTS:
                if not stripped.startswith(label):
                    continue
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    values = _numbers(parts[1])
                    if values:
                        found[key] = values[0]
                break
    return found


def energy_decomposition_residual_hartree(parts: dict) -> float | None:
    """|Σ компонент − Total energy| в Hartree; None, если данных не хватает."""
    if not parts or "total" not in parts:
        return None
    total = 0.0
    for key, _ in _ENERGY_COMPONENTS:
        if key == "total":
            continue
        total += parts.get(key, 0.0)
    return abs(total - parts["total"])


def parse_mulliken(outfile: str) -> dict | None:
    """
    Mulliken Population Analysis в LSD-постановке:

     #  Atom  Element  Kind  Atomic population (alpha,beta) Net charge  Spin moment
           1     Cu       1         5.500000     5.500000     0.000000     0.000000
     # Total charge and spin       11.000000    11.000000    -0.000000    -0.000000

    Возвращает {'charges': [...], 'spins': [...], 'total_charge', 'total_spin'}.
    Берётся последний блок (финальная геометрия).
    """
    charges, spins = [], []
    total_charge = total_spin = None
    inside = False
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            low = line.lower()
            if "atomic population" in low and "spin moment" in low:
                inside = True
                charges, spins = [], []
                total_charge = total_spin = None
                continue
            if not inside:
                continue
            if "total charge and spin" in low:
                values = _numbers(line)
                if len(values) >= 4:
                    total_charge, total_spin = values[2], values[3]
                inside = False
                continue
            tokens = line.split()
            if len(tokens) < 7:
                if charges:
                    inside = False
                continue
            try:
                index = int(tokens[0])
                charge = float(tokens[5])
                spin = float(tokens[6])
            except ValueError:
                if charges:
                    inside = False
                continue
            if index != len(charges) + 1:
                inside = False
                continue
            charges.append(charge)
            spins.append(spin)
    if not charges:
        return None
    if total_charge is None:
        total_charge = sum(charges)
    if total_spin is None:
        total_spin = sum(spins)
    return {"charges": [_clean(v) for v in charges],
            "spins": [_clean(v) for v in spins],
            "total_charge": _clean(total_charge),
            "total_spin": _clean(total_spin)}


_DIPOLE_RE = re.compile(r"X=\s*(\S+)\s+Y=\s*(\S+)\s+Z=\s*(\S+)\s+Total=\s*(\S+)")


def parse_dipole(outfile: str) -> dict | None:
    """
    Дипольный момент из секции ELECTRIC/MAGNETIC MOMENTS:

      Dipole moment [Debye]
        X= -0.5660638E-06 Y= -0.6494626E-05 Z= -0.5209988E-05  Total=      0.0000083

    При PERIODIC NONE и вакуумной ячейке диполь изолированного кластера
    определён корректно и сравним с СВЧ-спектроскопией и другими кодами.
    """
    last = None
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Total=" not in line or "X=" not in line:
                continue
            m = _DIPOLE_RE.search(line)
            if not m:
                continue
            try:
                last = [float(m.group(i)) for i in range(1, 5)]
            except ValueError:
                continue
    if last is None:
        return None
    return {"dipole_debye": last[:3], "dipole_total_debye": last[3]}


_OCC_EIG_RE = re.compile(r"Eigenvalues of the occupied subspace spin\s+(\d+)", re.I)
_UNOCC_EIG_RE = re.compile(r"Lowest Eigenvalues of the unoccupied subspace spin\s+(\d+)", re.I)
_FERMI_RE = re.compile(r"Fermi Energy \[eV\]\s*:\s*(-?\d+\.\d+)")


def parse_eigenvalues(outfile: str, hartree_to_ev: float) -> dict | None:
    """
    Собственные значения занятого подпространства (Ha) и низшая вакантная
    орбиталь от OT-эйгенсолвера; Fermi Energy CP2K печатает сразу в eV.

    Даёт ε_HOMO/ε_LUMO → купмансовские оценки IP = −ε_HOMO и EA = −ε_LUMO
    (сопоставимы с фотоэлектронной спектроскопией) и компактный занятый
    спектр: 11 чисел на Cu₂, ~5·n на кластер.
    """
    occupied, unoccupied = {}, {}
    fermi = None
    mode = None  # ("occ"|"unocc", спин)
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _OCC_EIG_RE.search(line)
            if m:
                mode = ("occ", int(m.group(1)))
                occupied[mode[1]] = []
                continue
            m = _UNOCC_EIG_RE.search(line)
            if m:
                mode = ("unocc", int(m.group(1)))
                unoccupied[mode[1]] = []
                continue
            if mode is None:
                continue
            stripped = line.strip()
            if not stripped:
                mode = None
                continue
            m = _FERMI_RE.search(stripped)
            if m:
                fermi = float(m.group(1))
                mode = None
                continue
            values = _numbers(stripped)
            if values and len(values) == len(stripped.split()):
                target = occupied if mode[0] == "occ" else unoccupied
                target[mode[1]].extend(values)
    if not occupied:
        return None

    alpha = occupied.get(1) or []
    beta = occupied.get(2) or alpha
    res = {"orbital_energies_ev": {
        "alpha": [v * hartree_to_ev for v in alpha],
        "beta": [v * hartree_to_ev for v in beta],
    }}
    if alpha or beta:
        homo_h = max(alpha + beta)
        res["homo_ev"] = homo_h * hartree_to_ev
        res["ip_koopmans_ev"] = -res["homo_ev"]
    unocc_values = [v for values in unoccupied.values() for v in values]
    if unocc_values:
        res["lumo_ev"] = min(unocc_values) * hartree_to_ev
        res["ea_koopmans_ev"] = -res["lumo_ev"]
    if fermi is not None:
        res["fermi_ev"] = fermi
    return res


_S2_RE = re.compile(r"Ideal and single determinant S\*\*2\s*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)")
_ABS_SPIN_RE = re.compile(r"Integrated absolute spin density\s*:\s*(-?\d+\.\d+)")


def parse_spin_diagnostics(outfile: str) -> dict:
    """
      Integrated absolute spin density  :   0.0000000000
      Ideal and single determinant S**2 :   0.000000      -0.000000

    Отклонение <S²> от идеала — спиновое загрязнение (для Fe-кластеров это
    признак нефизического решения); интегральная |m| — доля дробного спина.
    """
    s2 = ideal = abs_spin = None
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _ABS_SPIN_RE.search(line)
            if m:
                abs_spin = float(m.group(1))
                continue
            m = _S2_RE.search(line)
            if m:
                ideal = float(m.group(1))
                s2 = float(m.group(2))
    res = {}
    if s2 is not None:
        res["s2"] = _clean(s2)
        res["s2_ideal"] = _clean(ideal)
    if abs_spin is not None:
        res["abs_spin_density"] = _clean(abs_spin)
    return res


def force_stats(forces_ev_ang) -> dict:
    """max и RMS по атомам: насколько структура ещё не в минимуме."""
    if not forces_ev_ang:
        return {}
    norms = [float(np.linalg.norm(f)) for f in forces_ev_ang]
    return {"force_max_ev_ang": max(norms),
            "force_rms_ev_ang": float(np.sqrt(np.mean(np.square(norms))))}


def collect_physical_props(outfile: str, forces_ev_ang=None,
                           hartree_to_ev: float = 27.211386245988) -> dict:
    """
    Компактный набор физических наблюдаемых из ОДНОГО лога (тот же прогон,
    которым мастер сертифицирует плотность). Ключи, которых CP2K не напечатал,
    в результат не попадают.
    """
    props = {}
    components = parse_energy_decomposition(outfile)
    if components:
        props["energy_components_hartree"] = components
        residual = energy_decomposition_residual_hartree(components)
        if residual is not None:
            props["energy_components_residual_hartree"] = residual
    mulliken = parse_mulliken(outfile)
    if mulliken:
        props["charges_mulliken"] = mulliken["charges"]
        props["spins_mulliken"] = mulliken["spins"]
        props["mulliken_total_charge"] = mulliken["total_charge"]
        props["mulliken_total_spin"] = mulliken["total_spin"]
    props.update(parse_dipole(outfile) or {})
    props.update(parse_eigenvalues(outfile, hartree_to_ev) or {})
    props.update(parse_spin_diagnostics(outfile))
    props.update(force_stats(forces_ev_ang))
    return props


# ---------- Парсинг AO-матриц (перенесено из master-worker.py) ----------
# ============================================================
# Разбор файла волновой функции CP2K (-RESTART.wfn)
# ============================================================
# Формат установлен зондом test/probe_wfn.py на CP2K 2024.3:
#   * Fortran unformatted, маркеры длины 4 байта;
#   * запись 20 байт (5 int32): nspin, ?, nao, ?, ?
#   * блоки 8*nao байт — столбцы матрицы коэффициентов, всего nspin*n_occ штук;
#   * запись 8*nspin*n_occ байт — собственные значения;
#   * координат, имени элемента, базиса и функционала в файле НЕТ.
WFN_RECORD_LIMIT = 64 * 1024 * 1024


class WfnError(Exception):
    """Состояние (wfn) не принято: файл не той структуры или не от этой задачи."""


def read_fortran_records(path: str, limit_bytes: int = WFN_RECORD_LIMIT) -> list:
    """
    Читает файл как цепочку Fortran unformatted записей (маркеры 4 байта).
    Возвращает [(offset, length), ...]; [] — если файл не такой структуры.
    """
    with open(path, "rb") as f:
        data = f.read(limit_bytes)
    records = []
    off = 0
    n = len(data)
    while off + 8 <= n:
        (length,) = struct.unpack_from("<i", data, off)
        if length < 0 or off + 8 + length > n:
            break
        (tail,) = struct.unpack_from("<i", data, off + 4 + length)
        if tail != length:
            break
        records.append((off, length))
        off += 8 + length
    return records


def parse_wfn_header(path: str) -> dict:
    """
    Синтаксический разбор присланного -RESTART.wfn БЕЗ запуска CP2K.

    Даёт: nspin, nao (число AO-функций) и число столбцов матрицы коэффициентов
    (= nspin * n_occ). Смена элемента, базиса, псевдопотенциала с другим q или
    числа атомов меняет nao, а значит и размер со структурой файла.

    :return: {ok, reason, size, nspin, nao, columns, header_ints, records}
    """
    info = {"ok": False, "reason": "", "size": 0, "nspin": None, "nao": None,
            "columns": 0, "header_ints": None, "records": []}
    if not path or not os.path.exists(path):
        info["reason"] = "файл состояния не найден"
        return info

    size = os.path.getsize(path)
    info["size"] = size
    if size < 32 or size % 4 != 0:
        info["reason"] = f"размер {size} не похож на wfn"
        return info

    records = read_fortran_records(path)
    lengths = [ln for _off, ln in records]
    info["records"] = lengths
    if not records:
        info["reason"] = "не Fortran unformatted (нет валидных записей)"
        return info

    if lengths[0] == 20:
        with open(path, "rb") as f:
            f.seek(records[0][0] + 4)
            head = f.read(20)
        ints = list(struct.unpack("<5i", head))
        info["header_ints"] = ints
        if ints[0] in (1, 2):
            info["nspin"] = ints[0]
        if 0 < ints[2] <= 100000:
            info["nao_header"] = ints[2]

    col_len = max(lengths)
    if col_len % 8 != 0:
        info["reason"] = f"длина блоков {col_len} не кратна 8"
        return info
    info["nao"] = col_len // 8
    info["columns"] = lengths.count(col_len)
    if info["nao"] <= 0 or info["columns"] <= 0:
        info["reason"] = "не нашёл матрицу коэффициентов"
        return info
    if info.get("nao_header") and info["nao_header"] != info["nao"]:
        info["reason"] = (f"nao из заголовка ({info['nao_header']}) != "
                          f"nao из блоков ({info['nao']})")
        return info

    info["ok"] = True
    return info


def parse_cp2k_system_info(log_path: str) -> dict:
    """Авторитетные ожидания из лога CP2K: nao, nMО, занятые орбитали, электроны."""
    pats = {
        "nao": r"Number of orbital functions\s*:?\s*(\d+)",
        "nmo": r"Number of molecular orbitals\s*:?\s*(\d+)",
        "n_occ": r"Number of occupied orbitals\s*:?\s*(\d+)",
        "nelec": r"Number of electrons\s*:?\s*(\d+)",
        "nspin": r"Number of spin channels\s*:?\s*(\d+)",
    }
    info = {}
    if not os.path.exists(log_path):
        return info
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for key, pat in pats.items():
                if key in info:
                    continue
                m = re.search(pat, line, re.IGNORECASE)
                if m:
                    info[key] = int(m.group(1))
    return info


def parse_scf_status(log_path: str) -> dict:
    """
    Сертификат сходимости из лога: «*** SCF run converged in N steps ***»
    (в противовес «*** SCF run NOT converged ***»), плюс последняя невязка OT
    (для диагностики: если она чуть выше EPS_SCF, виден хвост сходимости).
    """
    out = {"converged": False, "steps": None, "not_converged": False, "last_conv": None}
    if not os.path.exists(log_path):
        return out
    conv_re = re.compile(r"SCF run converged in\s+(\d+)\s+steps", re.IGNORECASE)
    ot_row_re = re.compile(r"^\s*\d+\s+OT\s+(?:DIIS|CG|SD|BFGS)\s+(.*)$", re.IGNORECASE)
    float_re = re.compile(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = conv_re.search(line)
            if m:
                out["converged"] = True
                out["steps"] = int(m.group(1))
            if re.search(r"SCF run NOT converged", line, re.IGNORECASE):
                out["not_converged"] = True
            m = ot_row_re.match(line)
            if m:
                # невязка — число перед энергией (< -1 Ha), как в parse_first_ot_step
                vals = [float(v) for v in float_re.findall(m.group(1))]
                neg = [i for i, v in enumerate(vals) if v < -1.0]
                if neg and neg[0] > 0:
                    out["last_conv"] = vals[neg[0] - 1]
    return out


def check_restart_reading(log_path: str, wfn_name: str = "") -> dict:
    """
    Проверяет, что CP2K РЕАЛЬНО прочитал присланный restart, а не откатился
    на ATOMIC-гад. Иначе мастер «проверял» бы не состояние воркера.
    """
    res = {"mentioned": False, "warnings": [], "lines": []}
    if not os.path.exists(log_path):
        return res
    warn_keys = ("NOT COMPATIBLE", "INCOMPATIBLE", "DIFFERENT NUMBER", "ATOMIC GUESS",
                 "COULD NOT READ", "CANNOT READ", "NOT FOUND", "IGNORED")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            up = line.upper()
            if ("RESTART" in up or "WFN" in up or "WAVEFUNCTION" in up) and \
                    ("READ" in up or "GUESS" in up or (wfn_name and wfn_name.upper() in up)):
                res["lines"].append(line.strip())
                res["mentioned"] = True
            if any(k in up for k in warn_keys) and ("RESTART" in up or "WFN" in up or
                                                   "WAVEFUNCTION" in up or "GUESS" in up):
                res["warnings"].append(line.strip())
    return res


def parse_first_ot_step(log_path: str) -> dict:
    """
    Лучшая попытка прочесть первую строку таблицы OT: невязка (градиент) и энергия
    ПРИСЛАННОГО состояния. Используется как информационная метрика; решение об
    приёмке принимается по parse_scf_status (+ энергия/силы собственного прогона).
    """
    out = {"grad": None, "energy_hartree": None, "line": None}
    if not os.path.exists(log_path):
        return out
    step_re = re.compile(r"^\s*1\s+OT\s+(?:DIIS|CG|SD|BFGS)\s+(.*)$", re.IGNORECASE)
    float_re = re.compile(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = step_re.match(line)
            if not m:
                continue
            vals = [float(v) for v in float_re.findall(m.group(1))]
            neg = [(i, v) for i, v in enumerate(vals) if v < -1.0]
            if neg:
                i, energy = neg[0]
                out["grad"] = vals[i - 1] if i > 0 else None
                out["energy_hartree"] = energy
                out["line"] = line.strip()
                return out
    return out


def verify_state_from_wfn(coords, wfn_path: str, cfg: dict, project: str = "master_state",
                          max_scf: int | None = None, timeout: int | None = None,
                          verbose: bool = False) -> dict:
    """
    ОДИН прогон CP2K ОТ присланного состояния (SCF_GUESS RESTART +
    WFN_RESTART_FILE_NAME): проверяет, что состояние стационарно для МОЕЙ задачи
    (элемент, базис, псевдопотенциал, ячейка, функционал, геометрия), и возвращает
    посчитанные мастером энергию, силы, gap, mag.

    :param max_scf: сколько шагов OT разрешено (по умолчанию RESTART_VERIFY_SCF = 3)
    :return: dict с ключами ok/reason/energy_*/forces_*/gap/mag/scf_*/sysinfo/...
    """
    workdir = tempfile.mkdtemp(prefix="master_state_")
    try:
        project = re.sub(r"[^a-zA-Z0-9_]", "_", project)
        max_scf = int(max_scf or RESTART_VERIFY_SCF)
        timeout = int(timeout or CP2K_TIMEOUT)

        local_wfn = os.path.join(workdir, f"{project}-RESTART.wfn")
        shutil.copyfile(wfn_path, local_wfn)
        submitted = parse_wfn_header(local_wfn)

        inp = os.path.join(workdir, f"{project}.inp")
        out = os.path.join(workdir, f"{project}.out")
        force_file = os.path.join(workdir, f"{project}.forces")

        text = render_cp2k_input(
            project, coords, cfg,
            run_type="ENERGY_FORCE",
            max_scf=max_scf,
            ignore_convergence=True,
            scf_guess="RESTART",
            wfn_restart_file=os.path.basename(local_wfn),
        )
        with open(inp, "w", encoding="utf-8") as f:
            f.write(text)

        cmd = cp2k_argv(cfg)
        t0 = time.time()
        try:
            with open(out, "w", encoding="utf-8") as fout:
                subprocess.run(cmd + ["-i", os.path.basename(inp)],
                               cwd=workdir, stdout=fout, stderr=subprocess.STDOUT,
                               text=True, timeout=timeout, env=cp2k_env(cfg))
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": f"таймаут {timeout} с при проверке состояния",
                    "submitted": submitted, "elapsed": time.time() - t0}
        except FileNotFoundError:
            return {"ok": False, "reason": f"команда CP2K '{cmd}' не найдена",
                    "submitted": submitted}
        elapsed = time.time() - t0

        if cfg.get("cleanup_cubes", CP2K_PARAMS["cleanup_cubes"]):
            cleanup_cube_files(workdir, project)

        res = {
            "ok": False, "reason": "",
            "submitted": submitted,
            "sysinfo": parse_cp2k_system_info(out),
            "scf": parse_scf_status(out),
            "restart": check_restart_reading(out, os.path.basename(local_wfn)),
            "first_step": parse_first_ot_step(out),
            "log_path": out,
            "elapsed": elapsed,
        }

        if verbose:
            print(f"    | [state] log: {out}")

        # 1) состояние реально прочитано, без предупреждений о несовместимости
        if res["restart"]["warnings"]:
            res["reason"] = ("CP2K предупредил о несовместимом restart: "
                             + " | ".join(res["restart"]["warnings"][:2]))
            return res
        if REQUIRE_RESTART_READ_CONFIRMATION and not res["restart"]["mentioned"]:
            res["reason"] = ("в логе нет подтверждения чтения restart "
                             "(возможен откат на ATOMIC-гад)")
            return res

        # 2) энергия собственного прогона
        try:
            energy_h = parse_energy(out)
        except Exception as e:
            res["reason"] = f"CP2K не выдал энергию: {e}"
            return res

        res["energy_hartree"] = energy_h
        res["energy_ev"] = energy_h * cfg["hartree_to_ev"]

        # 3) сертификат сходимости: состояние должно доводиться до SCF-сходимости
        #    за разрешённое число шагов
        if not res["scf"]["converged"]:
            conv = res["scf"].get("last_conv")
            eps = _num(cfg, "eps_scf")
            res["reason"] = (f"состояние не доводится до сходимости за {max_scf} шагов "
                             f"(последняя невязка {conv}, eps_scf={eps}); "
                             "не стационарно для этой геометрии/метода")
            return res

        # 4) силы и метаданные — считается мастером из того же состояния
        try:
            forces = read_forces(workdir, project, out)
        except Exception as e:
            res["reason"] = f"не удалось прочитать силы: {e}"
            return res
        if len(forces) != len(coords):
            res["reason"] = f"сил {len(forces)}, а атомов {len(coords)}"
            return res

        res["forces_ev_ang"] = [[fx * cfg["hartree_bohr_to_ev_ang"],
                                 fy * cfg["hartree_bohr_to_ev_ang"],
                                 fz * cfg["hartree_bohr_to_ev_ang"]] for fx, fy, fz in forces]
        res["gap"] = parse_homo_lumo(out, cfg["hartree_to_ev"])
        res["mag"] = parse_magnetic_moment(out)
        # Физические наблюдаемые — из этого же прогона мастера (не из чисел воркера).
        res["props"] = collect_physical_props(out, res["forces_ev_ang"],
                                              cfg["hartree_to_ev"])

        # 5) канонический wfn, который CP2K записал сам для этой задачи
        canon = local_wfn
        if os.path.exists(canon):
            res["canonical"] = parse_wfn_header(canon)

        res["ok"] = True
        return res
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------- Запуск CP2K (воркер) ----------
def _emit(cfg: dict, text: str):
    """Прогресс CP2K: в callback (лог воркера) или в stdout."""
    cb = cfg.get("progress")
    if callable(cb):
        cb(text)
    else:
        print(text, flush=True)


def run_cp2k_with_live_log(inp_file, out_file, timeout, workdir, cfg: dict):
    """
    Возвращает (is_cp2k_error, returncode).
    - returncode > 0: внутренняя ошибка CP2K (структуру надо пометить как сбойную).
    - returncode < 0: системная ошибка (SIGKILL/OOM), структуру помечать НЕЛЬЗЯ.
    - Таймаут/расхождение SCF: возвращаем (True, -1).
    - Успех: (False, 0).
    """
    interesting = re.compile(
        r"(SCF run converged|ENERGY\||ABORT|WARNING|error|Error|"
        r"GEO_OPT|step number|Informations at step)"
    )

    scf_step_re = re.compile(
        r"^\s*(\d+)\s+OT\s+(\S+)\s+([\d.Ee+-]+)\s+([\d.]+)\s+"
        r"([\d.Ee+-]+)\s+(-?\d+\.\d+)\s+(-?[\d.Ee+-]+)"
    )

    with open(out_file, "w", encoding="utf-8", errors="replace") as fout:
        try:
            proc = subprocess.Popen(
                cp2k_argv(cfg) + ["-i", os.path.basename(inp_file)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=workdir, env=cp2k_env(cfg),
                encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            _emit(cfg, f"    ! ОШИБКА: Команда '{cfg['cp2k_cmd']}' не найдена в PATH.")
            return (False, -2)

        start_time = time.time()
        last_print = start_time
        scf_header_shown = False

        scf_convergence_history = []
        MAX_STALLED_STEPS = 10
        last_lines = []

        try:
            for line in proc.stdout:
                fout.write(line)
                fout.flush()

                last_lines.append(line.rstrip())
                if len(last_lines) > 15:
                    last_lines.pop(0)

                # сырую шапку таблицы CP2K не печатаем — свою выдаём ниже
                noisy = "Update method" in line or line.strip().startswith("-----")
                if cfg.get("verbose_cp2k") and interesting.search(line) and not noisy:
                    _emit(cfg, f"    | {line.rstrip()}")
                    last_print = time.time()

                m_scf = scf_step_re.match(line)
                if m_scf:
                    step = int(m_scf.group(1))
                    time_s = float(m_scf.group(4))
                    conv = float(m_scf.group(5))
                    energy = float(m_scf.group(6))
                    change = float(m_scf.group(7))
                    scf_convergence_history.append(conv)
                    if cfg.get("verbose_cp2k"):
                        if not scf_header_shown:
                            _emit(cfg, "    | SCF (OT): шаг / сходимость / энергия")
                            _emit(cfg, f"    | {'шаг':>4} {'conv':>10} {'E, Ha':>16} "
                                       f"{'dE, Ha':>10} {'t, с':>6}")
                            _emit(cfg, "    | " + "─" * 52)
                            scf_header_shown = True
                        _emit(cfg, f"    | {step:4d} {conv:10.2e} {energy:16.8f} "
                                   f"{change:+10.2e} {time_s:6.1f}")
                        last_print = time.time()

                    if len(scf_convergence_history) > MAX_STALLED_STEPS:
                        recent = scf_convergence_history[-MAX_STALLED_STEPS:]
                        if min(recent) >= scf_convergence_history[-MAX_STALLED_STEPS - 1]:
                            _emit(cfg, f"    ! ОБНАРУЖЕНО РАСХОЖДЕНИЕ SCF на шаге {step}. Убиваю CP2K.")
                            proc.kill()
                            return (True, -1)

                if time.time() - last_print > 60:
                    elapsed = int(time.time() - start_time)
                    _emit(cfg, f"    | ... (тишина {elapsed} с, CP2K работает)")
                    last_print = time.time()

                if timeout is not None and (time.time() - start_time) > timeout:
                    _emit(cfg, f"    ! ТАЙМАУТ ({timeout} с) — убиваю CP2K")
                    proc.kill()
                    return (True, -1)

            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            return (True, -1)

        rc = proc.returncode

        if rc > 0:
            _emit(cfg, f"    ! CP2K завершился с кодом возврата {rc} (внутренняя ошибка).")
            _emit(cfg, "    ! Последние строки вывода:")
            for l in last_lines:
                _emit(cfg, f"      > {l}")
            _emit(cfg, f"    ! Полный лог сохранен в: {out_file}")
            return (True, rc)

        if rc < 0:
            _emit(cfg, f"    ! CP2K убит сигналом {-rc} (системная ошибка, вероятно OOM).")
            _emit(cfg, "    ! Структура НЕ будет помечена как сбойная.")
            for l in last_lines:
                _emit(cfg, f"      > {l}")
            _emit(cfg, f"    ! Полный лог сохранен в: {out_file}")
            return (False, rc)

        return (False, 0)


# ---------- Энергия изолированного атома ----------
def calculate_isolated_atom_energy(cfg: dict):
    """Считает (или берёт из кэша) энергию изолированного атома, eV."""
    _emit(cfg, "\n" + "=" * 60)
    _emit(cfg, f"ЭНЕРГИЯ ИЗОЛИРОВАННОГО АТОМА {cfg['element']}")
    _emit(cfg, "=" * 60)

    cache = cfg["isolated_atom_energy_file"]
    if os.path.exists(cache):
        try:
            with open(cache, "r") as f:
                energy_ev = float(f.read().strip())
            _emit(cfg, f"✓ из кэша: {energy_ev:.6f} eV")
            return energy_ev
        except Exception as e:
            _emit(cfg, f"⚠ ошибка кэша: {e}; пересчитываю")
            try:
                os.remove(cache)
            except OSError:
                pass

    project = f"{cfg['element']}_isolated_atom"
    inp_path = os.path.join(cfg["output_dir"], f"{project}.inp")
    out_path = os.path.join(cfg["output_dir"], f"{project}.out")

    half = resolve_cell_size([(0.0, 0.0, 0.0)], cfg) / 2.0
    coords = [(half, half, half)]
    write_cp2k_input(inp_path, project, coords, cfg)

    is_cp2k_error, rc = run_cp2k_with_live_log(
        inp_path, out_path, CP2K_TIMEOUT, cfg["output_dir"], cfg
    )
    if cfg.get("cleanup_cubes", CP2K_PARAMS["cleanup_cubes"]):
        cleanup_cube_files(cfg["output_dir"], project)
    if is_cp2k_error or rc != 0:
        _emit(cfg, "✗ не удалось посчитать изолированный атом — энергия связи не будет рассчитана")
        return None

    try:
        energy_hartree = parse_energy(out_path)
        energy_ev = energy_hartree * cfg["hartree_to_ev"]
        _emit(cfg, f"✓ E_iso = {energy_ev:.6f} eV ({energy_hartree:.8f} Ha)")
        with open(cache, "w") as f:
            f.write(f"{energy_ev:.8f}\n")
        return energy_ev
    except Exception as e:
        _emit(cfg, f"✗ ошибка разбора E_iso: {e}")
        return None


# ---------- Высокоуровневый API ----------
def build_xyz_record(coords, energy_ev, forces_ev_ang, n_atoms, dim, struct_id,
                     cfg: dict, gap_ev=None, mag_moment=None,
                     e_binding_per_atom=None, e_isolated_atom=None,
                     extra_comment=""):
    extra = ""
    if gap_ev is not None:
        extra += f"homo_lumo_gap={gap_ev:.6f} "
    if mag_moment is not None:
        extra += f"magnetic_moment={mag_moment:.6f} "
    if e_isolated_atom is not None:
        extra += f"e_isolated_atom={e_isolated_atom:.8f} "
    if e_binding_per_atom is not None:
        extra += f"binding_energy_per_atom={e_binding_per_atom:.8f} "
    extra += extra_comment

    comment = (
        f'Lattice="{cfg["lattice_str"]}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={energy_ev:.8f} '
        f'n_atoms={n_atoms} dim={dim} struct_id={struct_id} '
        f'{extra}'
        f'pbc="F F F"'
    )
    lines = [str(n_atoms), comment]
    species = cfg.get("species") or [cfg["element"]] * n_atoms
    for (x, y, z), (fx, fy, fz), symbol in zip(coords, forces_ev_ang, species):
        lines.append(
            f"{symbol} {x:.8f} {y:.8f} {z:.8f} "
            f"{fx:.8f} {fy:.8f} {fz:.8f}"
        )
    return "\n".join(lines) + "\n"


def run_dft_for_single_structure(coords, n_atoms, dim, struct_id, cfg,
                                 force_recompute=False, save_wfn=True):
    """
    Считает DFT для одной структуры с кэшированием.

    save_wfn=True — в результат кладётся путь к <project>-RESTART.wfn, который
    CP2K пишет сам; его воркер и отправляет мастеру.

    Возвращает dict
        {energy_ev, energy_hartree, forces_ev_ang, gap, mag, cached, elapsed, hash,
         scf_converged, scf_steps, wfn_path}
    или dict с ключом "error": True при ошибке.
    """
    safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", str(struct_id))
    ghash = _geometry_hash(coords, cfg)
    project = f"{cfg['element']}{n_atoms}_{safe_id}_{ghash}"

    inp_path = os.path.join(cfg["output_dir"], f"{project}.inp")
    out_path = os.path.join(cfg["output_dir"], f"{project}.out")
    result_json = os.path.join(cfg["output_dir"], f"{project}.json")

    # --- 1) Проверка кэша ---
    if not force_recompute and os.path.exists(result_json):
        try:
            with open(result_json, "r", encoding="utf-8") as f:
                cached = json.load(f)
            need_wfn = save_wfn and not cached.get("wfn_path")
            need_props = cached.get("props_version") != PROPS_VERSION
            if need_wfn or need_props:
                # кэш от старой версии/без нужных данных — пересчитываем
                print(f"  ⚠ {struct_id}: в кэше не хватает данных, пересчитываю...")
            else:
                print(f"  ✓ {struct_id}: взято из кэша (hash={ghash})")
                return cached
        except Exception as e:
            print(f"  ⚠ Ошибка чтения кэша {result_json}: {e}. Пересчитываю...")

    # --- 2) Запуск CP2K ---
    write_cp2k_input(inp_path, project, coords, cfg)
    t0 = time.time()
    is_cp2k_error, rc = run_cp2k_with_live_log(
        inp_path, out_path, CP2K_TIMEOUT, cfg["output_dir"], cfg
    )
    elapsed = time.time() - t0

    if cfg.get("cleanup_cubes", CP2K_PARAMS["cleanup_cubes"]):
        cleanup_cube_files(cfg["output_dir"], project)

    if rc != 0:
        return {
            "error": True,
            "rc": rc,
            "is_cp2k_error": is_cp2k_error,
            "elapsed": elapsed,
        }

    # --- 3) Парсинг ---
    try:
        energy = parse_energy(out_path)
        forces = read_forces(cfg["output_dir"], project, out_path)
        gap = parse_homo_lumo(out_path, cfg["hartree_to_ev"])
        mag = parse_magnetic_moment(out_path)
    except Exception as e:
        print(f"  ✗ Ошибка парсинга {struct_id}: {e}")
        return {"error": True, "rc": 0, "is_cp2k_error": True, "elapsed": elapsed}

    if len(forces) != n_atoms:
        print(f"  ✗ {struct_id}: ожидалось {n_atoms} атомов, получено {len(forces)}")
        return {"error": True, "rc": 0, "is_cp2k_error": True, "elapsed": elapsed}

    energy_ev = energy * cfg["hartree_to_ev"]
    forces_ev_ang = [
        [fx * cfg["hartree_bohr_to_ev_ang"],
         fy * cfg["hartree_bohr_to_ev_ang"],
         fz * cfg["hartree_bohr_to_ev_ang"]]
        for fx, fy, fz in forces
    ]

    scf = parse_scf_status(out_path)

    result = {
        "energy_ev": energy_ev,
        "energy_hartree": energy,
        "forces_ev_ang": forces_ev_ang,
        "gap": gap,
        "mag": mag,
        "cached": False,
        "elapsed": elapsed,
        "hash": ghash,
        "cell_size": resolve_cell_size(coords, cfg),
        "scf_converged": bool(scf["converged"]),
        "scf_steps": scf["steps"],
    }
    # Физические наблюдаемые из того же прогона: диполь, заряды/спины,
    # разложение энергии, занятый спектр, IP/EA по Купмансу.
    result.update(collect_physical_props(out_path, forces_ev_ang, cfg["hartree_to_ev"]))
    result["props_version"] = PROPS_VERSION

    # --- 3a) файл состояния, который CP2K пишет сам (режим restart-состояния) ---
    if save_wfn:
        wfn_path = os.path.join(cfg["output_dir"], f"{project}-RESTART.wfn")
        result["wfn_path"] = wfn_path if os.path.exists(wfn_path) else None

    # --- 4) Сохраняем кэш ---
    try:
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠ Не удалось сохранить кэш {result_json}: {e}")

    return result


def run_dft_for_structures(structures, xyz_path, done_ids, failed_ids, cfg: dict):
    with open(xyz_path, "a") as xyz:
        for idx, s in enumerate(structures):
            n_atoms = s["n_atoms"]
            dim = s["dim"]
            struct_id = s["struct_id"]

            if struct_id in done_ids:
                print(f"\n[{idx+1}/{len(structures)}] {struct_id} — уже посчитано, пропускаю")
                continue

            if struct_id in failed_ids:
                print(f"\n[{idx+1}/{len(structures)}] {struct_id} — ранее признано сбойным, пропускаю")
                continue

            coords = s["coords"]
            print(f"\n[{idx+1}/{len(structures)}] {struct_id} (n={n_atoms}, dim={dim})")

            res = run_dft_for_single_structure(coords, n_atoms, dim, struct_id, cfg)

            if res.get("error"):
                rc = res.get("rc", 0)
                elapsed = res.get("elapsed", 0)
                if rc > 0 or res.get("is_cp2k_error"):
                    print(f"  ✗ Внутренняя ошибка CP2K (код {rc}) за {elapsed:.1f} с")
                    print(f"  ! Помечаю {struct_id} как сбойную")
                    _mark_as_failed(struct_id, cfg["failed_ids_file"])
                    failed_ids.add(struct_id)
                else:
                    print(f"  ⚠ Системная ошибка (код {rc}) за {elapsed:.1f} с")
                    print(f"  ! Структура {struct_id} НЕ помечена как сбойная.")
                continue

            energy_ev = res["energy_ev"]
            forces_ev_ang = res["forces_ev_ang"]
            gap = res["gap"]
            mag = res["mag"]
            e_per_atom = energy_ev / n_atoms

            # Энергия связи на атом
            e_iso = cfg.get("e_isolated_atom")
            be_per_atom = None
            if e_iso is not None:
                be_per_atom = (e_iso * n_atoms - energy_ev) / n_atoms

            msg = f"  ✓ OK: E = {energy_ev:.6f} eV"
            if res.get("cached"):
                msg += " (из кэша)"
            else:
                msg += f" за {res['elapsed']:.1f} с"
            if be_per_atom is not None:
                msg += f", E_binding/atom = {be_per_atom:.6f} eV"
            else:
                msg += f", E/atom = {e_per_atom:.6f} eV"
            if gap is not None:
                msg += f", Gap = {gap:.4f} eV"
            if mag is not None:
                msg += f", MagMom = {mag:.4f} μB"
            print(msg)

            xyz.write(build_xyz_record(
                coords, energy_ev, forces_ev_ang, n_atoms, dim, struct_id,
                cfg, gap, mag,
            ))
            xyz.flush()
            done_ids.add(struct_id)


def _mark_as_failed(struct_id, filepath):
    with open(filepath, "a") as f:
        f.write(f"{struct_id}\n")


def read_failed_ids(filepath):
    failed = set()
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    failed.add(line)
    return failed
