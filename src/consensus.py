#!/usr/bin/env python3
"""
Консенсусные константы и расписание размеров — единственный источник правды.

Всё, что здесь лежит, обязано совпадать у всех узлов сети. Если два мастера
по-разному решают «какой размер кластера на высоте H», «сколько монет за
блок» или «каким профилем считать DFT», их блоки не пройдут проверку друг у
друга — сеть расколется. Поэтому эти числа НЕ читаются из config.json: их
меняют только вместе с версией протокола (сознательный форк).
"""

BLOCK_VERSION = 6

# ---------- деньги ----------
# Единица = 1 монета, дробей нет вообще (float в деньгах — дыра, см. AUDIT §L3).
COINBASE_VALUE = 50_000_000      # субсидия при start_atoms (см. SUBSIDY_PER_ATOM)
REWARD_VALUE = 25_000_000        # доля воркера при start_atoms
TX_FEE = 100_000
MAX_TXS_PER_BLOCK = 2_000
DEFAULT_CONFIRMATIONS = 1
# Зрелость coinbase — консенсус: выход блока N тратится не раньше N + 1.
COINBASE_MATURITY = 1

# ---------- расписание размеров ----------
ATOM_SCHEDULE = {
    "start_atoms": 2,     # первый размер кластера
    "base_blocks": 10,    # блоков на первый этап
    "exponent": 1.5,      # B(s) = base_blocks * s^exponent
    "max_atoms": 2_000,   # потолок размера (держится равным MAX_TASK_ATOMS)
}

# Размерности чередуются по ВЫСОТЕ (протокол v6): block seq → dim.
# Раньше размерность зависела от локального счётчика заданий мастера, поэтому
# два мастера выдавали разные dim на одной высоте — это нельзя проверить
# консенсусом. Теперь dim = DIM_CYCLE[(seq-1) % 2].
DIM_CYCLE = (3, 2)

# Предел парсера: n_atoms из заголовка больше этого не бывает.
MAX_N_ATOMS = 10_000
# Потолок задачи для приёмки (анти-DoS): n_atoms приходит из запроса.
MAX_TASK_ATOMS = 2_000
# Физический минимум между атомами (Å) — свойство генератора, не консенсус.
MIN_ATOM_DISTANCE_ANG = 0.5


# ---------- вычислительный профиль (консенсус) ----------
# Числа блока воспроизводимы только при одинаковой арифметике: одна сборка
# CP2K, одно число рангов, одни и те же сетка/функционал/решатель. Профиль —
# часть консенсуса: блок с чужим профилем чужой.
#
# Число OMP-потоков сюда НЕ входит: это локальная настройка производительности.
# Ранги входят: число рангов меняет разбиение сетки и порядок сумм, то есть
# сами числа, а потоки меняют только порядок внутри блочных операций.
CP2K_PROFILE = {
    "cutoff": 400,
    "rel_cutoff": 50,
    "eps_default": 1.0e-10,
    "eps_scf": 1.0e-6,
    "max_scf": 200,
    "box_padding": 6.0,
    "min_cell_size": 20.0,
    "max_cell_size": 100.0,
    "ot_minimizer": "DIIS",
    "ot_preconditioner": "FULL_SINGLE_INVERSE",
    "mpi_ranks": 1,
}
PROFILE_KEYS = tuple(sorted(CP2K_PROFILE))


def profile_mismatch(params) -> list[str]:
    """Какие ключи профиля не совпадают с сетью (пусто — блок наш)."""
    if not isinstance(params, dict):
        return ["cp2k_params — не объект"]
    bad = []
    for key in PROFILE_KEYS:
        if key not in params:
            bad.append(f"{key}: нет")
        elif params[key] != CP2K_PROFILE[key]:
            bad.append(f"{key}: {params[key]!r} != {CP2K_PROFILE[key]!r}")
    return bad


# ---------- расписание и награда ----------
def stage_for_height(height: int, formula_size: int = 1) -> dict:
    """
    Этап расписания для высоты `height` (сколько блоков уже принято).

    n_atoms = start + (s-1)*formula_size, blocks = base * s^exponent.
    `formula_size` — сумма количеств в composition (Cu:1, Al2O3:5).
    """
    base = max(1, int(ATOM_SCHEDULE["base_blocks"]))
    power = float(ATOM_SCHEDULE["exponent"])
    unit = max(1, int(formula_size or 1))
    start = max(int(ATOM_SCHEDULE["start_atoms"]), unit)
    max_atoms = int(ATOM_SCHEDULE["max_atoms"] or 0)

    acc, s = 0, 1
    while True:
        blocks = max(1, int(round(base * (s ** power))))
        n_atoms = start + (s - 1) * unit
        if max_atoms and n_atoms > max_atoms:
            n_atoms = max_atoms
        if height < acc + blocks:
            return {"stage": s, "n_atoms": n_atoms, "blocks": blocks,
                    "done": height - acc, "height": height}
        acc += blocks
        s += 1


def expected_n_atoms(height: int, formula_size: int = 1) -> int:
    return int(stage_for_height(height, formula_size)["n_atoms"])


def dim_for_height(height: int) -> int:
    """Размерность блока задаётся высотой: 3D, 2D, 3D, 2D…"""
    return int(DIM_CYCLE[int(height) % len(DIM_CYCLE)])


# ---------- эмиссия ----------
# Плата за полезную работу: 25 000 000 монет за атом-блок (при старте n=2 это
# прежние 50 000 000). Эмиссия пропорциональна посчитанным атомам, а реальный
# темп ограничен физикой: блок на n атомов требует n-атомного DFT и его
# проверки каждым пиром. Поэтому «напечатать» монеты, не считая, нельзя.
SUBSIDY_PER_ATOM = COINBASE_VALUE // int(ATOM_SCHEDULE["start_atoms"])


def subsidy_for(n_atoms: int) -> int:
    """Субсидия блока с n атомами (до комиссий)."""
    return int(SUBSIDY_PER_ATOM) * int(n_atoms)


def coinbase_split(n_atoms: int) -> tuple[int, int]:
    """
    Как субсидия делится в coinbase: (мастеру, воркеру).

    Половина — тому, кто считал (воркер), половина — тому, кто проверил и
    собрал блок (мастер). Раньше награда воркеру шла отдельным переводом из
    монет мастера: на больших кластерах мастер физически не мог её заплатить
    (награда росла, его баланс — нет). Теперь платит сеть, из эмиссии.
    """
    total = subsidy_for(n_atoms)
    worker = total // 2
    return total - worker, worker


# ---------- допуски ----------
def peer_tol_energy_ev(n_atoms: int) -> float:
    """
    Допуск на энергию при пересчёте блока пиром.

    Абсолютные 0.02 эВ — для димера; для большого кластера та же абсолютная
    величина означает всё более жёсткое требование к совпадению, поэтому
    допуск растёт линейно с числом атомов (0.1 мэВ на атом), но не меньше
    0.02 эВ. Подделка (чужая плотность/другая геометрия) ошибается на
    единицы эВ — такой допуск её не пропускает.
    """
    return max(0.02, 1e-4 * float(max(1, int(n_atoms))))
