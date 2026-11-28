#!/usr/bin/env python3
"""
Консенсусные константы и расписание размеров — единственный источник правды.

Всё, что здесь лежит, обязано совпадать у всех узлов сети. Если два мастера
по-разному решают «какой размер кластера на высоте H», «сколько монет за
блок» или «каким профилем считать DFT», их блоки не пройдут проверку друг у
друга — сеть расколется. Поэтому эти числа НЕ читаются из config.json: их
меняют только вместе с версией протокола (сознательный форк).
"""

import re

BLOCK_VERSION = 7

# ---------- деньги ----------
# Единица = 1 монета, дробей нет вообще (float в деньгах — дыра, см. AUDIT §L3).
COINBASE_VALUE = 50_000_000      # субсидия при start_atoms
REWARD_VALUE = 25_000_000        # доля воркера при start_atoms
# Комиссии НЕ В КОНСЕНСУСЕ: транзакция может платить любую комиссию, включая 0.
# Консенсус проверяет только «входы ≥ выходов» и что coinbase не больше
# субсидии + фактических комиссий блока. Число ниже — просто значение по
# умолчанию для кошелька (src/chain.py::DEFAULT_FEE), а не правило сети.
MAX_TXS_PER_BLOCK = 2_000
DEFAULT_CONFIRMATIONS = 1
# Зрелость coinbase — консенсус: выход блока N тратится не раньше N + 1.
COINBASE_MATURITY = 1

# atom_diameter попадает в заголовок и в block_id. В заголовке он хранится
# СТРОКОЙ с ровно тремя знаками после точки: "2.560". Число 2.56 в JSON
# нельзя записать как «2.560» — хвостовой ноль теряется, и «2.56» / «2.560» /
# «2» дают разные block_id у одного и того же блока. Строка снимает вопрос:
# у поля ровно одна запись, и она же — предмет консенсуса.
#
# Почему 3 знака: диаметр — металлическая длина связи, величина порядка
# единиц Å. Точность тоньше тысячной ангстрема физически бессмысленна (на
# этих масштабах главную роль играет соотношение неопределённостей).
#
# Лишние знаки ОТБРАСЫВАЮТСЯ, а не округляются: 2.5689 -> 2.568. Округление
# (2.569) молча меняло бы распределение геометрий, то есть master и worker
# считали бы разные кластеры при формально одном «диаметре».
DIAMETER_DECIMALS = 3
DIAMETER_UNITS = 10 ** DIAMETER_DECIMALS        # 1 Å = 1000 тысячных
# Консенсус: диаметр меньше 0.5 Å запрещён. min_dist = diameter, то есть
# такой «диаметр» ставит атомы ближе физического предела связи — это уже не
# атом, а точка, и геометрия не имеет смысла.
MIN_DIAMETER_ANG = 0.5
MIN_DIAMETER_UNITS = int(MIN_DIAMETER_ANG * DIAMETER_UNITS)
# Каноническая запись заголовка: "2.560", "12.000", без ведущих нулей.
DIAMETER_RE = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{%d}$" % DIAMETER_DECIMALS)


def diameter_milliunits(value) -> int:
    """Диаметр в тысячных долях Å, усечением (без округления).

    Разбор идёт по десятичной записи числа, а не по float-арифметике:
    int(2.5689 * 1000) на двоичных дробях даёт то 2568, то 2569.
    """
    from decimal import Decimal

    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("atom_diameter не задан")
    d = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    if not d.is_finite():
        raise ValueError(f"atom_diameter {value!r} - не конечное число")
    # int() от Decimal усекает к нулю: 2.5689 -> 2568, без округления.
    units = int(d * DIAMETER_UNITS)
    if units < MIN_DIAMETER_UNITS:
        raise ValueError(
            f"atom_diameter {value} Å меньше физического минимума "
            f"{MIN_DIAMETER_ANG} Å (min_dist = diameter)")
    return units


def canonical_diameter(value) -> float:
    """Каноническое число для расчётов: 3 знака, усечение."""
    return diameter_milliunits(value) / DIAMETER_UNITS


def canonical_diameter_text(value) -> str:
    """Каноническая запись для заголовка: ровно 3 знака после точки."""
    units = diameter_milliunits(value)
    return f"{units // DIAMETER_UNITS}.{units % DIAMETER_UNITS:03d}"


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
# Физический минимум между атомами (Å). Диаметр — это min_dist генератора,
# поэтому предел тот же, что у диаметра (консенсус, см. MIN_DIAMETER_ANG).
MIN_ATOM_DISTANCE_ANG = MIN_DIAMETER_ANG


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
    """
    Какие ключи профиля не совпадают с сетью (пусто — блок наш).

    Сверяем И значение, И тип: `400.0` вместо `400` — то же число, но другие
    байты JSON, а значит другой `block_id` у того же блока. Один профиль —
    одно каноническое представление.
    """
    if not isinstance(params, dict):
        return ["cp2k_params — не объект"]
    bad = []
    for key in PROFILE_KEYS:
        want = CP2K_PROFILE[key]
        if key not in params:
            bad.append(f"{key}: нет")
        elif type(params[key]) is not type(want) or params[key] != want:
            bad.append(f"{key}: {params[key]!r} != {want!r}")
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
# Эмиссия за блок фиксирована и не зависит от размера кластера: COINBASE_VALUE
# на блок, как субсидия в Bitcoin. Соблазн платить «за атом» выглядит честнее
# по работе, но это инфляция: чем крупнее кластер, тем больше монет печатает
# один и тот же блок, и денежная масса растёт вместе с расписанием. Плата за
# редкость крупных блоков — дело комиссий, а не эмиссии.


def subsidy_for(n_atoms: int) -> int:
    """Субсидия блока (до комиссий). Не зависит от n — эмиссия фиксирована."""
    return int(COINBASE_VALUE)


def coinbase_split(n_atoms: int) -> tuple[int, int]:
    """
    Как субсидия делится в coinbase: (мастеру, воркеру).

    25 000 000 — тому, кто считал (воркер), 25 000 000 — тому, кто проверил
    и собрал блок (мастер). Раньше награда воркеру шла отдельным переводом из
    монет мастера: блок на большом кластере было бы нечем оплатить. Теперь
    платит сеть — из фиксированной эмиссии этого блока.
    """
    total = subsidy_for(n_atoms)
    worker = total // 2
    return total - worker, worker


# ---------- состояние (wfn) ----------
def max_state_bytes(n_atoms: int) -> int:
    """
    Бюджет файла состояния для кластера из n атомов.

    Файл -RESTART.wfn — матрица плотности в AO-базисе: размер ~ nao²/2 · 4 Б,
    nao ≈ 32 на атом (Cu/MOLOPT). Значит объём растёт как n²: 9.5 КБ при n=2,
    ~7 МБ при 55, ~0.6 ГБ при 500, ~2.4 ГБ при 1000.

    Берём двойной запас на базисные наборы потяжелее и не опускаемся ниже
    64 МБ: маленьким кластерам лимит не мешает, а большой запрос отбивается
    до того, как съест память.
    """
    n = max(1, int(n_atoms))
    estimate = 2 * 4 * (32 * n) ** 2 // 2
    return max(64 * 1024 * 1024, int(estimate))


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
