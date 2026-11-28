#!/usr/bin/env python3
"""
Консенсусные константы и расписание размеров — единственный источник правды.

Всё, что здесь лежит, обязано совпадать у всех узлов сети. Если два мастера
по-разному решают «какой размер кластера на высоте H» или «сколько монет за
блок», их блоки не пройдут проверку друг у друга — сеть расколется. Поэтому
эти числа НЕ читаются из config.json: их меняют только вместе с версией
протокола (сознательный форк).

Здесь же — проверка того, что блок соответствует расписанию: её выполняет и
мастер (при выдаче и приёмке), и каждый пир (при проверке цепочки).
"""

BLOCK_VERSION = 5

# ---------- деньги ----------
# Единица = 1 монета, дробей нет вообще (float в деньгах — дыра, см. AUDIT §L3).
COINBASE_VALUE = 50_000_000
REWARD_VALUE = 25_000_000
TX_FEE = 100_000
MAX_TXS_PER_BLOCK = 2_000
DEFAULT_CONFIRMATIONS = 1
# Зрелость coinbase — консенсус: выход блока N тратится не раньше N + 1.
COINBASE_MATURITY = 1

# ---------- расписание размеров ----------
# Этап s покрывает B(s) = base_blocks * s^exponent блоков, размер кластера
# растёт на формульную единицу за этап: n = start + (s-1)*unit.
ATOM_SCHEDULE = {
    "start_atoms": 2,     # первый размер кластера
    "base_blocks": 10,    # блоков на первый этап
    "exponent": 1.5,      # B(s) = base_blocks * s^exponent
    # Потолок размера: дальше расписание упирается в него. 0 — без потолка,
    # но тогда мастер отказался бы принимать задачи (MAX_TASK_ATOMS), а пиры
    # принимали бы блоки, которые никто не может собрать. Держим равным
    # MAX_TASK_ATOMS, чтобы расписание и приёмка не расходились.
    "max_atoms": 2_000,
}

# Размерности чередуются: объёмная (3) и плоская (2).
# Консенсус проверяет только принадлежность этому набору: точная размерность
# блока зависит от локального счётчика заданий мастера, а не от высоты,
# поэтому «dim = DIM_CYCLE[height % 2]» развёл бы сеть.
DIM_CYCLE = (3, 2)

# Абсолютный предел парсера: n_atoms из заголовка больше этого не бывает.
# Он выше потолка расписания намеренно — это защита от мусора в файле,
# а не правило эмиссии.
MAX_N_ATOMS = 10_000

# Потолок задачи для приёмки (анти-DoS): n_atoms приходит из запроса, поэтому
# всё, что больше, отбивается до аллокаций numpy и запуска CP2K.
MAX_TASK_ATOMS = 2_000

# Физический минимум между атомами (Å). Не консенсус сам по себе: консенсус —
# что геометрия равна Generate(seed), а генератор этот минимум соблюдает.
MIN_ATOM_DISTANCE_ANG = 0.5


def stage_for_height(height: int, formula_size: int = 1) -> dict:
    """
    Этап расписания для высоты `height` (сколько блоков уже принято).

    n_atoms = start + (s-1)*formula_size, blocks = base * s^exponent.
    `formula_size` — сумма количеств в composition (Cu:1, Al2O3:5).

    Функция чистая и одинаковая у всех: мастер выдаёт по ней задания,
    а пиры проверяют, что блок собран по ней же.
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
