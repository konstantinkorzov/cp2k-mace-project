#!/usr/bin/env python3
"""
Модуль генерации случайных кластеров Cu_n (n = 2..N) в 1D, 2D, 3D.

Публичные функции:
  * generate_structures(config) -> str          — пакетная генерация (старая схема);
  * generate_random_cluster(n_atoms, dim, cfg)  — ОДНА случайная структура заданного
    размера. Её вызывает воркер (worker.py), когда мастер присылает только желаемое
    число атомов в кластере.

Все константы берутся из cfg (dict), который передаёт главный модуль/мастер.
"""

import os
import random

import numpy as np


# ============================================================
# Параметры генерации структур — константы (в config.json их нет).
# ============================================================
BOX_MARGIN = 1.5               # куб генерации = max_dist * n^(1/3) * BOX_MARGIN
# Раньше было max_dist*sqrt(n)*4: для 55 атомов это ~118 Å — атомы разлетались
# в облако, а ячейка CP2K упиралась в предел. Кластер растёт как n^(1/3).
RANDOM_SEED = 42               # воспроизводимость
MAX_ATTEMPTS_PER_ATOM = 10000  # попыток на размещение одного атома

# Расстояния задаются через диаметр атома (config.json: atom_diameter, Å):
#   min_dist = diameter      — ближе друг к другу атомы не ставим;
#   max_dist = diameter * 2  — у каждого атома должен быть ХОТЯ БЫ ОДИН сосед
#                              не дальше этого расстояния. Это НЕ ограничение
#                              на расстояния до всех остальных атомов: дальние
#                              атомы могут быть дальше max_dist.
MIN_DIST_FACTOR = 1.0
MAX_DIST_FACTOR = 2.0


def atom_diameter(cfg: dict) -> float:
    """Диаметр из cfg. Канонизируется так же, как в заголовке блока
    (consensus: 3 знака, усечение), иначе локальный прогон с диаметром
    «2.5689» дал бы другую геометрию, чем приёмка с «2.568»."""
    raw = cfg.get("atom_diameter")
    if raw is None or raw == "":
        raw = 2.0
    try:
        import consensus
    except ImportError:                      # запуск без src в sys.path
        return float(raw)
    # ValueError (мусор, диаметр < минимума) намеренно не глотается:
    # молча посчитать кластер с другим диаметром — это другой датасет.
    return consensus.canonical_diameter(raw)


def composition(cfg: dict) -> dict:
    """Формульная единица системы, например {'Al': 2, 'O': 3}."""
    comp = cfg.get("composition") or {}
    if comp:
        return {str(k): int(v) for k, v in comp.items()}
    return {str(cfg.get("element") or "X"): 1}


def formula_size(cfg: dict) -> int:
    """Сколько атомов в одной формульной единице."""
    return sum(composition(cfg).values())


def assign_species(n_atoms: int, cfg: dict, seed: int | None = None) -> list[str]:
    """
    Раскладывает формулу по `n_atoms` атомам и перемешивает.

    Например, Al2O3 при n_atoms=10 — это 4 Al и 6 O в случайном порядке:
    координаты генерируются без учёта сорта, поэтому любая перестановка
    даёт новую валидную структуру того же состава.
    """
    comp = composition(cfg)
    unit = sum(comp.values())
    if n_atoms % unit != 0:
        raise ValueError(f"n_atoms={n_atoms} не кратно формуле {comp} ({unit} атомов)")
    mult = n_atoms // unit
    species = []
    for symbol, count in comp.items():
        species.extend([symbol] * (count * mult))
    random.Random(seed).shuffle(species)
    return species


def min_dist(cfg: dict) -> float:
    return atom_diameter(cfg) * MIN_DIST_FACTOR


def max_dist(cfg: dict) -> float:
    return atom_diameter(cfg) * MAX_DIST_FACTOR


def _compute_box_size(n_atoms: int, cfg: dict) -> float:
    """Куб генерации: растёт как радиус шара (n^(1/3)), а не как sqrt(n)."""
    return max_dist(cfg) * (n_atoms ** (1.0 / 3.0)) * BOX_MARGIN


def _random_point(dim, box_size, rng=random):
    """Случайная точка. rng — модуль random (legacy) или random.Random (потокобезопасно)."""
    if dim == 1:
        return np.array([rng.uniform(0, box_size), 0.0, 0.0])
    elif dim == 2:
        return np.array([rng.uniform(0, box_size), rng.uniform(0, box_size), 0.0])
    elif dim == 3:
        return np.array([rng.uniform(0, box_size),
                         rng.uniform(0, box_size),
                         rng.uniform(0, box_size)])
    else:
        raise ValueError("dim должен быть 1, 2 или 3")

def generate_extra_structures(config: dict, n_atoms: int, count: int, start_index: int = 0) -> str:
    """
    Генерирует `count` дополнительных структур для заданного n_atoms
    и возвращает путь к отдельному XYZ-файлу с ними.
    
    struct_id: f"{n_atoms}_{dim}_{start_index+i}_seed{seed}_d{d}_extra"
    """
    random_seed = RANDOM_SEED
    atom_diameter = config["atom_diameter"]
    dims = config["dimensions_per_n"].get(str(n_atoms), [3])
    # config["dimensions_per_n"] может быть с int-ключами, если из JSON уже приведён
    if isinstance(list(config["dimensions_per_n"].keys())[0], int):
        dims = config["dimensions_per_n"].get(n_atoms, [3])

    output_xyz = config["output_xyz"].replace(
        ".xyz", f"_extra_n{n_atoms}.xyz"
    )

    random.seed(random_seed + 1000 * (start_index + 1))

    if os.path.exists(output_xyz):
        os.remove(output_xyz)

    total_written = 0
    for dim in dims:
        box_size = _compute_box_size(n_atoms, config)
        structures_with_id = []
        for i in range(count):
            idx = start_index + i
            struct_id = f"{n_atoms}_{dim}_{idx}_seed{random_seed}_d{atom_diameter}_extra"
            try:
                s, _ = _generate_structure_sequential(n_atoms, dim, config)
                structures_with_id.append((struct_id, s))
            except RuntimeError as e:
                print(f"  ⚠ {e}")
                continue
        _write_xyz(structures_with_id, n_atoms, dim, box_size, config["element"], output_xyz)
        total_written += len(structures_with_id)

    print(f"[extra] Сгенерировано {total_written} структур для n={n_atoms} → {output_xyz}")
    return output_xyz
    
def generate_random_cluster(n_atoms: int, dim: int, cfg: dict, seed: int | None = None,
                            max_attempts: int = 1) -> np.ndarray:
    """
    Генерирует ОДНУ случайную структуру из `n_atoms` атомов размерности `dim`
    и возвращает np.ndarray формы (n_atoms, 3).

    Использует те же параметры, что и пакетная генерация:
      atom_diameter из cfg; min_dist = diameter, max_dist = 2*diameter (константы модуля)

    :param seed: если задан, фиксирует генератор (для воспроизводимости тестов).
    :param max_attempts: сколько раз повторить генерацию при неудаче размещения атомов.
    :raises RuntimeError: если структуру собрать не удалось.
    """
    if dim not in (1, 2, 3):
        raise ValueError("dim должен быть 1, 2 или 3")

    last_error = None
    for attempt in range(max(1, max_attempts)):
        # Локальный генератор, а не random.seed(): мастер вызывает генерацию из
        # потоков FastAPI, и общий RNG давал бы гонку и невоспроизводимые кадры.
        rng = random.Random(seed + attempt if seed is not None else None)
        try:
            points, _box_size = _generate_structure_sequential(n_atoms, dim, cfg, rng=rng)
            return np.asarray(points, dtype=np.float64)
        except RuntimeError as e:
            last_error = e
    raise RuntimeError(f"Не удалось сгенерировать кластер n={n_atoms}, dim={dim}: {last_error}")


def _generate_structure_sequential(n_atoms: int, dim: int, cfg: dict, rng=random):
    """
    Размещает атомы последовательно. Правило: расстояние до БЛИЖАЙШЕГО соседа
    должно лежать в [min_dist, max_dist]. Расстояния до дальних атомов не
    ограничиваются и могут быть больше max_dist.
    """
    box_size = _compute_box_size(n_atoms, cfg)
    d_min = min_dist(cfg)
    d_max = max_dist(cfg)

    points = [_random_point(dim, box_size, rng)]

    for atom_idx in range(1, n_atoms):
        placed = False
        for _ in range(MAX_ATTEMPTS_PER_ATOM):
            candidate = _random_point(dim, box_size, rng)
            nearest = min(float(np.linalg.norm(candidate - p)) for p in points)
            if d_min <= nearest <= d_max:
                points.append(candidate)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Не удалось разместить атом {atom_idx + 1} из {n_atoms} "
                f"для dim={dim} за {MAX_ATTEMPTS_PER_ATOM} попыток. "
                f"BOX_SIZE={box_size:.2f} Å, MIN_DIST={d_min:.2f}, MAX_DIST={d_max:.2f}."
            )

    points = np.array(points)
    points = points - points.mean(axis=0) + box_size / 2.0
    return points, box_size


def _write_xyz(structures_with_id, n_atoms, dim, box_size, element, path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, "a") as f:
        for struct_id, points in structures_with_id:
            n = len(points)
            comment = (
                f'Lattice="{box_size:.4f} 0.0 0.0 0.0 {box_size:.4f} 0.0 0.0 0.0 {box_size:.4f}" '
                f'Properties=species:S:1:pos:R:3 '
                f'n_atoms={n_atoms} dim={dim} struct_id={struct_id} '
                f'pbc="F F F"'
            )
            f.write(f"{n}\n{comment}\n")
            for p in points:
                f.write(f"{element} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def generate_structures(config: dict) -> str:
    """
    Публичная точка входа. Генерирует структуры и пишет их в XYZ.

    :param config: словарь с параметрами (см. config.json).
    :return: путь к записанному XYZ-файлу.
    """
    random_seed = RANDOM_SEED
    atom_diameter = config["atom_diameter"]
    structures_per_n = {int(k): v for k, v in config["structures_per_n"].items()}
    dimensions_per_n = {int(k): v for k, v in config["dimensions_per_n"].items()}

    output_xyz = config["output_xyz"]
    random.seed(random_seed)

    print("Параметры генерации:")
    print(f"  ATOM_DIAMETER = {atom_diameter} Å")
    print(f"  RANDOM_SEED = {random_seed}")
    print(f"  STRUCTURES_PER_N = {structures_per_n}")
    print(f"  DIMENSIONS_PER_N = {dimensions_per_n}")
    print()

    if os.path.exists(output_xyz):
        os.remove(output_xyz)

    total_written = 0

    for n_atoms, n_structs in structures_per_n.items():
        dims = dimensions_per_n.get(n_atoms, [3])
        box_size = _compute_box_size(n_atoms, config)
        for dim in dims:
            print(f"Генерация {n_structs} структур для n={n_atoms}, dim={dim}...")
            structures_with_id = []
            for i in range(n_structs):
                struct_id = f"{n_atoms}_{dim}_{i}_seed{random_seed}_d{atom_diameter}"
                try:
                    s, _ = _generate_structure_sequential(n_atoms, dim, config)
                    structures_with_id.append((struct_id, s))
                except RuntimeError as e:
                    print(f"  ⚠ {e}")
                    continue
            _write_xyz(structures_with_id, n_atoms, dim, box_size, config["element"], output_xyz)
            total_written += len(structures_with_id)
            print(f"  ✓ Записано {len(structures_with_id)} структур")

    print(f"\nГотово. Всего записано {total_written} структур в {output_xyz}")
    return output_xyz


# Версия генератора геометрии: входит в заголовок блока (geom_version).
# Пиры обязаны воспроизвести геометрию тем же алгоритмом.
GEOM_VERSION = 1


def generate_for_seed(seed: int, n_atoms: int, dim: int, cfg: dict) -> np.ndarray:
    """
    ЕДИНСТВЕННЫЙ путь генерации: мастер, воркер (через выданное) и проверка
    блока используют его же. Геометрия — только функция seed и параметров
    системы: никаких шаблонов и внешних структур.
    """
    return generate_random_cluster(n_atoms, dim, cfg, seed=int(seed), max_attempts=5)
