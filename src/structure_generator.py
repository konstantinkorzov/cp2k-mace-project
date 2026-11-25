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
    return float(cfg.get("atom_diameter") or 2.0)


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


def _random_point(dim, box_size):
    if dim == 1:
        return np.array([random.uniform(0, box_size), 0.0, 0.0])
    elif dim == 2:
        return np.array([random.uniform(0, box_size), random.uniform(0, box_size), 0.0])
    elif dim == 3:
        return np.array([random.uniform(0, box_size),
                         random.uniform(0, box_size),
                         random.uniform(0, box_size)])
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
        if seed is not None:
            random.seed(seed + attempt)
        try:
            points, _box_size = _generate_structure_sequential(n_atoms, dim, cfg)
            return np.asarray(points, dtype=np.float64)
        except RuntimeError as e:
            last_error = e
    raise RuntimeError(f"Не удалось сгенерировать кластер n={n_atoms}, dim={dim}: {last_error}")


def _generate_structure_sequential(n_atoms: int, dim: int, cfg: dict):
    """
    Размещает атомы последовательно. Правило: расстояние до БЛИЖАЙШЕГО соседа
    должно лежать в [min_dist, max_dist]. Расстояния до дальних атомов не
    ограничиваются и могут быть больше max_dist.
    """
    box_size = _compute_box_size(n_atoms, cfg)
    d_min = min_dist(cfg)
    d_max = max_dist(cfg)

    points = [_random_point(dim, box_size)]

    for atom_idx in range(1, n_atoms):
        placed = False
        for _ in range(MAX_ATTEMPTS_PER_ATOM):
            candidate = _random_point(dim, box_size)
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


# ============================================================
# ШАБЛОН (config.json → template_xyz): генерация ВОКРУГ заданной структуры
# ============================================================
# Мастер читает XYZ с образцовой геометрией и передаёт её воркеру в задании.
# Воркер не генерирует случайный клубок, а берёт шаблон, подгоняет число атомов
# под n этапа, слегка смещает атомы (σ ≈ 0.08·min_dist), расталкивает слипшиеся
# пары и случайно поворачивает кластер. Внутренние расстояния при повороте не
# меняются, поэтому структура остаётся «около» шаблона, но каждая задача даёт
# новую геометрию — иначе все блоки этапа были бы одним и тем же кадром.

TEMPLATE_NOISE_FRACTION = 0.08   # σ смещения атома = доля min_dist
TEMPLATE_REPAIR_STEPS = 200      # итераций расталкивания слипшихся пар
TEMPLATE_DECIMALS = 4            # точность координат в задании (Å)


def load_template_xyz(path: str) -> dict:
    """
    Читает XYZ-файл (один или несколько кадров подряд).

    Возвращает {'frames': [{'symbols': [...], 'coords': np.ndarray(n,3)}],
                'source': имя файла}.

    Понимает обычный XYZ и extended XYZ: важны только первые четыре колонки
    (символ, x, y, z); строка-комментарий и Lattice игнорируются.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"файл шаблона не найден: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    frames = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            continue          # мусорная строка между кадрами — пропускаем
        if i < len(lines):
            i += 1            # строка-комментарий
        symbols, coords = [], []
        for _ in range(n_atoms):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            if len(parts) < 4:
                continue
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
            symbols.append(parts[0])
        if coords and len(coords) == n_atoms:
            frames.append({"symbols": symbols,
                           "coords": np.asarray(coords, dtype=float)})

    if not frames:
        raise ValueError(f"в {path} не найдено ни одного кадра XYZ")
    return {"frames": frames, "source": os.path.basename(path)}


def template_min_distance(template: dict) -> float:
    """Минимальное межатомное расстояние по всем кадрам шаблона (Å)."""
    best = float("inf")
    for frame in (template or {}).get("frames") or []:
        pts = np.asarray(frame["coords"], dtype=float)
        if len(pts) < 2:
            continue
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        best = min(best, float(d.min()))
    return best


def template_payload(template: dict) -> dict:
    """Компактное представление шаблона для задания воркеру (JSON)."""
    return {
        "source": template.get("source"),
        "frames": [{
            "symbols": list(frame["symbols"]),
            "coords": [[round(float(x), TEMPLATE_DECIMALS) for x in atom]
                       for atom in frame["coords"]],
        } for frame in template.get("frames") or []],
    }


def _unit_vector(dim: int, rng: random.Random) -> np.ndarray:
    if dim == 1:
        return np.array([1.0 if rng.random() < 0.5 else -1.0, 0.0, 0.0])
    if dim == 2:
        a = rng.uniform(0.0, 2.0 * np.pi)
        return np.array([np.cos(a), np.sin(a), 0.0])
    while True:
        v = np.array([rng.gauss(0.0, 1.0) for _ in range(3)])
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            return v / norm


def _flatten(pts: np.ndarray, dim: int) -> np.ndarray:
    """Обнуляет координаты вне размерности (2D → z=0, 1D → y=z=0)."""
    if dim == 1:
        pts[:, 1] = 0.0
        pts[:, 2] = 0.0
    elif dim == 2:
        pts[:, 2] = 0.0
    return pts


def _next_surface_atom(pts: np.ndarray, dim: int, d_min: float, d_max: float,
                       rng: random.Random, attempts: int = 2000) -> np.ndarray:
    """Новый атом на поверхности шаблона: снаружи, но в контакте с соседом."""
    dist = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    order = np.argsort(dist)
    surface = order[len(order) // 2:]
    if len(surface) == 0:
        surface = order
    for _ in range(attempts):
        parent = pts[int(rng.choice(list(surface)))]
        cand = parent + _unit_vector(dim, rng) * rng.uniform(d_min, d_max)
        nearest = float(np.linalg.norm(pts - cand, axis=1).min())
        if d_min * 0.95 <= nearest <= d_max:
            return cand
    raise RuntimeError(
        f"шаблон: не удалось дорастить кластер (d_min={d_min:.2f}, d_max={d_max:.2f})")


def _fit_size(coords: np.ndarray, n_atoms: int, dim: int, d_min: float,
              d_max: float, rng: random.Random) -> np.ndarray:
    """Подгоняет число атомов шаблона под n_atoms: отрезает или доращивает."""
    pts = np.array(coords, dtype=float, copy=True)
    _flatten(pts, dim)

    if len(pts) > n_atoms:
        # отрезаем компактный фрагмент: n атомов, ближайших к случайному атому
        anchor = pts[rng.randrange(len(pts))]
        order = np.argsort(np.linalg.norm(pts - anchor, axis=1))
        pts = pts[order[:n_atoms]]

    while len(pts) < n_atoms:
        pts = np.vstack([pts, _next_surface_atom(pts, dim, d_min, d_max, rng)])
    return pts


def _perturb(pts: np.ndarray, sigma: float, dim: int,
             rng: random.Random) -> np.ndarray:
    noise = np.array([[rng.gauss(0.0, sigma) for _ in range(3)] for _ in pts])
    return _flatten(pts + noise, dim)


def _repair_clashes(pts: np.ndarray, d_min: float, dim: int,
                    steps: int = TEMPLATE_REPAIR_STEPS) -> np.ndarray:
    """Расталкивает пары ближе d_min, не ломая остальную геометрию."""
    pts = np.array(pts, dtype=float, copy=True)
    for _ in range(steps):
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
        if dist[i, j] >= d_min:
            break
        vec = pts[i] - pts[j]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            vec, norm = np.array([1.0, 0.0, 0.0]), 1.0
        shift = (d_min - float(dist[i, j])) / 2.0 + 1e-4
        pts[i] += vec / norm * shift
        pts[j] -= vec / norm * shift
        _flatten(pts, dim)
    return pts


def _repair_dangling(pts: np.ndarray, d_max: float, dim: int,
                     steps: int = 50) -> np.ndarray:
    """Подтягивает атомы без соседей: после шума и 2D-проекции они отрываются."""
    pts = np.array(pts, dtype=float, copy=True)
    if len(pts) < 2:
        return pts
    limit = d_max * 0.95
    for _ in range(steps):
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)
        idx = int(np.argmax(nearest))
        if nearest[idx] <= limit:
            break
        j = int(np.argmin(d[idx]))
        vec = pts[idx] - pts[j]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            vec, norm = np.array([1.0, 0.0, 0.0]), 1.0
        pts[idx] = pts[j] + vec / norm * limit
        _flatten(pts, dim)
    return pts


def _random_rotation(pts: np.ndarray, dim: int, rng: random.Random) -> np.ndarray:
    """Жёсткий поворот: внутренние расстояния не меняются."""
    if dim == 1 or len(pts) == 0:
        return pts
    if dim == 2:
        a = rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(a), np.sin(a)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        u1, u2, u3 = rng.random(), rng.random(), rng.random()
        w = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
        x = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
        y = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
        z = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    return pts @ rot.T


def _template_geometry_ok(pts: np.ndarray, d_min: float, d_max: float) -> bool:
    if not np.isfinite(pts).all():
        return False
    if len(pts) < 2:
        return True
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    if float(d.min()) < d_min * 0.95:
        return False     # остались слипшиеся атомы
    if float(d.min(axis=1).max()) > d_max * 1.05:
        return False     # есть оторванный атом без соседей
    return True


def generate_from_template(n_atoms: int, dim: int, cfg: dict, template: dict,
                           seed: int | None = None,
                           max_attempts: int = 5) -> np.ndarray:
    """
    ОДНА структура рядом с шаблоном: подгонка размера → шум → поворот.

    :raises RuntimeError: если не удалось получить геометрию, проходящую те же
        проверки, что и случайная генерация (min_dist ≤ d ≤ max_dist).
    """
    if dim not in (1, 2, 3):
        raise ValueError("dim должен быть 1, 2 или 3")
    frames = (template or {}).get("frames") or []
    if not frames:
        raise RuntimeError("шаблон пуст — нечего деформировать")

    rng = random.Random(seed)
    d_min, d_max = min_dist(cfg), max_dist(cfg)
    same_size = [f for f in frames if len(f["coords"]) == n_atoms]
    pool = same_size or frames
    base = np.asarray(pool[rng.randrange(len(pool))]["coords"], dtype=float)

    for attempt in range(max(1, max_attempts)):
        pts = _fit_size(base, n_atoms, dim, d_min, d_max, rng)
        sigma = TEMPLATE_NOISE_FRACTION * d_min * (1.0 + 0.5 * attempt)
        pts = _perturb(pts, sigma, dim, rng)
        pts = _repair_clashes(pts, d_min, dim)
        pts = _repair_dangling(pts, d_max, dim)
        pts = _random_rotation(pts, dim, rng)
        if _template_geometry_ok(pts, d_min, d_max):
            box_size = _compute_box_size(n_atoms, cfg)
            return _flatten(pts - pts.mean(axis=0) + box_size / 2.0, dim)

    raise RuntimeError(
        f"шаблон {template.get('source')}: не удалось получить валидную структуру "
        f"n={n_atoms}, dim={dim} за {max_attempts} попыток")