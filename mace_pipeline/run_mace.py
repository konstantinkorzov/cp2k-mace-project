#!/usr/bin/env python3
"""
MACE-пайплайн, отдельно от мастера.

1. Читает блокчейн элемента: results/<element>/blocks/*.zip (dft.json).
2. Собирает train.xyz (extxyz: energy eV, forces eV/Å).
3. Fine-tune MACE (mace_run_train, foundation_model из конфига).
4. Ищет минимумы BFGS по случайным стартовым кластерам (find_minimum.py).
5. Строит HTML со структурами (xyz_to_html.py).

Запуск (из любого места):

    python3 mace_pipeline/run_mace.py
    python3 mace_pipeline/run_mace.py --config mace_pipeline/config.json
    python3 mace_pipeline/run_mace.py --element Fe --skip-train --model path/to.model

Мастер это не импортирует и не запускает: MACE живёт только здесь.
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
SRC = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

import chain as chain_mod          # noqa: E402
import find_minimum                 # noqa: E402
import xyz_to_html                  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    for key in ("results_dir", "output_dir"):
        if cfg.get(key) and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.normpath(os.path.join(base, cfg[key]))
    return cfg


def read_blocks(element: str, results_dir: str) -> list[dict]:
    element_dir = os.path.join(results_dir, element)
    frames = []
    for b in chain_mod.list_blocks(element_dir):
        try:
            header, dft, _txs, _names = chain_mod.read_block(b["path"])
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ блок {b['name']}: {e}")
            continue
        geom = dft.get("geometry")
        energy = dft.get("energy_ev")
        if not geom or energy is None:
            continue
        frames.append({
            "geometry": geom,
            "energy": float(energy),
            "forces": dft.get("forces_ev_ang"),
            "seq": b["seq"],
        })
    return frames


def write_dataset(frames: list[dict], element: str, path: str):
    import numpy as np
    from ase import Atoms
    from ase.io import write
    atoms_list = []
    for fr in frames:
        atoms = Atoms(symbols=[element] * len(fr["geometry"]),
                      positions=np.asarray(fr["geometry"], dtype=float))
        atoms.info["energy"] = fr["energy"]
        if fr.get("forces") and len(fr["forces"]) == len(fr["geometry"]):
            atoms.arrays["forces"] = np.asarray(fr["forces"], dtype=float)
        atoms_list.append(atoms)
    write(path, atoms_list, format="extxyz")


def train_model(cfg: dict, dataset: str, output_dir: str) -> str | None:
    os.makedirs(output_dir, exist_ok=True)
    name = f"mace_{cfg['element']}"
    cmd = [
        "mace_run_train",
        "--name", name,
        "--train_file", dataset,
        "--energy_key", "energy",
        "--forces_key", "forces",
        "--model", "MACE",
        "--device", str(cfg["device"]),
        "--seed", str(cfg["seed"]),
        "--max_num_epochs", str(cfg["max_num_epochs"]),
        "--lr", str(cfg["lr"]),
        "--batch_size", str(cfg["batch_size"]),
        "--valid_fraction", str(cfg["valid_fraction"]),
        "--eval_interval", str(cfg["eval_interval"]),
        "--energy_weight", str(cfg["energy_weight"]),
        "--forces_weight", str(cfg["forces_weight"]),
        "--work_dir", output_dir,
        "--log_dir", os.path.join(output_dir, "logs"),
    ]
    foundation = str(cfg.get("foundation_model") or "").strip()
    if foundation:
        cmd += ["--foundation_model", foundation]
    else:
        # без foundation-весов MACE требует атомные энергии E0s
        cmd += ["--E0s", str(cfg.get("E0s", "average"))]
        print("  (foundation_model пуст — обучение с нуля, E0s=average)")
    print("  $ " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("✗ mace_run_train не найден (нужен mace-torch)")
        return None
    except subprocess.CalledProcessError as e:
        print(f"✗ mace_run_train завершился с кодом {e.returncode}")
        return None

    models = sorted(glob.glob(os.path.join(output_dir, "**", "*.model"), recursive=True),
                    key=os.path.getmtime)
    return models[-1] if models else None


def generate_cluster(element: str, n_atoms: int, dim: int, count: int,
                     min_d: float, max_d: float, seed: int) -> list:
    from ase import Atoms
    rnd = random.Random(seed)
    box = max_d * (n_atoms ** 0.5) * 2.0
    out = []
    for _ in range(count):
        for _attempt in range(200):
            pts = []
            for _i in range(n_atoms):
                pts.append([rnd.uniform(0, box) if dim >= 1 else 0.0,
                            rnd.uniform(0, box) if dim >= 2 else 0.0,
                            rnd.uniform(0, box) if dim == 3 else 0.0])
            ok = True
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    d = sum((pts[i][k] - pts[j][k]) ** 2 for k in range(3)) ** 0.5
                    if d < min_d:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append(Atoms(symbols=[element] * n_atoms, positions=pts))
                break
    return out


def main():
    ap = argparse.ArgumentParser(description="MACE-пайплайн поверх блокчейна (отдельно от мастера)")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--element", default=None, help="переопределить элемент из конфига")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--model", default=None, help="готовая .model (пропустить обучение)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-search", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.element:
        cfg["element"] = args.element
    # element — слаг системы и часть путей (results/<element>): без проверки
    # "../../.." читает и пишет вне каталога проекта
    import re as _re
    if not _re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{0,63}", str(cfg.get("element") or "")):
        raise SystemExit("элемент/система: латиница, цифры, . _ - (например cu3, al2o3)")
    if args.results_dir:
        cfg["results_dir"] = os.path.abspath(args.results_dir)
    if args.output_dir:
        cfg["output_dir"] = os.path.abspath(args.output_dir)

    element = cfg["element"]
    print("=" * 70)
    print(f"MACE-пайплайн: элемент {element}")
    print(f"  блокчейн : {os.path.join(cfg['results_dir'], element, 'blocks')}")
    print(f"  output   : {cfg['output_dir']}")
    print("=" * 70)

    frames = read_blocks(element, cfg["results_dir"])
    if not frames:
        print("✗ В блокчейне нет кадров с geometry/energy — обучать не на чем")
        return 1
    print(f"Кадров из блокчейна: {len(frames)} (блоки {frames[0]['seq']}..{frames[-1]['seq']})")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    dataset = os.path.join(cfg["output_dir"], f"{element}_train.xyz")
    write_dataset(frames, element, dataset)
    print(f"Датасет: {dataset}")

    model = args.model
    if not model and not args.skip_train:
        model = train_model(cfg, dataset, cfg["output_dir"])
        if not model:
            return 1
        print(f"Модель: {model}")
    elif not model and args.skip_train:
        print("✗ --skip-train без --model")
        return 1

    if args.skip_search:
        print("Поиск пропущен (--skip-search)")
        return 0

    from mace.calculators import MACECalculator
    from ase.io import write

    search = cfg["search"]
    starts = generate_cluster(element, int(search["n_atoms"]), int(search["dim"]),
                              int(search["count"]), float(search["min_dist"]),
                              float(search["max_dist"]), int(cfg["seed"]))
    if not starts:
        print("✗ Не удалось сгенерировать стартовые структуры")
        return 1
    start_xyz = os.path.join(cfg["output_dir"], f"{element}_start.xyz")
    write(start_xyz, starts, format="extxyz")
    print(f"Стартовых структур: {len(starts)} → {start_xyz}")

    calc = MACECalculator(model_paths=model, device=cfg["device"])
    best_xyz = os.path.join(cfg["output_dir"], f"{element}_best.xyz")
    logfile = os.path.join(cfg["output_dir"], f"{element}_bfgs.log")
    results = find_minimum.run_find_minimum(
        calc=calc, xyz_path=start_xyz, output_xyz=best_xyz, logfile=logfile,
        fmax=float(search["fmax"]), steps=int(search["steps"]),
    )
    for n, r in sorted(results.items()):
        print(f"  n={n}: E={r['energy']:.6f} eV, fmax={r['fmax']:.4f}")

    all_opt = os.path.splitext(best_xyz)[0] + "_all_optimized.xyz"
    if os.path.exists(all_opt):
        html = os.path.join(cfg["output_dir"], cfg.get("html_name", "structures.html"))
        xyz_to_html.structure_to_html(all_opt, html)
        print(f"HTML: {html}")

    print("Готово")
    return 0


if __name__ == "__main__":
    sys.exit(main())
