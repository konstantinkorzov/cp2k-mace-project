#!/usr/bin/env python3
"""Проверка генерации по шаблону (без CP2K)."""
import os
import sys
import tempfile
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import structure_generator as sg

CFG = {"atom_diameter": 2.0, "composition": {"Cu": 1}, "element": "Cu"}
D_MIN = sg.min_dist(CFG)
D_MAX = sg.max_dist(CFG)

# 1) строим «образец»: 13-атомный кластер штатным генератором
rng_coords = sg.generate_random_cluster(13, 3, CFG, seed=7, max_attempts=5)
print("образец:", rng_coords.shape, "min d =", round(float(
    np.linalg.norm(rng_coords[:, None] - rng_coords[None], axis=-1)[
        ~np.eye(13, dtype=bool)].min()), 3))

tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "seed13.xyz")
with open(path, "w") as f:
    f.write(f"13\ncomment Lattice=\"20 0 0 0 20 0 0 0 20\" Properties=species:S:1:pos:R:3\n")
    for p in rng_coords:
        f.write(f"Cu {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

tpl = sg.load_template_xyz(path)
print("кадров:", len(tpl["frames"]), "| payload frames:", len(sg.template_payload(tpl)["frames"]))
print("min d шаблона:", round(sg.template_min_distance(tpl), 3))

errors = []
for n in (2, 6, 13, 20, 55):
    for dim in (3, 2):
        for seed in range(3):
            try:
                pts = sg.generate_from_template(n, dim, CFG, tpl, seed=seed)
            except Exception as e:
                errors.append(f"n={n} dim={dim} seed={seed}: {e}")
                continue
            d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
            np.fill_diagonal(d, np.inf)
            dmin, dmax = float(d.min()), float(d.min(axis=1).max())
            ok = (pts.shape == (n, 3) and np.isfinite(pts).all()
                  and dmin >= D_MIN * 0.95 and dmax <= D_MAX * 1.05)
            if dim == 2:
                ok = ok and np.allclose(pts[:, 2], pts[0, 2])
            print(f"n={n:2d} dim={dim} seed={seed}: min={dmin:.3f} maxn={dmax:.3f} "
                  f"{'OK' if ok else 'FAIL'}")
            if not ok:
                errors.append(f"n={n} dim={dim} seed={seed}: dmin={dmin:.3f} dmax={dmax:.3f}")

# 2) проверим, что структуры РАЗНЫЕ (не один и тот же кадр)
a = sg.generate_from_template(13, 3, CFG, tpl, seed=1)
b = sg.generate_from_template(13, 3, CFG, tpl, seed=2)
print("разные кадры:", not np.allclose(a, b))

# 3) шаблон из 2 атомов + этап n=2
p2 = os.path.join(tmpdir, "dimer.xyz")
with open(p2, "w") as f:
    f.write("2\ndimer\nCu 0 0 0\nCu 2.5 0 0\n")
t2 = sg.load_template_xyz(p2)
for seed in range(3):
    pts = sg.generate_from_template(2, 3, CFG, t2, seed=seed)
    d = float(np.linalg.norm(pts[0] - pts[1]))
    print(f"димер seed={seed}: d={d:.3f}")

print("\nОШИБКИ:", errors if errors else "нет")
sys.exit(1 if errors else 0)
