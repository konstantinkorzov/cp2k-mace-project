#!/usr/bin/env python3
"""
Зонд формата CP2K -RESTART.wfn (диагностика, только чтение).

Цель: на вашем CP2K 2024.3 зафиксировать, что именно лежит в файле волновой
функции, чтобы мастер мог делать МЕХАНИЧЕСКУЮ проверку (число базисных функций,
число МО, спиновых каналов) до физической проверки (градиент/энергия).

Запуск внутри контейнера, где есть CP2K:

    python3 probe_wfn.py /path/to/PROJECT-RESTART.wfn [--log /path/to/PROJECT.out]

Что печатает:
  * размер файла, тип (похож на Fortran unformatted или нет);
  * hexdump первых 256 байт;
  * все ASCII-строки внутри (в CP2K там бывает имя проекта/маркеры формата);
  * кандидаты на целочисленные поля заголовка (nao, nmo, nspin) — скан по
    4- и 8-байтовым little-endian целым в разумном диапазоне;
  * разбор цепочки Fortran-записей (маркеры длины 4 или 8 байт);
  * если передан --log, сверку с тем, что CP2K написал про систему:
    число орбитальных функций, число электронов, число занятых орбиталей,
    а также сообщения о чтении restart и предупреждения о несовместимости.
"""

import argparse
import os
import re
import struct
import sys


def hexdump(data: bytes, width: int = 16, limit: int = 256) -> str:
    out = []
    for off in range(0, min(len(data), limit), width):
        chunk = data[off:off + width]
        hexs = " ".join(f"{b:02x}" for b in chunk)
        asci = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"  {off:08x}  {hexs:<{width * 3}}  |{asci}|")
    return "\n".join(out)


def ascii_strings(data: bytes, min_len: int = 4, limit: int = 40) -> list:
    found = re.findall(rb"[ -~]{%d,}" % min_len, data)
    return [s.decode("ascii", "replace") for s in found[:limit]]


def int_candidates(data: bytes, limit_each: int = 12) -> dict:
    """Скан целых little-endian: значения, похожие на размерности задачи."""
    res4, res8 = [], []
    for off in range(0, max(0, len(data) - 4), 4):
        v = struct.unpack_from("<i", data, off)[0]
        if 1 <= v <= 4096:
            res4.append((off, v))
    for off in range(0, max(0, len(data) - 8), 4):
        v = struct.unpack_from("<q", data, off)[0]
        if 1 <= v <= 4096:
            res8.append((off, v))
    return {"int32": res4[:limit_each], "int64": res8[:limit_each]}


def fortran_records(data: bytes, marker: int) -> list:
    recs = []
    off = 0
    fmt = "<i" if marker == 4 else "<q"
    n = len(data)
    while off + 2 * marker <= n:
        (length,) = struct.unpack_from(fmt, data, off)
        if length < 0 or off + 2 * marker + length > n:
            break
        (tail,) = struct.unpack_from(fmt, data, off + marker + length)
        if tail != length:
            break
        recs.append(length)
        off += 2 * marker + length
        if len(recs) > 64:
            break
    return recs


def parse_log(logpath: str):
    """Достаёт из лога CP2K размерности системы и сообщения про restart."""
    pats = {
        "orbital_functions": r"Number of orbital functions\s*:?\s*(\d+)",
        "electrons": r"Number of electrons\s*:?\s*([\d.]+)",
        "occupied": r"Number of occupied orbitals\s*:?\s*(\d+)",
        "molecular_orbitals": r"Number of molecular orbitals\s*:?\s*(\d+)",
        "basis_functions": r"Number of basis functions\s*:?\s*(\d+)",
    }
    info = {}
    reading = []
    warnings = []
    if not os.path.exists(logpath):
        return {"missing": logpath}
    with open(logpath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for key, pat in pats.items():
                m = re.search(pat, line, re.IGNORECASE)
                if m and key not in info:
                    info[key] = m.group(1)
            up = line.upper()
            if "RESTART" in up and ("WAVEFUNCTION" in up or "WFN" in up or "READ" in up):
                reading.append(line.strip())
            if any(w in up for w in ("NOT COMPATIBLE", "INCOMPATIBLE", "DIFFERENT NUMBER",
                                     "ATOMIC GUESS", "COULD NOT READ", "IGNOR")):
                if "RESTART" in up or "WFN" in up or "WAVEFUNCTION" in up:
                    warnings.append(line.strip())
    return {"info": info, "reading": reading[:10], "warnings": warnings[:10]}


def main():
    ap = argparse.ArgumentParser(description="Зонд формата CP2K -RESTART.wfn")
    ap.add_argument("wfn", help="путь к PROJECT-RESTART.wfn")
    ap.add_argument("--log", help="путь к PROJECT.out (для сверки размерностей)")
    ap.add_argument("--bytes", type=int, default=4096, help="сколько байт читать (по умолчанию 4096)")
    args = ap.parse_args()

    if not os.path.exists(args.wfn):
        print(f"✗ файл не найден: {args.wfn}")
        return 2

    size = os.path.getsize(args.wfn)
    walk_bytes = 64 * 1024 * 1024
    with open(args.wfn, "rb") as f:
        raw = f.read(min(size, walk_bytes))
    head = raw[:max(args.bytes, 4096)]

    print("=" * 72)
    print(f"файл:   {args.wfn}")
    print(f"размер: {size} байт ({size / 1024 / 1024:.2f} МБ)")
    print("=" * 72)

    print("\n--- hexdump (первые 256 байт) ---")
    print(hexdump(head))

    strs = ascii_strings(head)
    print(f"\n--- ASCII-строки внутри (первые {len(strs)}) ---")
    for s in strs[:20]:
        print(f"  {s!r}")

    print("\n--- кандидаты на целые поля заголовка ---")
    cand = int_candidates(head)
    print("  int32 (offset, value):", ", ".join(f"{o}:{v}" for o, v in cand["int32"]))
    print("  int64 (offset, value):", ", ".join(f"{o}:{v}" for o, v in cand["int64"]))

    print("\n--- разбор как Fortran unformatted (по первым "
          f"{min(len(raw), walk_bytes) // 1024} КБ) ---")
    for marker in (4, 8):
        recs = fortran_records(raw, marker)
        print(f"  маркер {marker} байт: записей разобрано {len(recs)}"
              + (f", длины: {recs[:16]}" if recs else " (не похоже)"))

    print("\n--- делимость размера (намекает на плотный массив float64/float32) ---")
    for width, name in ((8, "float64/int64"), (4, "float32/int32")):
        q, r = divmod(size, width)
        print(f"  size % {width} = {r}  ({'делится' if r == 0 else 'НЕ делится'}, ~{q} элементов {name})")
    if size % 8 == 0:
        total = size // 8
        print(f"  если все 8-байтовые: {total} чисел; проверьте комбинации "
              f"nao*nmo (например 26*26=676, 40*40=1600)")

    if args.log:
        print("\n--- сверка с логом CP2K ---")
        info = parse_log(args.log)
        if "missing" in info:
            print(f"  ✗ лог не найден: {info['missing']}")
        else:
            for k, v in info["info"].items():
                print(f"  {k}: {v}")
            print("  строки про чтение restart:")
            for l in info["reading"] or ["    (нет — проверьте, что лог от этого прогона)"]:
                print(f"    {l}")
            print("  ВАЖНО — предупреждения о несовместимости restart:")
            for l in info["warnings"] or ["    (нет)"]:
                print(f"    ⚠ {l}")
            print("\n  Вывод: если в логе есть 'Number of orbital functions' и это число\n"
                  "  совпадает с найденным в заголовке файла кандидатом — фиксируем его\n"
                  "  как nao и дальше мастер проверяет элемент/базис/q механически.")

    print("\nГотово. Пришлите этот вывод — по нему зафиксируем точный набор полей.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
