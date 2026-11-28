#!/usr/bin/env python3
"""
Индекс цепочки на SQLite (stdlib, без внешних зависимостей).

Зачем: `chain.list_blocks()` раньше на КАЖДЫЙ вызов читал все zip-блоки и
считал sha256 каждого файла — O(размер цепочки × размер блока). Теперь это
SELECT, а файл читается один раз: когда блок появился или изменился
(размер/mtime).

Источник правды — по-прежнему ФАЙЛЫ. Индекс производный: его можно удалить,
он соберётся заново; при любой ошибке SQLite данные берутся из файлов
напрямую (см. _from_files), так что недоступность индекса не ломает узел.

Почему SQLite, а не LevelDB/LMDB (как в Bitcoin): у нас порядок сотни тысяч
строк, а нужны транзакции, агрегаты (балансы по адресам) и выборка по seq —
это ровно SQL. Нативная KV-библиотека потребовала бы сборки под каждую
платформу и не дала бы ничего при нашем объёме.
"""

import os
import sqlite3
import threading

INDEX_NAME = "index.db"
SCHEMA_VERSION = "1"

# Один замок на все операции с индексом в процессе: SQLite сам сериализует
# записи, но нам нужно ещё и «прочитать-сравнить-записать» делать атомарно.
_LOCK = threading.RLock()


def db_path(element_dir: str) -> str:
    return os.path.join(element_dir, INDEX_NAME)


# ---------------------------------------------------------------- схема
def _connect(element_dir: str) -> sqlite3.Connection:
    path = db_path(element_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS meta("
                     "key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS blocks("
                     "name TEXT PRIMARY KEY, seq INTEGER NOT NULL, "
                     "system TEXT NOT NULL, n_atoms INTEGER NOT NULL, "
                     "block_id TEXT NOT NULL, sha256 TEXT NOT NULL, "
                     "size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL)")
        conn.execute("CREATE INDEX IF NOT EXISTS blocks_seq ON blocks(seq)")
        row = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if row is None:
            conn.execute("INSERT INTO meta(key, value) VALUES('schema', ?)",
                         (SCHEMA_VERSION,))
        elif str(row[0]) != SCHEMA_VERSION:
            raise RuntimeError(f"index schema {row[0]} != {SCHEMA_VERSION}")
    return conn


# ---------------------------------------------------------------- файлы
def _scan(directory: str) -> dict[str, tuple[int, int]]:
    """name -> (size, mtime_ns) для файлов, похожих на блок."""
    out = {}
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if not entry.is_file() or not entry.name.endswith(".zip"):
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                out[entry.name] = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        return {}
    return out


def _signature(chain, path: str) -> tuple[str, int, str, str] | None:
    """(system, n_atoms, block_id, sha256) блока; None — имя не по формату."""
    name = os.path.basename(path)
    m = chain.BLOCK_NAME_RE.match(name)
    if not m or not chain.SYSTEM_RE.match(m.group(2)):
        return None
    return (m.group(2), int(m.group(3)),
            chain._block_id_of_file(path), chain.sha256_file(path))


def _from_files(element_dir: str) -> list[dict]:
    """Резервный путь: читаем файлы как до появления индекса."""
    import chain

    d = chain.blocks_dir(element_dir)
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        m = chain.BLOCK_NAME_RE.match(name)
        if not m or not chain.SYSTEM_RE.match(m.group(2)):
            continue
        path = os.path.join(d, name)
        out.append({"seq": int(m.group(1)), "system": m.group(2),
                    "n_atoms": int(m.group(3)), "path": path, "name": name,
                    "sha256": chain.sha256_file(path),
                    "block_id": chain._block_id_of_file(path)})
    out.sort(key=lambda b: b["seq"])
    return out


# ---------------------------------------------------------------- синхронизация
def _sync(conn: sqlite3.Connection, element_dir: str,
          seen: dict[str, tuple[int, int]]) -> None:
    """Приводит таблицу blocks в соответствие с файлами на диске."""
    import chain

    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = {r["name"]: r for r in conn.execute("SELECT * FROM blocks")}
        add = [n for n in seen if n not in rows]
        gone = [n for n in rows if n not in seen]
        changed = [n for n in seen
                   if n in rows and (rows[n]["size"] != seen[n][0]
                                     or rows[n]["mtime_ns"] != seen[n][1])]
        for name in gone + changed:
            conn.execute("DELETE FROM blocks WHERE name=?", (name,))
        d = chain.blocks_dir(element_dir)
        for name in add + changed:
            path = os.path.join(d, name)
            sig = _signature(chain, path)
            if sig is None:
                continue
            system, n_atoms, block_id, sha = sig
            m = chain.BLOCK_NAME_RE.match(name)
            conn.execute(
                "INSERT OR REPLACE INTO blocks"
                "(name, seq, system, n_atoms, block_id, sha256, size, mtime_ns)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (name, int(m.group(1)), system, n_atoms, block_id, sha,
                 seen[name][0], seen[name][1]))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def list_blocks(element_dir: str) -> list[dict]:
    """Список блоков (по возрастанию seq) — как chain.list_blocks()."""
    import chain

    d = chain.blocks_dir(element_dir)
    if not os.path.isdir(d):
        return []
    seen = _scan(d)
    try:
        with _LOCK:
            conn = _connect(element_dir)
            try:
                _sync(conn, element_dir, seen)
                rows = [dict(r) for r in
                        conn.execute("SELECT * FROM blocks ORDER BY seq, name")]
            finally:
                conn.close()
        out = [{"seq": r["seq"], "system": r["system"], "n_atoms": r["n_atoms"],
                "path": os.path.join(d, r["name"]), "name": r["name"],
                "sha256": r["sha256"], "block_id": r["block_id"]}
               for r in rows]
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[Index] ! индекс недоступен ({type(e).__name__}: {e}) — "
              f"читаю блоки напрямую")
        return _from_files(element_dir)


def rebuild(element_dir: str) -> dict:
    """Сносит index.db и строит заново. Возвращает {blocks: N}."""
    path = db_path(element_dir)
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass
    rows = list_blocks(element_dir)
    return {"blocks": len(rows)}
