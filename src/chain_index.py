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
        conn.execute("CREATE TABLE IF NOT EXISTS utxo("
                     "txid TEXT NOT NULL, vout INTEGER NOT NULL, "
                     "value INTEGER NOT NULL, address TEXT NOT NULL, "
                     "height INTEGER NOT NULL, coinbase INTEGER NOT NULL DEFAULT 0, "
                     "PRIMARY KEY(txid, vout)) WITHOUT ROWID")
        conn.execute("CREATE INDEX IF NOT EXISTS utxo_address ON utxo(address)")
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


# ---------------------------------------------------------------- UTXO
ZERO_HASH = "0" * 64


def _meta(conn) -> dict:
    return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}


def _set_applied(conn, seq: int, block_id: str) -> None:
    for key, value in (("applied_seq", str(int(seq))),
                       ("applied_id", str(block_id))):
        conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, value))


def _row_of(out: dict) -> tuple[int, str, int, int] | None:
    if not isinstance(out, dict):
        return None
    value = out.get("value")
    value = value if type(value) is int else 0
    addr = out.get("address") if isinstance(out.get("address"), str) else ""
    return value, addr, 0, 0


def _apply_tx(conn, tx: dict, height: int) -> None:
    """Применяет транзакцию блока к таблице utxo (как _utxo_apply в chain)."""
    import chain

    if not isinstance(tx, dict):
        return
    tid = tx.get("txid") or chain.txid_of(tx)
    if tx.get("type") != "coinbase":
        for i in tx.get("inputs") or []:
            try:
                vout = int(i.get("vout", -1))
            except (TypeError, ValueError):
                vout = -1
            conn.execute("DELETE FROM utxo WHERE txid=? AND vout=?",
                         (str(i.get("txid") or ""), vout))
    is_cb = 1 if tx.get("type") == "coinbase" else 0
    for vout, out in enumerate(tx.get("outputs") or []):
        row = _row_of(out)
        if row is None:
            continue
        value, addr, _h, _cb = row
        conn.execute("INSERT OR REPLACE INTO utxo"
                     "(txid, vout, value, address, height, coinbase) "
                     "VALUES(?,?,?,?,?,?)",
                     (tid, vout, value, addr, int(height), is_cb))


def _apply_block(conn, path: str, seq: int, block_id: str) -> None:
    import chain

    try:
        txs = chain.read_block_txs(path)
    except Exception as e:  # noqa: BLE001
        # Битый блок молча выпадать из UTXO не должен: мастер обязан
        # отказаться собирать новые блоки на неверном взгляде.
        chain.BROKEN_BLOCKS.add(path)
        print(f"[Index] ✗ блок {os.path.basename(path)} не читается ({e}) — "
              f"новые блоки строить нельзя, нужен verify_chain/восстановление")
        txs = []
    for tx in txs:
        _apply_tx(conn, tx, seq)
    _set_applied(conn, seq, block_id)


def _check_readable(blocks: list[dict]) -> None:
    """Проверка целостности, когда нумерация блоков уже с дырками.

    В здоровой цепочке этот путь не выполняется (он O(n) по чтению zip):
    вызывается только когда раскладка высот не 1..N, то есть UTXO всё равно
    нельзя считать доверенным.
    """
    import chain

    for b in blocks:
        try:
            chain.read_block_txs(b["path"])
        except Exception as e:  # noqa: BLE001
            chain.BROKEN_BLOCKS.add(b["path"])
            print(f"[Index] ✗ блок {b['name']} не читается ({e})")


def _rebuild_to(conn, blocks: list[dict], upto_seq: int) -> None:
    """Полная перестройка UTXO до высоты upto_seq (редкий путь)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM utxo")
        _set_applied(conn, 0, ZERO_HASH)
        for b in blocks:
            if b["seq"] > int(upto_seq):
                continue
            _apply_block(conn, b["path"], b["seq"], b["block_id"])
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _ensure_synced(conn, blocks: list[dict]) -> None:
    """Догоняет UTXO до цепочки или перестраивает при любом расхождении."""
    seqs = [b["seq"] for b in blocks]
    if seqs != list(range(1, len(blocks) + 1)):
        # дырка/дубль высот: инкрементально применять нельзя
        _check_readable(blocks)
        _rebuild_to(conn, blocks, len(blocks))
        return
    applied_seq, applied_id = (int(_meta(conn).get("applied_seq") or 0),
                               str(_meta(conn).get("applied_id") or ZERO_HASH))
    if applied_seq > len(blocks):
        _rebuild_to(conn, blocks, len(blocks))
        return
    if applied_seq:
        cur = blocks[applied_seq - 1]
        if cur["block_id"] != applied_id:
            _rebuild_to(conn, blocks, len(blocks))
            return
    tail = blocks[applied_seq:]
    if not tail:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        for b in tail:
            _apply_block(conn, b["path"], b["seq"], b["block_id"])
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


class UtxoView:
    """Ленивый UTXO: чтение из индекса по ключу, изменения — в overlay.

    Заменяет словарь размером во всю цепочку там, где нужны единичные
    обращения (проверка входов транзакции, сборка блока). Полная
    материализация (items/len/iter) осталась для кошелька и балансов.
    """

    def __init__(self, element_dir: str, conn, before_seq=None, parent=None):
        self.element_dir = element_dir
        self._conn = conn
        self._parent = parent
        self._overlay: dict[tuple, dict | None] = {}
        self.before_seq = before_seq
        self._owns = parent is None

    # --- чтение
    def _row(self, key):
        key = (str(key[0]), int(key[1]))
        if key in self._overlay:
            return self._overlay[key]
        if self._parent is not None:
            return self._parent._row(key)
        cur = self._conn.execute(
            "SELECT value, address, height, coinbase FROM utxo "
            "WHERE txid=? AND vout=?", key)
        row = cur.fetchone()
        return dict(row) if row else None

    def get(self, key, default=None):
        row = self._row(key)
        return default if row is None else row

    def __contains__(self, key):
        return self._row(key) is not None

    def __getitem__(self, key):
        row = self._row(key)
        if row is None:
            raise KeyError(key)
        return row

    def pop(self, key, *default):
        row = self._row(key)
        if row is None:
            if default:
                return default[0]
            raise KeyError(key)
        self._overlay[(str(key[0]), int(key[1]))] = None
        return row

    def __setitem__(self, key, row):
        self._overlay[(str(key[0]), int(key[1]))] = dict(row)

    def child(self) -> "UtxoView":
        """Отдельный слой изменений поверх этого вида (снимок для расчёта)."""
        return UtxoView(self.element_dir, self._conn, self.before_seq, parent=self)

    # --- полная материализация
    def _material(self) -> dict:
        base = self._parent._material() if self._parent is not None else {}
        if self._parent is None:
            for r in self._conn.execute("SELECT txid, vout, value, address, "
                                        "height, coinbase FROM utxo"):
                base[(r["txid"], int(r["vout"]))] = {
                    "value": int(r["value"]), "address": r["address"],
                    "height": int(r["height"]), "coinbase": bool(r["coinbase"])}
        for key, row in self._overlay.items():
            if row is None:
                base.pop(key, None)
            else:
                base[key] = row
        return base

    def items(self):
        return self._material().items()

    def values(self):
        return self._material().values()

    def keys(self):
        return self._material().keys()

    def __len__(self):
        return len(self._material())

    def __iter__(self):
        return iter(self._material())

    def close(self) -> None:
        if self._owns:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def utxo_view(element_dir: str, before_seq: int | None = None) -> UtxoView:
    """Вид на UTXO. before_seq — состояние ПЕРЕД блоком с этой высотой."""
    blocks = list_blocks(element_dir)
    with _LOCK:
        conn = _connect(element_dir)
        try:
            _ensure_synced(conn, blocks)
            applied = int(_meta(conn).get("applied_seq") or 0)
            if before_seq is not None and int(before_seq) - 1 < applied:
                _rebuild_to(conn, blocks, max(0, int(before_seq) - 1))
        except Exception:
            conn.close()
            raise
    return UtxoView(element_dir, conn, before_seq=before_seq)


def balances_by_address(element_dir: str) -> dict[str, int]:
    """Суммы по адресам одним SQL-агрегатом (для кошелька/панели)."""
    blocks = list_blocks(element_dir)
    with _LOCK:
        conn = _connect(element_dir)
        try:
            _ensure_synced(conn, blocks)
            return {r["address"]: int(r["total"]) for r in conn.execute(
                "SELECT address, SUM(value) AS total FROM utxo GROUP BY address")}
        finally:
            conn.close()
