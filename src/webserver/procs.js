'use strict';

/**
 * Менеджер процессов для веб-панели.
 *
 * Запускает master (src/main.py), worker (src/worker.py) и MACE-пайплайн
 * (mace_pipeline/run_mace.py) дочерними процессами, следит за портами и
 * каталогами результатов так, чтобы несколько мастеров могли жить в одном
 * контейнере одновременно.
 *
 * Правила:
 *   * мастер всегда получает СВОБОДНЫЙ порт (по умолчанию с 8000 вверх);
 *   * у каждого мастера свой каталог результатов: results, results-8001, ...
 *     (каталог, в котором уже работает живой main.py, не переиспользуется —
 *     проверяется по <dir>/main.pid);
 *   * пароли передаются через окружение (MASTER_PASSWORD / WORKER_PASSWORD),
 *     чтобы не светиться в ps;
 *   * stdout+stderr каждого процесса пишутся в .webui/logs/<id>.log.
 */

const fs = require('fs');
const path = require('path');
const net = require('net');
const http = require('http');
const { spawn } = require('child_process');

const ROOT = path.resolve(__dirname, '..', '..');
const LOG_DIR = path.join(ROOT, '.webui', 'logs');
fs.mkdirSync(LOG_DIR, { recursive: true });

const processes = new Map(); // id -> record
let counter = 0;

function isAlive(pid) {
    if (!pid || !Number.isInteger(pid)) return false;
    try {
        process.kill(pid, 0);
        return true;
    } catch (e) {
        return false;
    }
}

function portFree(port) {
    return new Promise((resolve) => {
        const srv = net.createServer();
        srv.once('error', () => resolve(false));
        srv.once('listening', () => srv.close(() => resolve(true)));
        srv.listen(port, '0.0.0.0');
    });
}

function usedPorts() {
    const set = new Set();
    for (const rec of processes.values()) {
        if (rec.port && isAlive(rec.pid)) set.add(rec.port);
    }
    return set;
}

async function pickPort(from) {
    const used = usedPorts();
    for (let p = from; p < from + 100; p++) {
        if (used.has(p)) continue;
        // eslint-disable-next-line no-await-in-loop
        if (await portFree(p)) return p;
    }
    throw new Error('нет свободных портов в диапазоне ' + from + '..' + (from + 100));
}

/** Живой ли master.py в этом каталоге (по main.pid). */
function dirBusy(dir) {
    try {
        const pid = parseInt(fs.readFileSync(path.join(dir, 'main.pid'), 'utf8').trim(), 10);
        return Number.isInteger(pid) && isAlive(pid);
    } catch (e) {
        return false;
    }
}

function pickDir(port) {
    const candidates = ['results', `results-${port}`];
    for (const name of candidates) {
        const abs = path.join(ROOT, name);
        if (!dirBusy(abs)) return abs;
    }
    return path.join(ROOT, `results-${port}-${Date.now() % 10000}`);
}

function resolveDir(dir) {
    if (!dir) return null;
    return path.isAbsolute(dir) ? dir : path.join(ROOT, dir);
}

/** Порт мастера по умолчанию из src/config.json (ключ "port"). */
function configuredMasterPort() {
    try {
        const raw = JSON.parse(fs.readFileSync(path.join(ROOT, 'src', 'config.json'), 'utf8'));
        const entry = raw.port;
        const value = (entry && typeof entry === 'object' && 'value' in entry) ? entry.value : entry;
        const port = parseInt(value, 10);
        return Number.isInteger(port) && port > 0 && port < 65536 ? port : 8000;
    } catch (e) {
        return 8000;
    }
}

function start(rec, cmd, args, extraEnv) {
    const logFile = path.join(LOG_DIR, rec.id + '.log');
    const fd = fs.openSync(logFile, 'a');
    fs.writeSync(fd, `\n=== ${new Date().toISOString()} запуск: ${cmd} ${args.join(' ')}\n`);

    const child = spawn(cmd, args, {
        cwd: ROOT,
        env: Object.assign({}, process.env, extraEnv || {}),
        stdio: ['ignore', fd, fd],
        detached: false,
    });
    try { fs.closeSync(fd); } catch (e) { /* уже закрыт */ }

    rec.pid = child.pid;
    rec.logFile = logFile;
    rec.startedAt = Date.now();
    rec.exitCode = null;
    rec.endedAt = null;

    child.on('exit', (code, signal) => {
        rec.exitCode = code;
        rec.signal = signal;
        rec.endedAt = Date.now();
    });
    child.on('error', (err) => {
        rec.error = String(err);
        rec.endedAt = Date.now();
    });

    processes.set(rec.id, rec);
    return rec;
}

function addRecord(kind, name) {
    counter += 1;
    return {
        id: `${kind}-${counter}`,
        kind,
        name,
    };
}

async function startMaster(opts = {}) {
    const password = String(opts.password || 'master');
    let port = parseInt(opts.port, 10);
    if (!Number.isInteger(port) || port <= 0) {
        port = await pickPort(configuredMasterPort());
    } else if (!(await portFree(port))) {
        throw new Error(`порт ${port} занят — укажите другой или оставьте «авто»`);
    }

    let dir = resolveDir(opts.dir);
    if (dir) {
        if (dirBusy(dir)) throw new Error(`в каталоге ${dir} уже работает master`);
    } else {
        dir = pickDir(port);
    }

    const rec = addRecord('master', opts.name || `мастер :${port}`);
    rec.port = port;
    rec.dir = dir;
    rec.password = password;

    const args = ['src/main.py', '--port', String(port), '--results-dir', dir];
    const peers = (opts.peers || []).map((p) => String(p).trim()).filter(Boolean);
    for (const peer of peers) args.push('--peer', peer);
    rec.peers = peers;

    start(rec, 'python3', args, { MASTER_PASSWORD: password });
    return publicRecord(rec);
}

async function startWorker(opts = {}) {
    let name = String(opts.name || '').trim();
    if (!name) {
        let n = 1;
        const taken = new Set([...processes.values()]
            .filter((r) => r.kind === 'worker' && isAlive(r.pid)).map((r) => r.name));
        while (taken.has(`worker${n}`)) n += 1;
        name = `worker${n}`;
    }
    if ([...processes.values()].some((r) => r.kind === 'worker' && r.name === name
        && isAlive(r.pid))) {
        throw new Error(`воркер с именем ${name} уже запущен (имя = кошелёк)`);
    }

    const masters = (opts.masters || []).map((m) => String(m).trim()).filter(Boolean);
    if (!masters.length) throw new Error('не выбран ни один адрес мастера');

    const password = String(opts.password || 'worker');
    const rec = addRecord('worker', name);
    rec.masters = masters;
    rec.password = password;

    const args = ['src/worker.py', '--name', name];
    for (const url of masters) args.push('--master', url);

    start(rec, 'python3', args, { WORKER_PASSWORD: password, WORKER_MASTERS: masters.join(',') });
    return publicRecord(rec);
}

function startMace(opts = {}) {
    const args = ['mace_pipeline/run_mace.py'];
    if (opts.element) args.push('--element', String(opts.element));
    if (opts.resultsDir) args.push('--results-dir', resolveDir(opts.resultsDir));
    if (opts.outputDir) args.push('--output-dir', resolveDir(opts.outputDir));
    if (opts.model) args.push('--model', String(opts.model));
    if (opts.skipTrain) args.push('--skip-train');
    if (opts.skipSearch) args.push('--skip-search');

    const rec = addRecord('mace', 'MACE ' + (opts.element || ''));
    rec.args = args.slice(1);
    start(rec, 'python3', args, {});
    return publicRecord(rec);
}

function find(id) {
    return processes.get(id) || null;
}

function stop(id, signal = 'SIGTERM') {
    const rec = find(id);
    if (!rec) throw new Error('процесс не найден');
    if (!isAlive(rec.pid)) throw new Error('процесс уже завершён');
    try { process.kill(rec.pid, signal); } catch (e) { /* уже нет */ }
    setTimeout(() => {
        if (isAlive(rec.pid)) {
            try { process.kill(rec.pid, 'SIGKILL'); } catch (e) { /* уже нет */ }
        }
    }, 5000).unref();
    return publicRecord(rec);
}

function publicRecord(rec) {
    return {
        id: rec.id,
        kind: rec.kind,
        name: rec.name,
        pid: rec.pid || null,
        port: rec.port || null,
        dir: rec.dir || null,
        masters: rec.masters || [],
        peers: rec.peers || [],
        args: rec.args || null,
        alive: isAlive(rec.pid),
        startedAt: rec.startedAt || null,
        endedAt: rec.endedAt || null,
        exitCode: rec.exitCode,
        error: rec.error || null,
    };
}

function list() {
    const out = [];
    for (const rec of processes.values()) out.push(publicRecord(rec));
    out.sort((a, b) => (a.startedAt || 0) - (b.startedAt || 0));
    return out;
}

/** Порции лога: offset → {offset, chunk}. */
function readLog(id, offset) {
    const rec = find(id);
    if (!rec) throw new Error('процесс не найден');
    const file = path.join(LOG_DIR, id + '.log');
    let stat;
    try {
        stat = fs.statSync(file);
    } catch (e) {
        return { offset: 0, chunk: '' };
    }
    let from = Number.isFinite(offset) && offset >= 0 ? offset : 0;
    if (from > stat.size) from = 0;
    let length = stat.size - from;
    const MAX = 256 * 1024;
    if (length > MAX) from = stat.size - MAX;
    length = stat.size - from;
    if (length <= 0) return { offset: stat.size, chunk: '' };

    const fd = fs.openSync(file, 'r');
    const buf = Buffer.alloc(length);
    fs.readSync(fd, buf, 0, length, from);
    fs.closeSync(fd);
    return { offset: stat.size, chunk: buf.toString('utf8') };
}

/** Каталоги результатов, которые панель знает: свои процессы + results*, . */
function knownResultsDirs() {
    const dirs = new Set();
    for (const rec of processes.values()) {
        if (rec.dir && fs.existsSync(rec.dir)) dirs.add(rec.dir);
    }
    try {
        for (const name of fs.readdirSync(ROOT)) {
            if (/^results/.test(name) && fs.statSync(path.join(ROOT, name)).isDirectory()) {
                dirs.add(path.join(ROOT, name));
            }
        }
    } catch (e) { /* нет каталога */ }
    return [...dirs].sort();
}

/** Один GET-запрос JSON (без зависимостей, только для локальной сети). */
function getJson(url, timeoutMs = 1200) {
    return new Promise((resolve) => {
        const req = http.get(url, { timeout: timeoutMs }, (res) => {
            let body = '';
            res.on('data', (c) => { body += c; });
            res.on('end', () => {
                if (res.statusCode !== 200) return resolve(null);
                try { resolve(JSON.parse(body)); } catch (e) { resolve(null); }
            });
        });
        req.on('timeout', () => { req.destroy(); resolve(null); });
        req.on('error', () => resolve(null));
    });
}

/**
 * Мастера, запущенные вне панели (например, из терминала): ищем по портам.
 * Признак мастера — ответ /chain/height вида {element, height, head_sha256}.
 */
async function discoverMasters(from = 8000, to = 8010) {
    const skip = new Set([...processes.values()]
        .filter((r) => r.kind === 'master' && isAlive(r.pid)).map((r) => r.port));
    const ports = [];
    for (let port = from; port <= to; port += 1) {
        if (!skip.has(port)) ports.push(port);
    }
    // параллельно: последовательный опрос 10 портов занимал ~10 с
    const found = await Promise.all(ports.map(async (port) => {
        const h = await getJson(`http://127.0.0.1:${port}/chain/height`);
        if (!h || typeof h.element !== 'string') return null;
        const [queue, peers] = await Promise.all([
            getJson(`http://127.0.0.1:${port}/queue-status`),
            getJson(`http://127.0.0.1:${port}/admin/peers`),
        ]);
        return {
            port,
            url: `http://127.0.0.1:${port}`,
            element: h.element,
            height: h.height,
            head: h.head_sha256,
            pending: queue ? queue.pending : null,
            active: queue ? queue.active : null,
            peers: peers ? (peers.peers || []) : [],
        };
    }));
    return found.filter(Boolean);
}

/** Самый свежий .out CP2K в каталоге результатов (для вкладки «Воркер»). */
function latestCp2kLog(dir, tailBytes = 64 * 1024) {
    const base = resolveDir(dir);
    if (!base) throw new Error('не указан каталог результатов');
    const files = [];
    const walk = (d, depth) => {
        if (depth > 3) return;
        let entries = [];
        try { entries = fs.readdirSync(d, { withFileTypes: true }); } catch (e) { return; }
        for (const e of entries) {
            const p = path.join(d, e.name);
            if (e.isDirectory()) walk(p, depth + 1);
            else if (e.name.endsWith('.out')) {
                try {
                    const st = fs.statSync(p);
                    files.push({ path: p, size: st.size, mtime: st.mtimeMs });
                } catch (err) { /* пропускаем */ }
            }
        }
    };
    walk(base, 0);
    files.sort((a, b) => b.mtime - a.mtime);
    if (!files.length) return { file: null, content: '', files: [] };

    const newest = files[0];
    const from = Math.max(0, newest.size - tailBytes);
    const fd = fs.openSync(newest.path, 'r');
    const buf = Buffer.alloc(newest.size - from);
    fs.readSync(fd, buf, 0, buf.length, from);
    fs.closeSync(fd);

    return {
        file: path.relative(base, newest.path),
        size: newest.size,
        mtime: newest.mtime,
        truncated: from > 0,
        content: buf.toString('utf8'),
        files: files.slice(0, 20).map((f) => ({
            file: path.relative(base, f.path), size: f.size, mtime: f.mtime,
        })),
    };
}

module.exports = {
    ROOT,
    list,
    find,
    stop,
    startMaster,
    startWorker,
    startMace,
    readLog,
    knownResultsDirs,
    latestCp2kLog,
    discoverMasters,
    configuredMasterPort,
    pickPort,
    portFree,
    isAlive,
};
