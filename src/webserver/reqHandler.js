'use strict';

const fs = require('fs');
const path = require('path');
const http = require('http');
const { execFile } = require('child_process');

const procs = require('./procs');

// ==================== helpers ====================

function fetchJson(url, timeoutMs = 2500) {
    return new Promise((resolve) => {
        const req = http.get(url, { timeout: timeoutMs }, (res) => {
            let body = '';
            res.on('data', (chunk) => { body += chunk; });
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    try { resolve(JSON.parse(body)); } catch (e) { resolve(null); }
                } else {
                    resolve(null);
                }
            });
        });
        req.on('timeout', () => { req.destroy(); resolve(null); });
        req.on('error', () => resolve(null));
    });
}

function runPython(args, env, timeoutMs = 60000) {
    return new Promise((resolve) => {
        execFile('python3', args, {
            cwd: procs.ROOT,
            env: Object.assign({}, process.env, env || {}),
            maxBuffer: 8 * 1024 * 1024,
            timeout: timeoutMs,
        }, (err, stdout, stderr) => {
            const text = String(stdout || '').trim();
            let parsed = null;
            try { parsed = JSON.parse(text); } catch (e) { parsed = null; }
            if (parsed) return resolve(parsed);
            resolve({
                ok: false,
                error: (stderr || '').trim() || (err && err.message) || 'пустой ответ',
            });
        });
    });
}

function isInsideRoot(p) {
    const abs = path.resolve(p);
    return abs === procs.ROOT || abs.startsWith(procs.ROOT + path.sep);
}

function postJson(url, payload, method = 'POST', timeoutMs = 4000) {
    return new Promise((resolve) => {
        const body = JSON.stringify(payload);
        const req2 = http.request(url, {
            method,
            headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) },
            timeout: timeoutMs,
        }, (r) => {
            let data = '';
            r.on('data', (c) => { data += c; });
            r.on('end', () => resolve({ status: r.statusCode, body: data }));
        });
        req2.on('timeout', () => { req2.destroy(); resolve({ status: 0, body: '' }); });
        req2.on('error', () => resolve({ status: 0, body: '' }));
        req2.write(body);
        req2.end();
    });
}

const normaliseUrl = (u) => String(u || '').trim().replace(/\/+$/, '');
let _lastGossip = 0;

function bad(res, code, message) {
    return res.status(code).json({ ok: false, error: message });
}

// ==================== routes ====================

exports.handle = function (app) {
    global.paths = app.locals.paths;

    app.get('/', onMain);

    // старые эндпоинты панели main.py (оставлены для совместимости)
    app.get('/api/config', onGetConfig);
    app.post('/api/config', onSaveConfig);
    app.get('/api/status', onStatus);
    app.get('/api/cluster', onClusterStatus);
    app.get('/api/log', onLog);
    app.get('/download', onDownload);

    // новые: процессы
    app.get('/api/overview', onOverview);
    app.post('/api/master/start', onMasterStart);
    app.post('/api/worker/start', onWorkerStart);
    app.post('/api/process/stop', onProcessStop);
    app.get('/api/process/log', onProcessLog);
    app.get('/api/cp2k/log', onCp2kLog);

    // сеть мастеров
    app.get('/api/network', onNetwork);
    app.post('/api/master/peers', onMasterPeers);

    // кошелёк
    app.post('/api/wallet/info', onWalletInfo);
    app.post('/api/wallet/chains', onWalletChains);
    app.post('/api/wallet/list', onWalletList);
    app.post('/api/wallet/create', onWalletCreate);
    app.post('/api/wallet/send', onWalletSend);
    app.post('/api/wallet/passwd', onWalletPasswd);
    app.post('/api/wallet/newkey', onWalletNewKey);
    app.post('/api/wallet/tx', onWalletTx);

    // MACE
    app.post('/api/mace/run', onMaceRun);
    app.post('/api/mace/stop', onProcessStop);
};

function onMain(req, res) {
    res.render('pages/index', { lang: 'ru' });
}

// ---------- Обзор ----------

async function masterInfo(rec) {
    const base = `http://127.0.0.1:${rec.port}`;
    const [height, queue, status, peers] = await Promise.all([
        fetchJson(base + '/chain/height'),
        fetchJson(base + '/queue-status'),
        fetchJson(base + '/admin/status'),
        fetchJson(base + '/admin/peers'),
    ]);
    return {
        reachable: Boolean(height || queue),
        height: height ? height.height : null,
        head: height ? height.head_sha256 : null,
        pending: queue ? queue.pending : null,
        active: queue ? queue.active : null,
        verifying: queue ? queue.verifying : null,
        completed: queue ? queue.completed : null,
        failed: queue ? queue.failed : null,
        workers: status ? (status.workers || []).length : null,
        paused: status ? status.paused : null,
        peers: peers ? (peers.peers || []) : [],
        template: height ? (height.template || null) : null,
    };
}

async function onOverview(req, res) {
    try {
        const all = procs.list();
        const masters = [];
        for (const rec of all.filter((r) => r.kind === 'master' && r.alive)) {
            try {
                masters.push(Object.assign({}, rec, { info: await masterInfo(rec) }));
            } catch (e) {
                masters.push(Object.assign({}, rec, { info: null }));
            }
        }
        res.json({
            ok: true,
            processes: all,
            masters,
            external: await procs.discoverMasters(8000, 8010),
            dirs: procs.knownResultsDirs(),
            root: procs.ROOT,
        });
    } catch (e) {
        bad(res, 500, String(e));
    }
}

// ---------- Запуск/остановка ----------

async function onMasterStart(req, res) {
    try {
        const body = req.body || {};
        const rec = await procs.startMaster({
            port: body.port,
            dir: body.dir,
            wallet: body.wallet,
            password: body.password || '',
            peers: Array.isArray(body.peers) ? body.peers : [],
            name: body.name,
        });
        res.json({ ok: true, process: rec });
    } catch (e) {
        bad(res, 400, e.message || String(e));
    }
}

async function onWorkerStart(req, res) {
    try {
        const body = req.body || {};
        const rec = await procs.startWorker({
            name: body.name,
            wallet: body.wallet,
            password: body.password || '',
            masters: Array.isArray(body.masters) ? body.masters : [],
        });
        res.json({ ok: true, process: rec });
    } catch (e) {
        bad(res, 400, e.message || String(e));
    }
}

function onProcessStop(req, res) {
    try {
        const id = (req.body || {}).id;
        if (!id) return bad(res, 400, 'нужен id процесса');
        res.json({ ok: true, process: procs.stop(id) });
    } catch (e) {
        bad(res, 400, e.message || String(e));
    }
}

function onProcessLog(req, res) {
    try {
        const id = req.query.id;
        if (!id) return bad(res, 400, 'нужен id');
        const offset = parseInt(req.query.offset, 10);
        res.json(Object.assign({ ok: true }, procs.readLog(id, offset)));
    } catch (e) {
        bad(res, 404, e.message || String(e));
    }
}

function onCp2kLog(req, res) {
    try {
        const dir = req.query.dir || (global.paths && global.paths.RESULTS_DIR);
        const tail = Math.min(parseInt(req.query.tail, 10) || 64 * 1024, 512 * 1024);
        res.json(Object.assign({ ok: true }, procs.latestCp2kLog(dir, tail)));
    } catch (e) {
        bad(res, 404, e.message || String(e));
    }
}

// ---------- Сеть мастеров ----------

async function onNetwork(req, res) {
    try {
        const found = new Map(); // url -> entry
        const peerLists = new Map(); // url мастера -> его список пиров

        for (const rec of procs.list().filter((r) => r.kind === 'master' && r.alive)) {
            const url = `http://127.0.0.1:${rec.port}`;
            const info = await masterInfo(rec);
            found.set(url, {
                url, source: 'эта панель', ok: info.reachable,
                height: info.height, peers: info.peers | 0,
                pid: rec.pid, dir: rec.dir,
            });
            peerLists.set(url, (info.peers || []).map(normaliseUrl));
        }

        // мастера, запущенные вне панели (из терминала)
        for (const ext of await procs.discoverMasters(8000, 8010)) {
            if (!found.has(ext.url)) {
                found.set(ext.url, {
                    url: ext.url, source: 'вне панели', ok: true,
                    height: ext.height, peers: (ext.peers || []).length,
                });
                peerLists.set(ext.url, (ext.peers || []).map(normaliseUrl));
            }
        }

        // адреса, которые мастер сообщает воркеру (его список пиров)
        const seeds = [...found.keys()];
        for (const url of seeds) {
            const peers = await fetchJson(url + '/admin/peers');
            if (!peerLists.has(url)) peerLists.set(url, []);
            for (const peer of (peers && peers.peers) || []) {
                const clean = normaliseUrl(peer);
                peerLists.get(url).push(clean);
                if (found.has(clean)) {
                    found.get(clean).source = found.get(clean).source + ', пир';
                    continue;
                }
                const h = await fetchJson(clean + '/chain/height');
                found.set(clean, {
                    url: clean, source: 'пир мастера', ok: Boolean(h),
                    height: h ? h.height : null, peers: 0,
                });
            }
        }

        // Обмен адресами (как в p2p): сообщаем каждому мастеру об известных
        // нам узлах, которых у него нет. Не чаще раза в 30 с, по одному адресу.
        const gossip = [];
        if (Date.now() - _lastGossip > 30000) {
            _lastGossip = Date.now();
            const all = [...found.keys()];
            for (const [masterUrl, known] of peerLists) {
                const knownSet = new Set(known);
                const candidate = all.find((u) => u !== masterUrl && !knownSet.has(u));
                if (!candidate) continue;
                const r = await postJson(masterUrl + '/admin/peers', { url: candidate });
                if (r.status === 200) gossip.push({ master: masterUrl, peer: candidate });
            }
        }

        res.json({ ok: true, nodes: [...found.values()], gossip });
    } catch (e) {
        bad(res, 500, String(e));
    }
}

async function onMasterPeers(req, res) {
    try {
        const body = req.body || {};
        const port = parseInt(body.port, 10);
        if (!port) return bad(res, 400, 'нужен port мастера');
        if (!body.url) return bad(res, 400, 'нужен url пира');

        const url = `http://127.0.0.1:${port}/admin/peers`;
        const text = await postJson(url, { url: String(body.url).trim() },
            body.action === 'del' ? 'DELETE' : 'POST');
        let body2 = null;
        try { body2 = JSON.parse(text.body); } catch (e) { body2 = null; }
        if (text.status === 200 && body2) return res.json({ ok: true, peers: body2.peers });
        if (text.status === 404 && body.action === 'del') {
            return res.json({ ok: true, peers: null, note: 'пир уже удалён' });
        }
        return bad(res, 502, `мастер ответил ${text.status}: ${text.body.slice(0, 200)}`);
    } catch (e) {
        bad(res, 500, String(e));
    }
}

// ---------- Кошелёк ----------

function walletRoot(body) {
    const raw = (body && body.resultsRoot) || path.join(procs.ROOT, 'results');
    const abs = path.resolve(raw);
    if (!isInsideRoot(abs)) throw new Error('каталог результатов должен быть внутри проекта');
    return abs;
}

async function onWalletInfo(req, res) {
    try {
        const body = req.body || {};
        if (!body.password) return bad(res, 400, 'нужен пароль кошелька');
        const root = walletRoot(body);
        const args = ['src/webwallet.py', 'info', '--results-root', root];
        if (body.name) args.push('--name', String(body.name));
        if (body.element) args.push('--element', String(body.element));
        for (const url of (body.masters || [])) {
            if (url) args.push('--master', String(url));
        }
        const out = await runPython(
            args,
            { WEBUI_WALLET_PASSWORD: String(body.password) }, 20000);
        res.json(out);
    } catch (e) {
        bad(res, 500, String(e));
    }
}

async function onWalletList(req, res) {
    // Какие кошельки вообще есть: только имена, без секретов.
    try {
        const body = req.body || {};
        const out = await runPython(
            ['src/webwallet.py', 'list', '--results-root', walletRoot(body)], {}, 20000);
        res.json(out);
    } catch (e) {
        bad(res, 500, String(e));
    }
}

async function onWalletCreate(req, res) {
    // Явное создание: имя + пароль. Ничего не создаётся само.
    try {
        const body = req.body || {};
        const name = String(body.name || '').trim();
        if (!name) return bad(res, 400, 'нужно имя кошелька');
        if (!body.password) return bad(res, 400, 'нужен пароль кошелька');
        const args = ['src/webwallet.py', 'create', '--results-root', walletRoot(body),
            '--name', name];
        const out = await runPython(
            args, { WEBUI_WALLET_PASSWORD: String(body.password) }, 30000);
        res.json(out);
    } catch (e) {
        bad(res, 500, String(e));
    }
}

async function onWalletChains(req, res) {
    // Какие элементы (системы) вообще есть в этом каталоге результатов: у
    // каждого своя цепочка — в кошельке это как разные металлы.
    try {
        const body = req.body || {};
        const out = await runPython(
            ['src/webwallet.py', 'chains', '--results-root', walletRoot(body)], {}, 20000);
        res.json(out);
    } catch (e) {
        bad(res, 500, String(e));
    }
}

async function onWalletSend(req, res) {
    try {
        const body = req.body || {};
        if (!body.password) return bad(res, 400, 'нужен пароль кошелька');
        const amount = Number(body.amount);
        if (!body.to || !(amount > 0)) return bad(res, 400, 'нужны адрес и сумма');

        const root = walletRoot(body);
        const args = ['src/webwallet.py', 'send', '--results-root', root,
            '--to', String(body.to), '--amount', String(amount)];
        if (body.fee !== undefined && body.fee !== null && body.fee !== '') {
            args.push('--fee', String(body.fee));
        }
        if (body.name) args.push('--name', String(body.name));
        if (body.element) args.push('--element', String(body.element));
        const masters = (body.masters || []).map((m) => String(m).trim()).filter(Boolean);
        for (const url of masters) args.push('--master', url);

        const out = await runPython(args, { WEBUI_WALLET_PASSWORD: String(body.password) }, 90000);
        res.json(out);
    } catch (e) {
        bad(res, 500, String(e));
    }
}

async function onWalletNewKey(req, res) {
    try {
        const body = req.body || {};
        if (!body.password) return bad(res, 400, 'нужен пароль кошелька');
        const args = ['src/webwallet.py', 'newkey',
            '--results-root', walletRoot(body)];
        if (body.name) args.push('--name', String(body.name));
        if (body.element) args.push('--element', String(body.element));
        const out = await runPython(
            args, { WEBUI_WALLET_PASSWORD: String(body.password) }, 30000);
        res.json(out);
    } catch (e) {
        bad(res, 500, String(e));
    }
}

async function onWalletPasswd(req, res) {
    try {
        const body = req.body || {};
        if (!body.password || !body.newPassword) {
            return bad(res, 400, 'нужны старый и новый пароли');
        }
        const root = walletRoot(body);
        const args = ['src/webwallet.py', 'passwd', '--results-root', root,
            '--change-to', String(body.newPassword)];
        if (body.name) args.push('--name', String(body.name));
        if (body.element) args.push('--element', String(body.element));
        const out = await runPython(args, { WEBUI_WALLET_PASSWORD: String(body.password) }, 30000);
        res.json(out);
    } catch (e) {
        bad(res, 500, String(e));
    }
}

async function onWalletTx(req, res) {
    try {
        const body = req.body || {};
        if (!body.txid) return bad(res, 400, 'нужен txid');
        const root = walletRoot(body);
        const args = ['src/webwallet.py', 'tx', '--results-root', root,
            '--txid', String(body.txid)];
        if (body.element) args.push('--element', String(body.element));
        for (const url of (body.masters || [])) {
            if (url) args.push('--master', String(url));
        }
        const out = await runPython(args, {}, 20000);
        res.json(out);
    } catch (e) {
        bad(res, 500, String(e));
    }
}

// ---------- MACE ----------

function onMaceRun(req, res) {
    try {
        const body = req.body || {};
        const rec = procs.startMace({
            element: body.element,
            resultsDir: body.resultsDir,
            outputDir: body.outputDir,
            model: body.model,
            skipTrain: Boolean(body.skipTrain),
            skipSearch: Boolean(body.skipSearch),
        });
        res.json({ ok: true, process: rec });
    } catch (e) {
        bad(res, 400, e.message || String(e));
    }
}

// ==================== старые обработчики ====================

function onGetConfig(req, res) {
    try {
        const raw = fs.readFileSync(global.paths.CONFIG_PATH, 'utf8');
        res.json({ ok: true, config: JSON.parse(raw) });
    } catch (e) {
        bad(res, 500, String(e));
    }
}

function onSaveConfig(req, res) {
    try {
        const incoming = req.body && req.body.config;
        if (!incoming || typeof incoming !== 'object') return bad(res, 400, 'config must be an object');

        const raw = fs.readFileSync(global.paths.CONFIG_PATH, 'utf8');
        const current = JSON.parse(raw);
        // сохраняем владельца файла: панель работает root'ом в контейнере,
        // а проект в bind-mount принадлежит пользователю (иначе файл станет root-овым)
        let owner = null;
        try { owner = fs.statSync(global.paths.CONFIG_PATH); } catch (e) { owner = null; }
        const updated = {};
        for (const key of Object.keys(current)) {
            const cur = current[key];
            const inc = incoming[key];
            if (cur && typeof cur === 'object' && 'value' in cur && 'help' in cur) {
                updated[key] = {
                    value: (inc && typeof inc === 'object' && 'value' in inc) ? inc.value : cur.value,
                    help: cur.help,
                };
            } else {
                updated[key] = (inc !== undefined) ? inc : cur;
            }
        }
        const tmp = global.paths.CONFIG_PATH + '.tmp';
        fs.writeFileSync(tmp, JSON.stringify(updated, null, 2), 'utf8');
        if (owner) {
            try { fs.chownSync(tmp, owner.uid, owner.gid); } catch (e) { /* без прав — пропускаем */ }
        }
        fs.renameSync(tmp, global.paths.CONFIG_PATH);
        res.json({ ok: true, config: updated });
    } catch (e) {
        bad(res, 500, String(e));
    }
}

function onStatus(req, res) {
    const all = procs.list();
    res.json({ ok: true, processes: all });
}

async function onClusterStatus(req, res) {
    const port = parseInt(req.query.port, 10) || 8000;
    const queue = await fetchJson(`http://127.0.0.1:${port}/queue-status`);
    res.json({ ok: true, master_reachable: Boolean(queue), queue });
}

function onLog(req, res) {
    try {
        const id = req.query.id || (procs.list().find((p) => p.kind === 'master') || {}).id;
        if (!id) return res.json({ ok: true, offset: 0, chunk: '' });
        const offset = parseInt(req.query.offset, 10);
        res.json(Object.assign({ ok: true }, procs.readLog(id, offset)));
    } catch (e) {
        bad(res, 500, String(e));
    }
}

function onDownload(req, res) {
    const file = global.paths.ARCHIVE_FILE;
    if (!fs.existsSync(file)) return res.status(404).send('artifacts.zip not found yet');
    res.download(file, 'artifacts.zip');
}
