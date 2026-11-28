'use strict';

/**
 * Аутентификация панели: вход по паролю, scrypt-хэш в SQLite (results/webui.db).
 * Пароль передаётся в Python только через env (не в argv, иначе виден в `ps`).
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');

const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const RESULTS_DIR = path.join(PROJECT_ROOT, 'results');
const SCRIPT = path.join(PROJECT_ROOT, 'src', 'webadmin.py');

const COOKIE = 'webui_session';
const TTL_MS = 12 * 60 * 60 * 1000;     // 12 часов
const LOGIN_MAX = 5;                    // попыток
const LOGIN_WINDOW_MS = 60 * 1000;      // за минуту

const sessions = new Map();             // token -> {user, exp}
const attempts = new Map();             // ip -> [timestamps]

function runAdmin(args, env) {
    return new Promise((resolve) => {
        execFile('python3', [SCRIPT].concat(args), {
            cwd: PROJECT_ROOT,
            env: Object.assign({}, process.env, env || {}),
            timeout: 30000,
        }, (err, stdout) => {
            let out = null;
            try { out = JSON.parse(String(stdout || '').trim()); } catch (e) { out = null; }
            if (!out) {
                out = { ok: false, error: (err && err.message) || 'webadmin.py: пустой ответ' };
            }
            resolve(out);
        });
    });
}

async function init() {
    const out = await runAdmin(['init', '--root', RESULTS_DIR, '--name', 'admin']);
    if (!out.ok) {
        console.error('Панель: не удалось инициализировать администратора:', out.error);
        return;
    }
    if (out.created && out.password) {
        // Пароль НЕ печатаем в лог контейнера (аудит, L4): логи уезжают в
        // journald/CI и копируются вместе с отчётами. Он пишется только в
        // файл 0600, а если файл не пишется — просим задать пароль через env.
        const file = path.join(RESULTS_DIR, 'webui_admin_password.txt');
        try {
            fs.writeFileSync(file, out.password + '\n', { mode: 0o600 });
            console.log(`Панель: создан admin, пароль записан в ${file} (0600)`);
        } catch (e) {
            console.error('Панель: создан admin, но файл с паролем записать не удалось ' +
                `(${e.message}). Задайте WEBUI_ADMIN_PASSWORD и пересоздайте admin.`);
        }
    } else if (out.created) {
        console.log('Панель: admin создан из WEBUI_ADMIN_PASSWORD');
    }
}

// ---------- сессии ----------

function setCookie(res, req, token, maxAgeSec) {
    const secure = req.secure ? '; Secure' : '';
    res.setHeader('Set-Cookie',
        `${COOKIE}=${token}; Path=/; HttpOnly; SameSite=Strict; ` +
        `Max-Age=${maxAgeSec}${secure}`);
}

function tokenOf(req) {
    const raw = String(req.headers.cookie || '');
    for (const part of raw.split(';')) {
        const idx = part.indexOf('=');
        if (idx < 0) continue;
        if (part.slice(0, idx).trim() === COOKIE) {
            return decodeURIComponent(part.slice(idx + 1).trim());
        }
    }
    return null;
}

function current(req) {
    const token = tokenOf(req);
    if (!token) return null;
    const rec = sessions.get(token);
    if (!rec) return null;
    if (rec.exp < Date.now()) {
        sessions.delete(token);
        return null;
    }
    return rec;
}

function issue(req, res, user) {
    const token = crypto.randomBytes(32).toString('hex');
    sessions.set(token, { user, exp: Date.now() + TTL_MS });
    setCookie(res, req, token, Math.floor(TTL_MS / 1000));
}

// ---------- защита от перебора ----------

function tooManyAttempts(ip) {
    const now = Date.now();
    const arr = (attempts.get(ip) || []).filter((t) => now - t < LOGIN_WINDOW_MS);
    attempts.set(ip, arr);
    return arr.length >= LOGIN_MAX;
}

function noteAttempt(ip) {
    const arr = attempts.get(ip) || [];
    arr.push(Date.now());
    attempts.set(ip, arr);
}

// ---------- маршруты ----------

function register(app) {
    app.get('/login', (req, res) => {
        if (current(req)) return res.redirect('/');
        res.render('pages/login', { lang: 'ru', error: null });
    });

    app.post('/login', async (req, res) => {
        const ip = req.ip || 'unknown';
        if (tooManyAttempts(ip)) {
            return res.status(429).render('pages/login', {
                lang: 'ru', error: 'слишком много попыток — подождите минуту',
            });
        }
        const password = String((req.body && req.body.password) || '');
        const out = await runAdmin(['check', '--root', RESULTS_DIR, '--name', 'admin'],
                                   { WEBUI_PASSWORD: password });
        if (!out.ok) {
            noteAttempt(ip);
            return res.status(401).render('pages/login', {
                lang: 'ru', error: 'неверный пароль',
            });
        }
        attempts.delete(ip);
        issue(req, res, 'admin');
        res.redirect('/');
    });

    app.get('/logout', (req, res) => {
        const token = tokenOf(req);
        if (token) sessions.delete(token);
        setCookie(res, req, '', 0);
        res.redirect('/login');
    });

    app.get('/password', (req, res) => {
        res.render('pages/password', { lang: 'ru', error: null, ok: null });
    });

    app.post('/password', async (req, res) => {
        const body = req.body || {};
        const oldP = String(body.old_password || '');
        const newP = String(body.new_password || '');
        const confirm = String(body.confirm_password || '');
        const render = (error, ok) => res.render('pages/password', { lang: 'ru', error, ok });

        if (newP.length < 8) return render('новый пароль короче 8 символов', null);
        if (newP !== confirm) return render('пароли не совпадают', null);

        const out = await runAdmin(['passwd', '--root', RESULTS_DIR, '--name', 'admin'],
                                   { WEBUI_OLD_PASSWORD: oldP, WEBUI_NEW_PASSWORD: newP });
        if (!out.ok) return render(out.error || 'не удалось сменить пароль', null);

        // после смены пароля старые сессии недействительны
        sessions.clear();
        issue(req, res, 'admin');
        return render(null, 'пароль изменён');
    });
}

function middleware(req, res, next) {
    if (req.path === '/login' || req.path === '/favicon.ico') return next();
    if (current(req)) return next();
    if (req.path.startsWith('/api/')) {
        return res.status(401).json({ ok: false, error: 'нужен вход в панель' });
    }
    return res.redirect('/login');
}

module.exports = { init, register, middleware };
