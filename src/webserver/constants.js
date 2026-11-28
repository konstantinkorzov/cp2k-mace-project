'use strict';

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

exports.SSL_PORT = parseInt(process.env.WEBUI_SSL_PORT || '8443', 10);

const KEY_PATH = path.join(__dirname, 'privkey.pem');
const CERT_PATH = path.join(__dirname, 'fullchain.pem');

// Имена, под которые выпускается сертификат. CN уже не смотрят современные
// браузеры — нужен SAN. Дополнительные имена (домен, внешний IP) передаются
// через WEBUI_TLS_SAN="example.com,203.0.113.7".
const TLS_CN = process.env.WEBUI_TLS_CN || 'localhost';

function sanEntries() {
    const extra = String(process.env.WEBUI_TLS_SAN || '')
        .split(',').map((s) => s.trim()).filter(Boolean);
    const names = ['DNS:localhost', 'IP:127.0.0.1', `DNS:${TLS_CN}`];
    for (const item of extra) {
        names.push(/^[0-9.]+$/.test(item) ? `IP:${item}` : `DNS:${item}`);
    }
    return [...new Set(names)].join(',');
}

/** Есть ли в сертификате SAN (без него браузер ругается на имя, а не только на доверие). */
function certHasSan() {
    if (!fs.existsSync(CERT_PATH)) return false;
    try {
        const out = execFileSync('openssl',
            ['x509', '-in', CERT_PATH, '-noout', '-ext', 'subjectAltName'],
            { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 10000 });
        return /DNS:|IP Address:/.test(out);
    } catch (e) {
        return false;
    }
}

function generateCert() {
    const dir = path.dirname(CERT_PATH);
    fs.mkdirSync(dir, { recursive: true });
    execFileSync('openssl', [
        'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '3650',
        '-subj', `/CN=${TLS_CN}`,
        '-addext', `subjectAltName=${sanEntries()}`,
        '-keyout', KEY_PATH, '-out', CERT_PATH,
    ], { stdio: 'ignore', timeout: 60000 });
    try { fs.chmodSync(KEY_PATH, 0o600); } catch (e) { /* not fatal */ }
    try { fs.chmodSync(CERT_PATH, 0o644); } catch (e) { /* not fatal */ }
}

/**
 * Самоподписанный сертификат для HTTPS панели.
 *
 * Ключ НЕ в репозитории (gitignore): если файлов нет или сертификат старый
 * (без SAN), он создаётся при запуске — openssl есть в образе. Без openssl
 * панель работает по HTTP.
 */
function ensureCert() {
    try {
        if (fs.existsSync(KEY_PATH) && certHasSan()) {
            return { key: fs.readFileSync(KEY_PATH), cert: fs.readFileSync(CERT_PATH) };
        }
        generateCert();
        console.log(`Создан самоподписанный сертификат (CN=${TLS_CN}, SAN=${sanEntries()})`);
        return { key: fs.readFileSync(KEY_PATH), cert: fs.readFileSync(CERT_PATH) };
    } catch (e) {
        console.warn('HTTPS отключён: сертификат не создан (' + e.message + ')');
        return null;
    }
}

exports.SSL_options = ensureCert();
