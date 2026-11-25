'use strict';

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

exports.SSL_PORT = parseInt(process.env.WEBUI_SSL_PORT || '8443', 10);

const KEY_PATH = path.join(__dirname, 'privkey.pem');
const CERT_PATH = path.join(__dirname, 'fullchain.pem');

/**
 * Самоподписанный сертификат для локального HTTPS.
 *
 * Приватный ключ НЕ хранится в репозитории: если файлов нет, они создаются
 * при первом запуске (openssl есть в образе). Если openssl недоступен —
 * панель просто работает по HTTP.
 */
function ensureCert() {
    if (fs.existsSync(KEY_PATH) && fs.existsSync(CERT_PATH)) {
        return { key: fs.readFileSync(KEY_PATH), cert: fs.readFileSync(CERT_PATH) };
    }
    try {
        execFileSync('openssl', [
            'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '3650',
            '-subj', '/CN=localhost',
            '-keyout', KEY_PATH, '-out', CERT_PATH,
        ], { stdio: 'ignore', timeout: 30000 });
        console.log('Создан самоподписанный сертификат для HTTPS (src/webserver/*.pem)');
        return { key: fs.readFileSync(KEY_PATH), cert: fs.readFileSync(CERT_PATH) };
    } catch (e) {
        console.warn('HTTPS отключён: нет сертификата и openssl недоступен (' + e.message + ')');
        return null;
    }
}

exports.SSL_options = ensureCert();
