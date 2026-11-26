'use strict';

const path = require('path');
const g_constants = require('./constants');

const https = require('https');
const http = require('http');
const express = require('express');
const auth = require('./auth');

const app = express();

// --- Абсолютные пути проекта ---
// __dirname    = .../src/webserver
// PROJECT_ROOT = .../cp2k-mace-project
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const RESULTS_DIR = path.join(PROJECT_ROOT, 'results');
const CONFIG_PATH = path.join(PROJECT_ROOT, 'src', 'config.json');
const MAIN_SCRIPT = path.join(PROJECT_ROOT, 'src', 'main.py');
const LOG_FILE = path.join(RESULTS_DIR, 'main.log');
const PID_FILE = path.join(RESULTS_DIR, 'main.pid');
const STATUS_FILE = path.join(RESULTS_DIR, 'main.status.json');
const ARCHIVE_FILE = path.join(RESULTS_DIR, 'artifacts.zip');

app.locals.paths = {
    PROJECT_ROOT, RESULTS_DIR, CONFIG_PATH, MAIN_SCRIPT,
    LOG_FILE, PID_FILE, STATUS_FILE, ARCHIVE_FILE,
};

// body parser: JSON для API, urlencoded для формы входа
app.use(express.json({ limit: '8mb' }));
app.use(express.urlencoded({ extended: false, limit: '16kb' }));

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Аутентификация панели: без входа доступны только /login и /favicon.ico.
// Пароль (scrypt-хэш) лежит в results/webui.db, см. src/webadmin.py.
auth.init();
auth.register(app);
app.use(auth.middleware);

// статика — уже после проверки сессии
app.use(express.static(path.join(__dirname, 'static_pages')));

app.use((err, req, res, next) => {
    console.error('Express error:', err);
    res.status(500).send('Something broke!');
});

require('./reqHandler.js').handle(app);

// --- HTTPS (если есть сертификат) и HTTP для простого входа ---
if (g_constants.SSL_options) {
    const httpsServer = https.createServer(g_constants.SSL_options, app);
    httpsServer.listen(g_constants.SSL_PORT, () => {
        console.log(`Панель (HTTPS): https://127.0.0.1:${g_constants.SSL_PORT}`);
    });
} else {
    console.log('Панель (HTTPS) не запущена: нет сертификата');
}

const HTTP_PORT = parseInt(process.env.WEBUI_PORT || '8080', 10);
http.createServer(app).listen(HTTP_PORT, () => {
    console.log(`Панель (HTTP):  http://127.0.0.1:${HTTP_PORT}`);
});
