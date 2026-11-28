/**
 * Деньги в крупной единице DFT: 1 DFT = 1 000 000 базовых единиц.
 *
 * Один файл на две стороны: в браузере он даёт window.Money, на сервере —
 * module.exports (панель использует require). Консенсус по-прежнему считает
 * в целых базовых единицах; здесь только показ и разбор ввода.
 */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.Money = factory();
}(typeof self !== 'undefined' ? self : this, function () {
    var TICKER = 'DFT';
    var DECIMALS = 6;
    var BASE = 1000000;

    function fmt(value, opts) {
        opts = opts || {};
        if (value === null || value === undefined || value === '') return '—';
        var n = Math.trunc(Number(value));
        if (!isFinite(n)) return '—';
        var sign = n < 0 ? '−' : (opts.signed && n > 0 ? '+' : '');
        var abs = Math.abs(n);
        var whole = Math.floor(abs / BASE);
        var frac = String(abs % BASE).padStart(DECIMALS, '0').replace(/0+$/, '');
        return sign + whole + (frac ? '.' + frac : '')
            + (opts.ticker ? ' ' + TICKER : '');
    }

    /** Строка в DFT → базовые единицы (целое). null — ввод некорректный. */
    function parse(text) {
        var s = String(text === null || text === undefined ? '' : text)
            .trim().replace(',', '.');
        if (!/^\d+(\.\d{0,6})?$/.test(s)) return null;
        var parts = s.split('.');
        var whole = parts[0] || '0';
        var frac = (parts[1] || '') + '000000';
        var units = Number(whole) * BASE + Number(frac.slice(0, DECIMALS));
        return Number.isSafeInteger(units) ? units : null;
    }

    return { TICKER: TICKER, DECIMALS: DECIMALS, BASE: BASE, fmt: fmt, parse: parse };
}));
