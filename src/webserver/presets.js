'use strict';
/**
 * Элементы для панели. Источник правды — src/elements.json: там символ,
 * русское имя, диаметр атома (металлическая длина связи), псевдопотенциал
 * GTH и валентность. Этот же файл читает cp2k_runner (потенциалы и базис),
 * поэтому панель и CP2K не могут разойтись.
 */
const fs = require('fs');
const path = require('path');

const ELEMENTS_PATH = path.join(__dirname, '..', 'elements.json');

function loadElements() {
    const raw = JSON.parse(fs.readFileSync(ELEMENTS_PATH, 'utf8'));
    const list = [];
    const presets = {};
    for (const [symbol, d] of Object.entries(raw.elements || {})) {
        list.push({
            symbol,
            name_ru: d.name_ru || symbol,
            atom_diameter: d.atom_diameter,
            pseudopotential: d.pseudopotential,
            valence: d.valence,
        });
        presets[symbol] = {
            name: symbol,
            composition: { [symbol]: 1 },
            atom_diameter: d.atom_diameter,
        };
    }
    return { basis: raw.basis || '', list, presets };
}

const { basis: BASIS, list: ELEMENT_LIST, presets: ELEMENT_PRESETS } = loadElements();

function applyElementPreset(configPath, element) {
    const preset = ELEMENT_PRESETS[element];
    if (!preset) throw new Error(`неизвестный элемент: ${element}`);
    const raw = fs.readFileSync(configPath, 'utf8');
    const cfg = JSON.parse(raw);
    let owner = null;
    try { owner = fs.statSync(configPath); } catch (e) { owner = null; }
    const prev = cfg.system;
    const help = (prev && typeof prev === 'object' && prev.help) ? prev.help : '';
    cfg.system = { value: preset, help };
    fs.writeFileSync(configPath, JSON.stringify(cfg, null, 2) + '\n');
    // панель в контейнере работает root'ом, проект в bind-mount — чужой
    try { if (owner) fs.chownSync(configPath, owner.uid, owner.gid); } catch (e) { /* ignore */ }
    return preset;
}

module.exports = { ELEMENTS_PATH, BASIS, ELEMENT_LIST, ELEMENT_PRESETS, applyElementPreset };
