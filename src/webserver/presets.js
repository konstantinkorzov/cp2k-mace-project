'use strict';
/**
 * Готовые элементы для панели: простые системы в один клик.
 *
 * Диаметр — металлическая длина связи (Cu 2.56, Ag 2.89, Au 2.88, Fe 2.48,
 * Al 2.86 Å). Для генератора это min_dist; max_dist = 2 × диаметр. Значения
 * попали в config.json → system.atom_diameter и попадают в заголовок блока,
 * поэтому смена диаметра = другой датасет (для одной системы диаметр должен
 * быть одинаковым у всех мастеров).
 *
 * Псевдопотенциалы и базисы всех пяти элементов есть в образе CP2K:
 *   GTH-PBE-q11 (Cu, Ag, Au), GTH-PBE-q16 (Fe), GTH-PBE-q3 (Al),
 *   DZVP-MOLOPT-SR-GTH.
 *
 * Составные системы (Al2O3, NaCl) — не пресеты: их задают в config.json
 * вручную (composition + atom_diameter).
 */
const fs = require('fs');

const ELEMENT_PRESETS = {
    Cu: { name: 'Cu', composition: { Cu: 1 }, atom_diameter: 2.56 },
    Ag: { name: 'Ag', composition: { Ag: 1 }, atom_diameter: 2.89 },
    Au: { name: 'Au', composition: { Au: 1 }, atom_diameter: 2.88 },
    Fe: { name: 'Fe', composition: { Fe: 1 }, atom_diameter: 2.48 },
    Al: { name: 'Al', composition: { Al: 1 }, atom_diameter: 2.86 },
};

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

module.exports = { ELEMENT_PRESETS, applyElementPreset };
