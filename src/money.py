#!/usr/bin/env python3
"""
Отображение денег в «крупной» единице DFT.

Консенсус считает в целых базовых единицах (int), как и раньше: 1 монета =
1 базовая единица, эмиссия 50 000 000 за блок. Для человека это неудобно,
поэтому всё, что видит пользователь, показывается в DFT:

    1 DFT = 1 000 000 базовых единиц (как 1 BTC = 10^8 сатоши)

Дробная часть — до 6 знаков, хвостовые нули убираются: 25 000 000 → "25",
100 000 → "0.1", 1 → "0.000001". Ввод — тоже в DFT, но разбор строго
строковый (никаких float): "12.345678" → 12 345 678 базовых единиц.
"""

TICKER = "DFT"
DECIMALS = 6
BASE = 10 ** DECIMALS          # 1 DFT в базовых единицах


def fmt(value, ticker: bool = False, signed: bool = False) -> str:
    """Базовые единицы → строка в DFT (без хвостовых нулей)."""
    value = int(value)
    sign = "-" if value < 0 else ("+" if signed and value > 0 else "")
    whole, frac = divmod(abs(value), BASE)
    text = str(whole)
    if frac:
        text += "." + f"{frac:0{DECIMALS}d}".rstrip("0")
    return f"{sign}{text}" + (f" {TICKER}" if ticker else "")


def parse(text, field: str = "сумма") -> int:
    """
    Строка в DFT → базовые единицы. Строго, без float и без округления:
    «12.3456789» — ошибка, а не 12.345678.
    """
    raw = str(text if text is not None else "").strip().replace(",", ".")
    if not raw:
        raise ValueError(f"{field}: пусто")
    if raw.startswith(("-", "+")):
        # сумма — величина: знак задаёт направление транзакции, а не число
        raise ValueError(f"{field}: без знака, {raw!r}")
    if raw.count(".") > 1:
        raise ValueError(f"{field}: некорректное число {raw!r}")
    whole, _, frac = raw.partition(".")
    if whole in ("", "+", "-"):
        whole = whole + "0" if whole.startswith(("+", "-")) else "0"
    if not whole.lstrip("+-").isdigit():
        raise ValueError(f"{field}: некорректное число {raw!r}")
    if frac and not frac.isdigit():
        raise ValueError(f"{field}: некорректная дробная часть {raw!r}")
    if len(frac) > DECIMALS:
        raise ValueError(f"{field}: больше {DECIMALS} знаков после точки "
                         f"({raw!r}); 1 {TICKER} = 10^{DECIMALS} базовых единиц")
    sign = -1 if whole.startswith("-") else 1
    whole_digits = whole.lstrip("+-")
    units = (int(whole_digits) * BASE + int(frac.ljust(DECIMALS, "0"))) * sign
    return units


def fmt_money(value, **kw) -> str:
    return fmt(value, ticker=True, **kw)
