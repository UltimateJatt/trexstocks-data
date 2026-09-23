"""Small shared helpers: time, files, symbols."""
import json
import math
from datetime import datetime, date
from zoneinfo import ZoneInfo

from . import config

ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> date:
    return now_et().date()


def in_window(t: datetime, start, end) -> bool:
    return (t.hour, t.minute) >= start and (t.hour, t.minute) <= end


def display_symbol(yahoo: str) -> str:
    """RY.TO -> RY, BRK-B -> BRK-B"""
    return yahoo[:-3] if yahoo.endswith(".TO") else yahoo


def currency_of(yahoo: str) -> str:
    return "CAD" if yahoo.endswith(".TO") else "USD"


def clean(x):
    """Turn NaN/inf into None so JSON stays valid; round floats."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, 4)
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item"):  # numpy scalar
        return clean(x.item())
    return x


def load_state(name: str, default=None):
    p = config.STATE / name
    if not p.exists():
        return default
    with open(p) as f:
        return json.load(f)


def save_state(name: str, data) -> None:
    config.STATE.mkdir(parents=True, exist_ok=True)
    with open(config.STATE / name, "w") as f:
        json.dump(clean(data), f, indent=1, sort_keys=False)


def log(*a):
    print(f"[{now_et():%H:%M:%S ET}]", *a, flush=True)
