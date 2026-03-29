from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def make_dataset(days: int = 900, skus: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    rows = []

    for sku_idx in range(1, skus + 1):
        base = 45 + sku_idx * 15
        trend = np.linspace(0, sku_idx * 12, days)
        weekly = 8 * np.sin(2 * np.pi * np.arange(days) / 7)
        yearly = 12 * np.sin(2 * np.pi * np.arange(days) / 365.25 + sku_idx)
        promotion = rng.binomial(1, 0.12, size=days)
        temp = 24 + 8 * np.sin(2 * np.pi * np.arange(days) / 365.25) + rng.normal(0, 1.8, size=days)
        economic_index = 100 + 0.03 * np.arange(days) + rng.normal(0, 1.1, size=days)
        price = 18 + sku_idx * 1.7 + rng.normal(0, 0.6, size=days) - promotion * 1.2
        holiday_flag = ((dates.month == 11) & (dates.day > 20)) | ((dates.month == 12) & (dates.day < 28))
        noise = rng.normal(0, 5 + sku_idx, size=days)
        demand = (
            base
            + trend
            + weekly
            + yearly
            + promotion * (10 + sku_idx * 1.8)
            - 1.1 * price
            + 0.18 * economic_index
            - 0.35 * np.abs(temp - 24)
            + holiday_flag.astype(int) * (14 + sku_idx)
            + noise
        )
        demand = np.maximum(np.round(demand), 0)

        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "sku": f"SKU-{sku_idx:03d}",
                    "category": "Core" if sku_idx <= 3 else "Seasonal",
                    "region": ["North", "South", "East", "West", "Central", "Online"][sku_idx - 1],
                    "sales": int(demand[i]),
                    "price": round(float(price[i]), 2),
                    "promotion": int(promotion[i]),
                    "temperature": round(float(temp[i]), 2),
                    "economic_index": round(float(economic_index[i]), 2),
                    "holiday_flag": int(holiday_flag[i]),
                }
            )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "sample_demand.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_dataset()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
