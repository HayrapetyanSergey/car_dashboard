import pandas as pd


def load_data(csv_path: str = "car_data.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Clean make/model/color/body_type text
    for col in ["make", "model", "color", "body_type"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})
                .str.replace(r"\s+", " ", regex=True)
            )

    # Numeric cleaning
    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    return df


def make_options(df: pd.DataFrame):
    if "make" not in df.columns:
        return []
    makes = sorted([m for m in df["make"].dropna().unique() if str(m).strip() != ""])
    return [{"label": m, "value": m} for m in makes]


def model_options_for_make(df: pd.DataFrame, make_value: str | None):
    if not make_value or "make" not in df.columns or "model" not in df.columns:
        return []
    d = df.dropna(subset=["make", "model"]).copy()
    d = d[d["make"] == make_value]
    models = sorted([m for m in d["model"].dropna().unique() if str(m).strip() != ""])
    return [{"label": m, "value": m} for m in models]


def year_bounds(df: pd.DataFrame, default_min=1950, default_max=2026):
    if "year" not in df.columns or df["year"].dropna().empty:
        return default_min, default_max
    ymin = int(df["year"].dropna().min())
    ymax = int(df["year"].dropna().max())
    return ymin, ymax
