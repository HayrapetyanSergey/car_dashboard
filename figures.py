import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_BG = "#0b141a"


def _base_dark_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        margin=dict(l=15, r=15, t=75, b=55),
        font=dict(family="Arial", size=12),
        title={"x": 0.5, "xanchor": "center"},
    )
    return fig


# =========================
# 0) GLOBAL TREEMAP (ALL MAKES)
# =========================
def global_make_model_treemap(
    df: pd.DataFrame,
    top_makes: int = 22,
    top_models_per_make: int = 14,
):
    """
    Treemap (ALL cars):
      - level 1: make
      - level 2: model
    Colorful like the example: each make gets a discrete color, models inherit it.
    To keep it readable, we limit to:
      - Top N makes by total listings
      - Top K models within each make
    """
    if not {"make", "model"}.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(title="Missing columns: make/model")
        return _base_dark_layout(fig)

    d = df.dropna(subset=["make", "model"]).copy()
    d["make"] = d["make"].astype(str).str.strip()
    d["model"] = d["model"].astype(str).str.strip()
    d = d[(d["make"] != "") & (d["model"] != "")]
    if d.empty:
        fig = go.Figure()
        fig.update_layout(title="No make/model data")
        return _base_dark_layout(fig)

    # Top makes
    make_counts = d["make"].value_counts().head(top_makes)
    top_make_list = make_counts.index.tolist()
    d = d[d["make"].isin(top_make_list)]

    # Top models per make
    rows = []
    for mk in top_make_list:
        dm = d[d["make"] == mk]
        mcounts = dm["model"].value_counts().head(top_models_per_make)
        tmp = pd.DataFrame({"make": mk, "model": mcounts.index, "count": mcounts.values})
        rows.append(tmp)

    plot_df = pd.concat(rows, ignore_index=True)
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Not enough data for treemap after filtering")
        return _base_dark_layout(fig)

    # Discrete colors per make (colorful)
    palette = (
        px.colors.qualitative.Set3
        + px.colors.qualitative.Pastel
        + px.colors.qualitative.Safe
        + px.colors.qualitative.Vivid
        + px.colors.qualitative.G10
    )
    color_map = {mk: palette[i % len(palette)] for i, mk in enumerate(top_make_list)}

    fig = px.treemap(
        plot_df,
        path=["make", "model"],
        values="count",
        color="make",
        color_discrete_map=color_map,
        title="Make → Model treemap (All cars)",
    )

    fig.update_traces(
        texttemplate="%{label}<br>%{value}",
        hovertemplate="%{label}<br>Listings: %{value}<extra></extra>",
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.15)")),
    )
    # tighter margins for full-width vibe
    fig.update_layout(margin=dict(l=6, r=6, t=70, b=10))
    return _base_dark_layout(fig)


# =========================
# 1) Top 20 Models (BAR)
# =========================
def top_models_bar(df: pd.DataFrame, make_value: str | None, n: int = 20):
    if not make_value:
        fig = go.Figure()
        fig.update_layout(title="Select a make to see Top 20 models")
        return _base_dark_layout(fig)

    if "make" not in df.columns or "model" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Missing columns: make/model")
        return _base_dark_layout(fig)

    d = df.dropna(subset=["make", "model"]).copy()
    d["make"] = d["make"].astype(str).str.strip()
    d["model"] = d["model"].astype(str).str.strip()
    d = d[d["make"] == make_value]

    if d.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data for make: {make_value}")
        return _base_dark_layout(fig)

    top = (
        d.groupby("model", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(n)
    )

    fig = px.bar(
        top,
        x="model",
        y="count",
        text="count",
        title=f"Top {n} Models for {make_value} (Listings count)",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title="Listings", range=[0, int(top["count"].max()) * 1.15])
    fig.update_xaxes(title="Model", tickangle=30)
    return _base_dark_layout(fig)


# =========================
# 2) Price trends (ONLY affected by year slider)
# =========================
def price_trends(df: pd.DataFrame, make_value: str | None, model_value: str | None, year_range=None):
    if not make_value:
        fig = go.Figure()
        fig.update_layout(title="Select a make (and choose a model) to see price trends")
        return _base_dark_layout(fig)

    d = df.dropna(subset=["make", "year", "price"]).copy()
    d["make"] = d["make"].astype(str).str.strip()
    d = d[d["make"] == make_value]

    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d = d.dropna(subset=["year", "price"])

    if year_range:
        y0, y1 = year_range
        d = d[(d["year"] >= y0) & (d["year"] <= y1)]

    if d.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No price/year data for {make_value} in selected years")
        return _base_dark_layout(fig)

    make_ts = d.groupby("year", as_index=False)["price"].mean()
    make_ts["series"] = f"{make_value} (avg)"

    frames = [make_ts]
    if model_value:
        dm = d.dropna(subset=["model"]).copy()
        dm["model"] = dm["model"].astype(str).str.strip()
        dm = dm[dm["model"] == model_value]
        if not dm.empty:
            model_ts = dm.groupby("year", as_index=False)["price"].mean()
            model_ts["series"] = f"{make_value} {model_value} (avg)"
            frames.append(model_ts)

    plot_df = pd.concat(frames, ignore_index=True)

    fig = px.line(
        plot_df,
        x="year",
        y="price",
        color="series",
        markers=True,
        title="Average Price Over Time (Make vs Selected Model)",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True, title="Average price")
    fig.update_xaxes(title="Year")
    return _base_dark_layout(fig)


# =========================
# 3) Gauge (NOT affected by year slider)
# =========================
def model_price_gauge(df: pd.DataFrame, make_value: str | None, model_value: str | None):
    if not make_value:
        fig = go.Figure()
        fig.update_layout(title="Select a make to see price gauge")
        return _base_dark_layout(fig)

    d = df.dropna(subset=["make", "price"]).copy()
    d["make"] = d["make"].astype(str).str.strip()
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d = d.dropna(subset=["price"])
    d = d[d["make"] == make_value]

    title_suffix = make_value
    if model_value and "model" in d.columns:
        d = d.dropna(subset=["model"]).copy()
        d["model"] = d["model"].astype(str).str.strip()
        d = d[d["model"] == model_value]
        title_suffix = f"{make_value} {model_value}"

    if d.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No price data for {title_suffix}")
        return _base_dark_layout(fig)

    pmin = float(d["price"].min())
    pavg = float(d["price"].mean())
    pmax = float(d["price"].max())
    axis_max = max(pmax, 1.0) * 1.05

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pavg,
            number={"prefix": "$", "valueformat": ".2s"},
            delta={"reference": pmin, "valueformat": ".2s", "prefix": "+$"},
            title={"text": f"Avg / Min / Max Price — {title_suffix}"},
            gauge={
                "axis": {"range": [0, axis_max], "tickformat": ".2s"},
                "bar": {"thickness": 0.35},
                "threshold": {"line": {"color": "red", "width": 4}, "value": pmax},
                "steps": [
                    {"range": [0, pmin], "color": "rgba(255,255,255,0.08)"},
                    {"range": [pmin, pavg], "color": "rgba(99,110,250,0.35)"},
                    {"range": [pavg, pmax], "color": "rgba(99,110,250,0.65)"},
                ],
            },
        )
    )
    return _base_dark_layout(fig)


# =========================
# 4) Heatmap: Color × Body Type (NOT affected by year slider)
# =========================
def color_bodytype_heatmap(df: pd.DataFrame, make_value: str | None, model_value: str | None,
                          mode: str = "count", top_colors: int = 12, top_body_types: int = 8):
    if not make_value:
        fig = go.Figure()
        fig.update_layout(title="Select a make to see Color × Body Type heatmap")
        return _base_dark_layout(fig)

    needed = {"make", "color", "body_type"}
    if not needed.issubset(set(df.columns)):
        fig = go.Figure()
        fig.update_layout(title="Missing columns (need make, color, body_type)")
        return _base_dark_layout(fig)

    d = df.dropna(subset=["make", "color", "body_type"]).copy()
    d["make"] = d["make"].astype(str).str.strip()
    d["color"] = d["color"].astype(str).str.strip()
    d["body_type"] = d["body_type"].astype(str).str.strip()
    d = d[d["make"] == make_value]

    title_suffix = make_value
    if model_value and "model" in d.columns:
        d = d.dropna(subset=["model"]).copy()
        d["model"] = d["model"].astype(str).str.strip()
        d = d[d["model"] == model_value]
        title_suffix = f"{make_value} {model_value}"

    if d.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No color/body_type data for {title_suffix}")
        return _base_dark_layout(fig)

    top_c = d["color"].value_counts().head(top_colors).index
    top_b = d["body_type"].value_counts().head(top_body_types).index
    d = d[d["color"].isin(top_c) & d["body_type"].isin(top_b)]

    if d.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Not enough data after top-N filtering for {title_suffix}")
        return _base_dark_layout(fig)

    pivot = pd.pivot_table(
        d, index="body_type", columns="color", values="make", aggfunc="size", fill_value=0
    )

    body_order = d["body_type"].value_counts().index.tolist()
    color_order = d["color"].value_counts().index.tolist()
    pivot = pivot.reindex(index=body_order, columns=color_order)

    z = pivot.to_numpy(dtype=float)

    if mode == "percent":
        total = z.sum()
        z = (z / total * 100.0) if total > 0 else z
        text = np.where(z > 0, np.round(z, 1).astype(str) + "%", "")
        title = f"Color × Body Type (Share %) — {title_suffix}"
        colorbar_title = "%"
    else:
        text = np.where(z > 0, z.astype(int).astype(str), "")
        title = f"Color × Body Type (Count) — {title_suffix}"
        colorbar_title = "Count"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=text,
            texttemplate="%{text}",
            hovertemplate="Body: %{y}<br>Color: %{x}<br>Value: %{z}<extra></extra>",
            colorbar={"title": colorbar_title},
        )
    )

    fig.update_layout(title=title)
    fig.update_xaxes(title="Color", tickangle=25)
    fig.update_yaxes(title="Body Type")
    return _base_dark_layout(fig)


# =========================
# 5) Annual mileage intensity (NOT affected by year slider)
# =========================
def annual_mileage_intensity_bar(df: pd.DataFrame, make_value: str | None, model_value: str | None, top_n: int = 15):
    if not make_value:
        fig = go.Figure()
        fig.update_layout(title="Select a make to see annual mileage intensity")
        return _base_dark_layout(fig)

    required = {"make", "model", "mileage", "year"}
    if not required.issubset(set(df.columns)):
        fig = go.Figure()
        fig.update_layout(title="Missing columns (need make, model, mileage, year)")
        return _base_dark_layout(fig)

    d = df.dropna(subset=["make", "model", "mileage", "year"]).copy()
    d["make"] = d["make"].astype(str).str.strip()
    d["model"] = d["model"].astype(str).str.strip()
    d = d[d["make"] == make_value]

    d["mileage"] = pd.to_numeric(d["mileage"], errors="coerce")
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d.dropna(subset=["mileage", "year"])
    if d.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No valid mileage/year values for {make_value}")
        return _base_dark_layout(fig)

    current_year = 2026
    age = (current_year - d["year"] + 1).clip(lower=1)
    d["annual_km"] = d["mileage"] / age
    d = d[(d["annual_km"] >= 0) & (d["annual_km"] <= 200000)]
    if d.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No valid annual_km after filtering for {make_value}")
        return _base_dark_layout(fig)

    agg = (
        d.groupby("model", as_index=False)
        .agg(annual_km=("annual_km", "median"), n=("annual_km", "size"))
        .sort_values("annual_km", ascending=False)
        .head(top_n)
    )

    default_color = "rgba(99,110,250,0.85)"
    highlight_color = "rgba(255,180,0,0.95)"
    colors = [default_color] * len(agg)
    if model_value and model_value in set(agg["model"]):
        idx = agg.index[agg["model"] == model_value][0]
        colors[list(agg.index).index(idx)] = highlight_color

    agg["label"] = agg.apply(lambda r: f'{int(r["annual_km"]):,} km/yr (n={int(r["n"])})', axis=1)

    fig = go.Figure(
        data=[
            go.Bar(
                x=agg["annual_km"],
                y=agg["model"],
                orientation="h",
                text=agg["label"],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=12, color="white"),
                marker_color=colors,
                hovertemplate="Model: %{y}<br>Annual km: %{x:,.0f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(title=f"Annual Mileage Intensity (km/year) — Top {top_n} models ({make_value})")
    fig.update_xaxes(title="Median annual mileage (km/year)")
    fig.update_yaxes(title="Model", autorange="reversed")
    return _base_dark_layout(fig)


# =========================
# 6) Specs distribution (full width)
# =========================
def specs_distribution_bar(df: pd.DataFrame, make_value: str | None, model_value: str | None, top_n_each: int = 10):
    if not make_value:
        fig = go.Figure()
        fig.update_layout(title="Select a make to see specs distribution")
        return _base_dark_layout(fig)

    needed = {"make", "transmission", "engine_type", "drive_type", "interior_material"}
    if not needed.issubset(set(df.columns)):
        fig = go.Figure()
        fig.update_layout(title="Missing columns for specs distribution")
        return _base_dark_layout(fig)

    d = df.copy()
    d["make"] = d["make"].astype(str).str.strip()
    d = d[d["make"] == make_value]

    title_suffix = make_value
    if model_value and "model" in d.columns:
        d = d.dropna(subset=["model"]).copy()
        d["model"] = d["model"].astype(str).str.strip()
        d = d[d["model"] == model_value]
        title_suffix = f"{make_value} {model_value}"

    if d.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data for {title_suffix}")
        return _base_dark_layout(fig)

    def _vc(col: str, group_label: str) -> pd.DataFrame:
        s = d[col].dropna().astype(str).str.strip()
        s = s[~s.isin(["nan", "None", ""])]
        if s.empty:
            return pd.DataFrame(columns=["group", "value", "count"])
        vc = s.value_counts().head(top_n_each)
        out = vc.reset_index()
        out.columns = ["value", "count"]
        out["group"] = group_label
        return out[["group", "value", "count"]]

    parts = [
        _vc("transmission", "Transmission"),
        _vc("engine_type", "Engine type"),
        _vc("drive_type", "Drive type"),
        _vc("interior_material", "Interior material"),
    ]
    plot_df = pd.concat(parts, ignore_index=True)
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No valid grouped data for {title_suffix}")
        return _base_dark_layout(fig)

    plot_df["label"] = plot_df["group"] + "<br>" + plot_df["value"]

    group_order = ["Transmission", "Engine type", "Drive type", "Interior material"]
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=group_order, ordered=True)
    plot_df = plot_df.sort_values(["group", "count"], ascending=[True, False])

    fig = px.bar(
        plot_df,
        x="label",
        y="count",
        color="group",
        text="count",
        title=f"Specs distribution (Top {top_n_each} per group) — {title_suffix}",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=5, r=5, t=85, b=110), bargap=0.25, legend_title_text="Group")
    fig.update_xaxes(title="", tickangle=0, showgrid=False)
    fig.update_yaxes(title="Listings count", rangemode="tozero")
    return _base_dark_layout(fig)
