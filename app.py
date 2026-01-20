import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update

from figures import (
    global_make_model_treemap,
    top_models_bar,
    price_trends,
    model_price_gauge,
    color_bodytype_heatmap,
    annual_mileage_intensity_bar,
    specs_distribution_bar,
)

DATA_PATH = "/home/sergey/Desktop/car_dashboard/car_data.csv"
df = pd.read_csv(DATA_PATH)

# basic cleanup
for c in ["make", "model", "color", "body_type", "transmission", "engine_type", "drive_type", "interior_material"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

def make_options(_df):
    vals = sorted([x for x in _df["make"].dropna().unique() if str(x).strip() not in ["", "nan", "None"]])
    return [{"label": v, "value": v} for v in vals]

def model_options_for_make(_df, make_value):
    if not make_value:
        return []
    d = _df[_df["make"] == make_value].dropna(subset=["model"])
    vals = sorted([x for x in d["model"].unique() if str(x).strip() not in ["", "nan", "None"]])
    return [{"label": v, "value": v} for v in vals]

# years for slider
if "year" in df.columns:
    yy = pd.to_numeric(df["year"], errors="coerce").dropna()
    ymin = int(yy.min()) if not yy.empty else 1950
    ymax = int(yy.max()) if not yy.empty else 2023
else:
    ymin, ymax = 1950, 2023

app = Dash(__name__)
app.title = "Car Dashboard"

APP_BG = "#0b141a"
CARD_BG = "rgba(255,255,255,0.02)"
CARD_BORDER = "1px solid rgba(255,255,255,0.08)"

app.layout = html.Div(
    style={
        "backgroundColor": APP_BG,
        "minHeight": "100vh",
        "padding": "18px",
        "color": "white",
        "fontFamily": "Arial",
    },
    children=[
        html.H1("Car Dashboard", style={"textAlign": "center", "marginBottom": "6px"}),
        html.Div(
            "Top treemap shows ALL makes → models. Then pick make and model for the charts.",
            style={"textAlign": "center", "opacity": 0.8, "marginBottom": "10px"},
        ),

        # 0) Treemap
        html.Div(
            style={
                "border": CARD_BORDER,
                "background": CARD_BG,
                "borderRadius": "14px",
                "padding": "12px",
                "marginBottom": "14px",
            },
            children=[
                dcc.Graph(
                    id="global_treemap_fig",
                    config={"displayModeBar": False, "responsive": True},
                    style={"width": "100%", "height": "460px"},
                )
            ],
        ),

        # ✅ Make dropdown (NO default selection)
        html.Div(
            style={"display": "flex", "justifyContent": "center", "marginBottom": "12px"},
            children=[
                html.Div(
                    style={"width": "420px"},
                    children=[
                        html.Div("Make", style={"textAlign": "center", "marginBottom": "6px", "fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="make_dd",
                            options=make_options(df),
                            value=None,                       # ✅ IMPORTANT: no default selected
                            placeholder="Select make…",       # ✅
                            clearable=True,
                            className="dark-dd",              # ✅ CSS hook
                            style={
                                "textAlign": "center",   # centers typed/selected text (works in most cases)
                            },
                        ),
                    ],
                )
            ],
        ),

        # 1) Top20 bar
        dcc.Graph(
            id="top_models_fig",
            config={"displayModeBar": False, "responsive": True},
            style={"width": "100%", "height": "460px"},
        ),
# ✅ Model selector (centered, between 2nd and 3rd charts)
html.Div(
    style={
        "display": "flex",
        "justifyContent": "center",
        "marginTop": "18px",
        "marginBottom": "12px",
    },
    children=[
        html.Div(
            style={"width": "420px"},
            children=[
                html.Div(
                    "Model (optional)",
                    style={
                        "textAlign": "center",
                        "fontWeight": "bold",
                        "marginBottom": "6px",
                        "color": "white",
                    },
                ),
                dcc.Dropdown(
                    id="model_dd",
                    options=[],
                    value=None,
                    placeholder="Select model (or click a bar above)...",
                    clearable=True,
                    className="dark-dd",
                    style={
                            "textAlign": "center",   # centers typed/selected text (works in most cases)
                        },
                ),
            ],
        )
    ],
),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px", "marginTop": "14px"},
            children=[
                # Left card
                html.Div(
                    style={"border": CARD_BORDER, "background": CARD_BG, "borderRadius": "14px", "padding": "12px"},
                    children=[
                        html.Div("Year range (affects only trends)", style={"textAlign": "center", "marginTop": "8px", "opacity": 0.85}),
                        dcc.RangeSlider(
                            id="year_slider",
                            min=ymin,
                            max=ymax,
                            value=[ymin, ymax],
                            step=1,
                            marks={ymin: str(ymin), ymax: str(ymax)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        dcc.Graph(
                            id="trend_fig",
                            config={"displayModeBar": False, "responsive": True},
                            style={"width": "100%", "height": "420px"},
                        ),
                    ],
                ),

                # Right card
                html.Div(
                    style={"border": CARD_BORDER, "background": CARD_BG, "borderRadius": "14px", "padding": "12px"},
                    children=[
                        dcc.Graph(
                            id="gauge_fig",
                            config={"displayModeBar": False, "responsive": True},
                            style={"width": "100%", "height": "520px"},
                        )
                    ],
                ),
            ],
        ),

        # Row: heatmap + mileage
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px", "marginTop": "14px"},
            children=[
                html.Div(
                    style={"border": CARD_BORDER, "background": CARD_BG, "borderRadius": "14px", "padding": "12px"},
                    children=[dcc.Graph(id="heatmap_fig", config={"displayModeBar": False, "responsive": True},
                                        style={"width": "100%", "height": "430px"})],
                ),
                html.Div(
                    style={"border": CARD_BORDER, "background": CARD_BG, "borderRadius": "14px", "padding": "12px"},
                    children=[dcc.Graph(id="mileage_fig", config={"displayModeBar": False, "responsive": True},
                                        style={"width": "100%", "height": "430px"})],
                ),
            ],
        ),

        # 6 full width
        html.Div(
            style={"marginTop": "16px", "marginLeft": "-18px", "marginRight": "-18px"},
            children=[
                dcc.Graph(
                    id="specs_dist_fig",
                    config={"displayModeBar": False, "responsive": True},
                    style={"width": "100%", "height": "620px"},
                )
            ],
        ),

        dcc.Store(id="selected_model_store", data=None),
    ],
)


# -----------------------
# Callbacks
# -----------------------

@app.callback(
    Output("global_treemap_fig", "figure"),
    Input("make_dd", "value"),
)
def update_global_treemap(_):
    return global_make_model_treemap(df, top_makes=22, top_models_per_make=14)


@app.callback(
    Output("top_models_fig", "figure"),
    Input("make_dd", "value"),
)
def update_top_models(make_value):
    return top_models_bar(df, make_value, n=20)


@app.callback(
    Output("model_dd", "options"),
    Input("make_dd", "value"),
)
def update_model_options(make_value):
    return model_options_for_make(df, make_value)


# ✅ FIX circular dependency: ONE callback updates BOTH store and dropdown value
@app.callback(
    Output("selected_model_store", "data"),
    Output("model_dd", "value"),
    Input("model_dd", "value"),
    Input("top_models_fig", "clickData"),
    State("selected_model_store", "data"),
)
def sync_model_selection(model_dd_value, bar_click, current_store):
    ctx = callback_context
    if not ctx.triggered:
        return current_store, current_store

    trig = ctx.triggered[0]["prop_id"]

    if trig == "top_models_fig.clickData" and bar_click:
        x = bar_click["points"][0].get("x")
        if x:
            return x, x

    if trig == "model_dd.value":
        return model_dd_value, model_dd_value

    return current_store, current_store


@app.callback(
    Output("trend_fig", "figure"),
    Output("gauge_fig", "figure"),
    Output("heatmap_fig", "figure"),
    Output("mileage_fig", "figure"),
    Output("specs_dist_fig", "figure"),
    Input("make_dd", "value"),
    Input("selected_model_store", "data"),
    Input("year_slider", "value"),
)
def update_lower_charts(make_value, model_value, year_range):
    fig2 = price_trends(df, make_value, model_value, year_range=year_range)
    fig3 = model_price_gauge(df, make_value, model_value)
    fig4 = color_bodytype_heatmap(df, make_value, model_value, mode="count")
    fig5 = annual_mileage_intensity_bar(df, make_value, model_value, top_n=15)
    fig6 = specs_distribution_bar(df, make_value, model_value, top_n_each=10)
    return fig2, fig3, fig4, fig5, fig6


if __name__ == "__main__":
    app.run(debug=True)
