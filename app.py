"""
Provides an Overview of the historical FDM print dataset and an interactive
Data Explorer with database-backed filtering and Plotly visualizations.
"""

import json
import logging
import pathlib
import sqlite3

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH    = pathlib.Path("data/printing_data.db")
MODEL_PATH = pathlib.Path("models/roughness_model.joblib")
LOGS_DIR   = pathlib.Path("logs")

# rPET is excluded from prediction because all rPET records in the source
# dataset had zero (invalid) surface-roughness measurements and were therefore
# excluded from model training. The model has no valid basis for rPET predictions.
EXCLUDED_MATERIALS = {"rPET"}


def get_logger() -> logging.Logger:
    """Return the application logger, creating a file handler exactly once.

    Checking for existing handlers prevents Streamlit from adding duplicate
    handlers each time it reruns the script.
    """
    logger = logging.getLogger("fdm_predictor")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOGS_DIR / "application.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


logger = get_logger()

st.set_page_config(page_title="3D Print Quality Predictor", layout="wide")
logger.info("Application started / page reloaded")

# Page title and subtitle
st.title("3D Print Quality Predictor")
st.caption(
    "Helping LayerWorks 3D analyze historical FDM printing data "
    "and make data-driven print-quality decisions."
)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data Explorer", "Quality Predictor", "Model Performance"],
)

# Database helpers

@st.cache_resource
def load_model():
    """Load the trained pipeline from disk once and cache it for the session."""
    if not MODEL_PATH.exists():
        logger.error("Model loading failed: file not found at %s", MODEL_PATH)
        return None
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
        return pipeline
    except Exception as exc:
        logger.error("Model loading failed: %s", exc)
        st.error("The model file could not be loaded. Check the application log for details.")
        return None


def get_roughness_range() -> dict:
    """Return the min and max valid roughness values to bound the quality target input."""
    conn = get_connection()
    row = pd.read_sql_query(
        """
        SELECT
            MIN(roughness_avg_um) AS rq_min,
            MAX(roughness_avg_um) AS rq_max
        FROM print_records
        WHERE roughness_valid = 1
        """,
        conn,
    ).iloc[0]
    conn.close()
    return row.to_dict()


def get_distinct_numeric_options(column: str) -> list:
    """Return the sorted list of distinct values for a numeric feature column,
    using only records where roughness_valid = 1 (i.e. valid training data)."""
    conn = get_connection()
    rows = pd.read_sql_query(
        f"SELECT DISTINCT {column} FROM print_records WHERE roughness_valid = 1 ORDER BY {column}",
        conn,
    )
    conn.close()
    return sorted(rows[column].tolist())


def get_connection():
    """Return a read-only connection to the SQLite database."""
    if not DB_PATH.exists():
        logger.error("Database access failed: file not found at %s", DB_PATH)
        st.error("Database not found. Check the application log for details.")
        st.stop()
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_filter_options(column: str) -> list[str]:
    """Return the sorted unique values for a categorical column."""
    conn = get_connection()
    rows = pd.read_sql_query(
        f"SELECT DISTINCT {column} FROM print_records ORDER BY {column}",
        conn,
    )
    conn.close()
    return rows[column].tolist()


def query_records(
    printer_type: str | None,
    material: str | None,
    infill_pattern: str | None,
) -> pd.DataFrame:
    """Return filtered print records from the database.

    Parameterized placeholders are used for every user-supplied filter value
    so that query construction never involves string concatenation of input.
    """
    conditions = []
    params: list = []

    if printer_type:
        conditions.append("printer_type = ?")
        params.append(printer_type)
    if material:
        conditions.append("material = ?")
        params.append(material)
    if infill_pattern:
        conditions.append("infill_pattern = ?")
        params.append(infill_pattern)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM print_records {where}"

    conn = get_connection()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


# Overview page

def render_overview():
    conn = get_connection()

    total       = pd.read_sql_query("SELECT COUNT(*) AS n FROM print_records", conn).iloc[0, 0]
    valid       = pd.read_sql_query("SELECT COUNT(*) AS n FROM print_records WHERE roughness_valid = 1", conn).iloc[0, 0]
    n_materials = pd.read_sql_query("SELECT COUNT(DISTINCT material) AS n FROM print_records", conn).iloc[0, 0]
    n_patterns  = pd.read_sql_query("SELECT COUNT(DISTINCT infill_pattern) AS n FROM print_records", conn).iloc[0, 0]
    conn.close()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total print records",        total)
    col2.metric("Valid roughness records",     valid)
    col3.metric("Materials",                   n_materials)
    col4.metric("Infill patterns",             n_patterns)

    st.markdown("---")
    st.markdown(
        """
        **About this application**

        LayerWorks 3D uses historical FDM print data to explore relationships
        between printing parameters (such as layer thickness, infill density,
        print speed, and material) and measured surface roughness. A trained
        machine-learning model will later allow users to estimate surface
        roughness for new printing configurations.
        """
    )

    # System Status
    st.markdown("---")
    st.markdown("### System Status")

    db_ok    = False
    model_ok = False
    metrics_ok = False
    db_count = None

    # Check database with a simple read-only count query
    try:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        db_count = _conn.execute("SELECT COUNT(*) FROM print_records").fetchone()[0]
        _conn.close()
        db_ok = True
    except Exception:
        pass

    # Check model file loads without error
    try:
        if MODEL_PATH.exists():
            joblib.load(MODEL_PATH)
            model_ok = True
    except Exception:
        pass

    # Check metrics JSON is readable
    metrics_data = None
    try:
        _mp = pathlib.Path("models/model_metrics.json")
        if _mp.exists():
            metrics_data = json.loads(_mp.read_text(encoding="utf-8"))
            metrics_ok = True
    except Exception:
        pass

    def _status_badge(ok: bool) -> str:
        return ":white_check_mark: Available" if ok else ":x: Unavailable"

    s1, s2, s3 = st.columns(3)
    s1.markdown(f"**SQLite database**  \n{_status_badge(db_ok)}")
    s2.markdown(f"**Trained model**  \n{_status_badge(model_ok)}")
    s3.markdown(f"**Metrics file**  \n{_status_badge(metrics_ok)}")

    all_ok = db_ok and model_ok and metrics_ok
    if all_ok:
        st.success("Overall status: **Operational**")
    else:
        st.error("Overall status: **Degraded** — one or more components are unavailable.")

    # Maintenance information
    st.markdown("---")
    st.markdown("### Maintenance Information")
    maint_model_type    = metrics_data["model_type"]             if metrics_data else "—"
    maint_records_used  = metrics_data["records_used_for_modeling"] if metrics_data else "—"
    maint_model_file    = MODEL_PATH.name
    maint_db_count      = db_count if db_count is not None else "—"

    mi1, mi2, mi3, mi4 = st.columns(4)
    mi1.metric("Model type",              maint_model_type)
    mi2.metric("Modeling records",         maint_records_used)
    mi3.metric("Model file",              maint_model_file)
    mi4.metric("Database records",         maint_db_count)


# Data Explorer page

def render_data_explorer():
    st.header("Data Explorer")

    # Sidebar filters — values loaded from the database, not hardcoded
    printer_types   = ["All"] + load_filter_options("printer_type")
    materials       = ["All"] + load_filter_options("material")
    infill_patterns = ["All"] + load_filter_options("infill_pattern")

    st.sidebar.markdown("### Filters")
    sel_printer  = st.sidebar.selectbox("Printer type",    printer_types)
    sel_material = st.sidebar.selectbox("Material",        materials)
    sel_pattern  = st.sidebar.selectbox("Infill pattern",  infill_patterns)

    # Translate "All" to None so query_records skips those conditions
    pt  = None if sel_printer  == "All" else sel_printer
    mat = None if sel_material == "All" else sel_material
    pat = None if sel_pattern  == "All" else sel_pattern

    df = query_records(pt, mat, pat)

    if df.empty:
        st.warning("No records match the selected filters.")
        return

    # Summary metrics
    valid_df = df[df["roughness_valid"] == 1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matching records",         len(df))

    if valid_df.empty:
        col2.metric("Avg roughness (µm)", "—")
        col3.metric("Min roughness (µm)", "—")
        col4.metric("Max roughness (µm)", "—")
        st.info("No valid roughness measurements exist for the selected filters.")
    else:
        col2.metric("Avg roughness (µm)", f"{valid_df['roughness_avg_um'].mean():.3f}")
        col3.metric("Min roughness (µm)", f"{valid_df['roughness_avg_um'].min():.3f}")
        col4.metric("Max roughness (µm)", f"{valid_df['roughness_avg_um'].max():.3f}")

    # Interactive data table
    st.subheader("Filtered records")
    st.dataframe(df, use_container_width=True)

    if valid_df.empty:
        return

    st.markdown("---")

    # Chart 1 — Bar chart: average roughness by material
    avg_by_material = (
        valid_df.groupby("material", as_index=False)["roughness_avg_um"]
        .mean()
        .sort_values("roughness_avg_um", ascending=False)
    )
    fig1 = px.bar(
        avg_by_material,
        x="material",
        y="roughness_avg_um",
        title="Average Surface Roughness by Material",
        labels={"material": "Material", "roughness_avg_um": "Avg Roughness (µm)"},
        color="material",
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 — Scatter: print speed vs roughness, coloured by material
    fig2 = px.scatter(
        valid_df,
        x="speed_mm_s",
        y="roughness_avg_um",
        color="material",
        title="Print Speed vs. Surface Roughness",
        labels={
            "speed_mm_s":       "Print Speed (mm/s)",
            "roughness_avg_um": "Surface Roughness (µm)",
            "material":         "Material",
        },
        hover_data=["printer_type", "infill_pattern", "layer_thickness_mm"],
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 — Box plot: roughness distribution by layer thickness
    fig3 = px.box(
        valid_df,
        x="layer_thickness_mm",
        y="roughness_avg_um",
        title="Surface Roughness Distribution by Layer Thickness",
        labels={
            "layer_thickness_mm": "Layer Thickness (mm)",
            "roughness_avg_um":   "Surface Roughness (µm)",
        },
    )
    st.plotly_chart(fig3, use_container_width=True)


# Quality Predictor page

def render_quality_predictor():
    st.header("Surface Roughness Predictor")
    st.markdown(
        "Enter an FDM printing configuration to estimate the expected surface "
        "roughness. Lower roughness values represent a smoother measured surface "
        "in the source dataset."
    )

    model = load_model()
    if model is None:
        st.error(
            f"Model file not found or could not be loaded: `{MODEL_PATH}`. "
            "Run `src/train_model.py` to generate it."
        )
        return

    ranges = get_roughness_range()

    # Categorical options from the database, excluding rPET
    all_printer_types = load_filter_options("printer_type")
    conn = get_connection()
    all_materials = pd.read_sql_query(
        """
        SELECT DISTINCT material FROM print_records
        WHERE roughness_valid = 1
        ORDER BY material
        """,
        conn,
    )["material"].tolist()
    conn.close()
    all_patterns = load_filter_options("infill_pattern")

    st.markdown("### Configuration")
    col_left, col_right = st.columns(2)

    with col_left:
        sel_printer = st.selectbox("Printer type", all_printer_types)
        sel_material = st.selectbox(
            "Material",
            all_materials,
            help=(
                "rPET is unavailable: valid surface-roughness measurements "
                "were not available for that material in the source dataset, "
                "so it was excluded from model training."
            ),
        )
        sel_pattern = st.selectbox("Infill pattern", all_patterns)

    with col_right:
        lt_vals = get_distinct_numeric_options("layer_thickness_mm")
        sel_layer = st.select_slider(
            "Layer thickness (mm)",
            options=lt_vals,
            value=lt_vals[len(lt_vals) // 2],
        )
        id_vals = get_distinct_numeric_options("infill_density_pct")
        sel_infill = st.select_slider(
            "Infill density (%)",
            options=id_vals,
            value=id_vals[len(id_vals) // 2],
        )
        sp_vals = get_distinct_numeric_options("speed_mm_s")
        sel_speed = st.select_slider(
            "Print speed (mm/s)",
            options=sp_vals,
            value=sp_vals[len(sp_vals) // 2],
        )

    st.markdown("### Quality target")
    rq_mid = round((ranges["rq_min"] + ranges["rq_max"]) / 2, 2)
    target_roughness = st.number_input(
        "Maximum acceptable surface roughness (µm)",
        min_value=round(float(ranges["rq_min"]), 4),
        max_value=round(float(ranges["rq_max"]), 4),
        value=rq_mid,
        step=0.1,
        help="Decision-support target only. This value is not an input to the model.",
    )

    if st.button("Predict Surface Roughness", type="primary"):
        # Guard: rPET must never reach the model
        if sel_material in EXCLUDED_MATERIALS:
            st.error(
                "rPET cannot be used for prediction because valid surface-roughness "
                "measurements were not available for that material in the source dataset."
            )
            logger.warning("Prediction blocked: rPET selected by user")
            return

        config_summary = (
            f"printer={sel_printer}, material={sel_material}, "
            f"layer={sel_layer}mm, infill={sel_infill}%, "
            f"pattern={sel_pattern}, speed={sel_speed}mm/s"
        )
        logger.info("Prediction request: %s", config_summary)

        try:
            # Build the one-row input DataFrame expected by the pipeline
            input_df = pd.DataFrame([{
                "printer_type":       sel_printer,
                "material":           sel_material,
                "layer_thickness_mm": float(sel_layer),
                "infill_density_pct": int(sel_infill),
                "infill_pattern":     sel_pattern,
                "speed_mm_s":         int(sel_speed),
            }])

            prediction = float(model.predict(input_df)[0])
            logger.info(
                "Prediction success: %s → roughness=%.4f µm (target=%.4f µm)",
                config_summary, prediction, target_roughness,
            )
        except Exception as exc:
            logger.error("Prediction failed: %s | config: %s", exc, config_summary)
            st.error("An error occurred while generating the prediction. "
                     "Check the application log for details.")
            return

        st.markdown("---")
        st.markdown("### Prediction result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted roughness (µm)",  f"{prediction:.2f}")
        col2.metric("Target roughness (µm)",      f"{target_roughness:.2f}")
        diff = prediction - target_roughness
        col3.metric("Difference (µm)", f"{diff:+.2f}")

        if prediction <= target_roughness:
            st.success("Predicted result meets the selected surface-roughness target.")
        else:
            st.warning("Predicted result does not meet the selected surface-roughness target.")

        st.info(
            "This result is a model estimate based on historical experimental data "
            "from the source dataset. It should be used to support, rather than "
            "replace, engineering judgment."
        )

        # Historical context for the selected material
        st.markdown("### Historical context — " + sel_material)
        conn = get_connection()
        hist = pd.read_sql_query(
            """
            SELECT
                AVG(roughness_avg_um)   AS avg_roughness,
                MIN(roughness_avg_um)   AS min_roughness,
                MAX(roughness_avg_um)   AS max_roughness,
                COUNT(*)                AS n_samples
            FROM print_records
            WHERE roughness_valid = 1
              AND material = ?
            """,
            conn,
            params=(sel_material,),
        ).iloc[0]
        conn.close()

        if hist["n_samples"] == 0:
            st.info("No valid historical records found for this material.")
        else:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Avg historical roughness (µm)", f"{hist['avg_roughness']:.3f}")
            h2.metric("Min historical roughness (µm)", f"{hist['min_roughness']:.3f}")
            h3.metric("Max historical roughness (µm)", f"{hist['max_roughness']:.3f}")
            h4.metric("Valid historical samples",       int(hist["n_samples"]))


# Model Performance page

METRICS_PATH     = pathlib.Path("models/model_metrics.json")
PREDICTIONS_PATH = pathlib.Path("models/test_predictions.csv")


def render_model_performance():
    st.header("Model Performance")
    st.markdown(
        "This page evaluates the trained Random Forest regression model using "
        "held-out test data that was **not** used during model training."
    )

    # Load metrics JSON
    if not METRICS_PATH.exists():
        st.error(f"Metrics file not found: `{METRICS_PATH}`. Run `src/train_model.py` first.")
        return
    try:
        import json
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not read metrics file: {exc}")
        return

    # Load test predictions CSV
    if not PREDICTIONS_PATH.exists():
        st.error(f"Predictions file not found: `{PREDICTIONS_PATH}`. Run `src/evaluate_model.py` first.")
        return
    try:
        preds = pd.read_csv(PREDICTIONS_PATH)
    except Exception as exc:
        st.error(f"Could not read predictions file: {exc}")
        return

    # Model summary
    st.markdown("### Model summary")
    st.markdown(
        f"**Model type:** {metrics['model_type']}  \n"
        "The model uses six FDM process parameters — printer type, material, "
        "layer thickness, infill density, infill pattern, and print speed — "
        "to predict surface roughness (µm)."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records used",             metrics["records_used_for_modeling"])
    c2.metric("Training records",          metrics["training_records"])
    c3.metric("Test records",              metrics["testing_records"])
    c4.metric("Total cleaned records",     metrics["total_cleaned_records"])
    c5.metric("Excluded (zero roughness)", metrics["excluded_invalid_roughness"])

    st.markdown("---")

    # Main accuracy metrics
    st.markdown("### Holdout-test accuracy")
    rf = metrics["random_forest"]
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (µm)",  f"{rf['mae']:.4f}")
    m2.metric("RMSE (µm)", f"{rf['rmse']:.4f}")
    m3.metric("R²",         f"{rf['r2']:.4f}")
    st.markdown(
        "- **MAE** — average absolute difference between predicted and measured roughness.  \n"
        "- **RMSE** — similar to MAE but penalizes larger errors more heavily.  \n"
        "- **R²** — proportion of variation in surface roughness explained by the model "
        "(1.0 = perfect, 0.0 = no better than predicting the mean)."
    )

    st.markdown("---")

    # Baseline comparison
    st.markdown("### Baseline comparison")
    st.markdown(
        "The baseline model always predicts the training-set mean roughness, "
        "without learning any relationships between print settings and roughness. "
        "Any useful model must outperform this reference."
    )
    bl = metrics["baseline"]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Baseline MAE (µm)",  f"{bl['mae']:.4f}")
    b2.metric("RF MAE (µm)",         f"{rf['mae']:.4f}",  delta=f"{rf['mae']-bl['mae']:.4f}")
    b3.metric("Baseline RMSE (µm)", f"{bl['rmse']:.4f}")
    b4.metric("RF RMSE (µm)",        f"{rf['rmse']:.4f}", delta=f"{rf['rmse']-bl['rmse']:.4f}")

    bar_data = pd.DataFrame({
        "Metric":  ["MAE", "MAE", "RMSE", "RMSE"],
        "Model":   ["Baseline", "Random Forest", "Baseline", "Random Forest"],
        "Value":   [bl["mae"], rf["mae"], bl["rmse"], rf["rmse"]],
    })
    fig_bar = px.bar(
        bar_data,
        x="Metric", y="Value", color="Model",
        barmode="group",
        title="Baseline vs. Random Forest — MAE and RMSE (µm)",
        labels={"Value": "Error (µm)"},
        color_discrete_map={"Baseline": "#b0bec5", "Random Forest": "#1565c0"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Cross-validation
    st.markdown("### 5-fold cross-validation")
    st.markdown(
        "Cross-validation evaluates how consistently the model performs across "
        "different subsets of the training data, providing an additional check "
        "that the holdout-test results are not a statistical fluke."
    )
    cv = metrics["cross_validation"]
    v1, v2, v3 = st.columns(3)
    v1.metric("CV MAE (avg, µm)",  f"{cv['mae']:.4f}")
    v2.metric("CV RMSE (avg, µm)", f"{cv['rmse']:.4f}")
    v3.metric("CV R² (avg)",        f"{cv['r2']:.4f}")

    st.markdown("---")

    # Accuracy bands
    st.markdown("### Prediction accuracy bands")
    n = len(preds)
    bands = [0.5, 1.0, 1.5, 2.0]
    band_cols = st.columns(len(bands))
    for col, band in zip(band_cols, bands):
        within = int((preds["absolute_error_um"] <= band).sum())
        col.metric(f"Within ±{band} µm", f"{within}/{n}", f"{100*within/n:.1f}%")

    st.markdown("---")

    # Predicted vs Actual scatter
    st.markdown("### Predicted vs. Measured Surface Roughness")
    lo = float(min(preds["actual_roughness_um"].min(), preds["predicted_roughness_um"].min()))
    hi = float(max(preds["actual_roughness_um"].max(), preds["predicted_roughness_um"].max()))
    fig_scatter = px.scatter(
        preds,
        x="actual_roughness_um",
        y="predicted_roughness_um",
        hover_data=["material", "printer_type", "infill_pattern",
                    "layer_thickness_mm", "speed_mm_s", "absolute_error_um"],
        title="Predicted vs. Measured Surface Roughness",
        labels={
            "actual_roughness_um":    "Measured Roughness (µm)",
            "predicted_roughness_um": "Predicted Roughness (µm)",
        },
        opacity=0.7,
    )
    fig_scatter.add_shape(
        type="line", x0=lo, y0=lo, x1=hi, y1=hi,
        line=dict(color="crimson", width=1.5, dash="dash"),
    )
    fig_scatter.add_annotation(
        x=hi, y=hi, text="Perfect prediction",
        showarrow=False, xanchor="right", font=dict(color="crimson"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Residual histogram
    st.markdown("### Prediction Error Distribution")
    fig_hist = px.histogram(
        preds,
        x="residual_um",
        nbins=20,
        title="Prediction Error Distribution",
        labels={"residual_um": "Residual (Measured − Predicted) µm"},
        color_discrete_sequence=["#1565c0"],
    )
    fig_hist.add_vline(
        x=0, line_color="crimson", line_dash="dash",
        annotation_text="Zero error", annotation_position="top right",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Largest errors
    st.markdown("### Five largest prediction errors")
    worst_cols = [
        "material", "printer_type", "layer_thickness_mm", "infill_density_pct",
        "infill_pattern", "speed_mm_s",
        "actual_roughness_um", "predicted_roughness_um", "absolute_error_um",
    ]
    worst = preds.nlargest(5, "absolute_error_um")[worst_cols].reset_index(drop=True)
    st.dataframe(worst, use_container_width=True)

    st.markdown("---")

    # Limitations
    st.markdown("### Model limitations")
    st.markdown(
        "- Predictions are **estimates** based on a limited experimental dataset "
        "and should not be treated as guarantees of actual print quality.  \n"
        "- **17 records** with zero surface-roughness measurements were excluded "
        "from model development because zero roughness is not physically meaningful "
        "for FDM prints.  \n"
        "- **rPET** cannot be predicted because the source dataset did not contain "
        "valid roughness measurements for that material.  \n"
        "- The model is intended to **support engineering decisions** by providing "
        "data-driven roughness estimates. When exact surface quality is critical, "
        "physical testing of actual printed parts remains necessary."
    )


# Router

if page == "Overview":
    render_overview()
elif page == "Data Explorer":
    render_data_explorer()
elif page == "Quality Predictor":
    render_quality_predictor()
elif page == "Model Performance":
    render_model_performance()
else:
    st.info(f"**{page}** is not a recognised page.")
