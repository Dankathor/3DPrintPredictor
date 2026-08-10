"""
3D Print Quality Predictor
--------------------------
Streamlit application for exploring historical FDM print data and predicting
surface roughness from supported print settings.
"""

import json
import logging
import pathlib
import sqlite3
from contextlib import closing

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = pathlib.Path("data/printing_data.db")
MODEL_PATH = pathlib.Path("models/roughness_model.joblib")
METRICS_PATH = pathlib.Path("models/model_metrics.json")
PREDICTIONS_PATH = pathlib.Path("models/test_predictions.csv")
LOGS_DIR = pathlib.Path("logs")

# The source dataset contains rPET records with zero roughness values. Because
# the source documentation does not explain what those zero values represent,
# those records are excluded from model training and rPET is not offered as a
# prediction option.
EXCLUDED_MATERIALS = {"rPET"}

# Column names are never accepted directly from user input. These allowlists
# protect the few helper queries that need a dynamic SQL identifier.
CATEGORICAL_QUERY_COLUMNS = {"printer_type", "material", "infill_pattern"}
NUMERIC_QUERY_COLUMNS = {"layer_thickness_mm", "infill_density_pct", "speed_mm_s"}


def get_logger() -> logging.Logger:
    """Return the application logger, creating a file handler once."""
    logger = logging.getLogger("fdm_predictor")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOGS_DIR / "application.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


logger = get_logger()

st.set_page_config(page_title="3D Print Quality Predictor", layout="wide")

# Streamlit reruns the script after most interactions. Log one startup event per
# browser session instead of writing a new "started" event on every rerun.
if "startup_logged" not in st.session_state:
    logger.info("Application session started")
    st.session_state["startup_logged"] = True

st.title("3D Print Quality Predictor")
st.caption(
    "Analyze historical FDM print data and predict surface roughness from print settings."
)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data Explorer", "Quality Predictor", "Model Performance"],
)


# ---------------------------------------------------------------------------
# Database and model helpers
# ---------------------------------------------------------------------------

def _readonly_database_uri() -> str:
    """Return a SQLite URI that opens the database in read-only mode."""
    return f"{DB_PATH.resolve().as_uri()}?mode=ro"


def open_readonly_connection() -> sqlite3.Connection:
    """Open SQLite in read-only/query-only mode.

    This helper raises normal exceptions so it can also be used by status
    checks. User-facing database errors are handled by get_connection().
    """
    conn = sqlite3.connect(
        _readonly_database_uri(),
        uri=True,
        check_same_thread=False,
    )
    conn.execute("PRAGMA query_only = ON")
    return conn


def get_connection() -> sqlite3.Connection:
    """Return a read-only database connection or stop the current page safely."""
    if not DB_PATH.exists():
        logger.error("Database access failed: file not found at %s", DB_PATH)
        st.error("The database is unavailable. Check the application log for details.")
        st.stop()

    try:
        return open_readonly_connection()
    except Exception as exc:
        logger.error("Database access failed: %s", exc)
        st.error("The database could not be opened. Check the application log for details.")
        st.stop()


def _validated_column(column: str, allowed_columns: set[str]) -> str:
    """Return an allowlisted SQL column name or raise ValueError."""
    if column not in allowed_columns:
        raise ValueError(f"Unsupported query column: {column}")
    return column


@st.cache_resource
def _load_model_cached(model_path: str, modified_ns: int):
    """Load a model resource keyed by file path and modification time."""
    del modified_ns  # Used only as part of the Streamlit cache key.
    return joblib.load(model_path)


def load_model():
    """Load the trained pipeline and refresh the cache when the file changes."""
    if not MODEL_PATH.exists():
        logger.error("Model loading failed: file not found at %s", MODEL_PATH)
        return None

    try:
        modified_ns = MODEL_PATH.stat().st_mtime_ns
        pipeline = _load_model_cached(str(MODEL_PATH), modified_ns)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
        return pipeline
    except Exception as exc:
        logger.error("Model loading failed: %s", exc)
        return None


@st.cache_data(show_spinner=False)
def get_roughness_range() -> dict:
    """Return min/max positive roughness values for the decision target input."""
    with closing(get_connection()) as conn:
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
    return row.to_dict()


@st.cache_data(show_spinner=False)
def get_distinct_numeric_options(column: str) -> list:
    """Return supported numeric values from records used for modeling."""
    column = _validated_column(column, NUMERIC_QUERY_COLUMNS)

    with closing(get_connection()) as conn:
        rows = pd.read_sql_query(
            f"""
            SELECT DISTINCT {column}
            FROM print_records
            WHERE roughness_valid = 1
            ORDER BY {column}
            """,
            conn,
        )
    return rows[column].dropna().tolist()


@st.cache_data(show_spinner=False)
def load_filter_options(column: str) -> list[str]:
    """Return unique categorical values for Data Explorer filters."""
    column = _validated_column(column, CATEGORICAL_QUERY_COLUMNS)

    with closing(get_connection()) as conn:
        rows = pd.read_sql_query(
            f"SELECT DISTINCT {column} FROM print_records ORDER BY {column}",
            conn,
        )
    return rows[column].dropna().tolist()


@st.cache_data(show_spinner=False)
def load_prediction_options(column: str) -> list[str]:
    """Return categorical options represented in valid model-training data."""
    column = _validated_column(column, CATEGORICAL_QUERY_COLUMNS)

    with closing(get_connection()) as conn:
        rows = pd.read_sql_query(
            f"""
            SELECT DISTINCT {column}
            FROM print_records
            WHERE roughness_valid = 1
            ORDER BY {column}
            """,
            conn,
        )
    return rows[column].dropna().tolist()


def query_records(
    printer_type: str | None,
    material: str | None,
    infill_pattern: str | None,
) -> pd.DataFrame:
    """Return historical records matching optional user-selected filters."""
    conditions = []
    params: list[str] = []

    if printer_type:
        conditions.append("printer_type = ?")
        params.append(printer_type)
    if material:
        conditions.append("material = ?")
        params.append(material)
    if infill_pattern:
        conditions.append("infill_pattern = ?")
        params.append(infill_pattern)

    where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = "SELECT * FROM print_records" + where_clause

    with closing(get_connection()) as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def render_overview():
    with closing(get_connection()) as conn:
        total = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM print_records", conn
        ).iloc[0, 0]
        valid = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM print_records WHERE roughness_valid = 1", conn
        ).iloc[0, 0]
        n_materials = pd.read_sql_query(
            "SELECT COUNT(DISTINCT material) AS n FROM print_records", conn
        ).iloc[0, 0]
        n_patterns = pd.read_sql_query(
            "SELECT COUNT(DISTINCT infill_pattern) AS n FROM print_records", conn
        ).iloc[0, 0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total print records", total)
    col2.metric("Valid roughness records", valid)
    col3.metric("Materials", n_materials)
    col4.metric("Infill patterns", n_patterns)

    st.markdown("---")
    st.markdown(
        """
        **About this application**

        LayerWorks 3D uses this application to explore historical FDM print
        results and estimate surface roughness from selected print settings.
        The Data Explorer provides filtered records and interactive charts,
        while the Quality Predictor uses a trained Random Forest model to
        estimate roughness and compare it with a user-selected target.
        """
    )

    st.markdown("---")
    st.markdown("### System Status")

    db_ok = False
    model_ok = False
    metrics_ok = False
    db_count = None

    try:
        with closing(open_readonly_connection()) as conn:
            db_count = conn.execute(
                "SELECT COUNT(*) FROM print_records"
            ).fetchone()[0]
        db_ok = True
    except Exception as exc:
        logger.error("System status database check failed: %s", exc)

    try:
        if MODEL_PATH.exists():
            joblib.load(MODEL_PATH)
            model_ok = True
    except Exception as exc:
        logger.error("System status model check failed: %s", exc)

    metrics_data = None
    try:
        if METRICS_PATH.exists():
            metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            metrics_ok = True
    except Exception as exc:
        logger.error("System status metrics check failed: %s", exc)

    def status_badge(ok: bool) -> str:
        return ":white_check_mark: Available" if ok else ":x: Unavailable"

    s1, s2, s3 = st.columns(3)
    s1.markdown(f"**SQLite database**  \n{status_badge(db_ok)}")
    s2.markdown(f"**Trained model**  \n{status_badge(model_ok)}")
    s3.markdown(f"**Metrics file**  \n{status_badge(metrics_ok)}")

    if db_ok and model_ok and metrics_ok:
        st.success("Overall status: **Operational**")
    else:
        st.error("Overall status: **Degraded** — one or more components are unavailable.")

    st.markdown("---")
    st.markdown("### Maintenance Information")

    model_type = metrics_data["model_type"] if metrics_data else "—"
    records_used = (
        metrics_data["records_used_for_modeling"] if metrics_data else "—"
    )

    mi1, mi2, mi3, mi4 = st.columns(4)
    mi1.metric("Model type", model_type)
    mi2.metric("Modeling records", records_used)
    mi3.metric("Model file", MODEL_PATH.name)
    mi4.metric("Database records", db_count if db_count is not None else "—")


# ---------------------------------------------------------------------------
# Data Explorer
# ---------------------------------------------------------------------------

def render_data_explorer():
    st.header("Data Explorer")
    st.markdown(
        "Filter historical FDM print records and compare measured surface roughness."
    )

    printer_types = ["All"] + load_filter_options("printer_type")
    materials = ["All"] + load_filter_options("material")
    infill_patterns = ["All"] + load_filter_options("infill_pattern")

    st.sidebar.markdown("### Filters")
    sel_printer = st.sidebar.selectbox("Printer type", printer_types)
    sel_material = st.sidebar.selectbox("Material", materials)
    sel_pattern = st.sidebar.selectbox("Infill pattern", infill_patterns)

    printer = None if sel_printer == "All" else sel_printer
    material = None if sel_material == "All" else sel_material
    pattern = None if sel_pattern == "All" else sel_pattern

    df = query_records(printer, material, pattern)

    if df.empty:
        st.warning("No records match the selected filters.")
        return

    valid_df = df[df["roughness_valid"] == 1].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matching records", len(df))

    if valid_df.empty:
        col2.metric("Avg roughness (µm)", "—")
        col3.metric("Min roughness (µm)", "—")
        col4.metric("Max roughness (µm)", "—")
        st.info("No positive roughness measurements exist for the selected filters.")
    else:
        col2.metric("Avg roughness (µm)", f"{valid_df['roughness_avg_um'].mean():.3f}")
        col3.metric("Min roughness (µm)", f"{valid_df['roughness_avg_um'].min():.3f}")
        col4.metric("Max roughness (µm)", f"{valid_df['roughness_avg_um'].max():.3f}")

    st.subheader("Filtered records")

    display_columns = [
        "sample_no",
        "printer_type",
        "material",
        "layer_thickness_mm",
        "infill_density_pct",
        "infill_pattern",
        "speed_mm_s",
        "roughness_avg_um",
        "roughness_valid",
    ]
    display_names = {
        "sample_no": "Sample",
        "printer_type": "Printer Type",
        "material": "Material",
        "layer_thickness_mm": "Layer Thickness (mm)",
        "infill_density_pct": "Infill Density (%)",
        "infill_pattern": "Infill Pattern",
        "speed_mm_s": "Print Speed (mm/s)",
        "roughness_avg_um": "Surface Roughness (µm)",
        "roughness_valid": "Roughness Included",
    }
    display_df = df[display_columns].rename(columns=display_names)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("Show all dataset fields"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    if valid_df.empty:
        return

    st.markdown("---")

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
        labels={
            "material": "Material",
            "roughness_avg_um": "Avg Roughness (µm)",
        },
        color="material",
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        valid_df,
        x="speed_mm_s",
        y="roughness_avg_um",
        color="material",
        title="Print Speed vs. Surface Roughness",
        labels={
            "speed_mm_s": "Print Speed (mm/s)",
            "roughness_avg_um": "Surface Roughness (µm)",
            "material": "Material",
        },
        hover_data=["printer_type", "infill_pattern", "layer_thickness_mm"],
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(
        valid_df,
        x="layer_thickness_mm",
        y="roughness_avg_um",
        title="Surface Roughness Distribution by Layer Thickness",
        labels={
            "layer_thickness_mm": "Layer Thickness (mm)",
            "roughness_avg_um": "Surface Roughness (µm)",
        },
    )
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Quality Predictor
# ---------------------------------------------------------------------------

def render_quality_predictor():
    st.header("Surface Roughness Predictor")
    st.markdown(
        "Enter FDM print settings to estimate surface roughness. "
        "Lower values indicate a smoother measured surface."
    )

    model = load_model()
    if model is None:
        st.error(
            "The trained model is unavailable. Run `src/train_model.py` and "
            "check the application log if the problem continues."
        )
        return

    roughness_range = get_roughness_range()

    printer_types = load_prediction_options("printer_type")
    materials = [
        material
        for material in load_prediction_options("material")
        if material not in EXCLUDED_MATERIALS
    ]
    patterns = load_prediction_options("infill_pattern")

    layer_values = get_distinct_numeric_options("layer_thickness_mm")
    infill_values = get_distinct_numeric_options("infill_density_pct")
    speed_values = get_distinct_numeric_options("speed_mm_s")

    st.markdown("### Configuration")
    col_left, col_right = st.columns(2)

    with col_left:
        sel_printer = st.selectbox("Printer type", printer_types)
        sel_material = st.selectbox(
            "Material",
            materials,
            help=(
                "rPET is not available for prediction because its source records "
                "have zero roughness values whose meaning is not documented, so "
                "those records were excluded from model training."
            ),
        )
        sel_pattern = st.selectbox("Infill pattern", patterns)

    with col_right:
        sel_layer = st.select_slider(
            "Layer thickness (mm)",
            options=layer_values,
            value=layer_values[len(layer_values) // 2],
        )
        sel_infill = st.select_slider(
            "Infill density (%)",
            options=infill_values,
            value=infill_values[len(infill_values) // 2],
        )
        sel_speed = st.select_slider(
            "Print speed (mm/s)",
            options=speed_values,
            value=speed_values[len(speed_values) // 2],
        )

    st.markdown("### Surface roughness target")
    target_midpoint = round(
        (float(roughness_range["rq_min"]) + float(roughness_range["rq_max"])) / 2,
        2,
    )
    target_roughness = st.number_input(
        "Maximum acceptable surface roughness (µm)",
        min_value=round(float(roughness_range["rq_min"]), 4),
        max_value=round(float(roughness_range["rq_max"]), 4),
        value=target_midpoint,
        step=0.1,
        help="This target is used only for comparison and is not an input to the model.",
    )

    if st.button("Predict Surface Roughness", type="primary"):
        if sel_material in EXCLUDED_MATERIALS:
            st.error("This material is not supported by the trained model.")
            logger.warning("Prediction blocked for excluded material: %s", sel_material)
            return

        config_summary = (
            f"printer={sel_printer}, material={sel_material}, "
            f"layer={sel_layer}mm, infill={sel_infill}%, "
            f"pattern={sel_pattern}, speed={sel_speed}mm/s"
        )
        logger.info("Prediction request: %s", config_summary)

        try:
            input_df = pd.DataFrame(
                [
                    {
                        "printer_type": sel_printer,
                        "material": sel_material,
                        "layer_thickness_mm": float(sel_layer),
                        "infill_density_pct": float(sel_infill),
                        "infill_pattern": sel_pattern,
                        "speed_mm_s": float(sel_speed),
                    }
                ]
            )
            prediction = float(model.predict(input_df)[0])
            logger.info(
                "Prediction success: %s -> roughness=%.4f µm (target=%.4f µm)",
                config_summary,
                prediction,
                target_roughness,
            )
        except Exception as exc:
            logger.error("Prediction failed: %s | config: %s", exc, config_summary)
            st.error(
                "The prediction could not be generated. "
                "Check the application log for details."
            )
            return

        st.markdown("---")
        st.markdown("### Prediction result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted roughness (µm)", f"{prediction:.2f}")
        col2.metric("Target roughness (µm)", f"{target_roughness:.2f}")
        difference = prediction - target_roughness
        col3.metric("Difference (µm)", f"{difference:+.2f}")

        if prediction <= target_roughness:
            st.success("The predicted roughness meets the selected target.")
        else:
            st.warning("The predicted roughness does not meet the selected target.")

        st.info(
            "This prediction is an estimate based on the historical dataset. "
            "Use it as a planning aid, not as a replacement for physical testing."
        )

        st.markdown(f"### Historical context — {sel_material}")
        with closing(get_connection()) as conn:
            historical = pd.read_sql_query(
                """
                SELECT
                    AVG(roughness_avg_um) AS avg_roughness,
                    MIN(roughness_avg_um) AS min_roughness,
                    MAX(roughness_avg_um) AS max_roughness,
                    COUNT(*) AS n_samples
                FROM print_records
                WHERE roughness_valid = 1
                  AND material = ?
                """,
                conn,
                params=(sel_material,),
            ).iloc[0]

        if historical["n_samples"] == 0:
            st.info("No positive historical roughness records were found for this material.")
        else:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric(
                "Avg historical roughness (µm)",
                f"{historical['avg_roughness']:.3f}",
            )
            h2.metric(
                "Min historical roughness (µm)",
                f"{historical['min_roughness']:.3f}",
            )
            h3.metric(
                "Max historical roughness (µm)",
                f"{historical['max_roughness']:.3f}",
            )
            h4.metric("Historical samples", int(historical["n_samples"]))


# ---------------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------------

def render_model_performance():
    st.header("Model Performance")
    st.markdown(
        "These results are based on a held-out test set that was not used to train the model."
    )

    if not METRICS_PATH.exists():
        st.error(
            f"Metrics file not found: `{METRICS_PATH}`. Run `src/train_model.py` first."
        )
        return

    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Metrics file could not be read: %s", exc)
        st.error("The model metrics could not be loaded. Check the application log.")
        return

    if not PREDICTIONS_PATH.exists():
        st.error(
            f"Predictions file not found: `{PREDICTIONS_PATH}`. "
            "Run `src/evaluate_model.py` first."
        )
        return

    try:
        predictions = pd.read_csv(PREDICTIONS_PATH)
    except Exception as exc:
        logger.error("Prediction results could not be read: %s", exc)
        st.error("The prediction results could not be loaded. Check the application log.")
        return

    st.markdown("### Model summary")
    st.markdown(
        f"**Model type:** {metrics['model_type']}  \n"
        "The model uses six FDM process parameters — printer type, material, "
        "layer thickness, infill density, infill pattern, and print speed — "
        "to predict surface roughness (µm)."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records used", metrics["records_used_for_modeling"])
    c2.metric("Training records", metrics["training_records"])
    c3.metric("Test records", metrics["testing_records"])
    c4.metric("Total cleaned records", metrics["total_cleaned_records"])
    c5.metric("Excluded zero values", metrics["excluded_invalid_roughness"])

    st.markdown("---")
    st.markdown("### Test set accuracy")

    rf = metrics["random_forest"]
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (µm)", f"{rf['mae']:.4f}")
    m2.metric("RMSE (µm)", f"{rf['rmse']:.4f}")
    m3.metric("R²", f"{rf['r2']:.4f}")

    st.markdown(
        "- **MAE** — average absolute difference between predicted and measured roughness.  \n"
        "- **RMSE** — similar to MAE but gives more weight to larger errors.  \n"
        "- **R²** — proportion of variation in surface roughness explained by the model."
    )

    st.markdown("---")
    st.markdown("### Baseline comparison")
    st.markdown(
        "The baseline always predicts the training-set mean roughness. "
        "Comparing against it shows whether the Random Forest adds predictive value."
    )

    baseline = metrics["baseline"]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Baseline MAE (µm)", f"{baseline['mae']:.4f}")
    b2.metric(
        "RF MAE (µm)",
        f"{rf['mae']:.4f}",
        delta=f"{rf['mae'] - baseline['mae']:.4f}",
    )
    b3.metric("Baseline RMSE (µm)", f"{baseline['rmse']:.4f}")
    b4.metric(
        "RF RMSE (µm)",
        f"{rf['rmse']:.4f}",
        delta=f"{rf['rmse'] - baseline['rmse']:.4f}",
    )

    bar_data = pd.DataFrame(
        {
            "Metric": ["MAE", "MAE", "RMSE", "RMSE"],
            "Model": ["Baseline", "Random Forest", "Baseline", "Random Forest"],
            "Value": [
                baseline["mae"],
                rf["mae"],
                baseline["rmse"],
                rf["rmse"],
            ],
        }
    )
    fig_bar = px.bar(
        bar_data,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="Baseline vs. Random Forest — MAE and RMSE (µm)",
        labels={"Value": "Error (µm)"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 5-fold cross-validation")
    st.markdown(
        "5-fold cross-validation evaluates the model across five different "
        "splits of the training data. Similar results across the folds show "
        "that performance is not dependent on one particular split."
    )

    cv = metrics["cross_validation"]
    v1, v2, v3 = st.columns(3)
    v1.metric("CV MAE (avg, µm)", f"{cv['mae']:.4f}")
    v2.metric("CV RMSE (avg, µm)", f"{cv['rmse']:.4f}")
    v3.metric("CV R² (avg)", f"{cv['r2']:.4f}")

    st.markdown("---")
    st.markdown("### Prediction accuracy bands")

    n_predictions = len(predictions)
    bands = [0.5, 1.0, 1.5, 2.0]
    band_cols = st.columns(len(bands))
    for col, band in zip(band_cols, bands):
        within = int((predictions["absolute_error_um"] <= band).sum())
        col.metric(
            f"Within ±{band} µm",
            f"{within}/{n_predictions}",
            f"{100 * within / n_predictions:.1f}%",
        )

    st.markdown("---")
    st.markdown("### Predicted vs. Measured Surface Roughness")

    low = float(
        min(
            predictions["actual_roughness_um"].min(),
            predictions["predicted_roughness_um"].min(),
        )
    )
    high = float(
        max(
            predictions["actual_roughness_um"].max(),
            predictions["predicted_roughness_um"].max(),
        )
    )

    fig_scatter = px.scatter(
        predictions,
        x="actual_roughness_um",
        y="predicted_roughness_um",
        hover_data=[
            "material",
            "printer_type",
            "infill_pattern",
            "layer_thickness_mm",
            "speed_mm_s",
            "absolute_error_um",
        ],
        title="Predicted vs. Measured Surface Roughness",
        labels={
            "actual_roughness_um": "Measured Roughness (µm)",
            "predicted_roughness_um": "Predicted Roughness (µm)",
        },
        opacity=0.7,
    )
    fig_scatter.add_shape(
        type="line",
        x0=low,
        y0=low,
        x1=high,
        y1=high,
        line=dict(width=1.5, dash="dash"),
    )
    fig_scatter.add_annotation(
        x=high,
        y=high,
        text="Perfect prediction",
        showarrow=False,
        xanchor="right",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Prediction Error Distribution")
    fig_hist = px.histogram(
        predictions,
        x="residual_um",
        nbins=20,
        title="Prediction Error Distribution",
        labels={"residual_um": "Residual (Measured − Predicted) µm"},
    )
    fig_hist.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="Zero error",
        annotation_position="top right",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("### Five largest prediction errors")

    worst_columns = [
        "material",
        "printer_type",
        "layer_thickness_mm",
        "infill_density_pct",
        "infill_pattern",
        "speed_mm_s",
        "actual_roughness_um",
        "predicted_roughness_um",
        "absolute_error_um",
    ]
    worst = (
        predictions.nlargest(5, "absolute_error_um")[worst_columns]
        .reset_index(drop=True)
    )
    st.dataframe(worst, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Model limitations")
    st.markdown(
        "- Predictions are **estimates** based on a limited experimental dataset "
        "and should not be treated as guarantees of actual print quality.  \n"
        "- The dataset contains **17 records with a roughness value of zero**. "
        "The source documentation does not explain what those values represent, "
        "so they are retained in the cleaned data but excluded from model training.  \n"
        "- **rPET** is not available for prediction because all of its roughness "
        "records are among the zero-value records excluded from training.  \n"
        "- When exact surface quality is important, physical testing of printed "
        "parts remains necessary."
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Overview":
    render_overview()
elif page == "Data Explorer":
    render_data_explorer()
elif page == "Quality Predictor":
    render_quality_predictor()
elif page == "Model Performance":
    render_model_performance()
else:
    st.info(f"**{page}** is not a recognized page.")
