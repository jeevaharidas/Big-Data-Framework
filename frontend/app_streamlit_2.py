# app_streamlit.py — end-to-end Streamlit + PySpark app
# Run: streamlit run app_streamlit.py

import os
import io
import tempfile
import pandas as pd
import numpy as np
import streamlit as st

# ---------- Streamlit page config ----------
st.set_page_config(page_title="RetailRocket – End-to-End", layout="wide")
st.title("🛒 RetailRocket – Scalable Data Intelligence (PySpark)")

# ---------- Spark setup ----------
@st.cache_resource(show_spinner=False)
def get_spark():
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName("RR-Streamlit-AllInOne")
        .master("local[*]")  # assumes local Spark available
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.shuffle.partitions", "200")
    return spark

spark = get_spark()

# ---------- Helpers ----------
def _persist_upload(uploaded_file) -> str:
    """Persist a Streamlit UploadedFile to a temp path and return the path."""
    suffix = ".csv" if uploaded_file.name.lower().endswith(".csv") else ".parquet"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(uploaded_file.getbuffer())
        return tf.name

def read_any_to_spark(path: str):
    path = path.strip()
    if not path:
        raise ValueError("Empty path")
    if path.lower().endswith(".csv"):
        return spark.read.csv(path, header=True, inferSchema=True)
    if path.lower().endswith(".parquet") or os.path.isdir(path):
        return spark.read.parquet(path)
    # try CSV by default
    return spark.read.csv(path, header=True, inferSchema=True)

def spark_head(sdf, n=10) -> pd.DataFrame:
    return sdf.limit(n).toPandas()

def spark_stats(sdf, numeric_only=True) -> pd.DataFrame:
    cols = [f.name for f in sdf.schema.fields]
    if numeric_only:
        from pyspark.sql.types import NumericType
        cols = [f.name for f in sdf.schema.fields if isinstance(f.dataType, NumericType)]
    if not cols:
        return pd.DataFrame({"msg": ["No numeric columns to describe"]})
    return sdf.select(*cols).describe().toPandas()

def infer_feature_columns(sdf, exclude):
    # numeric + low-cardinality categoricals
    from pyspark.sql.types import NumericType, StringType
    num_cols, cat_cols = [], []
    for f in sdf.schema.fields:
        if f.name in exclude:
            continue
        if isinstance(f.dataType, NumericType):
            num_cols.append(f.name)
        elif isinstance(f.dataType, StringType):
            cat_cols.append(f.name)
    return num_cols, cat_cols

def download_button_df(df: pd.DataFrame, label: str, file_name: str):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(label, data=buf.getvalue(), file_name=file_name, mime="text/csv")

# --- Make Spark DataFrames Pandas-friendly by converting VectorUDT columns ---
def sanitize_for_pandas(sdf):
    """Convert all VectorUDT columns to array<double> so toPandas()/Arrow works."""
    from pyspark.ml.linalg import VectorUDT
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    vec_cols = [f.name for f in sdf.schema.fields if isinstance(f.dataType, VectorUDT)]
    if not vec_cols:
        return sdf
    return sdf.select(
        *[vector_to_array(c).alias(c) if c in vec_cols else F.col(c) for c in sdf.columns]
    )

# ---------- UI Tabs ----------
tab_upload, tab_preview, tab_sql, tab_ml, tab_predict = st.tabs(
    ["📤 Data Upload", "👀 Preview & Summary", "🧮 Query Builder (Spark SQL)", "🤖 ML Module", "📊 Predictions & Download"]
)

# Keep state
ss = st.session_state
ss.setdefault("view_name", "dataset")
ss.setdefault("data_path", "")
ss.setdefault("have_data", False)
ss.setdefault("model", None)
ss.setdefault("assembler_cols", [])
ss.setdefault("target_col", "")
ss.setdefault("pred_df_path", "")  # Parquet path for predictions

# ======================
# 1) Data Upload Module
# ======================
with tab_upload:
    st.subheader("Upload CSV or pick an existing path")
    col1, col2 = st.columns(2)
    with col1:
        up = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
        if st.button("Use uploaded file", type="primary") and up is not None:
            ss.data_path = _persist_upload(up)
            ss.have_data = True
            st.success(f"Saved to temp: {ss.data_path}")
    with col2:
        path_in = st.text_input("...or type an existing local path", value=ss.data_path or "")
        if st.button("Load path"):
            try:
                _ = read_any_to_spark(path_in)  # just to validate
                ss.data_path = path_in
                ss.have_data = True
                st.success(f"Path set: {ss.data_path}")
            except Exception as e:
                st.error(f"Failed to read path: {e}")

    st.caption("Once data is loaded, switch to **Preview & Summary** tab.")

# =========================
# 2) Data Preview & Summary
# =========================
with tab_preview:
    st.subheader("Preview & Summary")
    if not ss.have_data:
        st.info("Load data in the **Upload** tab first.")
    else:
        try:
            sdf = read_any_to_spark(ss.data_path)
            sdf.createOrReplaceTempView(ss.view_name)
            st.write(f"Temp view: `{ss.view_name}`")
            # Show schema
            with st.expander("Schema", expanded=False):
                st.code(sdf._jdf.schema().treeString())  # pretty JVM schema tree
            st.write("Top rows")
            st.dataframe(spark_head(sdf, 20), use_container_width=True)
            st.write("Basic stats (numeric columns)")
            st.dataframe(spark_stats(sdf), use_container_width=True)
        except Exception as e:
            st.error(f"Could not preview: {e}")

# ========================
# 3) Query Builder (Spark)
# ========================
with tab_sql:
    st.subheader("Run Spark SQL against your temp view")
    if not ss.have_data:
        st.info("Load data first.")
    else:
        default_sql = f"SELECT * FROM {ss.view_name} LIMIT 100"
        sql_text = st.text_area("SQL", value=default_sql, height=140)
        if st.button("Run SQL"):
            try:
                res = spark.sql(sql_text)
                pdf = res.limit(1000).toPandas()
                st.success(f"Returned {len(pdf)} rows (showing up to 1000).")
                st.dataframe(pdf, use_container_width=True)
                download_button_df(pdf, "Download CSV", "sql_result.csv")
            except Exception as e:
                st.error(f"Query failed: {e}")
        st.caption("Tip: You can `CREATE OR REPLACE TEMP VIEW myview AS SELECT ...` and query it later.")

# =================
# 4) ML Module
# =================
with tab_ml:
    st.subheader("Train a model (choose target & features)")
    if not ss.have_data:
        st.info("Load data first.")
    else:
        try:
            sdf = read_any_to_spark(ss.data_path)
            sdf = sdf.dropna(how="all")
            cols = [f.name for f in sdf.schema.fields]

            target_col = st.selectbox("Target column (classification)", options=cols, index=0 if cols else None, key="target_col")
            exclude_cols = [target_col] if target_col else []
            num_cols, cat_cols = infer_feature_columns(sdf, exclude=exclude_cols)

            st.write("Numeric features (detected):", num_cols[:10], "..." if len(num_cols) > 10 else "")
            st.write("Categorical features (detected):", cat_cols[:10], "..." if len(cat_cols) > 10 else "")

            # Allow user to pick a subset
            use_num = st.multiselect("Use numeric features", options=num_cols, default=num_cols[: min(10, len(num_cols))])
            use_cat = st.multiselect("Use categorical features", options=cat_cols, default=cat_cols[: min(5, len(cat_cols))])

            model_type = st.selectbox("Model", ["RandomForestClassifier", "LogisticRegression"], index=0)
            test_ratio = st.slider("Test ratio", 0.1, 0.5, 0.2, 0.05)

            if st.button("Train model", type="primary"):
                from pyspark.sql import functions as F
                from pyspark.ml import Pipeline
                from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
                from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
                from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

                # Basic cleanup: drop rows with null target
                sdf2 = sdf.filter(F.col(target_col).isNotNull())

                # How many classes?
                n_classes = sdf2.select(target_col).distinct().count()

                # Target indexer
                target_indexer = StringIndexer(inputCol=target_col, outputCol="__label", handleInvalid="skip")
                stages = [target_indexer]

                # Index + OHE categorical features
                idx_cols = [f"{c}__idx" for c in use_cat]
                ohe_cols = [f"{c}__oh" for c in use_cat]
                if use_cat:
                    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}__idx", handleInvalid="keep") for c in use_cat]
                    ohe = OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols)
                    stages += indexers + [ohe]

                # Assembler
                assembler_inputs = use_num + ohe_cols
                if not assembler_inputs:
                    st.error("Please select at least one feature.")
                    st.stop()

                assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_raw")
                scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True)
                stages += [assembler, scaler]

                # Model selection (binary vs multi-class)
                if model_type == "RandomForestClassifier":
                    clf = RandomForestClassifier(
                        featuresCol="features",
                        labelCol="__label",
                        numTrees=200,
                        maxDepth=20,
                        maxBins=128,
                        seed=42,
                    )
                else:
                    family = "binomial" if n_classes == 2 else "multinomial"
                    clf = LogisticRegression(featuresCol="features", labelCol="__label", family=family)

                stages += [clf]
                pipe = Pipeline(stages=stages)

                # Split, fit, predict
                train, test = sdf2.randomSplit([1.0 - test_ratio, test_ratio], seed=42)
                model = pipe.fit(train)
                preds = model.transform(test)

                # Metrics: binary vs multi-class
                if n_classes == 2:
                    aucroc = BinaryClassificationEvaluator(labelCol="__label", metricName="areaUnderROC").evaluate(preds)
                    aucpr = BinaryClassificationEvaluator(labelCol="__label", metricName="areaUnderPR").evaluate(preds)
                    st.success(f"Trained {model_type} (binary). AUROC={aucroc:.3f}, AUPRC={aucpr:.3f}")
                else:
                    acc = MulticlassClassificationEvaluator(labelCol="__label", metricName="accuracy").evaluate(preds)
                    f1 = MulticlassClassificationEvaluator(labelCol="__label", metricName="f1").evaluate(preds)
                    st.success(f"Trained {model_type} (multi-class, {n_classes} classes). Accuracy={acc:.3f}, F1={f1:.3f}")

                ss.model = model
                ss.assembler_cols = assembler_inputs
                st.session_state["__test_preds"] = preds  # keep around for dashboard

        except Exception as e:
            st.error(f"Training failed: {e}")

# =========================
# 5) Prediction Dashboard
# =========================
with tab_predict:
    st.subheader("Score & visualize predictions; download results")
    if not ss.model:
        st.info("Train a model in the **ML Module** tab first.")
    else:
        from pyspark.sql import functions as F
        from pyspark.ml.functions import vector_to_array

        # Pick which DataFrame to score: the uploaded one or an ad-hoc SQL view
        mode = st.radio("Score on", ["Uploaded dataset", "Spark SQL view"], horizontal=True)
        if mode == "Spark SQL view":
            view_name = st.text_input("View name", value=ss.view_name)
            try:
                sdf_score = spark.sql(f"SELECT * FROM {view_name}")
            except Exception as e:
                st.error(f"Could not read view: {e}")
                st.stop()
        else:
            sdf_score = read_any_to_spark(ss.data_path)

        # Transform
        preds = ss.model.transform(sdf_score)

        # Add convenient probability columns:
        # - p1 for binary (class 1)
        # - p_max for multi-class (max class probability)
        if "probability" in preds.columns:
            preds = preds.withColumn("prob_arr", vector_to_array("probability"))
            preds = preds.withColumn("p_max", F.array_max("prob_arr"))
            preds = preds.withColumn("p1", F.when(F.size("prob_arr") > 1, F.col("prob_arr")[1]))
            preds = preds.drop("prob_arr")

        # Show a small preview (convert vectors first)
        st.write("Scored preview")
        st.dataframe(sanitize_for_pandas(preds).limit(50).toPandas(), use_container_width=True)

        # Simple viz: histogram of p1 (binary) or p_max (multi-class)
        hist_col = "p1" if "p1" in preds.columns else ("p_max" if "p_max" in preds.columns else None)
        if hist_col:
            pdf_hist = preds.select(hist_col).sample(False, 0.05, seed=42).toPandas()
            counts, _ = np.histogram(pdf_hist[hist_col].fillna(0.0), bins=20, range=(0, 1))
            st.bar_chart(pd.DataFrame({"count": counts}))

        # Optional: if session_id & itemid exist, show top-K by session
        k = st.slider("Top-K per session (if session_id/itemid exist)", 1, 20, 5, 1)
        if set(["session_id", "itemid"]).issubset(set(preds.columns)) and hist_col:
            from pyspark.sql.window import Window
            score_col = hist_col
            w = Window.partitionBy("session_id").orderBy(F.col(score_col).desc())
            topk = (
                preds.select("session_id", "itemid", F.col(score_col).alias("score"))
                .withColumn("rk", F.row_number().over(w))
                .filter(F.col("rk") <= k)
                .drop("rk")
            )
            st.write("Top-K preview")
            st.dataframe(sanitize_for_pandas(topk).limit(100).toPandas(), use_container_width=True)

            # Downloads
            topk_pdf = sanitize_for_pandas(topk).toPandas()
            download_button_df(topk_pdf, "Download Top-K CSV", "topk.csv")

        # Full predictions download (convert vectors first; cap rows for memory safety)
        preds_pdf = sanitize_for_pandas(preds).limit(100000).toPandas()
        download_button_df(preds_pdf, "Download Predictions CSV (up to 100k rows)", "predictions.csv")

st.caption("✅ This app covers: Upload → Preview → SQL → Train (target+features) → Predict → Visualize → Download.")
