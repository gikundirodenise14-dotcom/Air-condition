import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Classroom Air-Quality Monitor", page_icon="🌬️", layout="wide")

FEATURES = ["gas_ppm", "temperature_c", "humidity_pct"]
CLASSES = ["Good", "Moderate", "Poor"]
COLORS = {"Good": "#2a9d8f", "Moderate": "#e9a820", "Poor": "#e76f51"}
ADVICE = {
    "Good": "Air is fine. No action needed.",
    "Moderate": "Air is getting stale. Open a window or door, or turn on ventilation.",
    "Poor": "Ventilate now and consider reducing the number of people in the room.",
}


@st.cache_data
def load_data(path="classroom_air_quality.csv"):
    return pd.read_csv(path)


@st.cache_resource
def train(df, depth):
    X, y = df[FEATURES], df["air_quality"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(Xtr, ytr)
    pred = model.predict(Xte)
    return model, accuracy_score(yte, pred), confusion_matrix(yte, pred, labels=CLASSES)


df = load_data()

with st.sidebar:
    st.header("Model settings")
    depth = st.slider("Tree depth", 1, 8, 4, help="Deeper trees fit more detail but can overfit.")
    st.caption("Trained on a simulated dataset. Replace the CSV with real MQ-135 and DHT22 readings for the final version.")

model, acc, cm = train(df, depth)

st.title("🌬️ Classroom Air-Quality Monitor")
st.write("A decision tree that classifies air as **Good**, **Moderate** or **Poor** from gas level, temperature and humidity.")

tab1, tab2, tab3, tab4 = st.tabs(["Live prediction", "Batch (CSV)", "Model", "Data"])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Sensor readings")
        gas = st.slider("Gas level (CO2-equivalent ppm)", 350, 3000, 800, 10)
        temp = st.slider("Temperature (°C)", 15.0, 40.0, 26.0, 0.1)
        hum = st.slider("Humidity (%)", 10.0, 95.0, 55.0, 0.5)
    X_new = pd.DataFrame([[gas, temp, hum]], columns=FEATURES)
    label = model.predict(X_new)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X_new)[0]))
    with c2:
        st.subheader("Prediction")
        st.markdown(
            f"<div style='background:{COLORS[label]};color:white;padding:28px;border-radius:14px;"
            f"text-align:center;font-size:2.2rem;font-weight:700'>{label}</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.info(ADVICE[label])
        st.caption("Confidence from the leaf the reading lands in:")
        for c in CLASSES:
            st.progress(float(proba.get(c, 0.0)), text=f"{c}: {proba.get(c, 0.0):.0%}")

with tab2:
    st.write("Upload a CSV with columns `gas_ppm`, `temperature_c`, `humidity_pct` to classify many readings at once.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        new = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in new.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            new["predicted_air_quality"] = model.predict(new[FEATURES])
            st.dataframe(new, width="stretch")
            st.bar_chart(new["predicted_air_quality"].value_counts())
            st.download_button("Download results", new.to_csv(index=False), "predictions.csv", "text/csv")

with tab3:
    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy", f"{acc:.1%}")
    m2.metric("Tree depth", model.get_depth())
    m3.metric("Leaves", model.get_n_leaves())
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Decision tree")
        fig, ax = plt.subplots(figsize=(18, 8))
        plot_tree(model, feature_names=FEATURES, class_names=list(model.classes_),
                  filled=True, rounded=True, fontsize=8, ax=ax)
        st.pyplot(fig)
    with right:
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(cm, index=[f"actual {c}" for c in CLASSES],
                                  columns=[f"pred {c}" for c in CLASSES]))
        st.subheader("Feature importance")
        st.bar_chart(pd.Series(model.feature_importances_, index=FEATURES))
    with st.expander("Rules as text (copy into the ESP32 code)"):
        st.code(export_text(model, feature_names=FEATURES))

with tab4:
    st.write(f"{len(df):,} rows")
    st.dataframe(df.head(200), width="stretch")
    st.subheader("Gas level by class")
    st.bar_chart(df.groupby("air_quality")["gas_ppm"].mean())
