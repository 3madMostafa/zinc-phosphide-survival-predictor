import streamlit as st
import pandas as pd
import pickle

# Load the pretrained Random Forest model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Survival Prediction", layout="wide")
st.title("Survival Prediction App")
st.write(
    "Provide subject data manually or upload a file to predict survival (1 = Survived, 0 = Died) after exposure."
)

# Get the list of features the model expects
feature_names = list(model.feature_names_in_)

# Input method selection
input_method = st.radio("Input Method", ["Upload File", "Manual Entry"], index=1, horizontal=True)

if input_method == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload an Excel (.xlsx) or CSV file with these columns:",
        type=["xlsx", "csv"], accept_multiple_files=False
    )
    if uploaded_file:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df)

        if st.button("Run Batch Prediction"):
            try:
                df_proc = df.copy()
                # Map interpret strings
                for col in df_proc.columns:
                    if df_proc[col].dtype == object:
                        vals = df_proc[col].astype(str).str.lower().str.strip()
                        if set(vals.dropna().unique()) <= {"low", "normal", "high"}:
                            df_proc[col] = vals.map({"high": 0, "low": 1, "normal": 2})
                df_proc = df_proc.fillna(-999)
                preds = model.predict(df_proc[feature_names])
                df["Prediction"] = preds
                st.subheader("Prediction Results")
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

else:
    st.subheader("Manual Data Entry")
    with st.form(key="manual_form", clear_on_submit=False):
        cols = st.columns(3)
        input_data = {}
        # Distribute features across columns for better layout
        for i, feature in enumerate(feature_names):
            col = cols[i % 3]
            # decide widget type based on feature name
            if "(1=" in feature or "(1=" in feature:
                # numeric mapping features
                options = [0, 1, 2] if "interpret" in feature or "ratio" in feature.lower() else [0, 1]
                input_data[feature] = col.selectbox(feature, options)
            elif "interpret" in feature:
                input_data[feature] = col.selectbox(f"{feature}", ["low", "normal", "high"])
            else:
                input_data[feature] = col.number_input(feature, value=0.0)
        submit = st.form_submit_button(label="Predict")

        if submit:
            try:
                df_manual = pd.DataFrame([input_data])
                # encode interpret text to numeric
                for col_name in df_manual.columns:
                    if df_manual[col_name].dtype == object:
                        df_manual[col_name] = (
                            df_manual[col_name].astype(str)
                            .str.lower()
                            .str.strip()
                            .map({"high": 0, "low": 1, "normal": 2})
                        )
                df_manual = df_manual.fillna(-999)
                result = model.predict(df_manual[feature_names])[0]
                if result == 1:
                    st.success("Prediction: Survived (1)")
                else:
                    st.error("Prediction: Died (0)")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
