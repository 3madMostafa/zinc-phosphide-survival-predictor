import streamlit as st
import pandas as pd
import pickle

# Load the pretrained Random Forest model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Survival Prediction App")
st.write(
    "Provide subject data manually or upload an Excel/CSV file containing all required features to predict survival (1 = Survived, 0 = Died)."
)

# Retrieve feature names expected by the model
feature_names = model.feature_names_in_

# Select input method
input_method = st.radio("Choose input method:", ["Upload file", "Manual entry"])

if input_method == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload an Excel (.xlsx) or CSV file with columns matching the model features:",
        type=["xlsx", "csv"]
    )
    if uploaded_file:
        # Read the uploaded data
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Input Data Preview")
        st.dataframe(df)

        if st.button("Run Batch Prediction"):
            try:
                # Prepare input data
                df_proc = df.copy()
                # Encode interpret columns: low, normal, high
                for col in df_proc.columns:
                    if df_proc[col].dtype == object:
                        df_proc[col] = df_proc[col].str.strip().str.lower()
                        if set(df_proc[col].dropna().unique()) <= {"low", "normal", "high"}:
                            mapping = {"high": 0, "low": 1, "normal": 2}
                            df_proc[col] = df_proc[col].map(mapping)
                # Fill missing values
                df_proc = df_proc.fillna(-999)

                # Predict
                preds = model.predict(df_proc[feature_names])
                df["Prediction"] = preds

                st.subheader("Prediction Results")
                st.dataframe(df)

                # Allow download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

else:
    st.subheader("Manual Data Entry")
    input_data = {}
    # Generate an input widget for each feature
    for feature in feature_names:
        label = feature
        if "(1=" in feature or "(1=" in feature:
            # Feature includes mapping info in its name
            input_data[feature] = st.selectbox(
                label,
                [0, 1, 2] if "interpret" in feature or "ratio" in feature.lower() else [0, 1]
            )
        elif "interpret" in feature:
            input_data[feature] = st.selectbox(
                f"{label} (low, normal, high)",
                ["low", "normal", "high"]
            )
        else:
            # Numeric input
            input_data[feature] = st.number_input(label, value=0.0)

    if st.button("Predict" ):
        try:
            df_manual = pd.DataFrame([input_data])
            # Encode interpret text to numeric
            for col in df_manual.columns:
                if df_manual[col].dtype == object:
                    df_manual[col] = df_manual[col].str.strip().str.lower()
                    mapping = {"high": 0, "low": 1, "normal": 2}
                    df_manual[col] = df_manual[col].map(mapping)
            df_manual = df_manual.fillna(-999)

            result = model.predict(df_manual[feature_names])[0]
            if result == 1:
                st.success("Prediction: Survived (1)")
            else:
                st.error("Prediction: Died (0)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
