import streamlit as st
import pandas as pd
import pickle

# Load the pretrained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Survival Prediction App")
st.write(
    "This application predicts survival (1 = Survived, 0 = Died) after exposure to zinc phosphide or similar toxins."
)

# Choose input method
ingest_method = st.radio("Select input method:", ("Manual Entry", "Upload Data File"))

if ingest_method == "Manual Entry":
    st.header("Manual Data Entry")
    # Define all required input features with clear labels and mappings
    age = st.number_input("Age (years)", min_value=0)
    sex = st.selectbox(
        "Sex (1 = Male, 2 = Female)",
        [1, 2]
    )
    time_since_exposure = st.selectbox(
        "Time since exposure (1 = 0.5–1h, 2 = 1–3h, 3 = >3h)",
        [1, 2, 3]
    )
    package_condition = st.selectbox(
        "Package condition (1 = Newly opened, 2 = Exposed to moisture)",
        [1, 2]
    )
    exposure_route = st.selectbox(
        "Exposure route (1 = Direct ingestion, 2 = Dissolved in water, 3 = On food)",
        [1, 2, 3]
    )
    dose_taken = st.number_input(
        "Dose taken (in sachets)",
        min_value=0.0,
        step=0.1
    )
    ne_given = st.selectbox(
        "Norepinephrine given? (0 = No, 1 = Yes)",
        [0, 1]
    )
    nacl_given = st.selectbox(
        "Sodium bicarbonate given? (0 = No, 1 = Yes)",
        [0, 1]
    )
    # Collect inputs into a DataFrame
    input_dict = {
        "Age (in years)": age,
        "Sex (1=Male, 2=Female)": sex,
        "Time since exposure (1=0.5-1h,2=1-3h,3=>3h)": time_since_exposure,
        "Package condition (1=Newly opened,2=Exposed to moisture)": package_condition,
        "Exposure route (1=Direct,2=In water,3=On food)": exposure_route,
        "Dose taken (sachets)": dose_taken,
        "NE_given (Norepinephrine)": ne_given,
        "NaHCO3_given (Sodium Bicarbonate)": nacl_given
    }
    input_df = pd.DataFrame([input_dict])

    if st.button("Run Prediction"):
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("Prediction: Survived (1)")
        else:
            st.error("Prediction: Died (0)")

else:
    st.header("Upload Data File")
    uploaded = st.file_uploader(
        "Upload an Excel (.xlsx) or CSV file with all required columns", 
        type=["xlsx", "csv"]
    )
    if uploaded is not None:
        # Read the uploaded file
        if uploaded.name.lower().endswith(".csv"):
            data = pd.read_csv(uploaded)
        else:
            data = pd.read_excel(uploaded)
        st.write("Input data preview:")
        st.dataframe(data)

        if st.button("Run Batch Prediction"):
            # Ensure the dataframe has the correct columns
            try:
                preds = model.predict(data)
                data["Prediction"] = preds
                st.write("Prediction results:")
                st.dataframe(data)

                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
