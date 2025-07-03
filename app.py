import streamlit as st
import pandas as pd
import pickle

# تحميل الموديل
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🩺 Survival Prediction App")

st.write("""
ادخل بيانات الحالة أو ارفع ملف بيانات، 
وسيخبرك الموديل إذا كانت الحالة Survived أو Died.
""")

# اختيارات
option = st.radio("اختيار طريقة الإدخال:", ("يدوي", "رفع ملف"))

if option == "يدوي":
    # هنا ضع الـ inputs اللي الموديل متدرب عليها
    Age = st.number_input("Age", min_value=0)
    Sex = st.selectbox("Sex", [1, 2])
    Time_since_exposure = st.selectbox("Time since exposure", [1, 2, 3])
    Package_condition = st.selectbox("Package condition", [1, 2])
    Exposure_route = st.selectbox("Exposure route", [1, 2, 3])
    Dose_taken = st.number_input("Dose taken", min_value=0.0)

    # باقي الأعمدة اللي استخدمتها ممكن تضيفها بنفس الشكل 👆

    if st.button("Predict"):
        # نعمل DataFrame بالبيانات
        input_df = pd.DataFrame([[
            Age, Sex, Time_since_exposure, Package_condition,
            Exposure_route, Dose_taken
        ]], columns=[
            "Age", "Sex", "Time_since_exposure", "Package_condition",
            "Exposure_route", "Dose_taken"
        ])
        pred = model.predict(input_df)[0]
        if pred == 1:
            st.success("✅ Survived")
        else:
            st.error("❌ Died")

else:
    uploaded_file = st.file_uploader("ارفع ملف Excel", type=["xlsx", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("📄 البيانات:")
        st.dataframe(df)

        if st.button("Predict"):
            preds = model.predict(df)
            df["Prediction"] = preds
            st.write("🔍 النتائج:")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ تحميل النتائج كـ CSV",
                csv,
                "predictions.csv",
                "text/csv"
            )
