import streamlit as st
import pandas as pd
import pickle
import time

# Page configuration with custom styling
st.set_page_config(
    page_title="MedPredict - Survival Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern healthcare theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E8B57;
        --secondary-color: #4169E1;
        --success-color: #28a745;
        --danger-color: #dc3545;
        --light-bg: #f8f9fa;
        --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2E8B57 0%, #4169E1 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: var(--card-shadow);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: var(--card-shadow);
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
        color: #333333;
    }
    
    .prediction-card.success {
        border-left-color: var(--success-color);
        background: linear-gradient(135deg, #d4edda 0%, #f8f9fa 100%);
        color: #155724;
    }
    
    .prediction-card.danger {
        border-left-color: var(--danger-color);
        background: linear-gradient(135deg, #f8d7da 0%, #f8f9fa 100%);
        color: #721c24;
    }
    
    .prediction-card h2 {
        color: inherit;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    .prediction-card p {
        color: inherit;
        margin-bottom: 0.5rem;
    }
    
    /* Input section styling */
    .input-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: var(--card-shadow);
        margin: 1rem 0;
        color: #333333;
    }
    
    .input-section h3 {
        color: #2E8B57;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    .input-section p {
        color: #666666;
        margin-bottom: 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--card-shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--card-shadow);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: var(--light-bg);
        color: #333333;
    }
    
    /* Ensure sidebar text is readable */
    .css-1d391kg {
        color: #333333;
    }
    
    .css-1d391kg h3 {
        color: #2E8B57;
    }
    
    /* Footer styling */
    .footer {
        background: var(--light-bg);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: var(--card-shadow);
        text-align: center;
        color: #333333;
    }
    
    /* Ensure metric text is readable */
    .css-1xarl3l {
        color: #333333;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: black;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: var(--card-shadow);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .prediction-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load the pretrained Random Forest model (Top 10 features)
@st.cache_resource
def load_model():
    try:
        with open("model_guaranteed_top10.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'model_guaranteed_top10.pkl' not found. Please ensure the model file is in the same directory.")
        st.stop()

model = load_model()

# Initialize session state for manual prediction results
if 'manual_prediction_result' not in st.session_state:
    st.session_state.manual_prediction_result = None
if 'manual_prediction_data' not in st.session_state:
    st.session_state.manual_prediction_data = None

# Get the top 10 features from the model
feature_names = list(model.feature_names_in_)

# Sidebar with information and settings
with st.sidebar:
    st.markdown("### 📊 About This App")
    st.markdown("""
    **MedPredict** uses machine learning to predict survival outcomes based on patient data.
    
    **Features:**
    - 📁 Batch prediction from files
    - ✏️ Manual data entry
    - 📊 Interactive results
    - 💾 Export predictions
    """)
    
    st.markdown("### 🔍 Model Information")
    st.markdown(f"**Features:** {len(feature_names)} (Top 10 Most Important)")
    st.markdown(f"**Model Type:** Random Forest")
    
    with st.expander("View Top 10 Features"):
        for i, feature in enumerate(feature_names, 1):
            st.write(f"{i}. {feature}")
    
    st.markdown("### ⚙️ Settings")
    show_confidence = st.checkbox("Show prediction confidence", value=False)
    auto_download = st.checkbox("Auto-download results", value=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏥 MedPredict</h1>
    <p>Advanced Survival Prediction System (Top 10 Features)</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 Batch Prediction", "✏️ Manual Entry"])

with tab1:
    st.markdown("""
    <div class="input-section">
        <h3>📁 Upload Your Data File</h3>
        <p>Upload an Excel (.xlsx) or CSV file containing patient data for batch prediction.</p>
        <p><strong>Note:</strong> Your file must contain the following 10 features (exact column names required):</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display required features
    st.markdown("**Required Features:**")
    cols = st.columns(2)
    for i, feature in enumerate(feature_names):
        col = cols[i % 2]
        col.write(f"• {feature}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["xlsx", "csv"],
            accept_multiple_files=False,
            help="Upload a CSV or Excel file with the required top 10 features"
        )
    
    with col2:
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
            st.info(f"📄 **File:** {uploaded_file.name}")
    
    if uploaded_file:
        try:
            # Load data based on file type
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate that all required features are present
            missing_features = set(feature_names) - set(df.columns)
            if missing_features:
                st.error(f"❌ Missing required features: {', '.join(missing_features)}")
                st.info("💡 Please ensure your data contains all 10 required features with exact column names.")
            else:
                # Filter to only the required features
                df_filtered = df[feature_names].copy()
                
                st.markdown("### 📊 Data Preview (Top 10 Features Only)")
                st.dataframe(df_filtered, use_container_width=True, height=300)
                
                # Show data statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df_filtered))
                with col2:
                    st.metric("Features", len(df_filtered.columns))
                with col3:
                    missing_data = df_filtered.isnull().sum().sum()
                    st.metric("Missing Values", missing_data)
                
                # Prediction button
                if st.button("🔮 Run Batch Prediction", key="batch_predict", use_container_width=True):
                    with st.spinner("🔄 Processing predictions..."):
                        try:
                            # Simulate processing time for better UX
                            time.sleep(1)
                            
                            df_proc = df_filtered.copy()
                            # Map interpret strings
                            for col in df_proc.columns:
                                if df_proc[col].dtype == object:
                                    vals = df_proc[col].astype(str).str.lower().str.strip()
                                    if set(vals.dropna().unique()) <= {"low", "normal", "high"}:
                                        df_proc[col] = vals.map({"high": 0, "low": 1, "normal": 2})
                            
                            df_proc = df_proc.fillna(-999)
                            preds = model.predict(df_proc)
                            
                            if show_confidence:
                                confidence = model.predict_proba(df_proc).max(axis=1)
                                df_filtered["Confidence"] = confidence
                            
                            df_filtered["Prediction"] = preds
                            df_filtered["Survival_Status"] = df_filtered["Prediction"].map({1: "Survived", 0: "Died"})
                            
                            st.markdown("### 🎯 Prediction Results")
                            
                            # Summary metrics
                            survived_count = (preds == 1).sum()
                            died_count = (preds == 0).sum()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("✅ Survived", survived_count)
                            with col2:
                                st.metric("❌ Died", died_count)
                            with col3:
                                survival_rate = (survived_count / len(preds)) * 100
                                st.metric("Survival Rate", f"{survival_rate:.1f}%")
                            
                            # Results table
                            st.dataframe(df_filtered, use_container_width=True, height=400)
                            
                            # Download option
                            csv = df_filtered.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="survival_predictions_top10.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            if auto_download:
                                st.success("✅ Predictions completed successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Batch prediction failed: {str(e)}")
                            st.info("💡 Please ensure your data contains all required features and is properly formatted.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

with tab2:
    st.markdown("""
    <div class="input-section">
        <h3>✏️ Manual Data Entry</h3>
        <p>Enter patient data manually for individual prediction using the top 10 most important features.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form(key="manual_form", clear_on_submit=False):
        st.markdown("#### 📝 Patient Information (Top 10 Features)")
        
        # Create a more organized layout
        cols = st.columns(2)  # Use 2 columns for better layout with 10 features
        input_data = {}
        
        # Distribute features across columns for better layout
        for i, feature in enumerate(feature_names):
            col = cols[i % 2]
            
            # Determine widget type based on feature name
            if "(1=" in feature or "interpret" in feature.lower():
                if "interpret" in feature.lower():
                    # Categorical features with text options
                    input_data[feature] = col.selectbox(
                        f"🔍 {feature}",
                        ["low", "normal", "high"],
                        help=f"Select the {feature.lower()} level"
                    )
                else:
                    # Binary features
                    options = [0, 1, 2] if "ratio" in feature.lower() else [0, 1]
                    input_data[feature] = col.selectbox(
                        f"📊 {feature}",
                        options,
                        help=f"Select value for {feature}"
                    )
            else:
                # Numerical features
                input_data[feature] = col.number_input(
                    f"📈 {feature}",
                    value=0.0,
                    help=f"Enter numerical value for {feature}"
                )
        
        # Centered submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit = st.form_submit_button(
                label="🔮 Generate Prediction",
                use_container_width=True
            )
        
        if submit:
            with st.spinner("🔄 Analyzing data..."):
                try:
                    # Simulate processing time
                    time.sleep(1)
                    
                    df_manual = pd.DataFrame([input_data])
                    
                    # Encode interpret text to numeric
                    for col_name in df_manual.columns:
                        if df_manual[col_name].dtype == object:
                            df_manual[col_name] = (
                                df_manual[col_name].astype(str)
                                .str.lower()
                                .str.strip()
                                .map({"high": 0, "low": 1, "normal": 2})
                            )
                    
                    df_manual = df_manual.fillna(-999)
                    result = model.predict(df_manual)[0]
                    
                    if show_confidence:
                        confidence = model.predict_proba(df_manual).max(axis=1)[0]
                        confidence_text = f" (Confidence: {confidence:.1%})"
                    else:
                        confidence_text = ""
                    
                    # Store results in session state for display outside form
                    st.session_state.manual_prediction_result = {
                        'prediction': result,
                        'confidence_text': confidence_text,
                        'input_data': input_data
                    }
                    
                    # Prepare data for download
                    result_data = input_data.copy()
                    result_data["Prediction"] = result
                    result_data["Survival_Status"] = "Survived" if result == 1 else "Died"
                    if show_confidence:
                        result_data["Confidence"] = confidence
                    
                    st.session_state.manual_prediction_data = pd.DataFrame([result_data])
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("💡 Please check your input values and ensure all fields are filled correctly.")
                    # Clear session state on error
                    st.session_state.manual_prediction_result = None
                    st.session_state.manual_prediction_data = None
    
    # Display prediction results outside the form
    if st.session_state.manual_prediction_result is not None:
        result_data = st.session_state.manual_prediction_result
        result = result_data['prediction']
        confidence_text = result_data['confidence_text']
        input_data_display = result_data['input_data']
        
        # Display results with styled cards
        if result == 1:
            st.markdown(f"""
            <div class="prediction-card success">
                <h2>✅ Prediction: Survived</h2>
                <p><strong>Outcome:</strong> Patient is predicted to survive{confidence_text}</p>
                <p><strong>Prediction Value:</strong> 1</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
        else:
            st.markdown(f"""
            <div class="prediction-card danger">
                <h2>❌ Prediction: Died</h2>
                <p><strong>Outcome:</strong> Patient is predicted to not survive{confidence_text}</p>
                <p><strong>Prediction Value:</strong> 0</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show input summary and download button outside form
        with st.expander("📋 View Input Summary"):
            input_df = pd.DataFrame([input_data_display])
            st.dataframe(input_df, use_container_width=True)
        
        # Download button (now outside the form)
        if st.session_state.manual_prediction_data is not None:
            csv_single = st.session_state.manual_prediction_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download This Prediction",
                data=csv_single,
                file_name="single_prediction_top10.csv",
                mime="text/csv",
                key="download_single_prediction"
            )
        
        # Option to clear results
        if st.button("🔄 Make Another Prediction", key="clear_results"):
            st.session_state.manual_prediction_result = None
            st.session_state.manual_prediction_data = None
            st.rerun()

# Footer
st.markdown("""
<div class="footer">
    <p>🏥 <strong>MedPredict</strong> - Advanced Medical Prediction System (Top 10 Features)</p>
    <p>Built with ❤️ using Streamlit | For educational and research purposes</p>
    <p><em>Always consult with healthcare professionals for medical decisions</em></p>
</div>
""", unsafe_allow_html=True)
