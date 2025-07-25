import streamlit as st
import pandas as pd
import pickle
import time
import numpy as np

# Page configuration with custom styling
st.set_page_config(
    page_title="MedPredict - Multi-Target Medical Prediction",
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
        --warning-color: #ffc107;
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
    
    /* Multi-target prediction cards */
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: var(--card-shadow);
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
        color: #333333;
    }
    
    .prediction-card.survival {
        border-left-color: var(--success-color);
        background: linear-gradient(135deg, #d4edda 0%, #f8f9fa 100%);
    }
    
    .prediction-card.mortality {
        border-left-color: var(--danger-color);
        background: linear-gradient(135deg, #f8d7da 0%, #f8f9fa 100%);
    }
    
    .prediction-card.duration {
        border-left-color: var(--warning-color);
        background: linear-gradient(135deg, #fff3cd 0%, #f8f9fa 100%);
    }
    
    .prediction-card.clinical {
        border-left-color: var(--secondary-color);
        background: linear-gradient(135deg, #cce7ff 0%, #f8f9fa 100%);
    }
    
    .prediction-card h3 {
        color: inherit;
        margin-top: 0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .prediction-card p {
        color: inherit;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .prediction-value {
        font-weight: bold;
        font-size: 1.3rem;
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
    
    .css-1xarl3l {
        color: #333333;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: var(--light-bg);
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: var(--card-shadow);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #111 !important;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Summary statistics styling */
    .summary-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: var(--card-shadow);
        text-align: center;
        border-top: 3px solid var(--primary-color);
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    .stat-label {
        color: #666666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
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

# Load the pretrained Multi-target model
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

# Get the feature names from the model
feature_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []

# Define target columns for multi-target prediction
target_columns = [
    "Survival (1=Survived, 0=Died)",
    "Hospital_stay_days", 
    "Ventilation_days",
    "ICU_stay_days",
    "Need_vasopressors (1=Yes, 0=No)"
]

# Target column display names and emojis
target_display = {
    "Survival (1=Survived, 0=Died)": {"name": "Survival Status", "emoji": "❤️", "type": "binary"},
    "Hospital_stay_days": {"name": "Hospital Stay Duration", "emoji": "🏥", "type": "numeric", "unit": "days"},
    "Ventilation_days": {"name": "Ventilation Duration", "emoji": "🫁", "type": "numeric", "unit": "days"},
    "ICU_stay_days": {"name": "ICU Stay Duration", "emoji": "🚨", "type": "numeric", "unit": "days"},
    "Need_vasopressors (1=Yes, 0=No)": {"name": "Vasopressor Requirement", "emoji": "💉", "type": "binary"}
}

# Sidebar with information and settings
with st.sidebar:
    st.markdown("### 📊 About This App")
    st.markdown("""
    **MedPredict Multi-Target** uses machine learning to predict multiple medical outcomes simultaneously:
    
    **Predictions:**
    - ❤️ Survival Status
    - 🏥 Hospital Stay Duration  
    - 🫁 Ventilation Requirement
    - 🚨 ICU Stay Requirement
    - 💉 Vasopressor Requirement
    """)
    
    st.markdown("### 🔍 Model Information")
    st.markdown(f"**Features:** {len(feature_names)}")
    st.markdown(f"**Model Type:** Multi-Target Predictor")
    st.markdown(f"**Targets:** {len(target_columns)} outcomes")
    
    if feature_names:
        with st.expander("View Input Features"):
            for i, feature in enumerate(feature_names, 1):
                st.write(f"{i}. {feature}")
    
    with st.expander("View Target Outcomes"):
        for target, info in target_display.items():
            st.write(f"{info['emoji']} {info['name']}")
    
    st.markdown("### ⚙️ Settings")
    show_confidence = st.checkbox("Show prediction confidence", value=False)
    auto_download = st.checkbox("Auto-download results", value=True)
    show_detailed_results = st.checkbox("Show detailed analysis", value=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏥 MedPredict Multi-Target</h1>
    <p>Advanced Multi-Outcome Medical Prediction System</p>
</div>
""", unsafe_allow_html=True)

# Helper function to format predictions
def format_predictions(predictions, target_cols):
    """Format multi-target predictions for display"""
    formatted_results = {}
    
    for i, target in enumerate(target_cols):
        pred_value = predictions[i] if isinstance(predictions[0], (list, np.ndarray)) else predictions
        
        if target in target_display:
            info = target_display[target]
            
            if info['type'] == 'binary':
                if target == "Survival (1=Survived, 0=Died)":
                    formatted_results[target] = {
                        'value': int(pred_value[i]) if isinstance(pred_value, (list, np.ndarray)) else int(pred_value),
                        'display': "Survived" if (pred_value[i] if isinstance(pred_value, (list, np.ndarray)) else pred_value) >= 0.5 else "Died",
                        'emoji': info['emoji'],
                        'name': info['name'],
                        'type': 'survival'
                    }
                else:  # Vasopressor
                    val = pred_value[i] if isinstance(pred_value, (list, np.ndarray)) else pred_value
                    formatted_results[target] = {
                        'value': int(val),
                        'display': "Required" if val >= 0.5 else "Not Required",
                        'emoji': info['emoji'],
                        'name': info['name'],
                        'type': 'clinical'
                    }
            else:  # numeric
                val = pred_value[i] if isinstance(pred_value, (list, np.ndarray)) else pred_value
                formatted_results[target] = {
                    'value': float(val),
                    'display': f"{val:.1f} {info['unit']}",
                    'emoji': info['emoji'],
                    'name': info['name'],
                    'type': 'duration'
                }
    
    return formatted_results

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 Batch Prediction", "✏️ Manual Entry"])

with tab1:
    st.markdown("""
    <div class="input-section">
        <h3>📁 Upload Your Data File</h3>
        <p>Upload an Excel (.xlsx) or CSV file containing patient data for batch multi-target prediction.</p>
        <p><strong>Note:</strong> Your file must contain all required input features.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display required features if available
    if feature_names:
        st.markdown("**Required Input Features:**")
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            col = cols[i % 3]
            col.write(f"• {feature}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["xlsx", "csv"],
            accept_multiple_files=False,
            help="Upload a CSV or Excel file with the required input features"
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
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Validate required features
            if feature_names:
                missing_features = set(feature_names) - set(df.columns)
                if missing_features:
                    st.error(f"❌ Missing required features: {', '.join(missing_features)}")
                    st.info("💡 Please ensure your data contains all required input features.")
                else:
                    # Filter to only the required features
                    df_filtered = df[feature_names].copy()
                    
                    st.markdown("### 📊 Data Preview")
                    st.dataframe(df_filtered, use_container_width=True, height=300)
                    
                    # Show data statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(df_filtered))
                    with col2:
                        st.metric("Input Features", len(df_filtered.columns))
                    with col3:
                        missing_data = df_filtered.isnull().sum().sum()
                        st.metric("Missing Values", missing_data)
                    
                    # Prediction button
                    if st.button("🔮 Run Multi-Target Prediction", key="batch_predict", use_container_width=True):
                        with st.spinner("🔄 Processing multi-target predictions..."):
                            try:
                                # Simulate processing time for better UX
                                time.sleep(1)
                                
                                df_proc = df_filtered.copy()
                                
                                # Process categorical columns
                                for col in df_proc.columns:
                                    if df_proc[col].dtype == object:
                                        vals = df_proc[col].astype(str).str.lower().str.strip()
                                        if set(vals.dropna().unique()) <= {"low", "normal", "high"}:
                                            df_proc[col] = vals.map({"high": 0, "low": 1, "normal": 2})
                                
                                # Fill missing values
                                df_proc = df_proc.fillna(-999)
                                
                                # Make predictions
                                predictions = model.predict(df_proc)
                                
                                # Handle different prediction formats
                                if predictions.ndim == 1:
                                    # Single target - reshape for consistency
                                    predictions = predictions.reshape(-1, 1)
                                
                                # Add predictions to dataframe
                                results_df = df_filtered.copy()
                                
                                for i, target in enumerate(target_columns):
                                    if i < predictions.shape[1]:
                                        pred_col = predictions[:, i]
                                        results_df[f"{target}_prediction"] = pred_col
                                        
                                        # Add formatted display
                                        if target in target_display:
                                            info = target_display[target]
                                            if info['type'] == 'binary':
                                                if target == "Survival (1=Survived, 0=Died)":
                                                    results_df[f"{target}_status"] = (pred_col >= 0.5).map({True: "Survived", False: "Died"})
                                                else:
                                                    results_df[f"{target}_status"] = (pred_col >= 0.5).map({True: "Required", False: "Not Required"})
                                            else:
                                                results_df[f"{target}_formatted"] = pred_col.round(1).astype(str) + f" {info['unit']}"
                                
                                st.markdown("### 🎯 Multi-Target Prediction Results")
                                
                                # Summary statistics
                                if predictions.shape[1] >= 1:  # Survival predictions available
                                    survival_preds = predictions[:, 0] >= 0.5
                                    survived_count = survival_preds.sum()
                                    died_count = len(survival_preds) - survived_count
                                    survival_rate = (survived_count / len(survival_preds)) * 100
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("✅ Survived", int(survived_count))
                                    with col2:
                                        st.metric("❌ Died", int(died_count))
                                    with col3:
                                        st.metric("Survival Rate", f"{survival_rate:.1f}%")
                                    with col4:
                                        if predictions.shape[1] > 1:
                                            avg_hospital_stay = predictions[:, 1].mean()
                                            st.metric("Avg Hospital Stay", f"{avg_hospital_stay:.1f} days")
                                
                                if show_detailed_results and predictions.shape[1] > 1:
                                    st.markdown("#### 📈 Detailed Statistics")
                                    
                                    # Create summary statistics for each target
                                    cols = st.columns(min(3, predictions.shape[1]))
                                    
                                    for i, target in enumerate(target_columns[:predictions.shape[1]]):
                                        if i < len(cols):
                                            with cols[i]:
                                                pred_values = predictions[:, i]
                                                info = target_display.get(target, {})
                                                
                                                if info.get('type') == 'binary':
                                                    positive_rate = (pred_values >= 0.5).mean() * 100
                                                    st.metric(
                                                        f"{info.get('emoji', '📊')} {info.get('name', target)[:20]}",
                                                        f"{positive_rate:.1f}%"
                                                    )
                                                else:
                                                    avg_value = pred_values.mean()
                                                    unit = info.get('unit', '')
                                                    st.metric(
                                                        f"{info.get('emoji', '📊')} Avg {info.get('name', target)[:15]}",
                                                        f"{avg_value:.1f} {unit}"
                                                    )
                                
                                # Results table
                                st.markdown("#### 📋 Detailed Results")
                                st.dataframe(results_df, use_container_width=True, height=400)
                                
                                # Download option
                                csv = results_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="📥 Download Multi-Target Predictions as CSV",
                                    data=csv,
                                    file_name="multi_target_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                if auto_download:
                                    st.success("✅ Multi-target predictions completed successfully!")
                                
                            except Exception as e:
                                st.error(f"❌ Batch prediction failed: {str(e)}")
                                st.info("💡 Please ensure your data contains all required features and is properly formatted.")
                                st.code(f"Error details: {str(e)}")
            else:
                st.warning("⚠️ Model feature names not available. Please ensure the model was trained properly.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

with tab2:
    st.markdown("""
    <div class="input-section">
        <h3>✏️ Manual Data Entry</h3>
        <p>Enter patient data manually for individual multi-target prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not feature_names:
        st.error("❌ Cannot display manual entry form - model features not available.")
    else:
        with st.form(key="manual_form", clear_on_submit=False):
            st.markdown("#### 📝 Patient Information")
            
            # Create organized layout for input features
            cols = st.columns(2)
            input_data = {}
            
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
                    label="🔮 Generate Multi-Target Prediction",
                    use_container_width=True
                )
            
            if submit:
                with st.spinner("🔄 Analyzing data for multiple outcomes..."):
                    try:
                        # Simulate processing time
                        time.sleep(1)
                        
                        df_manual = pd.DataFrame([input_data])
                        
                        # Encode categorical text to numeric
                        for col_name in df_manual.columns:
                            if df_manual[col_name].dtype == object:
                                df_manual[col_name] = (
                                    df_manual[col_name].astype(str)
                                    .str.lower()
                                    .str.strip()
                                    .map({"high": 0, "low": 1, "normal": 2})
                                )
                        
                        df_manual = df_manual.fillna(-999)
                        predictions = model.predict(df_manual)
                        
                        # Handle prediction format
                        if predictions.ndim == 1:
                            predictions = predictions.reshape(1, -1)
                        
                        prediction_result = predictions[0]
                        
                        # Store results in session state for display outside form
                        st.session_state.manual_prediction_result = {
                            'predictions': prediction_result,
                            'input_data': input_data
                        }
                        
                        # Prepare data for download
                        result_data = input_data.copy()
                        
                        for i, target in enumerate(target_columns):
                            if i < len(prediction_result):
                                result_data[f"{target}_prediction"] = prediction_result[i]
                                
                                # Add formatted display
                                if target in target_display:
                                    info = target_display[target]
                                    if info['type'] == 'binary':
                                        if target == "Survival (1=Survived, 0=Died)":
                                            result_data[f"{target}_status"] = "Survived" if prediction_result[i] >= 0.5 else "Died"
                                        else:
                                            result_data[f"{target}_status"] = "Required" if prediction_result[i] >= 0.5 else "Not Required"
                                    else:
                                        result_data[f"{target}_formatted"] = f"{prediction_result[i]:.1f} {info['unit']}"
                        
                        st.session_state.manual_prediction_data = pd.DataFrame([result_data])
                    
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        st.info("💡 Please check your input values and ensure all fields are filled correctly.")
                        st.code(f"Error details: {str(e)}")
                        # Clear session state on error
                        st.session_state.manual_prediction_result = None
                        st.session_state.manual_prediction_data = None
        
        # Display prediction results outside the form
        if st.session_state.manual_prediction_result is not None:
            result_data = st.session_state.manual_prediction_result
            predictions = result_data['predictions']
            input_data_display = result_data['input_data']
            
            st.markdown("### 🎯 Multi-Target Prediction Results")
            
            # Display each prediction in a styled card
            for i, target in enumerate(target_columns):
                if i < len(predictions):
                    pred_value = predictions[i]
                    
                    if target in target_display:
                        info = target_display[target]
                        
                        if target == "Survival (1=Survived, 0=Died)":
                            is_survived = pred_value >= 0.5
                            card_class = "survival" if is_survived else "mortality"
                            status_text = "Survived" if is_survived else "Died"
                            icon = "✅" if is_survived else "❌"
                            
                            st.markdown(f"""
                            <div class="prediction-card {card_class}">
                                <h3>{icon} {info['emoji']} {info['name']}</h3>
                                <p class="prediction-value">Prediction: {status_text}</p>
                                <p>Probability: {pred_value:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if is_survived:
                                st.balloons()
                        
                        elif info['type'] == 'binary':
                            is_required = pred_value >= 0.5
                            status_text = "Required" if is_required else "Not Required"
                            icon = "⚠️" if is_required else "✅"
                            
                            st.markdown(f"""
                            <div class="prediction-card clinical">
                                <h3>{icon} {info['emoji']} {info['name']}</h3>
                                <p class="prediction-value">Prediction: {status_text}</p>
                                <p>Probability: {pred_value:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:  # numeric predictions
                            unit = info.get('unit', '')
                            
                            st.markdown(f"""
                            <div class="prediction-card duration">
                                <h3>📊 {info['emoji']} {info['name']}</h3>
                                <p class="prediction-value">Predicted Duration: {pred_value:.1f} {unit}</p>
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
                    label="📥 Download Multi-Target Prediction",
                    data=csv_single,
                    file_name="single_multi_target_prediction.csv",
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
    <p>🏥 <strong>MedPredict Multi-Target</strong> - Advanced Multi-Outcome Medical Prediction System</p>
    <p>Built with ❤️ using Streamlit | For educational and research purposes</p>
    <p><em>Always consult with healthcare professionals for medical decisions</em></p>
</div>
""", unsafe_allow_html=True)
