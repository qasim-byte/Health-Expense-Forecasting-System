# Streamlit app to predict medical insurance cost based on user inputs
import os
import joblib
import pandas as pd
import streamlit as st
from trainer import create_model, save_model
from data_cleaner import create_data_processor
from user_manager import UserManager
import base64
import io
import datetime
import json
import warnings
warnings.filterwarnings("ignore")

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Custom CSS for better styling
def local_css():
    st.markdown("""
    <style>
    /* --- SHARED THEME VARIABLES --- */
    :root {
        --glass-bg: rgba(128, 128, 128, 0.1);
        --glass-border: rgba(128, 128, 128, 0.2);
    }

    /* --- FROSTED GLASS SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: var(--glass-bg) !important;
        backdrop-filter: blur(15px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(15px) saturate(160%) !important;
        border-right: 1px solid var(--glass-border);
    }

    /* Keep sidebar text aligned with theme but ensure visibility */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--text-color) !important;
    }

    /* --- MAIN CONTENT ADJUSTMENTS --- */
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    .block-container {
        padding-top: 2rem !important;
    }

    /* --- COMPONENT STYLING --- */
    
    /* 1. Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--text-color);
        text-align: center;
        padding: 1.5rem;
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border-radius: 18px;
        border: 1px solid var(--glass-border);
    }

    /* 2. Prediction Box */
    .prediction-box {
        background: var(--glass-bg) !important; 
        backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid var(--glass-border) !important;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }

    /* 3. Prediction Amount - Cyan stays for pop, but with better contrast */
    .prediction-amount {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #00d4ff !important; 
        text-shadow: 1px 1px 10px rgba(0, 0, 0, 0.2);
    }

    /* 4. Notifications / Info Boxes */
    div[data-testid="stNotification"] {
        background-color: var(--secondary-background-color) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
    }

    /* Force all text in notifications to follow theme color */
    div[data-testid="stNotification"] * {
        color: var(--text-color) !important;
    }

    /* 5. Health Metrics & Liquid Glass */
    .health-metric, .suggestion-item {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border: 1px solid var(--glass-border);
    }

    .metric-label {
        font-weight: bold;
        color: var(--text-color);
        text-align: center;
    }

    /* Glass Tube visibility fix */
    .glass-tube {
        background: rgba(128, 128, 128, 0.1);
        border: 2px solid var(--glass-border);
    }

    /* --- PRINT FIX --- */
    @media print {
        .print-friendly {
            background: white !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)



# Function to generate liquid glass container HTML
def generate_liquid_glass_container(metric_name, value, fill_percentage, status="good"):
    """
    Generate HTML for a liquid glass container.
    
    Args:
        metric_name: Name of the metric (e.g., "Age", "BMI")
        value: Display value (e.g., "30 years", "25.5")
        fill_percentage: Percentage to fill the glass (0-100)
        status: "good", "warning", or "danger" for color coding
    """
    # Ensure fill percentage is within bounds
    fill_percentage = max(0, min(100, fill_percentage))
    
    return f"""
    <div class="liquid-glass-container">
        <div class="glass-tube">
            <div class="liquid-fill {status}" style="height: {fill_percentage}%"></div>
        </div>
        <div class="glass-base"></div>
        <div class="metric-label">{metric_name}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


# Function to calculate fill percentage and status for health metrics
def calculate_metric_fill(metric_name, value, patient_data):
    """
    Calculate fill percentage and status for a health metric.
    Returns (fill_percentage, status)
    """
    # Default values
    fill = 50
    status = "good"
    
    # Age: 0-120, higher age = higher fill (risk)
    if metric_name == "Age":
        age = float(value.split()[0]) if isinstance(value, str) else float(value)
        fill = min(100, age / 120 * 100)
        if age < 40:
            status = "good"
        elif age < 60:
            status = "warning"
        else:
            status = "danger"
    
    # BMI: 0-100, higher BMI = higher fill (risk)
    elif metric_name == "BMI":
        bmi = float(value) if isinstance(value, (int, float)) else float(value.split()[0])
        fill = min(100, bmi / 50 * 100)  # Normalize to 50 as max
        if bmi < 25:
            status = "good"
        elif bmi < 30:
            status = "warning"
        else:
            status = "danger"
    
    # Stress Level: 1-10 scale
    elif metric_name == "Stress Level":
        stress = float(value.split('/')[0]) if isinstance(value, str) else float(value)
        fill = stress / 10 * 100
        if stress < 4:
            status = "good"
        elif stress < 7:
            status = "warning"
        else:
            status = "danger"
    
    # Sleep Hours: 0-24, lower sleep = higher fill (risk)
    elif metric_name == "Sleep Hours":
        sleep = float(value.split()[0]) if isinstance(value, str) else float(value)
        # Inverse: less sleep = higher risk
        fill = max(0, 100 - (sleep / 8 * 100))  # 8 hours is ideal
        if sleep >= 7:
            status = "good"
        elif sleep >= 6:
            status = "warning"
        else:
            status = "danger"
    
    # Daily Steps: more steps = lower fill (better)
    elif metric_name == "Daily Steps":
        steps = int(value.replace(',', '')) if isinstance(value, str) else int(value)
        # More steps = lower risk
        fill = max(0, 100 - (steps / 10000 * 100))  # 10k steps is ideal
        if steps >= 8000:
            status = "good"
        elif steps >= 5000:
            status = "warning"
        else:
            status = "danger"
    
    # Activity Level: categorical
    elif metric_name == "Activity Level":
        activity = value.lower()
        if activity == "high":
            fill = 20
            status = "good"
        elif activity == "medium":
            fill = 50
            status = "warning"
        else:  # low
            fill = 80
            status = "danger"
    
    # For binary conditions (Yes/No)
    elif metric_name in ["Smoker", "Diabetes", "Hypertension", "Heart Disease", "Asthma"]:
        if value == "Yes":
            fill = 90
            status = "danger"
        else:
            fill = 10
            status = "good"
    
    # Insurance Coverage: higher coverage = lower fill (better)
    elif metric_name == "Coverage":
        coverage = float(value.replace('%', '')) if isinstance(value, str) else float(value)
        fill = 100 - coverage  # Lower coverage = higher risk
        if coverage >= 80:
            status = "good"
        elif coverage >= 50:
            status = "warning"
        else:
            status = "danger"
    
    # Previous Cost: higher cost = higher fill
    elif metric_name == "Previous Cost":
        cost = float(value.replace('$', '').replace(',', '')) if isinstance(value, str) else float(value)
        fill = min(100, cost / 20000 * 100)  # Normalize to $20k as max
        if cost < 5000:
            status = "good"
        elif cost < 15000:
            status = "warning"
        else:
            status = "danger"
    
    # Gender and Location: neutral
    elif metric_name in ["Gender", "Location"]:
        fill = 30
        status = "good"
    
    return fill, status


# Function to generate downloadable HTML report
def generate_html_report(patient_data, prediction, suggestions, background_image_path=None):
    """
    Generate a complete HTML report for download.
    """
    # Get current date
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    # Encode background image if available
    background_css = ""
    if background_image_path and os.path.exists(background_image_path):
        with open(background_image_path, "rb") as img_file:
            encoded_bg = base64.b64encode(img_file.read()).decode()
            background_css = f"""
            body {{
                background-image: url(data:image/png;base64,{encoded_bg});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                min-height: 100vh;
            }}
            """
    else:
        background_css = """
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        """
    
    # Generate liquid glass containers HTML
    liquid_glass_html = ""
    # Create metrics similar to the app
    metrics_list = [
        ("Age", f"{patient_data['age']} years"),
        ("Gender", patient_data['gender']),
        ("BMI", f"{patient_data['bmi']:.1f}"),
        ("Location", patient_data['city_type']),
        ("Smoker", patient_data['smoker']),
        ("Diabetes", patient_data['diabetes']),
        ("Hypertension", patient_data['hypertension']),
        ("Heart Disease", patient_data['heart_disease']),
        ("Asthma", patient_data['asthma']),
        ("Activity Level", patient_data['physical_activity_level']),
        ("Daily Steps", f"{patient_data['daily_steps']:,}"),
        ("Sleep Hours", f"{patient_data['sleep_hours']:.1f}"),
        ("Stress Level", f"{patient_data['stress_level']}/10"),
        ("Coverage", f"{patient_data['insurance_coverage_pct']}%"),
        ("Previous Cost", f"${patient_data['previous_year_cost']:,.2f}")
    ]
    
    for metric, value in metrics_list:
        fill, status = calculate_metric_fill(metric, value, patient_data)
        liquid_glass_html += generate_liquid_glass_container(metric, value, fill, status)
    
    # Generate suggestions HTML
    suggestions_html = ""
    if suggestions:
        suggestions_html = "<h3>Personalized Health Recommendations</h3><div class='suggestions'>"
        for i, suggestion in enumerate(suggestions, 1):
            suggestions_html += f"<div class='suggestion-item'><strong>{i}.</strong> {suggestion}</div>"
        suggestions_html += "</div>"
    
    # Calculate insurance details
    coverage_pct = patient_data['insurance_coverage_pct']
    out_of_pocket = prediction * (1 - coverage_pct/100)
    
    # CSS for the HTML report  # type: ignore
    css_styles = """
            /* Include all CSS from the app - simplified for HTML report */
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
            }}

            .liquid-glass-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                margin: 1rem 0.5rem;
                width: 120px;
            }}

            .glass-tube {{
                position: relative;
                width: 60px;
                height: 180px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 30px 30px 10px 10px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                overflow: hidden;
                box-shadow:
                    inset 0 0 20px rgba(0, 0, 0, 0.2),
                    0 8px 20px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(5px);
            }}

            .liquid-fill {{
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background: linear-gradient(to top, rgba(0, 212, 255, 0.8), rgba(0, 150, 255, 0.9));
                border-radius: 0 0 8px 8px;
                transition: height 1.5s ease-in-out;
                box-shadow: inset 0 0 10px rgba(255, 255, 255, 0.3);
            }}

            .liquid-fill.good {{
                background: linear-gradient(to top, rgba(46, 204, 113, 0.8), rgba(39, 174, 96, 0.9));
            }}

            .liquid-fill.warning {{
                background: linear-gradient(to top, rgba(241, 196, 15, 0.8), rgba(243, 156, 18, 0.9));
            }}

            .liquid-fill.danger {{
                background: linear-gradient(to top, rgba(231, 76, 60, 0.8), rgba(192, 57, 43, 0.9));
            }}

            .glass-base {{
                width: 80px;
                height: 10px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 5px;
                margin-top: -5px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            }}

            .metric-label {{
                margin-top: 10px;
                font-weight: bold;
                color: white;
                text-align: center;
                font-size: 0.9rem;
                text-shadow: 0 1px 2px rgba(0,0,0,0.5);
            }}

            .metric-value {{
                color: #00d4ff;
                font-weight: 800;
                font-size: 1.1rem;
                margin-top: 5px;
                text-align: center;
            }}

            .suggestions {{
                margin-top: 30px;
            }}

            .suggestion-item {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #00d4ff;
            }}

            .suggestion-item h4 {{
                margin: 0 0 8px 0;
                color: #00d4ff;
                font-size: 1.1rem;
            }}

            .suggestion-item p {{
                margin: 0;
                color: rgba(255, 255, 255, 0.9);
                line-height: 1.5;
            }}

            .insurance-analysis {{
                display: flex;
                gap: 20px;
                margin: 20px 0;
            }}

            .insurance-card {{
                flex: 1;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}

            .insurance-card h3 {{
                margin: 0 0 15px 0;
                color: #00d4ff;
                font-size: 1.2rem;
            }}

            .insurance-card .metric-amount {{
                font-size: 2rem;
                font-weight: bold;
                color: white;
                margin: 10px 0;
            }}

            .insurance-card p {{
                margin: 5px 0 0 0;
                color: rgba(255, 255, 255, 0.8);
                font-size: 0.9rem;
            }}

            .report-container {{
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 30px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}

            .report-header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }}

            .report-header h1 {{
                color: #00d4ff;
                font-size: 2.5rem;
                margin: 0 0 10px 0;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            }}

            .report-header p {{
                color: rgba(255, 255, 255, 0.8);
                font-size: 1.1rem;
                margin: 0;
            }}

            .patient-info {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 40px;
            }}

            .patient-info div {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}

            .patient-info h3 {{
                margin: 0 0 15px 0;
                color: #00d4ff;
                font-size: 1.3rem;
            }}

            .patient-info p {{
                margin: 8px 0;
                color: rgba(255, 255, 255, 0.9);
                font-size: 1rem;
            }}

            .patient-info strong {{
                color: white;
                font-weight: 600;
            }}

            .prediction-display {{
                text-align: center;
                margin: 40px 0;
                padding: 30px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                border: 2px solid rgba(255, 255, 255, 0.2);
            }}

            .prediction-display h2 {{
                color: #00d4ff;
                font-size: 1.8rem;
                margin: 0 0 20px 0;
            }}

            .prediction-amount {{
                font-size: 3rem;
                font-weight: bold;
                color: #00d4ff;
                margin: 20px 0;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            }}

            .prediction-display p {{
                color: rgba(255, 255, 255, 0.8);
                font-size: 1.1rem;
                margin: 15px 0 0 0;
                line-height: 1.5;
            }}

            .liquid-glass-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 20px;
                margin: 40px 0;
                justify-items: center;
            }}

            .insurance-analysis h2 {{
                text-align: center;
                color: #00d4ff;
                font-size: 1.8rem;
                margin: 40px 0 30px 0;
            }}

            .metric-amount {{
                font-size: 2.5rem;
                font-weight: bold;
                color: white;
                margin: 10px 0;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            }}

            .metric-amount.danger {{
                color: #e74c3c;
            }}

            .metric-amount.warning {{
                color: #f39c12;
            }}

            .metric-amount.good {{
                color: #27ae60;
            }}

            .metric-amount.info {{
                color: #3498db;
            }}
    """
    
    # Build the complete HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Health Insurance Report - {patient_data.get('patient_name', 'Patient')}</title>
        <style>
            {css_styles}
        </style>
    </head>
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid rgba(255, 255, 255, 0.2);
            }}

            .report-header h1 {{
                font-size: 2.5rem;
                margin-bottom: 10px;
                color: #00d4ff;
                text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
            }}

            .prediction-display {{
                text-align: center;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 30px;
                margin: 30px 0;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}

            .prediction-amount {{
                font-size: 4rem;
                font-weight: 800;
                color: #00d4ff;
                text-shadow: 0 0 15px rgba(0, 212, 255, 0.7);
                margin: 20px 0;
            }}

            .liquid-glass-grid {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 30px;
                margin: 40px 0;
            }}

            .patient-info {{
                display: flex;
                justify-content: space-between;
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}

            .insurance-analysis {{
                display: flex;
                gap: 20px;
                margin-top: 30px;
            }}

            .insurance-card {{
                flex: 1;
                background: rgba(255, 255, 255, 0.07);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}

            .suggestions {{
                margin-top: 30px;
            }}

            .suggestion-item {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                border-left: 5px solid #27AE60;
            }}

            @media print {{
                body {{
                    background: white !important;
                    color: black !important;
                }}
                .report-container {{
                    background: white !important;
                    color: black !important;
                    box-shadow: none !important;
                    border: 1px solid #ddd !important;
                }}
                .prediction-amount {{
                    color: #0066cc !important;
                }}
                .liquid-glass-container {{
                    color: black !important;
                }}
            }}

            {background_css}
        </style>
    </head>
    <body>
        <div class="report-container">
            <div class="report-header">
                <h1>🏥 Health Insurance Report</h1>
                <p>Generated on {current_date}</p>
            </div>
            
            <div class="patient-info">
                <div>
                    <h3>Patient Information</h3>
                    <p><strong>Name:</strong> {patient_data.get('patient_name', 'N/A')}</p>
                    <p><strong>Age:</strong> {patient_data['age']} years</p>
                    <p><strong>Gender:</strong> {patient_data['gender']}</p>
                </div>
                <div>
                    <h3>Report Summary</h3>
                    <p><strong>Prediction Date:</strong> {current_date}</p>
                    <p><strong>Insurance Type:</strong> {patient_data['insurance_type']}</p>
                    <p><strong>City Type:</strong> {patient_data['city_type']}</p>
                </div>
            </div>
            
            <div class="prediction-display">
                <h2>Predicted Annual Medical Cost</h2>
                <div class="prediction-amount">${prediction:,.2f}</div>
                <p>Based on your health profile and demographic information</p>
            </div>
            
            <h2 style="text-align: center; margin-top: 40px;">Health Metrics Visualization</h2>
            <div class="liquid-glass-grid">
                {liquid_glass_html}
            </div>
            
            <div style="margin-top: 50px;">
                <h2>Insurance Coverage Analysis</h2>
                <div class="insurance-analysis">
                    <div class="insurance-card">
                        <h3>Insurance Coverage</h3>
                        <div style="font-size: 3rem; font-weight: bold; color: #27AE60;">{coverage_pct}%</div>
                        <p>of medical costs covered</p>
                    </div>
                    <div class="insurance-card">
                        <h3>Estimated Out-of-Pocket</h3>
                        <div style="font-size: 3rem; font-weight: bold; color: #E74C3C;">${out_of_pocket:,.2f}</div>
                        <p>annual expense</p>
                    </div>
                </div>
            </div>
            
            {suggestions_html}
            
            <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.2); text-align: center;">
                <p>This report was generated by AI Health Insurance Advisor</p>
                <p>For more information, consult with your healthcare provider</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html


MODEL_PATH = "medical_cost_model.pkl"
DATA_PATH = "medical_cost_prediction_dataset.csv"

user_manager = UserManager()

class MedicalCostPredictor:
    def __init__(self):
        self.model = None

    def ensure_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            return

        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset not found: {DATA_PATH}. Please place the dataset file in this folder.")
            return

        st.info("Training model because the model file was not found.")
        loader = create_data_processor("loader", file_path=DATA_PATH)
        raw_data = loader.process()
        cleaner = create_data_processor("cleaner", data=raw_data)
        cleaned_data = cleaner.process()
        preprocessor = create_data_processor("preprocessor", data=cleaned_data)
        training_data = preprocessor.process()

        self.model = create_model("random_forest", data=training_data)
        rmse, r2 = self.model.train()
        save_model(self.model, MODEL_PATH)
        st.success(f"Model trained and saved to {MODEL_PATH}. RMSE={rmse:.2f}, R2={r2:.2f}")

    def predict_cost(self, input_data):
        if self.model is None:
            raise ValueError("Model not loaded.")
        return self.model.predict(input_data)[0]

    def get_personalized_suggestions(self, patient_data, predicted_cost):
        suggestions = []
        if patient_data['smoker'] == 'Yes':
            suggestions.append("Consider quitting smoking to potentially reduce medical costs by up to 20-30%.")
        if patient_data['bmi'] > 30:
            suggestions.append("Work on weight management through diet and exercise to lower BMI and associated health risks.")
        if patient_data['physical_activity_level'] == 'Low':
            suggestions.append("Increase physical activity to at least moderate levels for better health outcomes.")
        if patient_data['stress_level'] > 7:
            suggestions.append("Practice stress management techniques like meditation or yoga.")
        if patient_data['sleep_hours'] < 7:
            suggestions.append("Aim for 7-9 hours of sleep per night for optimal health.")
        if predicted_cost > 10000:
            suggestions.append("Consider consulting with an insurance advisor for better coverage options.")
        if patient_data['insurance_type'] == 'None':
            suggestions.append("Explore health insurance options to cover potential medical expenses.")
        return suggestions

predictor = MedicalCostPredictor()


def build_input_dataframe():
    st.markdown("### 👤 Patient Information")

    # Patient name display
    st.markdown(f"**👋 Welcome, {st.session_state.patient_name}!**")

    # Create organized sections with columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Basic Demographics")
        age = st.number_input("Age", min_value=0, max_value=120, value=30, help="Enter your current age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="Body Mass Index - you can calculate this or use our BMI calculator")

        st.markdown("#### 🏥 Health Conditions")
        smoker = st.selectbox("Do you smoke?", ["No", "Yes"], help="Current smoking status")
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], help="Do you have diabetes?")
        hypertension = st.selectbox("Hypertension", ["No", "Yes"], help="Do you have high blood pressure?")
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"], help="Do you have heart disease?")
        asthma = st.selectbox("Asthma", ["No", "Yes"], help="Do you have asthma?")

    with col2:
        st.markdown("#### 🏃‍♂️ Lifestyle & Activity")
        physical_activity_level = st.selectbox("Physical Activity Level",
                                             ["Low", "Medium", "High"],
                                             help="How active are you? Low: sedentary, Medium: moderate exercise, High: very active")
        daily_steps = st.number_input("Daily Steps", min_value=0, max_value=100000, value=5000,
                                    help="Approximate number of steps you take daily")
        sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, value=7.0, step=0.5,
                                    help="Average hours of sleep per night")
        stress_level = st.number_input("Stress Level (1-10)", min_value=1, max_value=10, value=5,
                                     help="Rate your average stress level (1=very low, 10=very high)")

        st.markdown("#### 🏛️ Insurance & Location")
        city_type = st.selectbox("City Type", ["Rural", "Semi-Urban", "Urban"],
                               help="Type of area where you live")
        insurance_type = st.selectbox("Insurance Type", ["None", "Government", "Private"],
                                    help="What type of health insurance do you have?")
        insurance_coverage_pct = st.number_input("Insurance Coverage (%)", min_value=0, max_value=100, value=50,
                                               help="What percentage of medical costs does your insurance cover?")

    st.markdown("#### 📈 Additional Health Information")
    col3, col4 = st.columns(2)
    with col3:
        doctor_visits_per_year = st.number_input("Doctor Visits per Year", min_value=0, max_value=100, value=2,
                                               help="How many times do you visit the doctor annually?")
        hospital_admissions = st.number_input("Hospital Admissions (Last Year)", min_value=0, max_value=100, value=0,
                                            help="How many times were you admitted to hospital last year?")
    with col4:
        medication_count = st.number_input("Number of Medications", min_value=0, max_value=100, value=0,
                                         help="How many different medications do you take regularly?")
        previous_year_cost = st.number_input("Previous Year Medical Cost ($)", min_value=0.0, value=0.0,
                                           help="Total medical expenses from last year (0 if none)")

    data = {
        "patient_name": st.session_state.patient_name,
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "smoker": smoker,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "asthma": asthma,
        "physical_activity_level": physical_activity_level,
        "daily_steps": daily_steps,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "doctor_visits_per_year": doctor_visits_per_year,
        "hospital_admissions": hospital_admissions,
        "medication_count": medication_count,
        "insurance_type": insurance_type,
        "insurance_coverage_pct": insurance_coverage_pct,
        "city_type": city_type,
        "previous_year_cost": previous_year_cost,
    }
    return pd.DataFrame([data])


def main():
    # Add custom CSS
    local_css()

    # Try to add background image (you can replace with your own image)
    # To add a custom background image, place a file called 'background.jpg' or 'background.png' in the same folder
    background_files = ['background.jpg', 'background.png', 'bg.jpg', 'bg.png']
    background_added = False

    for bg_file in background_files:
        if os.path.exists(bg_file):
            try:
                add_bg_from_local(bg_file)
                background_added = True
                break
            except:
                pass

    # If no background image found, use gradient
    if not background_added:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        """, unsafe_allow_html=True)

    

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.patient_id = None
        st.session_state.patient_name = None

    if not st.session_state.logged_in:
        show_auth_page()
    else:
        show_main_app()

def show_auth_page():
    

    st.markdown("## 🔐 Welcome to AI Health Insurance Advisor")
    
    message = "Please login to access your personalized health report and insurance cost prediction."
    st.info(message)
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

    with tab1:
        st.markdown("### Login to Your Account")
        with st.form("login_form"):
            patient_id = st.text_input("Patient ID", placeholder="Enter your patient ID")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            submitted = st.form_submit_button("🚀 Login", type="primary", use_container_width=True)
            if submitted:
                success, message = user_manager.login(patient_id, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.patient_id = patient_id
                    st.session_state.patient_name = message
                    st.success(f"🎉 Welcome back, {message}!")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

    with tab2:
        st.markdown("### Create New Account")
        with st.form("signup_form"):
            new_id = st.text_input("Choose Patient ID", placeholder="Create a unique patient ID")
            new_name = st.text_input("Full Name", placeholder="Enter your full name")
            new_password = st.text_input("Choose Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")

            submitted = st.form_submit_button("✅ Sign Up", type="primary", use_container_width=True)
            if submitted:
                if new_password != confirm_password:
                    st.error("❌ Passwords do not match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters long")
                elif not new_name.strip():
                    st.error("❌ Please enter your full name")
                else:
                    success, message = user_manager.signup(new_id, new_password, new_name.strip())
                    if success:
                        st.success(f"✅ Account created successfully! Welcome, {new_name.strip()}!")
                        st.info("💡 You can now login with your credentials")
                    else:
                        st.error(f"❌ {message}")

    

def show_main_app():
    st.sidebar.header(f"Welcome, {st.session_state.patient_name}")
    st.sidebar.write(f"Patient ID: {st.session_state.patient_id}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.patient_id = None
        st.session_state.patient_name = None
        st.rerun()

    # Show previous reports
    show_previous_reports()

    # Main prediction interface
    show_prediction_interface()

def show_previous_reports():
    reports = user_manager.get_reports(st.session_state.patient_id)

    if reports:
        st.header("📚 Your Previous Reports")
        for i, report in enumerate(reversed(reports)):
            with st.expander(f"📄 Report {len(reports)-i} - {report['timestamp'][:10]}"):
                st.write(f"**💰 Predicted Cost:** ${report['data']['prediction']:,.2f}")
                if 'suggestions' in report['data'] and report['data']['suggestions']:
                    st.write("**🎯 Suggestions:**")
                    for suggestion in report['data']['suggestions']:
                        st.write(f"• {suggestion}")

                # View report details button
                if st.button(f"🔍 View Full Report Details", key=f"view_{i}"):
                    st.markdown("---")
                    st.markdown("### 📋 Complete Report Details")

                    # Display all input data in organized format
                    input_data = report['data']['input_data']
                    st.markdown(f"**Patient:** {input_data.get('patient_name', 'N/A')}")
                    st.markdown(f"**Date:** {report['timestamp'][:10]}")

                    # Health metrics in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Basic Information:**")
                        st.write(f"- Age: {input_data.get('age', 'N/A')}")
                        st.write(f"- Gender: {input_data.get('gender', 'N/A')}")
                        st.write(f"- BMI: {input_data.get('bmi', 'N/A')}")
                        st.write(f"- Location: {input_data.get('city_type', 'N/A')}")

                    with col2:
                        st.markdown("**Insurance Details:**")
                        st.write(f"- Type: {input_data.get('insurance_type', 'N/A')}")
                        st.write(f"- Coverage: {input_data.get('insurance_coverage_pct', 'N/A')}%")
                        st.write(f"- Previous Cost: ${input_data.get('previous_year_cost', 0):,.2f}")

                    st.markdown("**Health Conditions:**")
                    conditions = ['smoker', 'diabetes', 'hypertension', 'heart_disease', 'asthma']
                    for condition in conditions:
                        status = input_data.get(condition, 'N/A')
                        st.write(f"- {condition.title()}: {status}")
    else:
        st.info("📝 No previous reports found. Create your first prediction below!")

def show_prediction_interface():
    st.markdown("---")
    st.markdown("## 🏥 Medical Cost Prediction")

    # Info about PDF functionality
    st.info("💡 **Pro Tip**: After generating your report, use `Ctrl+P` (or `Cmd+P` on Mac) to save it as a professional PDF!")

    predictor.ensure_model()
    if not os.path.exists(MODEL_PATH):
        return

    input_df = build_input_dataframe()

    # Print-friendly report container
    with st.container():
        st.markdown('<div class="print-friendly">', unsafe_allow_html=True)

        # Report Header
        st.markdown("### 📋 Patient Health Report")
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            st.markdown(f"**Patient:** {st.session_state.patient_name}")
        with col2:
            st.markdown(f"**ID:** {st.session_state.patient_id}")
        with col3:
            st.markdown(f"**Date:** {pd.Timestamp.now().strftime('%B %d, %Y')}")

        st.markdown("---")

        if st.button("🔮 Generate Prediction Report", type="primary", use_container_width=True):
            # Prepare data for model (exclude patient_name)
            model_input_df = input_df.drop(columns=['patient_name'])

            cleaner = create_data_processor("cleaner", data=model_input_df)
            cleaned_input = cleaner.process()
            preprocessor = create_data_processor("preprocessor", data=cleaned_input)
            preprocessed_input = preprocessor.process()
            prediction = predictor.predict_cost(preprocessed_input)

            # Store prediction data
            st.session_state['prediction'] = prediction
            st.session_state['input_data'] = input_df.iloc[0].to_dict()

            # Personalized Suggestions
            suggestions = predictor.get_personalized_suggestions(st.session_state['input_data'], prediction)

            # Display Results in Print-Friendly Format
            st.markdown("---")

            # Prediction Display
            st.markdown("""
            <div class="prediction-box">
                <h3>💰 Predicted Annual Medical Cost</h3>
                <div class="prediction-amount">${:,.2f}</div>
                <p>Based on your health profile and demographic information</p>
            </div>
            """.format(prediction), unsafe_allow_html=True)

            # Health Profile Summary
            st.markdown("### 📊 Health Profile Summary")
            patient_data = st.session_state['input_data']

            # Create metrics display
            metrics_data = {
                "Basic Info": [
                    ("Age", f"{patient_data['age']} years"),
                    ("Gender", patient_data['gender']),
                    ("BMI", f"{patient_data['bmi']:.1f}"),
                    ("Location", patient_data['city_type'])
                ],
                "Health Conditions": [
                    ("Smoker", patient_data['smoker']),
                    ("Diabetes", patient_data['diabetes']),
                    ("Hypertension", patient_data['hypertension']),
                    ("Heart Disease", patient_data['heart_disease']),
                    ("Asthma", patient_data['asthma'])
                ],
                "Lifestyle": [
                    ("Activity Level", patient_data['physical_activity_level']),
                    ("Daily Steps", f"{patient_data['daily_steps']:,}"),
                    ("Sleep Hours", f"{patient_data['sleep_hours']:.1f}"),
                    ("Stress Level", f"{patient_data['stress_level']}/10")
                ],
                "Insurance": [
                    ("Type", patient_data['insurance_type']),
                    ("Coverage", f"{patient_data['insurance_coverage_pct']}%"),
                    ("Previous Cost", f"${patient_data['previous_year_cost']:,.2f}")
                ]
            }

            # Create liquid glass containers for all metrics
            st.markdown('<div class="liquid-glass-grid">', unsafe_allow_html=True)
            
            # Flatten all metrics into a single list
            all_metrics = []
            for category, metrics in metrics_data.items():
                for metric, value in metrics:
                    all_metrics.append((metric, value))
            
            # Generate liquid glass containers for each metric
            for metric, value in all_metrics:
                fill_percentage, status = calculate_metric_fill(metric, value, patient_data)
                container_html = generate_liquid_glass_container(metric, value, fill_percentage, status)
                st.markdown(container_html, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Also show the categories as section headers with traditional metrics for reference
            st.markdown("---")
            st.markdown("### 📋 Detailed Health Metrics")
            cols = st.columns(2)
            for i, (category, metrics) in enumerate(metrics_data.items()):
                with cols[i % 2]:
                    st.markdown(f"**{category}**")
                    for metric, value in metrics:
                        st.markdown(f'<div class="health-metric"><strong>{metric}:</strong> {value}</div>', unsafe_allow_html=True)

            # Personalized Suggestions
            if suggestions:
                st.markdown("### 🎯 Personalized Health Recommendations")
                for i, suggestion in enumerate(suggestions, 1):
                    st.markdown(f'<div class="suggestion-item"><strong>{i}.</strong> {suggestion}</div>', unsafe_allow_html=True)

            # Insurance Information
            st.markdown("### 🏦 Insurance Coverage Analysis")
            coverage_pct = patient_data['insurance_coverage_pct']
            out_of_pocket = prediction * (1 - coverage_pct/100)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Insurance Coverage", f"{coverage_pct}%")
            with col2:
                st.metric("Estimated Out-of-Pocket", f"${out_of_pocket:,.2f}")

            st.markdown("**Recommended Insurance Provider:**")
            st.markdown("🔗 [Blue Cross Blue Shield](https://www.bcbs.com/) - A trusted provider with comprehensive coverage")

            # Save report to user data
            report_data = {
                'input_data': st.session_state['input_data'],
                'prediction': prediction,
                'suggestions': suggestions
            }
            user_manager.save_report(st.session_state.patient_id, report_data)

            # Print Instructions
            st.markdown("---")
            st.markdown("""
            ### 🖨️ Save as PDF
            **To save this report as a PDF:**
            1. Press `Ctrl + P` (or `Cmd + P` on Mac)
            2. Select 'Save as PDF' or 'Print to PDF'
            3. Choose your preferred settings and save

            This will create a professional PDF version of your personalized health report!
            """)

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
