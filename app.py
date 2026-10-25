from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ================= LOAD MODELS ================= #

# Diabetes models
diabetes_model = joblib.load("models/best_classification_model.pkl")
diabetes_preprocessor = joblib.load("models/preprocessor.pkl")

# Heart Failure models
hf_model = joblib.load("models/best_classification_model_hf.pkl")
hf_preprocessor = joblib.load("models/preprocessor_hf.pkl")

# Chronic Kidney Disease models

ckd_preprocessor = joblib.load("models/ckd_preprocessor.pkl")
ckd_model = joblib.load("models/ckd_model.pkl")

# ================= DIABETES HELPERS (UNCHANGED) ================= #

def get_severity_level(sev):
    if sev <= 15:
        return 'Low Severity'
    elif sev <= 30:
        return 'Moderate Severity'
    else:
        return 'High Severity'

def get_readmission(sev):
    if sev <= 15:
        return 'No Re-admission Needed'
    elif sev <= 30:
        return 'Monitor Closely'
    else:
        return 'Re-admission Recommended'

def get_lifestyle_suggestion(status, sev):
    if status == 'Non-Diabetic':
        return [
            'Maintain healthy lifestyle with balanced diet and regular exercise.',
            'Keep monitoring fasting glucose periodically.',
            'Stay active and manage weight.'
        ]
    else:
        if sev <= 15:
            return [
                'Maintain balanced diet with more fruits and fibre.',
                'Continue mild exercise (20–30 minutes/day).'
            ]
        elif sev <= 30:
            return [
                'Reduce sugar and processed food intake.',
                'Increase monitoring (weekly).',
                'Do moderate exercise (30–45 minutes/day).'
            ]
        else:
            return [
                'Immediate medical consultation.',
                'Strict diet and daily glucose monitoring.',
                'High-intensity supervised exercise as advised by doctor.'
            ]

def get_insurance_plan(status, sev):
    if status == 'Non-Diabetic':
        plan = 'No insurance needed – focus on healthy lifestyle'
        reason = ['Patient is non-diabetic, maintain healthy lifestyle.']
    else:
        if sev <= 15:
            plan = 'HealthGuard Basic Diabetes Plan'
            reason = ['Severity is low, basic coverage is sufficient.']
        elif sev <= 30:
            plan = 'WellLife Diabetes Care Plan'
            reason = [
                'Severity level is moderate.',
                'Requires periodic monitoring.',
                'Covers common diabetes-related medical visits.'
            ]
        else:
            plan = 'PremiumDiab Complications Coverage Plan'
            reason = [
                'High severity, risk of complications.',
                'Needs comprehensive coverage for treatment.',
                'Includes specialist visits and advanced therapies.'
            ]
    return plan, reason

# ================= HEART FAILURE HELPERS ================= #

def get_hf_severity_level(prob):
    if prob <= 30:
        return 'Low Severity'
    elif prob <= 60:
        return 'Moderate Severity'
    else:
        return 'High Severity'

def get_hf_readmission(prob):
    if prob <= 30:
        return 'Low chance of readmission'
    elif prob <= 60:
        return 'Moderate chance – monitor closely'
    else:
        return 'High chance – re-admission recommended'

def get_hf_lifestyle_suggestion(sev):
    if sev == 'Low Severity':
        return [
            'Maintain low-sodium diet',
            'Light physical activity (walking)',
            'Regular blood pressure monitoring'
        ]
    elif sev == 'Moderate Severity':
        return [
            'Strict sodium restriction',
            'Daily weight monitoring',
            'Moderate exercise as advised by doctor'
        ]
    else:
        return [
            'Immediate cardiologist consultation',
            'Strict medication adherence',
            'Possible hospitalization and close supervision'
        ]

def get_hf_insurance_plan(sev):
    if sev == 'Low Severity':
        return 'Basic Cardiac Care Plan', ['Covers routine cardiac checkups']
    elif sev == 'Moderate Severity':
        return 'Advanced Heart Care Plan', [
            'Covers diagnostics and frequent follow-ups'
        ]
    else:
        return 'Premium Heart Failure Coverage', [
            'ICU coverage',
            'Specialist consultations',
            'Advanced cardiac treatments'
        ]

# ================= CKD HELPERS ================= #

def get_ckd_severity_label(cls):
    return {
        0: "Mild CKD",
        1: "Moderate CKD",
        2: "Severe CKD"
    }[cls]

def get_readmission_label(prob):
    if prob <= 30:
        return "Low Risk of Readmission"
    elif prob <= 60:
        return "Moderate Risk – Monitor Closely"
    else:
        return "High Risk – Re-admission Recommended"

def get_ckd_lifestyle(sev):
    if sev == "Mild CKD":
        return [
            "Maintain healthy diet",
            "Drink adequate water",
            "Annual kidney check-up"
        ]
    elif sev == "Moderate CKD":
        return [
            "Low sodium & protein diet",
            "Strict BP and sugar control",
            "Frequent kidney monitoring"
        ]
    else:
        return [
            "Immediate nephrologist consultation",
            "Strict fluid & diet restriction",
            "Prepare for dialysis if advised"
        ]

def get_ckd_insurance(sev):
    if sev == "Mild CKD":
        return "Basic Renal Care Plan", ["Routine nephrology visits"]
    elif sev == "Moderate CKD":
        return "Advanced Kidney Care Plan", [
            "Covers diagnostics",
            "Specialist consultations"
        ]
    else:
        return "Comprehensive Renal Protection Plan", [
            "Dialysis coverage",
            "Hospitalization & ICU support"
        ]

# ================= ROUTES ================= #

@app.route("/")
def home():
    return render_template("home.html")

# ---------- DIABETES ---------- #

@app.route("/index")
def diabetes_index():
    return render_template("index_D.html")

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():

    user_input = {
        'Age': float(request.form['Age']),
        'Gender': request.form['Gender'],
        'Ethnicity': request.form['Ethnicity'],
        'Family_History_Diabetes': int(request.form['Family_History_Diabetes']),
        'BMI': float(request.form['BMI']),
        'Systolic_BP': float(request.form['Systolic_BP']),
        'Diastolic_BP': float(request.form['Diastolic_BP']),
        'Waist_Circumference': float(request.form['Waist_Circumference']),
        'Physical_Activity': int(request.form['Physical_Activity']),
        'Smoking_Status': request.form['Smoking_Status'],
        'Alcohol_Consumption': request.form['Alcohol_Consumption'],
        'Fasting_Glucose_mg_dL': float(request.form['Fasting_Glucose_mg_dL']),
        'HbA1c_Percent': float(request.form['HbA1c_Percent'])
    }

    all_features = diabetes_preprocessor.feature_names_in_
    df = pd.DataFrame([{f: user_input.get(f, 0) for f in all_features}])
    processed = diabetes_preprocessor.transform(df)

    prob = diabetes_model.predict_proba(processed)[0][1] * 100
    sev = prob * 0.5339
    status = 'Diabetic' if prob >= 50 else 'Non-Diabetic'

    insurance, insurance_reason = get_insurance_plan(status, sev)

    return render_template(
        "result.html",
        back_url="/index",
        status=status,
        confidence=round(prob, 2),
        severity=round(sev, 2),
        severity_level=get_severity_level(sev),
        readmission=get_readmission(sev),
        lifestyle=get_lifestyle_suggestion(status, sev),
        insurance=insurance,
        insurance_reason=insurance_reason
    )

# ---------- HEART FAILURE ---------- #

@app.route("/index_hf")
def heart_failure_index():
    return render_template("index_hf.html")

@app.route("/predict_hf", methods=["POST"])
def predict_heart_failure():

    user_input = {
        'Age': float(request.form['Age']),
        'Gender': request.form['Gender'],
        'Ethnicity': request.form['Ethnicity'],
        'BMI': float(request.form['BMI']),
        'Systolic_BP': float(request.form['Systolic_BP']),
        'Diastolic_BP': float(request.form['Diastolic_BP']),
        'Resting_Heart_Rate': float(request.form['Resting_Heart_Rate']),
        'Respiratory_Rate': float(request.form['Respiratory_Rate']),
        'Smoking_Status': request.form['Smoking_Status'],
        'Alcohol_Consumption': request.form['Alcohol_Consumption'],
        'Serum_Creatinine': float(request.form['Serum_Creatinine']),
        'eGFR': float(request.form['eGFR']),
        'BUN': float(request.form['BUN']),
        'Albumin': float(request.form['Albumin']),
        'Proteinuria': request.form['Proteinuria'],
        'HbA1c': float(request.form['HbA1c']),
        'Fasting_Glucose': float(request.form['Fasting_Glucose']),
        'Diabetes_Status': request.form['Diabetes_Status'],

        # defaults for labs not asked from user
        'UACR': 30,
        'Potassium': 4.2,
        'Sodium': 140,
        'Phosphate': 3.5,
        'Calcium': 9.5
    }

    all_features = hf_preprocessor.feature_names_in_
    df = pd.DataFrame([{f: user_input.get(f, 0) for f in all_features}])
    processed = hf_preprocessor.transform(df)

    prob = hf_model.predict_proba(processed)[0][1] * 100
    sev_level = get_hf_severity_level(prob)
    insurance, insurance_reason = get_hf_insurance_plan(sev_level)

    return render_template(
        "result.html",
        back_url="/index_hf",
        status="Heart Failure Risk",
        confidence=round(prob, 2),
        severity=round(prob, 2),
        severity_level=sev_level,
        readmission=get_hf_readmission(prob),
        lifestyle=get_hf_lifestyle_suggestion(sev_level),
        insurance=insurance,
        insurance_reason=insurance_reason
    )

# ---------- CHRONIC KIDNEY DISEASE ---------- #

@app.route("/index_ckd")
def ckd_index():
    return render_template("index_ckd.html")

@app.route("/predict_ckd", methods=["POST"])
def predict_ckd():

    # -------- 1. COLLECT FORM INPUT -------- #
    user_input = {
        "Age": float(request.form["Age"]),
        "Gender": request.form["Gender"],
        "Ethnicity": request.form.get("Ethnicity", "Unknown"),
        "BMI": float(request.form["BMI"]),
        "Serum_Creatinine": float(request.form["Serum_Creatinine"]),
        "eGFR": float(request.form["eGFR"]),
        "BUN": float(request.form["BUN"]),
        "Albumin": float(request.form["Albumin"]),
        "UACR": float(request.form["UACR"]),
        "Proteinuria": int(request.form["Proteinuria"]),
        "Potassium": float(request.form["Potassium"]),
        "Sodium": float(request.form["Sodium"]),
        "Phosphate": float(request.form["Phosphate"]),
        "Calcium": float(request.form["Calcium"]),
        "HbA1c": float(request.form["HbA1c"]),
        "Fasting_Glucose": float(request.form["Fasting_Glucose"]),
        "Diabetes_Status": request.form["Diabetes_Status"],
        "Systolic_BP": float(request.form["Systolic_BP"]),
        "Diastolic_BP": float(request.form["Diastolic_BP"]),
        "Smoking_Status": request.form["Smoking_Status"],
        "Alcohol_Consumption": request.form["Alcohol_Consumption"]
    }

    # -------- 2. DATAFRAME WITH EXACT FEATURE ORDER -------- #
    df = pd.DataFrame([user_input])
    df = df[ckd_preprocessor.feature_names_in_]

    # -------- 3. PREPROCESS -------- #
    X = ckd_preprocessor.transform(df)

    # -------- 4. PREDICT -------- #
    severity_class = ckd_model.predict(X)[0]
    readmission_prob = ckd_model.predict_proba(X)[0][1] * 100

    severity = get_ckd_severity_label(severity_class)
    readmission = get_readmission_label(readmission_prob)
    lifestyle = get_ckd_lifestyle(severity)
    insurance, insurance_reason = get_ckd_insurance(severity)

    # -------- 5. RENDER RESULT -------- #
    return render_template(
        "result.html",
        condition="Chronic Kidney Disease",
        severity=severity,
        confidence=f"{readmission_prob:.2f}%",
        readmission=readmission,
        lifestyle=lifestyle,
        insurance=insurance,
        insurance_reason=insurance_reason
    )
# ================= CHATBOT ROUTES ================= #

CHATBOT_DATA = {
    "ckd": {
        "symptoms": "Common CKD symptoms include fatigue, swelling of legs, frequent urination, and nausea.",
        "causes": "Major causes include diabetes, high blood pressure, and long-term kidney infections.",
        "prevention": "Control BP and sugar, drink water, avoid excessive painkillers.",
        "treatment": "Treatment includes medication, diet control, dialysis in severe cases.",
        "severity": "CKD severity ranges from Mild → Moderate → Severe based on kidney function."
    },
    "hf": {
        "symptoms": "Shortness of breath, fatigue, swollen ankles, rapid heartbeat.",
        "causes": "Caused by hypertension, coronary artery disease, diabetes.",
        "prevention": "Low salt diet, exercise, BP control, no smoking.",
        "treatment": "Medications, lifestyle changes, sometimes surgery.",
        "severity": "Heart failure severity is Low, Moderate, or High."
    },
    "diabetes": {
        "symptoms": "Frequent urination, excessive thirst, fatigue, blurred vision.",
        "causes": "Insulin resistance, genetics, unhealthy lifestyle.",
        "prevention": "Healthy diet, regular exercise, weight control.",
        "treatment": "Medication, insulin therapy, diet management.",
        "severity": "Severity depends on glucose levels and complications."
    }
}

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/chatbot_response", methods=["POST"])
def chatbot_response():
    data = request.json
    disease = data.get("disease")
    question = data.get("question")

    if disease and question:
        answer = CHATBOT_DATA[disease].get(question, "Sorry, I don't have information on that.")
        return {"reply": answer}

    return {"reply": "Please select an option."}

# ================= RUN ================= #

if __name__ == "__main__":
    app.run(debug=True)