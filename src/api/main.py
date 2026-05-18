import os
import json
import joblib
import pandas as pd
import warnings

from fastapi import FastAPI
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.schemas import LoanApplication
from src.training.feature_engineering import create_features

warnings.filterwarnings("ignore")

app = FastAPI(
    title="VaultSense AI",
    description="AI Powered Credit Risk Decisioning Platform",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/best_model.pkl")
threshold = joblib.load("models/threshold.pkl")

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "decisions.csv")

if os.path.exists("reports"):
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")


@app.get("/")
def root():
    return {
        "status": "VaultSense AI Running",
        "model_status": "Loaded",
        "version": "1.0"
    }


@app.get("/health")
def health():
    return {"api": "healthy"}


def get_risk_band(probability):
    if probability < 0.25:
        return "LOW"
    elif probability < 0.50:
        return "MEDIUM"
    return "HIGH"


def ml_decision(probability):
    if probability < 0.40:
        return "APPROVE"
    elif probability < 0.65:
        return "MANUAL REVIEW"
    return "REJECT"


def generate_reason_codes(data):
    reasons = []
    secured = data["has_collateral"] == 1

    if data["credit_score"] < 650:
        if secured:
            reasons.append(
                "Credit score is below preferred policy levels, but collateral partially reduces recovery risk."
            )
        else:
            reasons.append(
                "Low bureau credit score increases default risk, especially for unsecured lending."
            )

    if data["dti_ratio"] > 0.50:
        reasons.append(
            "High debt-to-income ratio indicates repayment stress."
        )

    if data["bureau_enquiries_6m"] >= 5:
        reasons.append(
            "Frequent recent bureau enquiries may indicate credit hunger."
        )

    if data["missed_payments_2y"] >= 2:
        reasons.append(
            "Past missed EMI payments indicate weak repayment behaviour."
        )

    if data["employment_years"] < 1:
        reasons.append(
            "Low employment stability increases income uncertainty."
        )

    if data["loan_amount_inr"] > data["annual_income_inr"] * 3:
        reasons.append(
            "Requested loan amount is high compared to annual income and requires policy review."
        )

    if data["has_collateral"] == 0 and data["loan_type"] in ["Personal_Loan", "MSME_Loan"]:
        reasons.append(
            "Unsecured exposure increases recovery risk."
        )

    if data["co_applicant_available"] == 0 and data["loan_amount_inr"] > 1000000:
        reasons.append(
            "Large loan without a co-applicant increases repayment dependency risk."
        )

    if data["nominee_available"] == 0:
        reasons.append(
            "Nominee details are missing and should be reviewed before disbursal."
        )

    if not reasons:
        reasons.append(
            "Applicant profile currently shows stable credit characteristics."
        )

    return reasons


def apply_policy_rules(data, base_decision):
    rules_triggered = []
    final_decision = base_decision

    if data["credit_score"] < 580:
        final_decision = "REJECT"
        rules_triggered.append(
            "Policy decline: credit score is below minimum acceptable bureau threshold."
        )

    if data["missed_payments_2y"] >= 4:
        final_decision = "REJECT"
        rules_triggered.append(
            "Policy decline: severe repayment delinquency observed in bureau history."
        )

    if data["loan_amount_inr"] > data["annual_income_inr"] * 3:
        if final_decision == "APPROVE":
            final_decision = "MANUAL REVIEW"
        rules_triggered.append(
            "Policy review: requested exposure exceeds three times annual income."
        )

    if data["dti_ratio"] > 0.55:
        if final_decision == "APPROVE":
            final_decision = "MANUAL REVIEW"
        rules_triggered.append(
            "Policy review: debt-to-income ratio exceeds internal comfort level."
        )

    if data["loan_type"] == "MSME_Loan" and data["has_collateral"] == 0:
        if final_decision == "APPROVE":
            final_decision = "MANUAL REVIEW"
        rules_triggered.append(
            "Policy review: unsecured MSME exposure requires additional credit scrutiny."
        )

    if data["loan_amount_inr"] > 1000000 and data["co_applicant_available"] == 0:
        if final_decision == "APPROVE":
            final_decision = "MANUAL REVIEW"
        rules_triggered.append(
            "Policy review: high-value loan without co-applicant requires officer review."
        )

    return final_decision, rules_triggered


def generate_recommendation(final_decision):
    if final_decision == "APPROVE":
        return (
            "Proceed with standard underwriting, subject to document verification, "
            "KYC validation, and routine credit officer approval."
        )

    if final_decision == "MANUAL REVIEW":
        return (
            "Send to credit officer for manual review. Verify income, bureau history, "
            "repayment capacity, collateral adequacy, and co-applicant strength."
        )

    return (
        "Application should not proceed without strong mitigation. Consider rejection "
        "or approval only with lower exposure, collateral, senior approval, and documented justification."
    )


def save_decision_log(
    application,
    probability,
    risk_band,
    ml_dec,
    final_dec,
    reasons,
    policy_rules,
    recommendation
):
    os.makedirs(LOG_DIR, exist_ok=True)

    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "customer_id": application.customer_id,
        "customer_name": application.customer_name,
        "age": application.age,
        "gender": application.gender,
        "education": application.education,
        "state": application.state,
        "urban_rural": application.urban_rural,

        "employment_type": application.employment_type,
        "employment_years": application.employment_years,
        "annual_income_inr": application.annual_income_inr,

        "loan_type": application.loan_type,
        "loan_purpose": application.loan_purpose,
        "loan_amount_inr": application.loan_amount_inr,
        "loan_tenure_months": application.loan_tenure_months,
        "interest_rate_pct": application.interest_rate_pct,

        "credit_score": application.credit_score,
        "num_existing_loans": application.num_existing_loans,
        "dti_ratio": application.dti_ratio,
        "ltv_ratio": application.ltv_ratio if application.ltv_ratio is not None else "",
        "has_collateral": application.has_collateral,
        "bureau_enquiries_6m": application.bureau_enquiries_6m,
        "missed_payments_2y": application.missed_payments_2y,
        "savings_account_balance_inr": application.savings_account_balance_inr,

        "co_applicant_available": application.co_applicant_available,
        "nominee_available": application.nominee_available,

        "probability_of_default": round(float(probability), 4),
        "risk_percentage": round(float(probability * 100), 2),
        "risk_band": risk_band,
        "ml_decision": ml_dec,
        "final_decision": final_dec,

        "reason_codes": " | ".join(reasons),
        "policy_rules_triggered": " | ".join(policy_rules) if policy_rules else "No policy rule override triggered.",
        "recommendation": recommendation,

        "model_version": "VaultSense-v1.0",
        "threshold_used": round(float(threshold), 4),
        "audit_status": "Logged for governance review"
    }

    log_df = pd.DataFrame([log_data])

    if os.path.exists(LOG_PATH):
        log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_PATH, index=False)


@app.post("/assess")
def assess_risk(application: LoanApplication):
    raw = application.dict()

    model_input = pd.DataFrame([raw])

    model_input = model_input.drop(
        columns=[
            "customer_id",
            "customer_name",
            "co_applicant_available",
            "nominee_available"
        ],
        errors="ignore"
    )

    model_input = create_features(model_input)

    probability = float(model.predict_proba(model_input)[0][1])

    risk_band = get_risk_band(probability)
    base_ml_decision = ml_decision(probability)
    final_decision, policy_rules = apply_policy_rules(raw, base_ml_decision)
    reasons = generate_reason_codes(raw)
    recommendation = generate_recommendation(final_decision)

    save_decision_log(
        application,
        probability,
        risk_band,
        base_ml_decision,
        final_decision,
        reasons,
        policy_rules,
        recommendation
    )

    return {
        "customer_id": application.customer_id,
        "customer_name": application.customer_name,
        "probability_of_default": round(probability, 4),
        "risk_percentage": round(probability * 100, 2),
        "risk_band": risk_band,
        "ml_decision": base_ml_decision,
        "decision": final_decision,
        "reason_codes": reasons,
        "policy_rules_triggered": policy_rules if policy_rules else ["No policy rule override triggered."],
        "recommendation": recommendation,
        "model_version": "VaultSense-v1.0",
        "threshold_used": round(float(threshold), 4)
    }


@app.get("/portfolio")
def get_portfolio():
    if not os.path.exists(LOG_PATH):
        return {
            "total_records": 0,
            "records": []
        }

    df = pd.read_csv(LOG_PATH)
    df = df.fillna("")

    return {
        "total_records": len(df),
        "records": df.tail(100).to_dict(orient="records")
    }


@app.get("/governance")
def get_governance():
    metrics_path = "reports/model_metrics.json"

    metrics = {}

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as file:
            metrics = json.load(file)

    audit_records = []

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = df.fillna("")
        audit_records = df.tail(50).to_dict(orient="records")

    return {
        "model_name": "VaultSense Credit Risk Engine",
        "model_version": "VaultSense-v1.0",
        "model_type": "Best selected from Logistic Regression, XGBoost, and LightGBM",
        "threshold": round(float(threshold), 4),
        "metrics": metrics,
        "shap_summary_url": "http://127.0.0.1:8000/reports/shap_summary.png",
        "shap_bar_url": "http://127.0.0.1:8000/reports/shap_bar.png",
        "audit_records": audit_records,
        "governance_notes": [
            "The ML model estimates Probability of Default.",
            "The final decision combines model score with internal credit policy rules.",
            "Policy rules may convert an approval into manual review or rejection.",
            "The audit trail stores the customer snapshot, model output, policy rules, final decision, model version, and timestamp.",
            "Credit officers must document overrides and high-risk approvals.",
            "Sensitive attributes should be monitored for fairness and compliance."
        ]
    }