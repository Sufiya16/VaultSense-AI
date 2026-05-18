import pandas as pd
import numpy as np


def create_features(df):

    df = df.copy()

    # =========================
    # LTV HANDLING
    # =========================

    df['ltv_missing_flag'] = df['ltv_ratio'].isnull().astype(int)

    df['ltv_ratio'] = df['ltv_ratio'].fillna(-1)

    # =========================
    # COMPETITION FEATURES
    # =========================

    df['loan_to_income_ratio'] = (
        df['loan_amount_inr']
        /
        (df['annual_income_inr'] + 1)
    )

    df['dti_credit_risk'] = (
        df['dti_ratio']
        /
        (df['credit_score'] / 700)
    )

    df['income_per_year_employed'] = (
        df['annual_income_inr']
        /
        (df['employment_years'] + 1)
    )

    # =========================
    # REAL WORLD BANK FEATURES
    # =========================

    df['bureau_risk_score'] = (
        df['bureau_enquiries_6m'] * 0.4
        +
        df['missed_payments_2y'] * 0.6
    )

    df['financial_stability_index'] = (
        df['savings_account_balance_inr']
        /
        (df['loan_amount_inr'] + 1)
    )

    df['existing_loan_burden'] = (
        df['num_existing_loans']
        *
        df['dti_ratio']
    )

    df['income_to_tenure_ratio'] = (
        df['annual_income_inr']
        /
        (df['loan_tenure_months'] + 1)
    )

    return df