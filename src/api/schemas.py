from pydantic import BaseModel
from typing import Optional


class LoanApplication(BaseModel):
    customer_id: str
    customer_name: str

    age: int
    gender: str
    education: str
    state: str
    urban_rural: str

    employment_type: str
    employment_years: int
    annual_income_inr: float

    loan_type: str
    loan_purpose: str
    loan_amount_inr: float
    loan_tenure_months: int
    interest_rate_pct: float

    credit_score: int
    num_existing_loans: int
    dti_ratio: float
    ltv_ratio: Optional[float] = None
    has_collateral: int
    bureau_enquiries_6m: int
    missed_payments_2y: int
    savings_account_balance_inr: float

    co_applicant_available: int = 0
    nominee_available: int = 0