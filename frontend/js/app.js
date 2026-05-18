async function predictRisk() {

    const payload = {

        customer_name:
            document.getElementById("customer_name").value,

        customer_id:
            document.getElementById("customer_id").value,

        age:
            parseInt(document.getElementById("age").value),

        gender:
            document.getElementById("gender").value,

        state:
            document.getElementById("state").value,

        urban_rural:
            document.getElementById("urban_rural").value,

        employment_type:
            document.getElementById("employment_type").value,

        employment_years:
            parseInt(document.getElementById("employment_years").value),

        annual_income_inr:
            parseFloat(document.getElementById("income").value),

        loan_type:
            document.getElementById("loan_type").value,

        loan_purpose:
            "General",

        loan_amount_inr:
            parseFloat(document.getElementById("loan_amount").value),

        loan_tenure_months:
            parseInt(document.getElementById("loan_tenure").value),

        interest_rate_pct:
            parseFloat(document.getElementById("interest_rate").value),

        credit_score:
            parseInt(document.getElementById("credit_score").value),

        num_existing_loans:
            parseInt(document.getElementById("existing_loans").value),

        dti_ratio:
            parseFloat(document.getElementById("dti").value),

        ltv_ratio:
            parseFloat(document.getElementById("ltv").value),

        savings_account_balance_inr:
            parseFloat(document.getElementById("savings").value),

        has_collateral:
            document.getElementById("collateral").value,
        co_applicant_available:
       parseInt(document.getElementById("co_applicant").value),

       nominee_available:
    parseInt(document.getElementById("nominee").value),
        bureau_enquiries_6m:
            parseInt(document.getElementById("bureau").value),

        missed_payments_2y:
            parseInt(document.getElementById("missed").value)
    };

    const response = await fetch(
        `${API_BASE}/assess`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        }
    );

    const result = await response.json();

    document.getElementById("output").innerHTML = `

        <h3>${result.decision}</h3>

        <p>
            <strong>Risk Score:</strong>
            ${result.risk_percentage}%
        </p>

        <p>
            <strong>Risk Band:</strong>
            ${result.risk_band}
        </p>

        <p>
            <strong>Recommendation:</strong>
            ${result.recommendation}
        </p>
    `;

    document.getElementById("reasons").innerHTML =
        result.reason_codes
        .map(w => `<div class="reason">⚠ ${w}</div>`)
        .join("");
}