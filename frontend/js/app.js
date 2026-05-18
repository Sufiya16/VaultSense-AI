let chartInstance = null;

function getNumber(id) {
    const value = document.getElementById(id).value;

    if (value === "" || isNaN(value)) {
        alert(`Please enter a valid value for ${id}.`);
        throw new Error(`Invalid value: ${id}`);
    }

    return Number(value);
}

function getText(id) {
    const value = document.getElementById(id).value.trim();

    if (value === "") {
        alert(`Please enter ${id}.`);
        throw new Error(`Missing value: ${id}`);
    }

    return value;
}

function getSelect(id) {
    const value = document.getElementById(id).value;

    if (value === "") {
        alert(`Please select ${id}.`);
        throw new Error(`Missing selection: ${id}`);
    }

    return value;
}

async function submitApplication() {
    const payload = {
        customer_id: getText("customer_id"),
        customer_name: getText("customer_name"),

        age: getNumber("age"),
        gender: getSelect("gender"),
        education: getSelect("education"),
        state: "MH",
        urban_rural: "Urban",

        employment_type: "Salaried",
        employment_years: getNumber("employment_years"),
        annual_income_inr: getNumber("income"),

        loan_type: getSelect("loan_type"),
        loan_purpose: "General",
        loan_amount_inr: getNumber("loan_amount"),
        loan_tenure_months: 24,
        interest_rate_pct: 10,

        credit_score: getNumber("credit_score"),
        num_existing_loans: 1,
        dti_ratio: getNumber("dti"),
        ltv_ratio: null,
        has_collateral: Number(getSelect("has_collateral")),

        bureau_enquiries_6m: getNumber("bureau"),
        missed_payments_2y: getNumber("missed"),
        savings_account_balance_inr: 50000,

        co_applicant_available: Number(getSelect("co_applicant_available")),
        nominee_available: Number(getSelect("nominee_available"))
    };

    const result = await assessRisk(payload);

    const riskClass =
        result.risk_band === "HIGH" ? "high" :
        result.risk_band === "MEDIUM" ? "medium" : "low";

    document.getElementById("result").innerHTML = `
        <p><b>Customer:</b> ${result.customer_name} (${result.customer_id})</p>

        <div class="metric ${riskClass}">
            ${result.risk_percentage}%
        </div>

        <p><b>Risk Classification:</b> ${result.risk_band}</p>
        <p><b>ML Recommendation:</b> ${result.ml_decision}</p>
        <p><b>Final Credit Decision:</b> ${result.decision}</p>

        <p><b>Recommendation:</b><br>${result.recommendation}</p>

        <p><b>Model Version:</b> ${result.model_version}</p>
        <p><b>Threshold Used:</b> ${result.threshold_used}</p>
    `;

    document.getElementById("warnings").innerHTML = `
        <h3>Model Reason Codes</h3>
        ${result.reason_codes.map(reason => `<div class="reason">${reason}</div>`).join("")}

        <h3>Policy Rules Triggered</h3>
        ${result.policy_rules_triggered.map(rule => `<div class="reason">${rule}</div>`).join("")}
    `;

    const ctx = document.getElementById("riskChart");

    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Default Risk", "Remaining Confidence"],
            datasets: [{
                data: [
                    result.risk_percentage,
                    100 - result.risk_percentage
                ],
                backgroundColor: [
                    "#3b82f6",
                    "#64748b"
                ]
            }]
        },
        options: {
            plugins: {
                legend: {
                    labels: {
                        color: "white"
                    }
                }
            }
        }
    });
}