async function loadGovernance() {
    const data = await getGovernance();

    document.getElementById("modelInfo").innerHTML = `
        <p><b>Model Name:</b> ${data.model_name}</p>
        <p><b>Model Version:</b> ${data.model_version}</p>
        <p><b>Model Type:</b> ${data.model_type}</p>
        <p><b>Decision Threshold:</b> ${data.threshold}</p>
        <p>
            This system separates model prediction from the final credit decision.
            The model estimates default probability, while policy rules apply
            additional banking controls.
        </p>
        <pre>${JSON.stringify(data.metrics, null, 2)}</pre>
    `;

    document.getElementById("governanceNotes").innerHTML =
        data.governance_notes
            .map(note => `<div class="reason">${note}</div>`)
            .join("");

    document.getElementById("shapPlots").innerHTML = `
     <h3>Global SHAP Feature Importance</h3>

     <img
        src="./assets/shap_bar.png"
        class="shap-img"
        alt="SHAP Feature Importance"
     />

    <h3 style="margin-top:30px;">
        SHAP Summary Plot
    </h3>

    <img
        src="./assets/shap_summary.png"
        class="shap-img"
        alt="SHAP Summary"
    />
`;

    const audits = data.audit_records || [];

    if (audits.length === 0) {
        document.getElementById("auditTable").innerHTML =
            "No audit records available yet.";
        return;
    }

    let html = `
        <table>
            <tr>
                <th>Time</th>
                <th>Customer</th>
                <th>Customer Snapshot</th>
                <th>Model Output</th>
                <th>Policy Decision</th>
                <th>Audit Status</th>
            </tr>
    `;

    audits.reverse().forEach(r => {
        html += `
            <tr>
                <td>${r.timestamp}</td>

                <td>
                    ${r.customer_name}<br>
                    <small>${r.customer_id}</small>
                </td>

                <td>
                    Loan: ${r.loan_type}<br>
                    Amount: INR ${Number(r.loan_amount_inr).toLocaleString()}<br>
                    Income: INR ${Number(r.annual_income_inr).toLocaleString()}<br>
                    Credit Score: ${r.credit_score}<br>
                    DTI: ${r.dti_ratio}
                </td>

                <td>
                    PD: ${r.risk_percentage}%<br>
                    Risk Band: ${r.risk_band}<br>
                    ML Decision: ${r.ml_decision}<br>
                    Model: ${r.model_version}
                </td>

                <td>
                    Final Decision: ${r.final_decision}<br>
                    Rules: ${r.policy_rules_triggered}
                </td>

                <td>${r.audit_status}</td>
            </tr>
        `;
    });

    html += "</table>";

    document.getElementById("auditTable").innerHTML = html;
}