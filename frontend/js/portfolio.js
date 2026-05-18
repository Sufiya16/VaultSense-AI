async function loadPortfolio() {
    const data = await getPortfolio();
    const records = data.records;

    if (!records || records.length === 0) {
        document.getElementById("portfolioTable").innerHTML =
            "No loan decisions have been recorded yet. Submit applications first.";
        return;
    }

    let html = `
        <table>
            <tr>
                <th>Time</th>
                <th>Customer</th>
                <th>Loan Type</th>
                <th>Amount</th>
                <th>Credit Score</th>
                <th>Risk %</th>
                <th>ML Decision</th>
                <th>Final Decision</th>
                <th>Policy Rules</th>
            </tr>
    `;

    records.reverse().forEach(r => {
        html += `
            <tr>
                <td>${r.timestamp}</td>
                <td>${r.customer_name}<br><small>${r.customer_id}</small></td>
                <td>${r.loan_type}</td>
                <td>INR ${Number(r.loan_amount_inr).toLocaleString()}</td>
                <td>${r.credit_score}</td>
                <td>${r.risk_percentage}%</td>
                <td>${r.ml_decision}</td>
                <td>${r.final_decision}</td>
                <td>${r.policy_rules_triggered}</td>
            </tr>
        `;
    });

    html += "</table>";

    document.getElementById("portfolioTable").innerHTML = html;
}