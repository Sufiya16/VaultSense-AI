const API_BASE = "http://127.0.0.1:8000";

async function assessRisk(payload) {
    const response = await fetch(`${API_BASE}/assess`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });
    return await response.json();
}

async function getPortfolio() {
    const response = await fetch(`${API_BASE}/portfolio`);
    return await response.json();
}

async function getGovernance() {
    const response = await fetch(`${API_BASE}/governance`);
    return await response.json();
}