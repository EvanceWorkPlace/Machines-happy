const form = document.getElementById("input-form");
const predictedSpan = document.getElementById("predictedValue");
const volatilitySpan = document.getElementById("volatility");
const confidenceSpan = document.getElementById("confidence");
const historyList = document.getElementById("historyList");

let lastPrediction = null;

// SUBMIT CURRENT VALUE
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const input = document.getElementById("currentValue");
  const value = parseFloat(input.value);

  if (isNaN(value)) return;

  const res = await fetch("/api/predict/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ multiplier: value })
  });

  const data = await res.json();

  if (data.predicted_value !== null) {
    predictedSpan.innerText = data.predicted_value.toFixed(2);
    confidenceSpan.innerText = data.confidence;
  } else {
    predictedSpan.innerText = "waiting for the data";
  }

  volatilitySpan.innerText = data.volatility;

  lastPrediction = data.predicted_value;

  input.value = "";
  document.getElementById("result-check").style.display = "block";
});


// CHECK ACTUAL RESULT
document.getElementById("checkBtn").addEventListener("click", async () => {
  const actualInput = document.getElementById("actualValue");
  const actual = parseFloat(actualInput.value);

  if (isNaN(actual) || lastPrediction === null) return;

  const res = await fetch("/api/check/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      predicted: lastPrediction,
      actual: actual
    })
  });
  async function loadStats() {
  const res = await fetch("/api/stats/");
  const data = await res.json();

  document.getElementById("accuracy").innerText = data.accuracy;
  document.getElementById("streak").innerText = data.streak;
  document.getElementById("streakType").innerText = data.streak_type || "—";
  }

  loadStats();
  
  const data = await res.json();

  // ADD TO HISTORY LIST
  const li = document.createElement("li");
  li.className = "list-group-item";
  li.innerHTML = `
    ⏱ ${data.time} |
    Pred: ${data.predicted.toFixed(2)} →
    Actual: ${data.actual.toFixed(2)} |
    <strong>${data.status}</strong>
  `;
  historyList.prepend(li);

  document.getElementById("resultStatus").innerText =
    `${data.status} (Δ ${data.difference})`;

  // CLEAR INPUTS
  actualInput.value = "";
  async function loadChart() {
  const res = await fetch("/api/chart/");
  const d = await res.json();

  new Chart(document.getElementById("predictionChart"), {
    type: "line",
    data: {
      labels: d.labels,
      datasets: [
        { label: "Predicted", data: d.predicted, borderWidth: 2 },
        { label: "Actual", data: d.actual, borderWidth: 2 }
      ]
    }
  });
}
});


