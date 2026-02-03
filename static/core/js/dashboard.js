async function fetchResults() {
  const res = await fetch('/api/results/');
  return res.json();
}

async function fetchSuggestion() {
  const res = await fetch('/api/suggestion/');
  return res.json();
}

function buildChart(ctx, data) {
  const labels = data.map((d) => new Date(d.timestamp).toLocaleString());
  const values = data.map((d) => d.multiplier);
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{ label: 'Multiplier', data: values, borderColor: 'blue', tension: 0.2 }]
    },
    options: { responsive: true }
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  const form = document.getElementById('result-form');
  const ctx = document.getElementById('chart').getContext('2d');
  let chart;

  async function reload() {
    const data = await fetchResults();
    if (chart) chart.destroy();
    chart = buildChart(ctx, data.reverse());
    const s = await fetchSuggestion();
    const body = document.getElementById('suggestion-body');
    body.innerText = `Recommended cashout: ${s.suggestion.recommended_min} - ${s.suggestion.recommended_max} (confidence ${s.suggestion.confidence})`;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const payload = {
      round_id: formData.get('round_id') || null,
      multiplier: parseFloat(formData.get('multiplier'))
    };
    await fetch('/api/results/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    form.reset();
    await reload();
  });

  await reload();
});
