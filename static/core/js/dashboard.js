
  const form = document.getElementById("result-form");
  const suggestionBody = document.getElementById("suggestion-body");
  const ctx = document.getElementById("chart").getContext("2d");

  let chart;
  const WINDOW = 10;

  function rollingStats(data, window) {
    const mean = [];
    const upper = [];
    const lower = [];

    for (let i = 0; i < data.length; i++) {
      if (i < window - 1) {
        mean.push(null);
        upper.push(null);
        lower.push(null);
        continue;
      }

      const slice = data.slice(i - window + 1, i + 1);
      const avg = slice.reduce((a, b) => a + b, 0) / slice.length;

      const variance =
        slice.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / slice.length;
      const std = Math.sqrt(variance);

      mean.push(avg);
      upper.push(avg + std);
      lower.push(avg - std);
    }

    return { mean, upper, lower };
  }

  async function fetchResults() {
    const res = await fetch("/api/results/");
    return await res.json();
  }

  async function fetchSuggestion() {
    const res = await fetch("/api/suggestion/");
    return await res.json();
  }

  async function renderChart() {
    const results = await fetchResults();
    const values = results.map(r => r.multiplier);
    const labels = values.map((_, i) => i + 1);

    const stats = rollingStats(values, WINDOW);

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Multiplier",
            data: values,
            borderWidth: 2,
            tension: 0.3
          },
          {
            label: "Rolling Mean",
            data: stats.mean,
            borderDash: [5, 5],
            borderWidth: 2
          },
          {
            label: "Upper Volatility Band",
            data: stats.upper,
            borderWidth: 1,
            borderDash: [2, 2]
          },
          {
            label: "Lower Volatility Band",
            data: stats.lower,
            borderWidth: 1,
            borderDash: [2, 2]
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "top" }
        },
        scales: {
          x: {
            title: { display: true, text: "Sequence Order" }
          },
          y: {
            title: { display: true, text: "Multiplier" }
          }
        }
      }
    });
  }

  async function renderSuggestion() {
    const data = await fetchSuggestion();
    if (data.streaks?.length) {
      suggestionBody.innerHTML += `
        <p><strong>Detected Patterns:</strong></p>
        <ul>
          ${data.streaks.map(s =>
            `<li>${s.type} (${s.length})</li>`
          ).join("")}
        </ul>
      `;
  }  
}

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    await fetch("/api/results/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        round_id: form.round_id.value,
        multiplier: parseFloat(form.multiplier.value)
      })
    });

    form.reset();
    renderChart();
    renderSuggestion();
  });

  renderChart();
  renderSuggestion();


  const heatCtx = document.getElementById("heatmap").getContext("2d");
  let heatChart;

  async function renderHeatmap() {
    const res = await fetch("/api/volatility-heatmap/");
    const data = await res.json();

    const values = data.volatility;
    const labels = values.map((_, i) => i + 1);

    const colors = values.map(v => {
      if (v === null) return "rgba(100,100,100,0.3)";
      if (v < 0.5) return "rgba(34,197,94,0.8)";     // low
      if (v < 1.5) return "rgba(250,204,21,0.8)";    // medium
      return "rgba(239,68,68,0.8)";                  // high
    });

    if (heatChart) heatChart.destroy();

    heatChart = new Chart(heatCtx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Volatility (Std Dev)",
          data: values,
          backgroundColor: colors
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { display: false },
          y: { title: { display: true, text: "σ" } }
        }
      }
    });
  }

  // hook into existing refresh cycle
  const oldRenderChart = renderChart;
  renderChart = async function () {
    await oldRenderChart();
    await renderHeatmap();
  };

  renderHeatmap();