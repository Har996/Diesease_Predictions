/* Model dashboard: interactive charts (Chart.js) and confusion-matrix heat map. */
(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("metrics-data").textContent);
  const PRIMARY = "#4636C9", SOFT = "#9A8FEA", INK = "#1D1A3B";

  // ---- confusion matrix (plain HTML table so it needs no library and is keyboard/screen-reader friendly)
  (function renderConfusion() {
    const host = document.getElementById("confusion");
    const labels = data.test.labels, cm = data.test.confusion_matrix;
    const table = document.createElement("table");
    table.className = "confusion";
    const head = table.createTHead().insertRow();
    head.appendChild(document.createElement("th"));
    labels.forEach((l) => { const th = document.createElement("th"); th.scope = "col"; const s = document.createElement("span"); s.textContent = l; th.appendChild(s); head.appendChild(th); });
    const body = table.createTBody();
    cm.forEach((row, i) => {
      const tr = body.insertRow();
      const th = document.createElement("th"); th.scope = "row"; th.textContent = labels[i]; tr.appendChild(th);
      const total = row.reduce((a, b) => a + b, 0) || 1;
      row.forEach((value, j) => {
        const td = tr.insertCell();
        const share = value / total;                       // row-normalised colour = recall view
        td.textContent = value || "";
        td.style.background = value ? "rgba(70,54,201," + (0.12 + 0.88 * share).toFixed(2) + ")" : "#fff";
        td.style.color = share > 0.55 ? "#fff" : INK;
        td.title = "Actual: " + labels[i] + " | Predicted: " + labels[j] + " | " + value + " of " + total + " (" + (share * 100).toFixed(0) + "%)";
        if (i === j) td.style.outline = "1px solid " + PRIMARY;
      });
    });
    host.appendChild(table);
  })();

  if (typeof Chart === "undefined") {
    ["chart-models", "chart-classes", "chart-importance"].forEach((id) => {
      const box = document.getElementById(id).parentElement;
      box.textContent = "Interactive charts need the Chart.js script (internet access). Static figures are available below.";
      box.style.height = "auto";
    });
    document.querySelector("details").open = true;
    return;
  }
  Chart.defaults.font.family = '"Public Sans", system-ui, sans-serif';
  Chart.defaults.color = INK;

  new Chart(document.getElementById("chart-models"), {
    type: "bar",
    data: {
      labels: data.candidates.map((c) => c.name),
      datasets: [
        { label: "Accuracy", data: data.candidates.map((c) => c.val.accuracy), backgroundColor: PRIMARY },
        { label: "Macro F1", data: data.candidates.map((c) => c.val.f1_macro), backgroundColor: SOFT },
      ],
    },
    options: { maintainAspectRatio: false, scales: { y: { min: 0, max: 1 } } },
  });

  const classes = Object.entries(data.dataset.class_counts);
  new Chart(document.getElementById("chart-classes"), {
    type: "bar",
    data: { labels: classes.map((c) => c[0]), datasets: [{ label: "Records", data: classes.map((c) => c[1]), backgroundColor: PRIMARY }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } } },
  });

  const imp = data.feature_importance.permutation.slice(0, 20);
  new Chart(document.getElementById("chart-importance"), {
    type: "bar",
    data: { labels: imp.map((f) => f.feature.replace(/_/g, " ")), datasets: [{ label: "Drop in macro-F1 when shuffled", data: imp.map((f) => f.importance), backgroundColor: PRIMARY }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } } },
  });
})();
