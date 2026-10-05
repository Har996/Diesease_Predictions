/* Symptom picker + prediction request. Works without JS too: the form falls back to a normal POST. */
(function () {
  "use strict";

  const form = document.getElementById("predict-form");
  if (!form) return;

  const search = document.getElementById("symptom-search");
  const items = Array.from(document.querySelectorAll("#symptom-list .symptom"));
  const emptyMsg = document.getElementById("symptom-empty");
  const counter = document.getElementById("symptom-count");
  const chips = document.getElementById("selected-chips");
  const clearBtn = document.getElementById("symptom-clear");
  const errorBox = document.getElementById("form-error");
  const loading = document.getElementById("loading");
  const submitBtn = document.getElementById("predict-btn");

  const checked = () => items.map((el) => el.querySelector("input")).filter((i) => i.checked);

  function filter() {
    const q = search.value.trim().toLowerCase();
    let visible = 0;
    items.forEach((el) => {
      const show = !q || el.dataset.label.includes(q);
      el.hidden = !show;
      if (show) visible += 1;
    });
    emptyMsg.hidden = visible !== 0;
  }

  function renderSelection() {
    const sel = checked();
    counter.textContent = sel.length + " selected";
    chips.replaceChildren();
    if (!sel.length) {
      const hint = document.createElement("span");
      hint.className = "text-muted-ink small";
      hint.textContent = "Nothing selected yet.";
      chips.appendChild(hint);
      return;
    }
    sel.forEach((input) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.append(input.nextElementSibling.textContent + " ");
      const remove = document.createElement("button");
      remove.type = "button";
      remove.setAttribute("aria-label", "Remove " + input.nextElementSibling.textContent);
      remove.textContent = "\u00d7";
      remove.addEventListener("click", () => { input.checked = false; renderSelection(); });
      chip.appendChild(remove);
      chips.appendChild(chip);
    });
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  search.addEventListener("input", filter);
  search.addEventListener("keydown", (e) => { if (e.key === "Enter") e.preventDefault(); });
  items.forEach((el) => el.querySelector("input").addEventListener("change", renderSelection));
  clearBtn.addEventListener("click", () => { items.forEach((el) => { el.querySelector("input").checked = false; }); renderSelection(); });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.hidden = true;

    const symptoms = checked().map((i) => i.value);
    if (!symptoms.length) { showError("Select at least one symptom."); return; }

    const payload = { symptoms };
    const age = form.querySelector("#age");
    const gender = form.querySelector("#gender");
    if (age && age.value !== "") payload.age = Number(age.value);
    if (gender && gender.value) payload.gender = gender.value;

    loading.hidden = false;
    submitBtn.disabled = true;
    try {
      const response = await fetch(form.action, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "The server could not process your request.");
      window.location.assign(data.result_url);
    } catch (err) {
      loading.hidden = true;
      submitBtn.disabled = false;
      showError(err instanceof TypeError ? "Could not reach the server. Check your connection and try again." : err.message);
    }
  });

  // Back/forward cache: never leave the overlay stuck on screen.
  window.addEventListener("pageshow", () => { loading.hidden = true; submitBtn.disabled = false; renderSelection(); });
  renderSelection();
})();
