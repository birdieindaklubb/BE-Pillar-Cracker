"use strict";

// The exact fixed PE 1.1.5 ring order used by pe115_pillarcracker.
const pillars = [
  { x: 42, z: 0 }, { x: 33, z: 24 }, { x: 12, z: 39 },
  { x: -12, z: 39 }, { x: -33, z: 24 }, { x: -42, z: 0 },
  { x: -33, z: -24 }, { x: -12, z: -39 }, { x: 12, z: -39 },
  { x: 33, z: -24 },
];

const ring = document.querySelector("#ring");
const picker = document.querySelector("#height-picker");
const selectionTitle = document.querySelector("#selection-title");
const selectionDetail = document.querySelector("#selection-detail");
const completion = document.querySelector("#completion");
const validation = document.querySelector("#validation");
const command = document.querySelector("#command");
const copyButton = document.querySelector("#copy");
const copyStatus = document.querySelector("#copy-status");
const scanMode = document.querySelector("#scan-mode");
const selectorInputs = document.querySelector("#selector-inputs");

// Shapes are 0 through 9, where the feature height is 76 + 3 * shape.
const observations = Array(10).fill(null);
let selectedPillar = null;
let measurementMode = "feature";

function measuredHeight(shape) {
  const featureHeight = 76 + 3 * shape;
  return measurementMode === "feature" ? featureHeight : featureHeight - 1;
}

function heightLabel(shape) {
  const suffix = measurementMode === "feature" ? "feature" : "obsidian";
  return `${measuredHeight(shape)} ${suffix}`;
}

function usedByAnotherPillar(shape) {
  return observations.some((value, index) => index !== selectedPillar && value === shape);
}

function makePillarButton(site, index) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "pillar";
  button.dataset.index = String(index);
  button.style.left = `${50 + site.x * 0.88}%`;
  button.style.top = `${50 + site.z * 0.88}%`;
  button.addEventListener("click", () => {
    selectedPillar = index;
    copyStatus.textContent = "";
    render();
  });
  ring.append(button);
  return button;
}

const pillarButtons = pillars.map(makePillarButton);

function renderPillars() {
  pillarButtons.forEach((button, index) => {
    const site = pillars[index];
    const value = observations[index];
    button.classList.toggle("selected", index === selectedPillar);
    button.classList.toggle("entered", value !== null);
    button.setAttribute("aria-pressed", String(index === selectedPillar));
    const valueText = value === null ? "Choose height" : heightLabel(value);
    button.innerHTML = `<span class="index">Pillar ${index}</span>`
      + `<span class="value">${valueText}</span>`
      + `<span class="coord">${site.x}, ${site.z}</span>`;
  });
}

function renderSelection() {
  picker.replaceChildren();
  if (selectedPillar === null) {
    selectionTitle.textContent = "Select a pillar";
    selectionDetail.textContent = "Choose one on the diagram to enter a height.";
    for (let shape = 0; shape < 10; ++shape) {
      const button = document.createElement("button");
      button.type = "button";
      button.disabled = true;
      button.textContent = measuredHeight(shape);
      picker.append(button);
    }
    return;
  }

  const site = pillars[selectedPillar];
  selectionTitle.textContent = `Pillar ${selectedPillar} at (${site.x}, ${site.z})`;
  selectionDetail.textContent = "Choose the measured Y value. A value already used by another pillar is unavailable.";
  for (let shape = 0; shape < 10; ++shape) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = measuredHeight(shape);
    button.title = heightLabel(shape);
    button.classList.toggle("selected", observations[selectedPillar] === shape);
    button.disabled = usedByAnotherPillar(shape);
    button.addEventListener("click", () => {
      observations[selectedPillar] = shape;
      copyStatus.textContent = "";
      render();
    });
    picker.append(button);
  }
}

function parseUnsigned(text, maximum) {
  const trimmed = text.trim();
  if (!/^(?:0x[0-9a-f]+|[0-9]+)$/i.test(trimmed)) {
    return null;
  }
  try {
    const value = BigInt(trimmed);
    return value <= maximum ? value : null;
  } catch {
    return null;
  }
}

function asHex(value) {
  return `0x${value.toString(16).toUpperCase()}`;
}

function renderSelectorInputs() {
  selectorInputs.replaceChildren();
  const mode = scanMode.value;
  if (mode === "all") {
    return;
  }

  const addInput = (id, label, hint) => {
    const wrapper = document.createElement("div");
    wrapper.className = "selector-input";
    const inputLabel = document.createElement("label");
    inputLabel.htmlFor = id;
    inputLabel.textContent = label;
    const input = document.createElement("input");
    input.id = id;
    input.type = "text";
    input.inputMode = "text";
    input.placeholder = "0x… or decimal";
    input.addEventListener("input", () => updateCommand());
    const small = document.createElement("small");
    small.textContent = hint;
    wrapper.append(inputLabel, input, small);
    selectorInputs.append(wrapper);
  };

  if (mode === "high16" || mode === "low16") {
    addInput("word", mode === "high16" ? "High 16 bits" : "Low 16 bits",
      "0 through 65535; this is a filter from another observation.");
  } else {
    addInput("range-start", "Range start", "First unsigned 32-bit seed.");
    addInput("range-count", "Range count", "Positive count that stays within the 32-bit seed space.");
  }
}

function selectorArgument() {
  const mode = scanMode.value;
  if (mode === "all") {
    return { value: "--all" };
  }
  if (mode === "high16" || mode === "low16") {
    const value = parseUnsigned(document.querySelector("#word").value, 0xffffn);
    if (value === null) {
      return { error: "Enter a valid 16-bit value for the selected seed filter." };
    }
    return { value: `--${mode} ${asHex(value)}` };
  }
  const start = parseUnsigned(document.querySelector("#range-start").value, 0xffff_ffffn);
  const count = parseUnsigned(document.querySelector("#range-count").value, 0x1_0000_0000n);
  if (start === null || count === null || count === 0n || start + count > 0x1_0000_0000n) {
    return { error: "Enter a non-empty range that stays within unsigned 32-bit seeds." };
  }
  return { value: `--range-start ${asHex(start)} --count ${asHex(count)}` };
}

function updateCommand() {
  const entered = observations.filter((height) => height !== null).length;
  completion.textContent = `${entered} / 10 pillars entered`;
  command.value = "";
  copyButton.disabled = true;
  validation.classList.remove("invalid");

  if (entered !== 10) {
    validation.textContent = "Enter every distinct pillar height to generate a command.";
    return;
  }
  const selector = selectorArgument();
  if (selector.error) {
    validation.textContent = selector.error;
    validation.classList.add("invalid");
    return;
  }
  const values = observations.map(measuredHeight).join(",");
  const heightFlag = measurementMode === "feature" ? "--heights" : "--obsidian-tops";
  command.value = `.\\build\\Release\\pe115_pillarcracker.exe ${heightFlag} ${values} ${selector.value}`;
  validation.textContent = "Ready. The command searches locally and reports every matching PE 1.1.5 seed.";
  copyButton.disabled = false;
}

function render() {
  renderPillars();
  renderSelection();
  updateCommand();
}

document.querySelectorAll('input[name="measure-mode"]').forEach((input) => {
  input.addEventListener("change", () => {
    measurementMode = document.querySelector('input[name="measure-mode"]:checked').value;
    copyStatus.textContent = "";
    render();
  });
});

scanMode.addEventListener("change", () => {
  copyStatus.textContent = "";
  renderSelectorInputs();
  updateCommand();
});

document.querySelector("#clear").addEventListener("click", () => {
  observations.fill(null);
  selectedPillar = null;
  copyStatus.textContent = "Heights cleared.";
  render();
});

document.querySelector("#copy").addEventListener("click", async () => {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(command.value);
    } else {
      command.focus();
      command.select();
      if (!document.execCommand("copy")) {
        throw new Error("browser copy command was rejected");
      }
    }
    copyStatus.textContent = "Command copied to the clipboard.";
  } catch {
    copyStatus.textContent = "Copy was blocked by the browser; select the command and copy it manually.";
    copyStatus.classList.add("invalid");
  }
});

renderSelectorInputs();
render();
