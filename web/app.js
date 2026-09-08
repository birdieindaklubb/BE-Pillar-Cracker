"use strict";

// Fixed PE 1.1.5 ring order. Height, radius, and cage state all describe one
// shuffled native shape; they are not independent random choices.
const pillars = [
  { x: 42, z: 0 }, { x: 33, z: 24 }, { x: 12, z: 39 },
  { x: -12, z: 39 }, { x: -33, z: 24 }, { x: -42, z: 0 },
  { x: -33, z: -24 }, { x: -12, z: -39 }, { x: 12, z: -39 },
  { x: 33, z: -24 },
];
const ALL_SHAPES = (1 << 10) - 1;
const CAGED_SHAPES = (1 << 1) | (1 << 2);
const observations = Array.from({ length: 10 }, () => ({
  height: null, radius: null, caged: null,
}));
const ring = document.querySelector("#ring");
const heightPicker = document.querySelector("#height-picker");
const radiusPicker = document.querySelector("#radius-picker");
const cagePicker = document.querySelector("#cage-picker");
const selectionTitle = document.querySelector("#selection-title");
const selectionDetail = document.querySelector("#selection-detail");
const completion = document.querySelector("#completion");
const validation = document.querySelector("#validation");
const command = document.querySelector("#command");
const copyButton = document.querySelector("#copy");
const copyStatus = document.querySelector("#copy-status");
const scanMode = document.querySelector("#scan-mode");
const selectorInputs = document.querySelector("#selector-inputs");
let selectedPillar = null;
let measurementMode = "feature";

function measuredHeight(shape) {
  return 76 + 3 * shape - (measurementMode === "obsidian" ? 1 : 0);
}
function radiusMask(radius) {
  let mask = 0;
  for (let shape = 0; shape < 10; ++shape) {
    if (Math.floor(shape / 3) + 2 === radius) mask |= 1 << shape;
  }
  return mask;
}
function allowedShapes(observation) {
  let mask = ALL_SHAPES;
  if (observation.height !== null) mask &= 1 << observation.height;
  if (observation.radius !== null) mask &= radiusMask(observation.radius);
  if (observation.caged !== null) {
    mask &= observation.caged ? CAGED_SHAPES : ALL_SHAPES & ~CAGED_SHAPES;
  }
  return mask;
}
function exactShape(observation) {
  const mask = allowedShapes(observation);
  return mask === 0 || (mask & (mask - 1)) !== 0 ? null : Math.log2(mask);
}
function hasConstraint(observation) {
  return observation.height !== null
    || observation.radius !== null || observation.caged !== null;
}
function usedByAnotherPillar(shape) {
  return observations.some((observation, index) => index !== selectedPillar
    && exactShape(observation) === shape);
}
function actionButton(label, selected, disabled, onClick, secondary = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.classList.toggle("selected", selected);
  button.classList.toggle("secondary-choice", secondary);
  button.disabled = disabled;
  button.addEventListener("click", onClick);
  return button;
}
function makePillarButton(site, index) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "pillar";
  button.style.left = (50 + site.x * 0.88) + "%";
  button.style.top = (50 + site.z * 0.88) + "%";
  button.addEventListener("click", () => {
    selectedPillar = index;
    copyStatus.textContent = "";
    render();
  });
  ring.append(button);
  return button;
}
const pillarButtons = pillars.map(makePillarButton);

function observationSummary(observation) {
  const values = [];
  if (observation.height !== null) values.push("H " + measuredHeight(observation.height));
  if (observation.radius !== null) values.push("R " + observation.radius);
  if (observation.caged !== null) values.push(observation.caged ? "cage" : "no cage");
  return values.length === 0 ? "Add observation" : values.join(" · ");
}
function renderPillars() {
  pillarButtons.forEach((button, index) => {
    const site = pillars[index];
    const observation = observations[index];
    button.classList.toggle("selected", index === selectedPillar);
    button.classList.toggle("entered", hasConstraint(observation));
    button.setAttribute("aria-pressed", String(index === selectedPillar));
    button.innerHTML = '<span class="index">Pillar ' + index + "</span>"
      + '<span class="value">' + observationSummary(observation) + "</span>"
      + '<span class="coord">' + site.x + ", " + site.z + "</span>";
  });
}
function emptyPickers() {
  heightPicker.replaceChildren();
  radiusPicker.replaceChildren();
  cagePicker.replaceChildren();
}
function renderSelection() {
  emptyPickers();
  if (selectedPillar === null) {
    selectionTitle.textContent = "Select a pillar";
    selectionDetail.textContent = "Choose one to record any known height, radius, or cage property.";
    [heightPicker, radiusPicker, cagePicker].forEach((picker) => {
      picker.append(actionButton("Select a pillar first", false, true, () => {}));
    });
    return;
  }
  const site = pillars[selectedPillar];
  const observation = observations[selectedPillar];
  selectionTitle.textContent = "Pillar " + selectedPillar + " at (" + site.x + ", " + site.z + ")";
  selectionDetail.textContent = "Add any measurements you know. Impossible PE 1.1.5 combinations are unavailable.";
  for (let shape = 0; shape < 10; ++shape) {
    const compatible = (allowedShapes({ ...observation, height: shape }) & (1 << shape)) !== 0;
    heightPicker.append(actionButton(String(measuredHeight(shape)),
      observation.height === shape, !compatible || usedByAnotherPillar(shape), () => {
        observation.height = shape;
        copyStatus.textContent = "";
        render();
      }));
  }
  heightPicker.append(actionButton("Clear height", observation.height === null, false, () => {
    observation.height = null;
    copyStatus.textContent = "";
    render();
  }, true));
  for (const radius of [2, 3, 4, 5]) {
    const proposed = { ...observation, radius };
    const compatible = allowedShapes(proposed) !== 0;
    const forbidden = !compatible
      || (exactShape(proposed) !== null && usedByAnotherPillar(exactShape(proposed)));
    radiusPicker.append(actionButton("Radius " + radius, observation.radius === radius,
      forbidden, () => {
        observation.radius = radius;
        copyStatus.textContent = "";
        render();
      }));
  }
  radiusPicker.append(actionButton("Unknown radius", observation.radius === null, false, () => {
    observation.radius = null;
    copyStatus.textContent = "";
    render();
  }, true));
  [["Caged", true], ["No cage", false]].forEach(([label, caged]) => {
    const proposed = { ...observation, caged };
    const compatible = allowedShapes(proposed) !== 0;
    const forbidden = !compatible
      || (exactShape(proposed) !== null && usedByAnotherPillar(exactShape(proposed)));
    cagePicker.append(actionButton(label, observation.caged === caged,
      forbidden, () => {
      observation.caged = caged;
      copyStatus.textContent = "";
      render();
    }));
  });
  cagePicker.append(actionButton("Unknown cage", observation.caged === null, false, () => {
    observation.caged = null;
    copyStatus.textContent = "";
    render();
  }, true));
}

function parseUnsigned(text, maximum) {
  const trimmed = text.trim();
  if (!/^(?:0x[0-9a-f]+|[0-9]+)$/i.test(trimmed)) return null;
  try {
    const value = BigInt(trimmed);
    return value <= maximum ? value : null;
  } catch {
    return null;
  }
}
function asHex(value) {
  return "0x" + value.toString(16).toUpperCase();
}
function renderSelectorInputs() {
  selectorInputs.replaceChildren();
  const mode = scanMode.value;
  if (mode === "all") return;
  const addInput = (id, label, hint) => {
    const wrapper = document.createElement("div");
    wrapper.className = "selector-input";
    const inputLabel = document.createElement("label");
    inputLabel.htmlFor = id;
    inputLabel.textContent = label;
    const input = document.createElement("input");
    input.id = id;
    input.type = "text";
    input.placeholder = "0x… or decimal";
    input.addEventListener("input", updateCommand);
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
    addInput("range-count", "Range count", "Positive count within the 32-bit seed space.");
  }
}
function selectorArgument() {
  const mode = scanMode.value;
  if (mode === "all") return { value: "--all" };
  if (mode === "high16" || mode === "low16") {
    const value = parseUnsigned(document.querySelector("#word").value, 0xffffn);
    if (value === null) return { error: "Enter a valid 16-bit value for the selected seed filter." };
    return { value: "--" + mode + " " + asHex(value) };
  }
  const start = parseUnsigned(document.querySelector("#range-start").value, 0xffff_ffffn);
  const count = parseUnsigned(document.querySelector("#range-count").value, 0x1_0000_0000n);
  if (start === null || count === null || count === 0n || start + count > 0x1_0000_0000n) {
    return { error: "Enter a non-empty range that stays within unsigned 32-bit seeds." };
  }
  return { value: "--range-start " + asHex(start) + " --count " + asHex(count) };
}
function observationArguments() {
  const result = [];
  observations.forEach((observation, index) => {
    const site = pillars[index];
    const coordinate = site.x + "," + site.z;
    if (observation.height !== null) {
      const fields = [coordinate, measuredHeight(observation.height)];
      if (observation.radius !== null) fields.push(observation.radius);
      if (observation.caged !== null && observation.radius !== null) {
        fields.push(observation.caged ? "CAGED" : "UNCAGED");
      }
      result.push((measurementMode === "feature" ? "--pillar " : "--pillar-top ")
        + fields.join(","));
      if (observation.caged !== null && observation.radius === null) {
        result.push("--pillar-cage " + coordinate + "," + (observation.caged ? "CAGED" : "UNCAGED"));
      }
    } else {
      if (observation.radius !== null) result.push("--pillar-radius " + coordinate + "," + observation.radius);
      if (observation.caged !== null) result.push("--pillar-cage " + coordinate + "," + (observation.caged ? "CAGED" : "UNCAGED"));
    }
  });
  return result;
}
function updateCommand() {
  const constrained = observations.filter(hasConstraint).length;
  const directHeights = observations.filter((observation) => observation.height !== null).length;
  completion.textContent = constrained + " / 10 pillars constrained · " + directHeights + " direct heights";
  command.value = "";
  copyButton.disabled = true;
  validation.classList.remove("invalid");
  if (constrained === 0) {
    validation.textContent = "Record at least one height, radius, or cage observation before generating a command.";
    return;
  }
  const selector = selectorArgument();
  if (selector.error) {
    validation.textContent = selector.error;
    validation.classList.add("invalid");
    return;
  }
  command.value = ".\\build\\Release\\pe115_pillarcracker.exe "
    + observationArguments().join(" ") + " " + selector.value;
  validation.textContent = directHeights === 10
    ? "Ready. Fully resolved heights retain the optimized inverse-shuffle scan."
    : constrained < 10
      ? "Ready, but this is a partial layout and may return very many seeds; use a range or seed-word filter where possible."
      : "Ready. Radius/cage observations use the exact generic shuffle path and can leave many candidates.";
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
  observations.forEach((observation) => {
    observation.height = null;
    observation.radius = null;
    observation.caged = null;
  });
  selectedPillar = null;
  copyStatus.textContent = "Observations cleared.";
  render();
});
document.querySelector("#copy").addEventListener("click", async () => {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(command.value);
    } else {
      command.focus();
      command.select();
      if (!document.execCommand("copy")) throw new Error("browser copy command was rejected");
    }
    copyStatus.textContent = "Command copied to the clipboard.";
  } catch {
    copyStatus.textContent = "Copy was blocked by the browser; select the command and copy it manually.";
    copyStatus.classList.add("invalid");
  }
});
renderSelectorInputs();
render();
