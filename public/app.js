"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const MAX_FILE_BYTES = 4 * 1024 * 1024;
const MAX_DRAWN_ATOMS = 3000;
const MAX_DRAWN_BOND = 2.5;
const RESIDUE_FIELDS = ["phi", "psi", "omega", "o_bond", "o_angle", "o_dihedral"];
const RESIDUE_DEFAULTS = { phi: -57, psi: -47, omega: 180, o_bond: 1.231, o_angle: 120.5, o_dihedral: "" };
const RESIDUE_LABELS = {
  phi: "phi", psi: "psi", omega: "omega",
  o_bond: "C=O bond length", o_angle: "CA-C-O angle", o_dihedral: "N-CA-C-O dihedral",
};
const CSV_COLUMNS = {
  phi: "phi", psi: "psi", omega: "omega",
  o_bond: "c_o_bond", o_angle: "ca_c_o_angle", o_dihedral: "n_ca_c_o_dihedral",
};
const THREE_TO_ONE = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G", HIS: "H", ILE: "I",
  LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V",
};
const ONE_TO_THREE = Object.fromEntries(Object.entries(THREE_TO_ONE).map(([three, one]) => [one, three]));

const BACKBONE_CYCLE = ["N", "CA", "C", "O"];
const DEFAULT_ATOMS = [...BACKBONE_CYCLE, ...BACKBONE_CYCLE];
const AXES = ["x", "y", "z"];

const state = {
  mode: null,
  downloads: [],
  structure: null,
  view: { yaw: 0.6, pitch: -0.35 },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmt(value, digits) {
  return value === null || value === undefined ? "-" : value.toFixed(digits);
}

function sentence(text) {
  return text.charAt(0).toUpperCase() + text.slice(1) + (/[.!?]$/.test(text) ? "" : ".");
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("Could not reach the conversion server. Check your connection and try again.");
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    let detail = data && data.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("; ");
    throw new Error(detail ? sentence(detail) : `The server returned an error (${response.status}).`);
  }
  return data;
}

function showError(message) {
  const box = $("#error");
  box.textContent = message;
  box.hidden = false;
}

function hideError() {
  $("#error").hidden = true;
}

function cell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

function fillTable(table, headers, rows) {
  table.replaceChildren();
  const thead = table.createTHead();
  const headRow = thead.insertRow();
  for (const { label, num, kind } of headers) {
    const th = document.createElement("th");
    if (kind) {
      const kindEl = document.createElement("span");
      kindEl.className = "col-kind";
      kindEl.textContent = kind;
      const nameEl = document.createElement("span");
      nameEl.className = "col-name";
      nameEl.textContent = label;
      th.append(kindEl, nameEl);
    } else {
      th.textContent = label;
    }
    if (num) th.classList.add("num");
    headRow.appendChild(th);
  }
  const tbody = table.createTBody();
  const fragment = document.createDocumentFragment();
  for (const row of rows) {
    const tr = document.createElement("tr");
    row.forEach((value, i) => {
      const className = [headers[i].num ? "num" : "", value === "-" ? "dim" : "", headers[i].cls ? headers[i].cls(value) : ""]
        .filter(Boolean).join(" ");
      tr.appendChild(cell(value, className));
    });
    fragment.appendChild(tr);
  }
  tbody.appendChild(fragment);
}

// ---------------------------------------------------------------------------
// Mode selection
// ---------------------------------------------------------------------------

function setMode(mode) {
  state.mode = mode;
  for (const button of $$(".choice")) {
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  }
  $("#panel-cartesian").hidden = mode !== "cartesian";
  $("#panel-internal").hidden = mode !== "internal";
  $("#results").hidden = true;
  hideError();
}

function selectTab(tab) {
  for (const other of tab.closest("[role='tablist']").querySelectorAll("[role='tab']")) {
    const selected = other === tab;
    other.setAttribute("aria-selected", String(selected));
    other.tabIndex = selected ? 0 : -1;
    document.getElementById(other.getAttribute("aria-controls")).hidden = !selected;
  }
  $("#results").hidden = true;
  hideError();
}

function setupTabs() {
  for (const list of $$("[role='tablist']")) {
    const tabs = Array.from(list.querySelectorAll("[role='tab']"));
    for (const tab of tabs) tab.addEventListener("click", () => selectTab(tab));
    list.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const current = tabs.indexOf(document.activeElement);
      if (current < 0) return;
      const next = tabs[(current + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      selectTab(next);
      next.focus();
    });
  }
}

// ---------------------------------------------------------------------------
// Atom-by-atom Cartesian form
// ---------------------------------------------------------------------------

function atomRows() {
  return $$("#atom-table tbody tr");
}

function nextBackboneAtom() {
  const names = atomRows().map((row) => row.querySelector("[data-field='name']").value.trim().toUpperCase());
  const lastBackbone = names.reverse().find((name) => BACKBONE_CYCLE.includes(name));
  if (!lastBackbone) return "N";
  return BACKBONE_CYCLE[(BACKBONE_CYCLE.indexOf(lastBackbone) + 1) % BACKBONE_CYCLE.length];
}

function refreshAtomTable() {
  let residue = 0;
  atomRows().forEach((row, i) => {
    const name = row.querySelector("[data-field='name']").value.trim().toUpperCase();
    if (i === 0 || name === "N") residue += 1;
    row.querySelector(".row-number").textContent = String(i + 1);
    row.querySelector(".row-residue").textContent = String(residue);
    const label = name || "atom";
    row.querySelector("[data-field='name']").setAttribute("aria-label", `Atom ${i + 1} name`);
    for (const axis of AXES) {
      row.querySelector(`[data-field='${axis}']`).setAttribute("aria-label", `Atom ${i + 1} (${label}) ${axis} coordinate`);
    }
    row.querySelector(".remove").setAttribute("aria-label", `Remove atom ${i + 1} (${label})`);
  });
  $("#add-next-atom").textContent = `Add next atom: ${nextBackboneAtom()}`;
}

function makeInput(field, value, placeholder) {
  const input = document.createElement("input");
  input.type = "text";
  input.dataset.field = field;
  input.value = value;
  input.placeholder = placeholder;
  input.autocomplete = "off";
  input.spellcheck = false;
  if (field !== "name") input.inputMode = "decimal";
  return input;
}

function addAtomRow(name = "", coords = ["", "", ""]) {
  const tr = document.createElement("tr");
  const number = cell("", "num row-number");
  const residue = cell("", "row-residue");
  tr.append(number, residue);

  const nameCell = document.createElement("td");
  nameCell.appendChild(makeInput("name", name, "name"));
  tr.appendChild(nameCell);

  AXES.forEach((axis, i) => {
    const td = document.createElement("td");
    td.appendChild(makeInput(axis, coords[i], axis));
    tr.appendChild(td);
  });

  const removeCell = document.createElement("td");
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "remove";
  remove.textContent = "×";
  remove.addEventListener("click", () => {
    if (atomRows().length > 1) tr.remove();
    else tr.querySelectorAll("input").forEach((input) => { if (input.dataset.field !== "name") input.value = ""; });
    refreshAtomTable();
  });
  removeCell.appendChild(remove);
  tr.appendChild(removeCell);

  $("#atom-table tbody").appendChild(tr);
  refreshAtomTable();
  return tr;
}

function resetAtomTable(atoms) {
  $("#atom-table tbody").replaceChildren();
  for (const atom of atoms) addAtomRow(atom.name, atom.coords);
}

function focusNextInput(input) {
  const inputs = $$("#atom-table tbody input");
  const index = inputs.indexOf(input);
  if (index < inputs.length - 1) {
    inputs[index + 1].focus();
    return;
  }
  const row = addAtomRow(nextBackboneAtom());
  row.querySelector("[data-field='x']").focus();
}

function handleAtomPaste(event) {
  const input = event.target;
  if (!input.matches("#atom-table input")) return;
  const tokens = event.clipboardData.getData("text").trim().split(/[\s,]+/);
  const numeric = tokens.filter((t) => t !== "" && !Number.isNaN(Number(t)));
  if (numeric.length < 3 || tokens.length > 4) return;
  event.preventDefault();
  const row = input.closest("tr");
  if (tokens.length === 4 && Number.isNaN(Number(tokens[0]))) row.querySelector("[data-field='name']").value = tokens[0];
  AXES.forEach((axis, i) => { row.querySelector(`[data-field='${axis}']`).value = numeric[numeric.length - 3 + i]; });
  refreshAtomTable();
}

function atomFormText() {
  const rows = atomRows();
  for (const input of $$("#atom-table input")) input.removeAttribute("aria-invalid");
  const problems = [];
  const lines = rows.map((row, i) => {
    const name = row.querySelector("[data-field='name']");
    const missing = [];
    if (!name.value.trim() || /\s/.test(name.value.trim())) {
      name.setAttribute("aria-invalid", "true");
      missing.push("a name");
    }
    for (const axis of AXES) {
      const input = row.querySelector(`[data-field='${axis}']`);
      if (input.value.trim() === "" || Number.isNaN(Number(input.value))) {
        input.setAttribute("aria-invalid", "true");
        missing.push(axis);
      }
    }
    if (missing.length) problems.push({ row: i + 1, name: name.value.trim() || "atom", missing });
    return `${name.value.trim()} ${AXES.map((axis) => row.querySelector(`[data-field='${axis}']`).value.trim()).join(" ")}`;
  });
  if (problems.length) {
    const first = problems[0];
    const more = problems.length > 1 ? ` (${problems.length - 1} more row${problems.length > 2 ? "s" : ""} also need filling in)` : "";
    $("#atom-table [aria-invalid='true']").focus();
    throw new Error(`Atom ${first.row} (${first.name}) needs ${first.missing.join(", ")}${more}.`);
  }
  if (rows.length < 2) throw new Error("Add at least 2 atoms: internal coordinates start with a bond.");
  return lines.join("\n");
}

function setupAtomForm() {
  resetAtomTable(DEFAULT_ATOMS.map((name) => ({ name })));
  const table = $("#atom-table");
  table.addEventListener("input", (event) => {
    event.target.removeAttribute("aria-invalid");
    if (event.target.dataset.field === "name") refreshAtomTable();
  });
  table.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || !event.target.matches("input")) return;
    event.preventDefault();
    if (event.metaKey || event.ctrlKey) convert("cartesian-form");
    else focusNextInput(event.target);
  });
  table.addEventListener("paste", handleAtomPaste);
  $("#add-residue").addEventListener("click", () => {
    const rows = BACKBONE_CYCLE.map((name) => addAtomRow(name));
    rows[0].querySelector("[data-field='x']").focus();
  });
  $("#add-next-atom").addEventListener("click", () => {
    addAtomRow(nextBackboneAtom()).querySelector("[data-field='x']").focus();
  });
  $("#add-other-atom").addEventListener("click", () => {
    addAtomRow().querySelector("[data-field='name']").focus();
  });
  $("#clear-atoms").addEventListener("click", () => {
    resetAtomTable(DEFAULT_ATOMS.map((name) => ({ name })));
    hideError();
  });
}

// ---------------------------------------------------------------------------
// Loading files
// ---------------------------------------------------------------------------

async function loadFile(input) {
  hideError();
  const file = input.files && input.files[0];
  input.value = "";
  if (!file) return;
  if (file.size > MAX_FILE_BYTES) {
    showError("That file is larger than 4 MB. Use the terminal app (python3 main.py) for very large structures.");
    return;
  }
  $(`#${input.dataset.target}`).value = await file.text();
}

// ---------------------------------------------------------------------------
// Per-residue table: phi, psi, omega and the carbonyl oxygen
// ---------------------------------------------------------------------------

function currentResidueValues() {
  return $$("#bb-table tbody tr").map((row) => {
    const values = {};
    for (const input of row.querySelectorAll("input")) values[input.dataset.field] = input.value;
    return values;
  });
}

function backboneCount() {
  const count = parseInt($("#bb-count").value, 10);
  return Number.isFinite(count) ? Math.min(Math.max(count, 1), 500) : 1;
}

function residueName(sequence, i) {
  return ONE_TO_THREE[sequence[i]] || "ALA";
}

function renderBackboneTable(previous = currentResidueValues()) {
  const count = backboneCount();
  const sequence = $("#bb-sequence").value.replace(/\s+/g, "").toUpperCase();
  const tbody = $("#bb-table tbody");
  tbody.replaceChildren();
  for (let i = 0; i < count; i++) {
    const tr = document.createElement("tr");
    const name = residueName(sequence, i);
    tr.appendChild(cell(`${name}${i + 1}`));
    for (const field of RESIDUE_FIELDS) {
      const td = document.createElement("td");
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.inputMode = "decimal";
      input.dataset.field = field;
      input.setAttribute("aria-label", `Residue ${i + 1} (${name}) ${RESIDUE_LABELS[field]}`);
      const unused = (field === "phi" && i === 0) || (field === "omega" && i === count - 1);
      input.disabled = unused;
      if (field === "o_dihedral") input.placeholder = "ψ+180";
      const kept = previous[i] && previous[i][field];
      input.value = unused ? "" : (kept !== undefined && kept !== "" ? kept : RESIDUE_DEFAULTS[field]);
      td.appendChild(input);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function syncBackboneForm() {
  const custom = $("#bb-preset").value === "custom";
  $("#bb-custom").hidden = !custom;
  if (custom) renderBackboneTable();
}

function parseResidueCsv(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim() && !line.trim().startsWith("#"));
  const header = (lines.shift() || "").split(",").map((h) => h.trim().toLowerCase());
  const column = (name) => header.indexOf(name);
  if (column("residue") < 0 || column("phi") < 0 || column("psi") < 0 || column("omega") < 0) {
    throw new Error("The residue table needs the columns: residue, phi, psi, omega, " +
      "c_o_bond, ca_c_o_angle, n_ca_c_o_dihedral.");
  }
  if (!lines.length) throw new Error("The residue table has no rows.");
  if (lines.length > 500) throw new Error("The web version handles up to 500 residues.");
  return lines.map((line) => {
    const cells = line.split(",").map((c) => c.trim());
    const values = { name: (cells[column("residue")].match(/^[A-Za-z]+/) || ["ALA"])[0].toUpperCase() };
    for (const field of RESIDUE_FIELDS) {
      const index = column(CSV_COLUMNS[field]);
      values[field] = index < 0 || cells[index] === undefined || cells[index] === "-" ? "" : cells[index];
    }
    return values;
  });
}

function fillResidueTable(rows) {
  $("#bb-count").value = String(rows.length);
  $("#bb-sequence").value = rows.map((row) => THREE_TO_ONE[row.name] || "A").join("");
  $("#bb-preset").value = "custom";
  $("#bb-custom").hidden = false;
  renderBackboneTable(rows);
  $("#results").hidden = true;
}

async function loadResidueFile(input) {
  hideError();
  const file = input.files && input.files[0];
  input.value = "";
  if (!file) return;
  try {
    if (file.size > MAX_FILE_BYTES) throw new Error("That file is larger than 4 MB.");
    fillResidueTable(parseResidueCsv(await file.text()));
  } catch (error) {
    showError(error.message);
  }
}

function residueRequestBody() {
  return currentResidueValues().map((values, i) => {
    const residue = {};
    for (const field of RESIDUE_FIELDS) {
      const raw = values[field];
      if (raw !== undefined && raw !== "" && Number.isNaN(Number(raw))) {
        throw new Error(`Residue ${i + 1}: ${RESIDUE_LABELS[field]} must be a number.`);
      }
      residue[field] = raw === undefined || raw === "" ? null : Number(raw);
    }
    return residue;
  });
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

async function convert(kind) {
  hideError();
  const button = document.querySelector(`[data-convert="${kind}"]`);
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Converting…";
  try {
    if (kind === "cartesian-form") {
      const text = atomFormText();
      renderResults("to-internal", await api("/api/cartesian-to-internal", { text }));
    } else if (kind === "cartesian") {
      const text = $("#cartesian-text").value;
      if (!text.trim()) throw new Error("Enter some coordinates first, or open a PDB file.");
      const data = await api("/api/cartesian-to-internal", { text, backbone_only: $("#backbone-only").checked });
      renderResults("to-internal", data);
    } else if (kind === "zmatrix") {
      const text = $("#zmatrix-text").value;
      if (!text.trim()) throw new Error("Enter a Z-matrix first, or open a .zmat file.");
      renderResults("to-cartesian", await api("/api/internal-to-cartesian", { text }));
    } else {
      const body = { count: backboneCount(), sequence: $("#bb-sequence").value };
      if ($("#bb-preset").value === "custom") {
        body.residues = residueRequestBody();
      } else {
        body.preset = $("#bb-preset").value;
      }
      renderResults("to-cartesian", await api("/api/backbone-to-cartesian", body));
    }
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

function verificationText(check) {
  const exp = (v) => v.toExponential(1);
  if (check.kind === "distances") {
    return `✓ Rebuilding Cartesian coordinates from these internal coordinates reproduces all ` +
      `${check.pairs.toLocaleString()} interatomic distances (largest deviation ${exp(check.max_deviation)} Å).`;
  }
  if (check.kind === "torsions") {
    return `✓ φ, ψ, ω and the oxygen angles measured back from these coordinates match your input ` +
      `(largest error ${exp(check.max_error)}°).`;
  }
  return `✓ Measured back from the coordinates, every bond, angle and dihedral matches ` +
    `(largest error ${exp(check.bond)} Å, ${exp(check.angle)}°, ${exp(check.dihedral)}°).`;
}

function fillResidueResults(table, torsions) {
  const withOxygen = torsions.some((t) => t.o_dihedral !== null);
  const headers = [{ label: "Residue" }, { label: "φ phi (°)", num: true }, { label: "ψ psi (°)", num: true },
    { label: "ω omega (°)", num: true }];
  if (withOxygen) {
    headers.push(
      { label: "C=O (Å)", num: true, kind: "bond length" },
      { label: "CA–C–O (°)", num: true, kind: "bond angle" },
      { label: "N–CA–C–O (°)", num: true, kind: "dihedral (torsion)" },
    );
  }
  headers.push({ label: "Region", cls: (v) => `region-${v}` });
  fillTable(table, headers, torsions.map((t) => [
    t.residue, fmt(t.phi, 1), fmt(t.psi, 1), fmt(t.omega, 1),
    ...(withOxygen ? [fmt(t.o_bond, 3), fmt(t.o_angle, 1), fmt(t.o_dihedral, 1)] : []),
    t.region,
  ]));
  return withOxygen;
}

function fillZmatrixTable(table, data) {
  const ref = (index) => (index === null ? "-" : `${index}:${data.zmatrix[index - 1].name}`);
  fillTable(table, [
    { label: "#", num: true }, { label: "Atom" }, { label: "Residue" },
    { label: "Bonded to" }, { label: "Length (Å)", num: true },
    { label: "Angle with" }, { label: "Angle (°)", num: true },
    { label: "Dihedral with" }, { label: "Dihedral (°)", num: true },
  ], data.zmatrix.map((e, i) => [
    String(i + 1), e.name, data.atoms[i].residue,
    ref(e.bond_ref), fmt(e.bond, 4), ref(e.angle_ref), fmt(e.angle, 2), ref(e.dihedral_ref), fmt(e.dihedral, 2),
  ]));
}

function renderResults(direction, data) {
  const toInternal = direction === "to-internal";
  const hasTorsions = data.torsions.length > 0;
  const perResidue = toInternal && hasTorsions;
  $("#verify").textContent = verificationText(data.check);

  const showSide = !toInternal && hasTorsions;
  $("#torsions-block").hidden = !showSide;
  $(".results-grid").classList.toggle("single", !showSide);
  if (showSide) fillResidueResults($("#torsions-table"), data.torsions);

  const details = $("#zmatrix-details");
  details.hidden = toInternal && !hasTorsions;
  details.open = false;

  if (perResidue) {
    $("#results-heading").textContent = "Result: internal coordinates per residue";
    const withOxygen = fillResidueResults($("#main-table"), data.torsions);
    $("#main-table-heading").textContent = withOxygen
      ? `φ, ψ, ω and oxygen for ${data.torsions.length} residues`
      : `φ, ψ and ω for ${data.torsions.length} residues (no O atoms in the input)`;
    $("#main-table-note").textContent = "φ = C(i−1)–N–CA–C, ψ = N–CA–C–N(i+1), ω = CA–C–N(i+1)–CA(i+1). " +
      "The oxygen is placed by the C=O length, the CA–C–O angle and the N–CA–C–O dihedral. " +
      "A dash means the angle needs a neighbouring residue that is not there.";
    fillZmatrixTable($("#zmatrix-table"), data);
  } else if (toInternal) {
    $("#results-heading").textContent = "Result: internal coordinates (Z-matrix)";
    $("#main-table-heading").textContent = `Z-matrix for ${data.atoms.length} atoms`;
    $("#main-table-note").textContent = "No protein backbone (N, CA, C) was found, so the general Z-matrix is shown. " +
      "Each atom is placed by a bond length to its bonded partner, an angle with that partner's partner, " +
      "and a dihedral with the next atom along the chain.";
    fillZmatrixTable($("#main-table"), data);
  } else {
    $("#results-heading").textContent = "Result: Cartesian coordinates";
    $("#main-table-heading").textContent = `Cartesian coordinates for ${data.atoms.length} atoms`;
    $("#main-table-note").textContent = "Atom 1 is placed at the origin, atom 2 on the +x axis and atom 3 in the " +
      "xy-plane: internal coordinates fix the shape, not the position or orientation.";
    fillTable($("#main-table"), [
      { label: "#", num: true }, { label: "Atom" }, { label: "Residue" },
      { label: "x (Å)", num: true }, { label: "y (Å)", num: true }, { label: "z (Å)", num: true },
    ], data.atoms.map((a, i) => [String(i + 1), a.name, a.residue, fmt(a.x, 3), fmt(a.y, 3), fmt(a.z, 3)]));
    fillZmatrixTable($("#zmatrix-table"), data);
  }

  state.downloads = data.downloads;
  $("#downloads").replaceChildren(...data.downloads.map((download, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tool";
    button.textContent = download.label;
    button.addEventListener("click", () => downloadResult(i));
    return button;
  }));

  $("#results").hidden = false;
  setStructure(data.atoms, data.bonds);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  $("#results").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}

function downloadResult(index) {
  const download = state.downloads[index];
  if (!download) return;
  const blob = new Blob([download.text], { type: "text/plain" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = download.filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

// ---------------------------------------------------------------------------
// Structure viewer
// ---------------------------------------------------------------------------

function elementOf(name) {
  const letter = (name.match(/[A-Za-z]/) || ["X"])[0].toUpperCase();
  return "NCOS".includes(letter) ? letter : "X";
}

function setStructure(atoms, bonds) {
  let shown = atoms.map((atom, index) => ({ ...atom, index }));
  if (shown.length > MAX_DRAWN_ATOMS) shown = shown.filter((a) => a.name === "CA");
  const keep = new Map(shown.map((a, i) => [a.index, i]));
  const centre = ["x", "y", "z"].map((k) => shown.reduce((sum, a) => sum + a[k], 0) / shown.length);
  const points = shown.map((a) => ({
    p: [a.x - centre[0], a.y - centre[1], a.z - centre[2]],
    element: elementOf(a.name),
  }));
  const radius = Math.max(1, ...points.map(({ p }) => Math.hypot(p[0], p[1], p[2])));
  const lines = [];
  for (const [i, j] of bonds) {
    if (!keep.has(i) || !keep.has(j)) continue;
    const a = points[keep.get(i)].p;
    const b = points[keep.get(j)].p;
    if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= MAX_DRAWN_BOND) lines.push([keep.get(i), keep.get(j)]);
  }
  state.structure = { points, lines, radius };
  drawStructure();
}

function drawStructure() {
  const svg = $("#viewer");
  const structure = state.structure;
  if (!structure || $("#results").hidden) return;
  const { width, height } = svg.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const { yaw, pitch } = state.view;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const scale = (0.44 * Math.min(width, height)) / structure.radius;
  const projected = structure.points.map(({ p, element }) => {
    const x1 = cy * p[0] + sy * p[2];
    const z1 = -sy * p[0] + cy * p[2];
    const y2 = cp * p[1] - sp * z1;
    const z2 = sp * p[1] + cp * z1;
    return { x: width / 2 + scale * x1, y: height / 2 - scale * y2, depth: z2 / structure.radius, element };
  });

  const ns = "http://www.w3.org/2000/svg";
  const fragment = document.createDocumentFragment();
  for (const [i, j] of structure.lines) {
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", projected[i].x.toFixed(1));
    line.setAttribute("y1", projected[i].y.toFixed(1));
    line.setAttribute("x2", projected[j].x.toFixed(1));
    line.setAttribute("y2", projected[j].y.toFixed(1));
    fragment.appendChild(line);
  }
  const atomRadius = Math.max(1.8, Math.min(6, 0.38 * scale));
  const order = projected.map((_, i) => i).sort((a, b) => projected[a].depth - projected[b].depth);
  for (const i of order) {
    const { x, y, depth, element } = projected[i];
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", x.toFixed(1));
    circle.setAttribute("cy", y.toFixed(1));
    circle.setAttribute("r", (atomRadius * (0.8 + 0.2 * (depth + 1))).toFixed(2));
    circle.setAttribute("class", `el-${element}`);
    circle.setAttribute("opacity", (0.55 + 0.45 * ((depth + 1) / 2)).toFixed(2));
    fragment.appendChild(circle);
  }
  svg.replaceChildren(fragment);
}

function rotate(dYaw, dPitch) {
  state.view.yaw += dYaw;
  state.view.pitch = Math.max(-1.5, Math.min(1.5, state.view.pitch + dPitch));
  requestAnimationFrame(drawStructure);
}

function setupViewer() {
  const svg = $("#viewer");
  let last = null;
  svg.addEventListener("pointerdown", (event) => {
    last = [event.clientX, event.clientY];
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (!last) return;
    rotate((event.clientX - last[0]) * 0.01, (event.clientY - last[1]) * 0.01);
    last = [event.clientX, event.clientY];
  });
  const stop = () => { last = null; };
  svg.addEventListener("pointerup", stop);
  svg.addEventListener("pointercancel", stop);
  svg.addEventListener("keydown", (event) => {
    const steps = { ArrowLeft: [-0.15, 0], ArrowRight: [0.15, 0], ArrowUp: [0, -0.15], ArrowDown: [0, 0.15] };
    if (!steps[event.key]) return;
    event.preventDefault();
    rotate(...steps[event.key]);
  });
  new ResizeObserver(() => drawStructure()).observe(svg);
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

async function loadProfile() {
  try {
    const profile = await api("/api/profile");
    $("#profile-name").textContent = profile.name;
    $("#profile-id").textContent = profile.id;
    $("#profile-structure").textContent = profile.structure;
  } catch (error) {
    showError(error.message);
  }
}

function init() {
  for (const button of $$(".choice")) button.addEventListener("click", () => setMode(button.dataset.mode));
  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input, textarea, select, [role='tab']") || event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.key === "1") setMode("cartesian");
    if (event.key === "2") setMode("internal");
  });

  setupTabs();
  setupAtomForm();

  for (const input of $$(".file-tool input[data-target]")) input.addEventListener("change", () => loadFile(input));

  for (const button of $$("[data-convert]")) button.addEventListener("click", () => convert(button.dataset.convert));
  for (const [textarea, kind] of [["#cartesian-text", "cartesian"], ["#zmatrix-text", "zmatrix"]]) {
    $(textarea).addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        convert(kind);
      }
    });
  }

  $("#bb-preset").addEventListener("change", syncBackboneForm);
  $("#bb-count").addEventListener("change", () => { if (!$("#bb-custom").hidden) renderBackboneTable(); });
  $("#bb-sequence").addEventListener("input", () => { if (!$("#bb-custom").hidden) renderBackboneTable(); });
  $("#bb-file").addEventListener("change", () => loadResidueFile($("#bb-file")));
  $("#bb-reset").addEventListener("click", () => renderBackboneTable([]));
  renderBackboneTable([]);

  setupViewer();
  loadProfile();
}

init();
