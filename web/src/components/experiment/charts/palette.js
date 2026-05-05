/**
 * Dark palette for the academic dashboard. Bars are thinner, type is
 * lighter, no bold weights anywhere by default.
 */
export const PALETTE = {
  bg: "#0d0f12",
  axis: "#9aa3ad",
  grid: "#222831",
  text: "#cdd5df",
  muted: "#6b7280",

  // class colours (slightly desaturated for the dark bg)
  PC:  "#5fa8d3", PE: "#e07a7a", PS: "#b388eb",
  PG:  "#e493b3", PI: "#5dc0d8", SM: "#7cc77c",
  Cer: "#e6a456", TAG: "#cdc15c", DAG: "#a07a5e",
  LPC: "#a8b2bd", CE:  "#9cc4d8", FA:  "#e8c598",

  pos: "#5fa8d3", neg: "#e07a7a",
  pass: "#7cc77c", fail: "#e07a7a",
  highlight: "#e6a456",
};

export const FRAGMENT_TYPE_COLOR = {
  precursor:    "#5fa8d3",
  head_charged: "#e07a7a",
  head_loss:    "#e6a456",
  fa_loss:      "#7cc77c",
  fa_anion:     "#b388eb",
  neutral_loss: "#5dc0d8",
  isotope:      "#a8b2bd",
};

export function classColor(classKey) {
  return PALETTE[classKey] || "#444444";
}

/**
 * d3-friendly margin object.
 */
export const MARGIN = { top: 14, right: 18, bottom: 38, left: 50 };
