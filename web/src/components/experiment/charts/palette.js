/**
 * Shared palette and chart styles, matching the paper's panels.
 */
export const PALETTE = {
  bg: "#ffffff",
  axis: "#333333",
  grid: "#dddddd",
  text: "#222222",
  muted: "#888888",

  // class colours
  PC:  "#1f77b4", PE: "#d62728", PS: "#9467bd",
  PG:  "#e377c2", PI: "#17becf", SM: "#2ca02c",
  Cer: "#ff7f0e", TAG: "#bcbd22", DAG: "#8c564b",
  LPC: "#7f7f7f", CE:  "#aec7e8", FA:  "#ffbb78",

  pos: "#1f77b4", neg: "#d62728",
  pass: "#2ca02c", fail: "#d62728",
  highlight: "#ff7f0e",
};

export const FRAGMENT_TYPE_COLOR = {
  precursor:    "#1f77b4",
  head_charged: "#d62728",
  head_loss:    "#ff7f0e",
  fa_loss:      "#2ca02c",
  fa_anion:     "#9467bd",
  neutral_loss: "#17becf",
  isotope:      "#7f7f7f",
};

export function classColor(classKey) {
  return PALETTE[classKey] || "#444444";
}

/**
 * d3-friendly margin object.
 */
export const MARGIN = { top: 14, right: 18, bottom: 38, left: 50 };
