/**
 * Library export serialisers for predicted records.
 *
 * Supported formats:
 *   - MSP   (NIST text format, read by NIST MS Search, MS-DIAL, GNPS)
 *   - JSON  (lossless, full record)
 *   - CSV   (precursor inclusion list + minimal MS1)
 *   - mzML  (minimal mzML stub for downstream tools)
 *   - skyline (transition list, .csv compatible with Skyline import)
 */

/** Build an MSP-format library string. */
export function toMSP(records) {
  const lines = [];
  for (const r of records) {
    lines.push(`Name: ${r.analyte} ${r.adduct}`);
    lines.push(`PrecursorMZ: ${r.precursorMz.toFixed(6)}`);
    lines.push(`Precursor_type: ${r.adduct}`);
    lines.push(`Ion_mode: ${r.polarity}`);
    lines.push(`Charge: ${r.polarity === "+" ? r.z : -r.z}`);
    lines.push(`Formula: ${formulaFromComp(r.composition)}`);
    lines.push(`MW: ${r.neutralMass.toFixed(6)}`);
    lines.push(`InstrumentType: ${r.analyserMode.toUpperCase()}`);
    lines.push(`CollisionEnergy: variable`);
    lines.push(`Comment: lavoisier_virtual; class=${r.analyteClass}; X=${r.X}; Y=${r.Y}; n=${r.n}; l=${r.l}; m=${r.m}; s=${r.s}`);
    lines.push(`Num Peaks: ${r.ms2.length}`);
    for (const p of r.ms2) {
      const inten = Math.max(1, Math.round(p.intensity * 999));
      lines.push(`${p.mz.toFixed(6)} ${inten} "${p.label}"`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

function formulaFromComp(c) {
  const order = ["C", "H", "N", "O", "P", "S", "Na", "K", "Cl", "F"];
  return order
    .filter((el) => c[el] > 0)
    .map((el) => (c[el] === 1 ? el : `${el}${c[el]}`))
    .join("");
}

/** Build a CSV inclusion list (Thermo / Bruker compatible). */
export function toInclusionListCSV(records) {
  const rows = [
    ["Compound", "Formula", "Adduct", "Precursor m/z", "Charge", "Polarity", "Intensity_pred", "Class"],
  ];
  for (const r of records) {
    rows.push([
      r.analyte,
      formulaFromComp(r.composition),
      r.adduct,
      r.precursorMz.toFixed(6),
      r.polarity === "+" ? r.z : -r.z,
      r.polarity,
      r.intensity.toExponential(3),
      r.analyteClass,
    ]);
  }
  return rows.map((row) => row.join(",")).join("\n");
}

/** Build a Skyline-style transition list. */
export function toSkylineTransitions(records) {
  const rows = [
    ["Molecule List", "Precursor Name", "Precursor Formula",
      "Precursor Adduct", "Precursor m/z", "Precursor Charge",
      "Product m/z", "Product Charge", "Note"],
  ];
  for (const r of records) {
    const formula = formulaFromComp(r.composition);
    for (const p of r.ms2.slice(0, 5)) {
      rows.push([
        r.analyteClass,
        r.analyte,
        formula,
        r.adduct,
        r.precursorMz.toFixed(6),
        r.polarity === "+" ? r.z : -r.z,
        p.mz.toFixed(6),
        r.polarity === "+" ? 1 : -1,
        p.label,
      ]);
    }
  }
  return rows.map((row) => row.join(",")).join("\n");
}

/** Lossless JSON. */
export function toJSON(records, designSpec) {
  return JSON.stringify(
    {
      meta: {
        generator: "Lavoisier Virtual Instrument v1",
        timestamp: new Date().toISOString(),
        record_count: records.length,
      },
      design: designSpec,
      records,
    },
    null,
    2
  );
}

/** Minimal mzML 1.1 stub (one spectrum per record's MS1). */
export function toMzMLLite(records, designSpec) {
  const xml = [];
  xml.push('<?xml version="1.0" encoding="utf-8"?>');
  xml.push('<indexedmzML xmlns="http://psi.hupo.org/ms/mzml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd">');
  xml.push('  <mzML version="1.1.0">');
  xml.push('    <cvList count="1"><cv id="MS" fullName="Proteomics Standards Initiative MS"/></cvList>');
  xml.push('    <fileDescription><fileContent><cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum"/></fileContent></fileDescription>');
  xml.push(`    <run id="virtual_lavoisier" defaultInstrumentConfigurationRef="IC1">`);
  xml.push(`      <spectrumList count="${records.length}">`);
  records.forEach((r, idx) => {
    const peaks = r.ms1.concat(r.ms2);
    xml.push(`        <spectrum id="scan=${idx + 1}" index="${idx}" defaultArrayLength="${peaks.length}">`);
    xml.push(`          <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>`);
    xml.push(`          <cvParam cvRef="MS" accession="MS:1000528" name="lowest observed m/z" value="${peaks[0]?.mz ?? 0}"/>`);
    xml.push(`          <cvParam cvRef="MS" accession="MS:1000527" name="highest observed m/z" value="${peaks[peaks.length - 1]?.mz ?? 0}"/>`);
    xml.push(`          <userParam name="lavoisier_virtual_compound" value="${r.analyte}${r.adduct}"/>`);
    xml.push(`        </spectrum>`);
  });
  xml.push('      </spectrumList>');
  xml.push('    </run>');
  xml.push('  </mzML>');
  xml.push('</indexedmzML>');
  return xml.join("\n");
}

/**
 * Trigger a browser download of the given content.
 */
export function downloadAs(filename, content, mime = "text/plain") {
  if (typeof window === "undefined") return;
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

export const FORMATS = [
  { key: "msp",  label: "NIST .msp",            ext: "msp",  mime: "text/plain", builder: toMSP },
  { key: "json", label: "Lavoisier .json",      ext: "json", mime: "application/json", builder: toJSON },
  { key: "csv",  label: "Inclusion list .csv",  ext: "csv",  mime: "text/csv",  builder: toInclusionListCSV },
  { key: "skyline", label: "Skyline transitions .csv", ext: "csv", mime: "text/csv", builder: toSkylineTransitions },
  { key: "mzml", label: "mzML lite .mzML",      ext: "mzML", mime: "application/xml", builder: toMzMLLite },
];
