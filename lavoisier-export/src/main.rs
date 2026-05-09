//! lavoisier-export — mzML → .lavoisier.json
//!
//! Reads one or more mzML files, computes S-entropy coordinates and partition
//! states for each MS2 precursor, annotates against a built-in lipid DB, then
//! writes a .lavoisier.json that the Lavoisier web dashboard can import.

use anyhow::{Context, Result};
use base64::prelude::*;
use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use flate2::read::ZlibDecoder;
use lavoisier_mass_computing::{PartitionState, SEntropyCoord, TernaryAddress};
use quick_xml::events::Event;
use quick_xml::Reader;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "lavoisier-export",
    about = "Analyse mzML files locally → .lavoisier.json for web visualisation",
    long_about = None,
)]
struct Args {
    /// Input mzML file(s)
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// Output path or directory (default: <input>.lavoisier.json)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Mass analyser type
    #[arg(short, long, default_value = "orbitrap")]
    analyser: String,

    /// Ion polarity [+ | -] (auto-detected from filename if omitted)
    #[arg(short, long)]
    polarity: Option<String>,

    /// Retention time window start (minutes)
    #[arg(long = "rt-start", default_value = "0.0")]
    rt_start: f64,

    /// Retention time window end (minutes)
    #[arg(long = "rt-end", default_value = "200.0")]
    rt_end: f64,

    /// Minimum MS1 peak intensity
    #[arg(long = "ms1-threshold", default_value = "1000")]
    ms1_threshold: f64,

    /// Minimum MS2 peak intensity
    #[arg(long = "ms2-threshold", default_value = "10")]
    ms2_threshold: f64,

    /// Mass accuracy for annotation (ppm)
    #[arg(long, default_value = "10.0")]
    ppm: f64,
}

// ── mzML scan data ────────────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct ScanData {
    scan_id: String,
    ms_level: u8,
    retention_time: f64,
    precursor_mz: Option<f64>,
    charge: Option<u8>,
    mz: Vec<f64>,
    intensity: Vec<f64>,
}

// ── mzML parser ───────────────────────────────────────────────────────────────

fn parse_mzml(path: &Path, rt_start: f64, rt_end: f64) -> Result<Vec<ScanData>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?;
    let buf_reader = std::io::BufReader::with_capacity(1 << 20, file);
    let mut xml = Reader::from_reader(buf_reader);

    let mut scans: Vec<ScanData> = Vec::new();
    let mut current: Option<ScanData> = None;

    // per-binaryDataArray state
    let mut in_binary = false;
    let mut is_mz_array = false;
    let mut is_intensity_array = false;
    let mut use_zlib = false;
    let mut use_32bit = false;

    let mut buf = Vec::new();

    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                match e.name().as_ref() {
                    b"spectrum" => {
                        let mut s = ScanData::default();
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"id" {
                                s.scan_id =
                                    String::from_utf8_lossy(&attr.value).to_string();
                            }
                        }
                        current = Some(s);
                        is_mz_array = false;
                        is_intensity_array = false;
                        use_zlib = false;
                        use_32bit = false;
                    }
                    b"binaryDataArray" => {
                        is_mz_array = false;
                        is_intensity_array = false;
                        use_zlib = false;
                        use_32bit = false;
                    }
                    b"cvParam" => {
                        if let Some(ref mut scan) = current {
                            let mut name = String::new();
                            let mut value = String::new();
                            let mut unit = String::new();
                            for attr in e.attributes().flatten() {
                                match attr.key.as_ref() {
                                    b"name" => {
                                        name =
                                            String::from_utf8_lossy(&attr.value).to_string()
                                    }
                                    b"value" => {
                                        value =
                                            String::from_utf8_lossy(&attr.value).to_string()
                                    }
                                    b"unitName" => {
                                        unit =
                                            String::from_utf8_lossy(&attr.value).to_string()
                                    }
                                    _ => {}
                                }
                            }
                            match name.as_str() {
                                "ms level" => {
                                    scan.ms_level = value.parse().unwrap_or(1)
                                }
                                "scan start time" => {
                                    let rt: f64 = value.parse().unwrap_or(0.0);
                                    scan.retention_time =
                                        if unit.eq_ignore_ascii_case("second") {
                                            rt / 60.0
                                        } else {
                                            rt
                                        };
                                }
                                "selected ion m/z" => {
                                    scan.precursor_mz = value.parse().ok()
                                }
                                "charge state" => {
                                    scan.charge = value.parse().ok()
                                }
                                "m/z array" => is_mz_array = true,
                                "intensity array" => is_intensity_array = true,
                                "zlib compression" => use_zlib = true,
                                "32-bit float" => use_32bit = true,
                                _ => {}
                            }
                        }
                    }
                    b"binary" => in_binary = true,
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_binary {
                    if let Some(ref mut scan) = current {
                        let text = e.unescape().unwrap_or_default();
                        let cleaned: String =
                            text.chars().filter(|c| !c.is_whitespace()).collect();
                        if !cleaned.is_empty() {
                            if let Ok(decoded) =
                                BASE64_STANDARD.decode(cleaned.as_bytes())
                            {
                                let data = if use_zlib {
                                    let mut dec = ZlibDecoder::new(decoded.as_slice());
                                    let mut out = Vec::new();
                                    let _ = dec.read_to_end(&mut out);
                                    out
                                } else {
                                    decoded
                                };
                                let values = if use_32bit {
                                    decode_f32(&data)
                                } else {
                                    decode_f64(&data)
                                };
                                if is_mz_array {
                                    scan.mz = values;
                                } else if is_intensity_array {
                                    scan.intensity = values;
                                }
                            }
                        }
                    }
                }
            }
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"binary" => in_binary = false,
                b"spectrum" => {
                    if let Some(scan) = current.take() {
                        if scan.retention_time >= rt_start
                            && scan.retention_time <= rt_end
                        {
                            scans.push(scan);
                        }
                    }
                    is_mz_array = false;
                    is_intensity_array = false;
                    use_zlib = false;
                    use_32bit = false;
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "XML parse error at byte {}: {}",
                    xml.buffer_position(),
                    e
                ))
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(scans)
}

fn decode_f32(data: &[u8]) -> Vec<f64> {
    let mut cur = std::io::Cursor::new(data);
    let mut out = Vec::with_capacity(data.len() / 4);
    while cur.position() + 4 <= data.len() as u64 {
        if let Ok(v) = cur.read_f32::<LittleEndian>() {
            out.push(v as f64);
        } else {
            break;
        }
    }
    out
}

fn decode_f64(data: &[u8]) -> Vec<f64> {
    let mut cur = std::io::Cursor::new(data);
    let mut out = Vec::with_capacity(data.len() / 8);
    while cur.position() + 8 <= data.len() as u64 {
        if let Ok(v) = cur.read_f64::<LittleEndian>() {
            out.push(v);
        } else {
            break;
        }
    }
    out
}

// ── S-entropy computation ────────────────────────────────────────────────────

const MASS_MIN: f64 = 50.0;
const MASS_MAX: f64 = 1500.0;
const LOG_MASS_MIN: f64 = 1.6989700043_f64; // log10(50)
const LOG_MASS_RANGE: f64 = 1.4771212547_f64; // log10(1500) - log10(50)

/// Map precursor m/z and RT to S-entropy coordinates.
///
/// S_k encodes molecular identity (inverse-log of mass)
/// S_t encodes chromatographic retention (linear RT)
/// S_e encodes fragmentation state (spectral Shannon entropy)
fn compute_sentropy(
    intensity_array: &[f64],
    precursor_mz: f64,
    rt: f64,
    rt_max: f64,
) -> SEntropyCoord {
    // S_k: inverse-log relationship with precursor mass
    let log_mz = precursor_mz.clamp(MASS_MIN, MASS_MAX).log10();
    let s_k = (1.0 - (log_mz - LOG_MASS_MIN) / LOG_MASS_RANGE).clamp(0.01, 0.99);

    // S_t: linear normalisation of retention time
    let s_t = if rt_max > 0.0 {
        (rt / rt_max).clamp(0.0, 1.0)
    } else {
        0.5
    };

    // S_e: normalised Shannon entropy of the fragment intensity distribution
    let s_e = spectral_entropy(intensity_array);

    SEntropyCoord::new_unchecked(s_k, s_t, s_e)
}

fn spectral_entropy(intensities: &[f64]) -> f64 {
    let n = intensities.len();
    if n <= 1 {
        return 0.5;
    }
    let total: f64 = intensities.iter().sum();
    if total <= 0.0 {
        return 0.5;
    }
    let h: f64 = intensities
        .iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| {
            let p = x / total;
            -p * p.ln()
        })
        .sum();
    let h_max = (n as f64).ln();
    if h_max > 0.0 {
        (h / h_max).clamp(0.01, 0.99)
    } else {
        0.5
    }
}

/// Entropy of the trit distribution in a ternary address (coordinate entropy).
fn partition_entropy(address: &str) -> f64 {
    if address.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 3];
    for c in address.chars() {
        match c {
            '0' => counts[0] += 1,
            '1' => counts[1] += 1,
            '2' => counts[2] += 1,
            _ => {}
        }
    }
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total as f64;
            -p * p.log2()
        })
        .sum::<f64>()
        / 3.0_f64.log2() // normalise to [0,1]
}

// ── Lipid database ────────────────────────────────────────────────────────────

/// Monoisotopic element masses (Da)
mod elem {
    pub const H: f64 = 1.00782503207;
    pub const C: f64 = 12.0;
    pub const N: f64 = 14.0030740048;
    pub const O: f64 = 15.99491461956;
    pub const P: f64 = 30.97376163;
    pub const PROTON: f64 = 1.00727646677;
    pub const SODIUM: f64 = 22.98921694;
    pub const NH4: f64 = 18.034374;
    pub const CL: f64 = 34.96885268;
}

/// Lipid class head-group contribution to formula C(X+a)H(2X-2Y+b)Nn_Oo_Pp
struct HeadGroup {
    class: &'static str,
    ca: i32,  // C offset (formula: C(X + ca))
    hb: i32,  // H offset (formula: H(2X - 2Y + hb))
    n: i32,
    o: i32,
    p: i32,
}

const HEAD_GROUPS: &[HeadGroup] = &[
    HeadGroup { class: "PC",  ca: 8,  hb: 16, n: 1, o: 8,  p: 1 },
    HeadGroup { class: "PE",  ca: 5,  hb: 8,  n: 1, o: 8,  p: 1 },
    HeadGroup { class: "PS",  ca: 6,  hb: 10, n: 1, o: 10, p: 1 },
    HeadGroup { class: "PG",  ca: 6,  hb: 9,  n: 0, o: 10, p: 1 },
    HeadGroup { class: "PI",  ca: 9,  hb: 13, n: 0, o: 13, p: 1 },
    HeadGroup { class: "LPC", ca: 8,  hb: 18, n: 1, o: 7,  p: 1 },
    HeadGroup { class: "LPE", ca: 5,  hb: 11, n: 1, o: 7,  p: 1 },
    HeadGroup { class: "TG",  ca: 3,  hb: 2,  n: 0, o: 6,  p: 0 },
    HeadGroup { class: "DG",  ca: 3,  hb: 4,  n: 0, o: 5,  p: 0 },
    // Sphingomyelin: X = total chain C (d18 backbone + FA)
    HeadGroup { class: "SM",  ca: 5,  hb: 13, n: 2, o: 6,  p: 1 },
    // Ceramide: Cer(X:Y) X includes the d18 backbone
    HeadGroup { class: "Cer", ca: 0,  hb: 1,  n: 1, o: 3,  p: 0 },
];

/// Carbon/double-bond ranges to enumerate per class
fn carbon_ranges(class: &str) -> (Vec<i32>, Vec<i32>) {
    match class {
        "PC" | "PE" | "PS" | "PG" | "PI" => (
            vec![28, 30, 32, 34, 36, 38, 40, 42],
            vec![0, 1, 2, 3, 4, 5, 6],
        ),
        "LPC" | "LPE" => (
            vec![14, 16, 18, 20, 22],
            vec![0, 1, 2],
        ),
        "TG" => (
            vec![42, 46, 48, 50, 52, 54, 56, 58],
            vec![0, 1, 2, 3, 4, 5, 6],
        ),
        "DG" => (
            vec![28, 30, 32, 34, 36, 38],
            vec![0, 1, 2, 3, 4],
        ),
        "SM" | "Cer" => (
            vec![32, 34, 36, 38, 40, 42, 44],
            vec![0, 1, 2, 3],
        ),
        _ => (vec![], vec![]),
    }
}

#[derive(Clone)]
struct LipidEntry {
    analyte: String,
    class: String,
    x: i32,
    y: i32,
    neutral_mass: f64,
    adduct: &'static str,
    adduct_abbr: &'static str,
    precursor_mz: f64,
    z: u8,
    polarity: char,
}

struct LipidDb {
    entries: Vec<LipidEntry>,
}

impl LipidDb {
    fn build() -> Self {
        let mut entries = Vec::with_capacity(4096);

        for hg in HEAD_GROUPS {
            let (carbons, dbs) = carbon_ranges(hg.class);
            for x in &carbons {
                for y in &dbs {
                    if *y > *x / 2 {
                        continue; // more double bonds than carbons/2 is unrealistic
                    }
                    let c_count = x + hg.ca;
                    let h_count = 2 * x - 2 * y + hg.hb;
                    if h_count < 1 {
                        continue;
                    }
                    let neutral_mass = (c_count as f64) * elem::C
                        + (h_count as f64) * elem::H
                        + (hg.n as f64) * elem::N
                        + (hg.o as f64) * elem::O
                        + (hg.p as f64) * elem::P;

                    let analyte = format!("{}({}:{})", hg.class, x, y);

                    // Positive-mode adducts — &(pat) dereferences the array ref
                    for &(adduct, adduct_abbr, dm) in &[
                        ("[M+H]+",   "+H",   elem::PROTON),
                        ("[M+Na]+",  "+Na",  elem::SODIUM),
                        ("[M+NH4]+", "+NH4", elem::NH4),
                    ] {
                        entries.push(LipidEntry {
                            analyte: analyte.clone(),
                            class: hg.class.to_string(),
                            x: *x,
                            y: *y,
                            neutral_mass,
                            adduct,
                            adduct_abbr,
                            precursor_mz: neutral_mass + dm,
                            z: 1,
                            polarity: '+',
                        });
                    }
                    // Negative-mode adducts
                    for &(adduct, adduct_abbr, dm) in &[
                        ("[M-H]-",   "-H",  -elem::PROTON),
                        ("[M+Cl]-",  "+Cl",  elem::CL),
                    ] {
                        entries.push(LipidEntry {
                            analyte: analyte.clone(),
                            class: hg.class.to_string(),
                            x: *x,
                            y: *y,
                            neutral_mass,
                            adduct,
                            adduct_abbr,
                            precursor_mz: neutral_mass + dm,
                            z: 1,
                            polarity: '-',
                        });
                    }
                }
            }
        }

        LipidDb { entries }
    }

    fn annotate(&self, mz: f64, polarity: char, ppm: f64) -> Option<&LipidEntry> {
        let tol = mz * ppm * 1e-6;
        self.entries
            .iter()
            .filter(|e| e.polarity == polarity)
            .min_by(|a, b| {
                let da = (a.precursor_mz - mz).abs();
                let db = (b.precursor_mz - mz).abs();
                da.partial_cmp(&db).unwrap()
            })
            .filter(|e| (e.precursor_mz - mz).abs() <= tol)
    }
}

// ── Record assembly ───────────────────────────────────────────────────────────

fn build_record(
    scan: &ScanData,
    ms2_peaks: &[[f64; 2]],
    scoord: SEntropyCoord,
    partition: PartitionState,
    address: &str,
    lipid: Option<&LipidEntry>,
    ms1_apex_intensity: f64,
    polarity: char,
    analyser: &str,
) -> Value {
    let p_entropy = partition_entropy(address);

    let (analyte, class, x, y, neutral_mass, adduct, adduct_abbr, prec_mz, z) =
        if let Some(l) = lipid {
            (
                l.analyte.clone(),
                l.class.clone(),
                l.x,
                l.y,
                l.neutral_mass,
                l.adduct.to_string(),
                l.adduct_abbr.to_string(),
                l.precursor_mz,
                l.z as i64,
            )
        } else {
            let pmz = scan.precursor_mz.unwrap_or(0.0);
            let nm = pmz - elem::PROTON;
            (
                format!("Unknown({:.4})", pmz),
                "Unknown".to_string(),
                0_i32,
                0_i32,
                nm,
                "[M+H]+".to_string(),
                "+H".to_string(),
                pmz,
                1_i64,
            )
        };

    let ms2_json: Vec<Value> = ms2_peaks
        .iter()
        .map(|p| json!([p[0], p[1]]))
        .collect();
    let peaks_json = ms2_json.clone();

    json!({
        "analyte":         analyte,
        "analyteClass":    class,
        "X":               x,
        "Y":               y,
        "composition":     {},
        "neutralMass":     neutral_mass,
        "adduct":          adduct,
        "adductAbbr":      adduct_abbr,
        "precursorMz":     prec_mz,
        "z":               z,
        "polarity":        polarity.to_string(),
        "intensity":       ms1_apex_intensity,
        "n":               partition.n,
        "l":               partition.l,
        "m":               partition.m,
        "s":               partition.s,
        "sentropy":        { "sk": scoord.s_k, "st": scoord.s_t, "se": scoord.s_e },
        "sentropyVec":     { "sk": scoord.s_k, "st": scoord.s_t, "se": scoord.s_e },
        "ternaryAddress":  address,
        "analyserMode":    analyser,
        "observable":      null,
        "shellDistribution": {},
        "partitionEntropy": p_entropy,
        "ms1":             [],
        "ms2":             ms2_json,
        "peaksAll":        peaks_json,
        "bitsTotal":       (address.len() as f64 * 3.0_f64.log2()) as i64,
    })
}

// ── Per-file pipeline ─────────────────────────────────────────────────────────

fn process_file(args: &Args, mzml_path: &Path, lipid_db: &LipidDb) -> Result<PathBuf> {
    let t0 = Instant::now();

    // Detect polarity from arg or filename
    let polarity_char: char = match &args.polarity {
        Some(p) if p == "-" => '-',
        Some(_) => '+',
        None => {
            let name = mzml_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            if name.contains("neg") || name.contains("negative") {
                '-'
            } else {
                '+'
            }
        }
    };

    eprintln!(
        "→ {} | polarity={} analyser={}",
        mzml_path.display(),
        polarity_char,
        args.analyser
    );

    // ── Stage 1: parse mzML ───────────────────────────────────────────────
    eprintln!("  [1/3] parsing mzML …");
    let scans = parse_mzml(mzml_path, args.rt_start, args.rt_end)?;

    let ms1_scans: Vec<&ScanData> = scans.iter().filter(|s| s.ms_level == 1).collect();
    let ms2_scans: Vec<&ScanData> = scans
        .iter()
        .filter(|s| s.ms_level == 2 && s.precursor_mz.is_some())
        .collect();

    eprintln!(
        "  parsed  MS1={} MS2={}",
        ms1_scans.len(),
        ms2_scans.len()
    );

    if ms2_scans.is_empty() {
        eprintln!("  warning: no MS2 scans found");
    }

    // Find the total RT range for normalisation
    let rt_max = scans
        .iter()
        .map(|s| s.retention_time)
        .fold(0.0_f64, f64::max)
        .max(1.0);

    // ── Stage 2: build records in parallel ───────────────────────────────
    eprintln!("  [2/3] computing S-entropy · annotation …");

    let records: Vec<Value> = ms2_scans
        .par_iter()
        .map(|scan| {
            let pmz = scan.precursor_mz.unwrap();

            // Filter MS2 peaks by intensity threshold
            let ms2_peaks: Vec<[f64; 2]> = scan
                .mz
                .iter()
                .zip(scan.intensity.iter())
                .filter(|(_, &i)| i >= args.ms2_threshold)
                .map(|(&m, &i)| [m, i])
                .collect();

            // S-entropy coordinates
            let scoord = compute_sentropy(
                &scan.intensity,
                pmz,
                scan.retention_time,
                rt_max,
            );

            // Partition state
            let partition = PartitionState::from_scoord(&scoord);

            // Ternary address (18 trits = 6 per dimension)
            let address = TernaryAddress::from_scoord(&scoord, 18)
                .map(|a| a.to_string())
                .unwrap_or_default();

            // Lipid annotation
            let lipid = lipid_db.annotate(pmz, polarity_char, args.ppm);

            // Look up MS1 apex intensity from the nearest preceding MS1 scan
            let ms1_scan_rt = scan.retention_time;
            let ms1_intensity = {
                let preceding = ms1_scans
                    .iter()
                    .rev()
                    .find(|&&s| s.retention_time <= ms1_scan_rt);
                if let Some(&ms1) = preceding {
                    ms1.mz
                        .iter()
                        .copied()
                        .zip(ms1.intensity.iter().copied())
                        .filter(|&(m, _)| ((m - pmz) / pmz * 1e6).abs() < 15.0)
                        .map(|(_, i)| i)
                        .reduce(f64::max)
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            };

            build_record(
                scan,
                &ms2_peaks,
                scoord,
                partition,
                &address,
                lipid,
                ms1_intensity,
                polarity_char,
                &args.analyser,
            )
        })
        .collect();

    eprintln!("  built {} records", records.len());

    // ── Stage 3: write JSON ───────────────────────────────────────────────
    eprintln!("  [3/3] writing output …");

    let out_path = resolve_output(mzml_path, &args.output)?;

    let envelope = json!({
        "version":    "1.0",
        "source":     "lavoisier-export",
        "sourceFile": mzml_path.file_name().unwrap_or_default().to_string_lossy(),
        "analyser":   args.analyser,
        "polarity":   polarity_char.to_string(),
        "runDate":    chrono_now(),
        "records":    records,
    });

    let json_bytes = serde_json::to_vec_pretty(&envelope)?;
    std::fs::write(&out_path, &json_bytes)?;

    let kb = json_bytes.len() / 1024;
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "  ✓  {} ({} kB, {:.1} s)",
        out_path.display(),
        kb,
        secs
    );

    Ok(out_path)
}

fn resolve_output(input: &Path, output_arg: &Option<PathBuf>) -> Result<PathBuf> {
    match output_arg {
        None => {
            let stem = input.file_stem().unwrap_or_default();
            let name = format!("{}.lavoisier.json", stem.to_string_lossy());
            Ok(input.with_file_name(name))
        }
        Some(p) if p.is_dir() => {
            let stem = input.file_stem().unwrap_or_default();
            let name = format!("{}.lavoisier.json", stem.to_string_lossy());
            Ok(p.join(name))
        }
        Some(p) => Ok(p.clone()),
    }
}

/// RFC 3339 timestamp without depending on chrono.
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Rough ISO-8601 UTC — accurate enough for the metadata field
    let (y, mo, d, h, mi, s) = epoch_to_ymd_hms(secs);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, mi, s)
}

fn epoch_to_ymd_hms(mut s: u64) -> (u64, u64, u64, u64, u64, u64) {
    let sec = s % 60;
    s /= 60;
    let min = s % 60;
    s /= 60;
    let hour = s % 24;
    s /= 24;
    // Gregorian calendar computation (works for 1970-2100)
    let z = s + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d, hour, min, sec)
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("lavoisier-export  building lipid database …");
    let lipid_db = LipidDb::build();
    eprintln!("  {} entries ({} classes)", lipid_db.entries.len(), HEAD_GROUPS.len());

    let mut any_error = false;
    for path in &args.input {
        if !path.exists() {
            eprintln!("error: file not found: {}", path.display());
            any_error = true;
            continue;
        }
        if let Err(e) = process_file(&args, path, &lipid_db) {
            eprintln!("error processing {}: {:#}", path.display(), e);
            any_error = true;
        }
    }

    if any_error {
        std::process::exit(1);
    }
    Ok(())
}
