// =====================================================================
// Experiment 0 — NCE invariance of the S-entropy address
//
// Claim under test: the S-entropy address (Sk, St, Se) is a property of
// the COMPOUND, not of the spectrum. If so, varying collision energy
// while holding the compound fixed should move the address less than
// changing compound does.
//
// Data: NIST AC_CAC_MSLibrary2020, 5328 HCD MS2 spectra, 332 compounds
// measured at all nine NCE levels (10,15,20,25,30,40,50,60,80 %).
//
// Criterion, stated before results are seen:
//   separation_ratio = mean_between / mean_within  >  2.0
//   and |pearson_r| < 0.3 for each axis against NCE
//
// The experiment can fail. If Se tracks NCE (more fragments at higher
// energy raises local entropy), the address is a property of the
// spectrum and the claim is refuted.
// =====================================================================

import lavoisier.acquire
import lavoisier.transform
import lavoisier.analyse

objective NCEInvariance:
    target: "does the S-entropy address survive collision-energy variation"
    criterion: "separation_ratio > 2.0 and |r| < 0.3 per axis"

dataset ACCAC:
    files: ["../oxford/public/ac_cac_lib2020_msp/AC_CAC_MSLibrary2020_V1D1B.msp"]
    format: "msp"
    instrument: "Orbitrap HCD"
    polarity: "P"
    min_peaks: 3

phase Read:
    scans = lavoisier.acquire.read_msp(dataset: ACCAC, min_peaks: 3)

phase Transform:
    coords = lavoisier.transform.sentropy(
        scans: scans,
        alpha: 1.0,
        beta: 1.0,
        k_neighbors: 5
    )

phase Structure:
    grouping = lavoisier.analyse.group_by(coords: coords, key: "compound")

phase Test:
    // Primary criterion: within-compound spread vs between-compound spread
    separation = lavoisier.analyse.separation(
        coords: coords,
        key: "compound",
        axes: ["s_k", "s_t", "s_e"],
        min_group: 9
    )

    // Secondary criterion: does any axis track collision energy?
    drift = lavoisier.analyse.drift(
        coords: coords,
        over: "nce",
        axes: ["s_k", "s_t", "s_e"]
    )

phase Controls:
    // Negative control: shuffling compound labels must collapse the ratio
    shuffled = lavoisier.analyse.shuffle_control(
        coords: coords,
        key: "compound",
        axes: ["s_k", "s_t", "s_e"],
        min_group: 9,
        seed: 20260818
    )

    // Comparison method: raw-spectrum cosine similarity across NCE.
    // Established practice, known to degrade with collision energy.
    baseline = lavoisier.analyse.baseline(
        scans: scans,
        key: "compound",
        over: "nce",
        tolerance: 0.01,
        min_group: 9
    )
