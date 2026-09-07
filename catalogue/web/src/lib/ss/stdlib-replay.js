// =====================================================================
//  The spectral operations, replayed.
//
//  These do NOT run in the browser and this file does not pretend they
//  do. read_msp opens a 60 MB NIST library; sentropy computes a
//  k-neighbour embedding over 5298 spectra. What ships instead is the
//  recorded workspace binding each op produced when the experiment was
//  actually run under Python, and every value carries `_replayed: true`
//  so the cell can badge it.
//
//  The contract that makes this honest: an op is keyed by the arguments
//  that DETERMINE its result. Edit one of those and the op REFUSES,
//  naming the argument and the value the fixture holds. A replayed op
//  never answers a question it was not asked.
// =====================================================================

import { RefusalError } from './stdlib.js'

// Recorded from shapeshifter-py/results/exp0_nce_invariance.*.json.
// Provenance for every number is in public/data/README.md.
export const FIXTURES = {
  scans: {
    key: { min_peaks: 3 },
    kind: 'scans',
    value: {
      n_scans: 5298,
      n_compounds: 727,
      source: 'AC_CAC_MSLibrary2020_V1D1B.msp',
      instrument: 'Orbitrap HCD',
      polarity: 'P',
      nce_levels: [10, 15, 20, 25, 30, 40, 50, 60, 80],
    },
  },
  coords: {
    key: { alpha: 1.0, beta: 1.0, k_neighbors: 5 },
    kind: 'coords',
    value: {
      n: 5298,
      axes: ['s_k', 's_t', 's_e'],
      alpha: 1.0,
      beta: 1.0,
      k_neighbors: 5,
    },
  },
  grouping: {
    key: { key: 'compound' },
    kind: 'object',
    value: {
      key: 'compound',
      n_groups: 727,
      n_items: 5298,
      min_group: 1,
      median_group: 8,
      max_group: 9,
    },
  },
  separation: {
    key: { key: 'compound', min_group: 9 },
    kind: 'object',
    value: {
      key: 'compound',
      axes: ['s_k', 's_t', 's_e'],
      n_groups: 313,
      mean_within: 1.154528421982867,
      mean_between: 1.0735511796040589,
      separation_ratio: 0.929861196279835,
    },
  },
  drift: {
    key: { over: 'nce' },
    kind: 'object',
    value: {
      over: 'nce',
      n: 5298,
      axes: {
        s_k: {
          slope: -0.020597497307482764,
          pearson_r: -0.32699660262134744,
          r_squared: 0.10692677812590341,
        },
        s_t: {
          slope: -1.706610906347266e-5,
          pearson_r: -0.008823682567485547,
          r_squared: 7.785737405174834e-5,
        },
        s_e: {
          slope: 0.005589525342575712,
          pearson_r: 0.38285909060726214,
          r_squared: 0.14658108326061978,
        },
      },
    },
  },
  shuffled: {
    key: { key: 'compound', min_group: 9, seed: 20260818 },
    kind: 'object',
    value: {
      key: 'compound',
      axes: ['s_k', 's_t', 's_e'],
      n_groups: 313,
      mean_within: 1.64011431204381,
      mean_between: 0.5690962785969732,
      separation_ratio: 0.3469857402120955,
      control: 'label_shuffle',
      seed: 20260818,
    },
  },
  baseline: {
    key: { key: 'compound', over: 'nce', tolerance: 0.01, min_group: 9 },
    kind: 'object',
    value: {
      metric: 'cosine_similarity',
      key: 'compound',
      over: 'nce',
      tolerance: 0.01,
      n_groups: 313,
      n_pairs: 11268,
      mean_within_compound: 0.5219657294740804,
      mean_adjacent_level: 0.9176754590228843,
    },
  },
}

/* Build a replayed op.

   `fixture` is the recorded binding. Every key in its `key` object is an
   argument whose value the fixture is conditional on: if the caller
   passes a different one, there is no honest answer to give and the op
   refuses. Arguments absent from the call take the fixture value, which
   is what the original program passed. */
function replayed(name, fixtureName) {
  return (args, env, ast, emit) => {
    const fx = FIXTURES[fixtureName]
    for (const [k, want] of Object.entries(fx.key)) {
      if (!args.has(k)) continue
      const got = args.get(k)
      if (got !== want)
        throw new RefusalError(
          name + ' is replayed from a recorded run, and that run used ' + k + ' = ' +
            JSON.stringify(want) + '. You asked for ' + JSON.stringify(got) +
            ', which is not covered by any shipped fixture. Running it would need ' +
            'the full NIST library and the Python implementation; returning the ' +
            'recorded number would be answering a different question. Set ' + k +
            ' back to ' + JSON.stringify(want) + ' to replay, or run this ' +
            'experiment under shapeshifter-py.'
        )
    }
    emit('warn', '  ' + name + ': replayed from a recorded run, not computed here')
    return [{ ...fx.value, _replayed: true, _op: name }, fx.kind]
  }
}

// Effects are declared honestly: the recorded run really did read files,
// and the compile-stage audit reports those inputs as it always did.
const R = (name, fixture, effects) => ({
  fn: replayed(name, fixture),
  effects,
  inputs: [],
  replayed: true,
})

export const REPLAY_OPS = {
  'lavoisier.acquire.read_msp': R('read_msp', 'scans', ['read']),
  'lavoisier.acquire.filter_scans': R('filter_scans', 'scans', ['pure']),
  'lavoisier.transform.sentropy': R('sentropy', 'coords', ['pure']),
  'lavoisier.analyse.group_by': R('group_by', 'grouping', ['pure']),
  'lavoisier.analyse.separation': R('separation', 'separation', ['pure']),
  'lavoisier.analyse.drift': R('drift', 'drift', ['pure']),
  'lavoisier.analyse.baseline': R('baseline', 'baseline', ['pure']),
  'lavoisier.analyse.shuffle_control': R('shuffle_control', 'shuffled', ['pure']),
}

export const REPLAYED_OP_NAMES = Object.keys(REPLAY_OPS)
