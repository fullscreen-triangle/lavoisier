import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import Head from "next/head";

const papers = [
  {
    title: "Categorical Compound Database",
    subtitle: "Oscillation-Counted Molecular Identification Through Ternary Phase Space Addressing",
    abstract:
      "Molecular database search in O(k) trie traversal, independent of database size. Each trit is one oscillation-counting observation. Validated on 39 NIST compounds; 5/6 chemical families show ternary cohesion without chemical knowledge encoded. 3,328× speedup over brute-force fingerprint search; 10⁹× projected at PubChem scale.",
    tag: "Mass Computing",
    tagColor: "primary",
  },
  {
    title: "Spectroscopic Derivation of the Chemical Elements",
    subtitle:
      "Atomic Structure and Virtual Instrumentation from Bounded Phase Space Geometry",
    abstract:
      "Derives partition coordinates (n, ℓ, m, s), capacity C(n)=2n², aufbau sequence, selection rules, and exclusion principle from a single axiom. Proves the Triple Equivalence and establishes computer hardware oscillators as physical spectrometers. Nine elements (H, C, Na, Si, Cl, Ar, Ca, Fe, Gd) validated with zero adjustable parameters.",
    tag: "Foundations",
    tagColor: "primaryDark",
  },
  {
    title: "Ion Trajectory Completion Mechanism",
    subtitle: "Partition Depth Trajectory Dynamics in Geometric Coordinate Systems",
    abstract:
      "Derives mass from first principles as accumulated partition residue. E = mc² emerges as a theorem. All four mass analyser equations (TOF, Quadrupole, Orbitrap, FT-ICR) derived from a single partition Lagrangian with different field topologies. Validated on 4,545 NIST entries.",
    tag: "Foundations",
    tagColor: "primaryDark",
  },
  {
    title: "Context-Based Spectral Database",
    subtitle:
      "Generative Phase Space Addressing for Mass Spectrometric Identification from First Principles",
    abstract:
      "A spectral database that generates ion trajectories on demand rather than storing measured spectra. The phase space structure IS the database. O(1) storage. O(k) query. Ternary trie + partition Lagrangian + ion-droplet bijection = complete identification without stored entries.",
    tag: "Generative DB",
    tagColor: "primary",
  },
  {
    title: "Purpose-Based Spectral Analysis",
    subtitle:
      "Domain-Constrained Generative Identification Through Oscillatory Resonance in Bounded Phase Space",
    abstract:
      "Comparison is oscillatory resonance, not algorithmic computation. Dual oscillatory paths (spectral + droplet) provide algorithm-free cross-validation. Purpose function maps domain context to phase space subregions with calculable Landauer cost. >95% reduction across metabolomics, glycomics, proteomics.",
    tag: "Purpose",
    tagColor: "primary",
  },
  {
    title: "Observation-Based Mass Computing",
    subtitle:
      "GPU Fragment Shaders as Physical Measurement Apparatus for Partition Synthesis in Bounded Phase Space",
    abstract:
      "The capstone. Proves that a GPU fragment shader evaluating partition functions IS a physical observation apparatus. Four-pass architecture. The compiled probe trained on GPU physical observables, not human labels. The complete framework runs on a laptop integrated GPU at O(1) memory.",
    tag: "Capstone",
    tagColor: "primaryDark",
  },
  {
    title: "The Force-Free Mass Spectrometer",
    subtitle:
      "GPU Fragment Shaders as Partition Depth Operators for Complete Ion Journey Synthesis Without Forces",
    abstract:
      "Every stage of the ion journey—ionisation, optics, analysis, fragmentation, detection, signal—is a partition depth operation. Each operation is one shader pass. Zero forces invoked anywhere. Validated: all analyser scaling laws at errors < 10⁻⁴, copper resistivity exact, BCS gap ratios within 5%.",
    tag: "Force-Free",
    tagColor: "primary",
  },
];

const PaperCard = ({ p, i }) => (
  <motion.article
    className="p-6 rounded-2xl border-2 border-solid border-dark/10 dark:border-light/10 bg-light dark:bg-dark
      hover:border-primary dark:hover:border-primaryDark transition-colors"
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    transition={{ delay: i * 0.05 }}
    viewport={{ once: true }}
  >
    <div
      className={`inline-block px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider mb-3
        ${
          p.tagColor === "primary"
            ? "bg-primary/10 text-primary dark:bg-primaryDark/10 dark:text-primaryDark"
            : "bg-primaryDark/10 text-primaryDark dark:bg-primary/10 dark:text-primary"
        }`}
    >
      {p.tag}
    </div>
    <h3 className="text-xl font-bold mb-1">{p.title}</h3>
    <p className="text-sm italic text-dark/60 dark:text-light/60 mb-3">{p.subtitle}</p>
    <p className="text-sm leading-relaxed text-dark/80 dark:text-light/80">
      {p.abstract}
    </p>
  </motion.article>
);

export default function Papers() {
  return (
    <>
      <Head>
        <title>Papers — Lavoisier</title>
      </Head>
      <TransitionEffect />
      <article className="flex flex-col min-h-screen text-dark dark:text-light">
        <Layout className="!pt-8">
          <AnimatedText
            text="The theoretical foundations."
            className="!text-6xl xl:!text-5xl lg:!text-5xl md:!text-4xl sm:!text-3xl mb-4"
          />
          <p className="text-lg text-dark/70 dark:text-light/70 mb-12 md:text-base">
            Seven papers. One axiom. Zero adjustable parameters.
          </p>

          <div className="grid grid-cols-2 gap-6 md:grid-cols-1 max-w-6xl">
            {papers.map((p, i) => (
              <PaperCard key={p.title} p={p} i={i} />
            ))}
          </div>
        </Layout>
      </article>
    </>
  );
}
