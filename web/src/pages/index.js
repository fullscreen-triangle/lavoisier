import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const Stat = ({ value, label }) => (
  <div className="flex flex-col items-center">
    <span className="text-5xl font-bold text-primary dark:text-primaryDark lg:text-4xl md:text-3xl">
      {value}
    </span>
    <span className="text-sm font-medium text-dark/60 dark:text-light/60 mt-1 text-center">
      {label}
    </span>
  </div>
);

const FeatureCard = ({ title, description, icon }) => (
  <motion.div
    className="p-6 rounded-2xl border-2 border-solid border-dark/10 dark:border-light/10 bg-light dark:bg-dark hover:border-primary dark:hover:border-primaryDark transition-colors"
    whileHover={{ y: -4 }}
  >
    <div className="text-3xl mb-3">{icon}</div>
    <h3 className="text-xl font-bold mb-2">{title}</h3>
    <p className="text-sm text-dark/70 dark:text-light/70 leading-relaxed">
      {description}
    </p>
  </motion.div>
);

export default function Home() {
  return (
    <>
      <Head>
        <title>Lavoisier — Force-Free Mass Spectrometry</title>
      </Head>

      <TransitionEffect />
      <article className="flex flex-col min-h-screen text-dark dark:text-light">
        <Layout className="!pt-8">
          {/* Hero */}
          <section className="flex flex-col items-center justify-center text-center py-20 md:py-12">
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="inline-flex items-center px-3 py-1 rounded-full border border-dark/20 dark:border-light/20 text-xs font-medium mb-6 text-dark/70 dark:text-light/70"
            >
              Bounded Phase Space Law · Partition Depth Minimisation
            </motion.div>

            <AnimatedText
              text="Mass spectrometry without forces."
              className="!text-6xl xl:!text-5xl lg:!text-5xl md:!text-4xl sm:!text-3xl !text-center"
            />

            <p className="mt-6 max-w-3xl text-lg md:text-base text-dark/80 dark:text-light/80 leading-relaxed">
              A browser-based analytical framework in which every stage of the ion journey—
              ionisation, optics, mass analysis, fragmentation, detection, signal—
              is a partition depth operation executed by a GPU fragment shader.
              No forces. No backend. No stored spectra.
            </p>

            <div className="mt-8 flex items-center gap-4 md:flex-col">
              <Link
                href="/tool"
                className="flex items-center rounded-lg border-2 border-solid bg-dark px-6 py-3 text-lg font-semibold
                  text-light hover:border-dark hover:bg-transparent hover:text-dark
                  dark:bg-light dark:text-dark dark:hover:border-light dark:hover:bg-dark dark:hover:text-light
                  md:py-2 md:text-base transition-colors"
              >
                Open the Tool →
              </Link>

              <Link
                href="/framework"
                className="flex items-center px-6 py-3 text-lg font-semibold underline underline-offset-4
                  hover:text-primary dark:hover:text-primaryDark md:py-2 md:text-base"
              >
                Read the Framework
              </Link>
            </div>
          </section>

          {/* Stats */}
          <section className="grid grid-cols-4 gap-8 py-16 border-t-2 border-b-2 border-dark/10 dark:border-light/10 md:grid-cols-2 md:gap-6">
            <Stat value="0" label="Forces invoked at any stage" />
            <Stat value="O(1)" label="GPU memory, independent of N" />
            <Stat value="25 MB" label="Total shader apparatus" />
            <Stat value="10⁹×" label="Speedup at PubChem scale" />
          </section>

          {/* Features */}
          <section className="py-16">
            <h2 className="text-3xl font-bold mb-2 text-center">The stack</h2>
            <p className="text-center text-dark/60 dark:text-light/60 mb-10 text-sm">
              Three layers. One axiom. Runs entirely in your browser.
            </p>

            <div className="grid grid-cols-3 gap-6 md:grid-cols-1">
              <FeatureCard
                icon="◉"
                title="Purpose Layer"
                description="Compiled probe selects which regions of S-entropy space to observe. Domain context constrains the partition depth landscape before any generation."
              />
              <FeatureCard
                icon="◈"
                title="Mass Computing"
                description="Ternary addresses encode both position and trajectory. Spectra are read from partition structure, not computed from dynamics."
              />
              <FeatureCard
                icon="▦"
                title="GPU Observation"
                description="Six shader passes: ionisation, optics, analysis, fragmentation, detection, signal. The fragment shader IS the instrument."
              />
            </div>
          </section>

          {/* Pipeline */}
          <section className="py-16 border-t-2 border-dark/10 dark:border-light/10">
            <h2 className="text-3xl font-bold mb-10 text-center">The force-free ion journey</h2>

            <div className="grid grid-cols-6 gap-2 md:grid-cols-3 sm:grid-cols-2">
              {[
                { n: 1, title: "Ionisation", subtitle: "partition creation" },
                { n: 2, title: "Ion Optics", subtitle: "−∇M descent" },
                { n: 3, title: "Mass Analysis", subtitle: "topology navigation" },
                { n: 4, title: "Fragmentation", subtitle: "M redistribution" },
                { n: 5, title: "Detection", subtitle: "partition completion" },
                { n: 6, title: "Signal", subtitle: "categorical state flux" },
              ].map((p, i) => (
                <motion.div
                  key={p.n}
                  className="p-4 rounded-lg border-2 border-solid border-dark/10 dark:border-light/10 text-center"
                  initial={{ opacity: 0, x: -10 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  viewport={{ once: true }}
                >
                  <div className="text-xs text-primary dark:text-primaryDark font-bold">
                    Pass {p.n}
                  </div>
                  <div className="text-base font-semibold mt-1">{p.title}</div>
                  <div className="text-xs text-dark/50 dark:text-light/50 mt-1">
                    {p.subtitle}
                  </div>
                </motion.div>
              ))}
            </div>
          </section>
        </Layout>
      </article>
    </>
  );
}
