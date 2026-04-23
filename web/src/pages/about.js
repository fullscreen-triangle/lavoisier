import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import Head from "next/head";
import Link from "next/link";

export default function About() {
  return (
    <>
      <Head>
        <title>About — Lavoisier</title>
      </Head>
      <TransitionEffect />
      <article className="flex flex-col min-h-screen text-dark dark:text-light">
        <Layout className="!pt-8">
          <AnimatedText
            text="About Lavoisier."
            className="!text-6xl xl:!text-5xl lg:!text-5xl md:!text-4xl sm:!text-3xl mb-8"
          />

          <div className="prose max-w-4xl text-lg leading-relaxed space-y-6 md:text-base">
            <p>
              Lavoisier is the computational implementation of the{" "}
              <Link href="/framework" className="text-primary dark:text-primaryDark underline">
                Bounded Phase Space Law
              </Link>{" "}
              applied to mass spectrometry. Named after Antoine Lavoisier,
              whose conservation principles dissolved the force-based phlogiston
              theory of combustion two centuries ago, this framework demonstrates
              that modern mass spectrometry similarly needs no forces.
            </p>

            <p>
              From a single axiom—physical systems occupy finite regions of
              phase space admitting partition and nesting—we derive oscillatory
              necessity, partition coordinates, the Triple Equivalence
              (oscillation ≡ counting ≡ partition), and ultimately the conclusion
              that a GPU fragment shader evaluating partition functions IS a
              physical observation apparatus, not a simulation of one.
            </p>

            <p>
              The entire ion journey—from electrospray ionisation to digitised
              signal—decomposes into six partition operations. Each operation is
              implemented as one fragment shader pass. The GPU is not accelerating
              the instrument; it{" "}
              <em className="text-primary dark:text-primaryDark not-italic font-semibold">
                is
              </em>{" "}
              the instrument, by the Processor-Oscillator Duality.
            </p>

            <h2 className="text-3xl font-bold mt-12 mb-4">Why this matters</h2>

            <ul className="space-y-3 list-disc pl-6">
              <li>
                <strong>Privacy by construction.</strong> No spectral data leaves
                your browser. Pharma, clinical, forensic, and regulatory
                requirements are satisfied without additional infrastructure.
              </li>
              <li>
                <strong>O(1) memory.</strong> The phase space structure is the
                database. Nothing is stored. Any compound&apos;s trajectory is
                generated on demand from the partition Lagrangian.
              </li>
              <li>
                <strong>Deterministic identification.</strong> The address is the
                identity. Two users running the same query get the same ternary
                address with the same resonance score. No probabilistic library
                matching.
              </li>
              <li>
                <strong>Runs on a laptop.</strong> The complete six-pass
                apparatus fits in ~25 MB of GPU memory. A 2 GB integrated GPU
                has 80× headroom.
              </li>
            </ul>

            <h2 className="text-3xl font-bold mt-12 mb-4">The philosophy</h2>

            <p>
              Forces are a useful shorthand that obscures the underlying
              structure. The Lorentz force is the Euler–Lagrange equation for
              partition depth minimisation. Mass is partition inertia. Charge is
              partition malformation. Current is categorical state propagation,
              not charge transport—proven by the twelve-order-of-magnitude ratio
              between signal velocity (10⁸ m/s) and drift velocity (10⁻⁴ m/s) in
              any conductor.
            </p>

            <p>
              Removing forces does not change any prediction. It reveals what
              mass spectrometry has always measured: the partition depth
              structure of bounded phase space.
            </p>
          </div>
        </Layout>
      </article>
    </>
  );
}
