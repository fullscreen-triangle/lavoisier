import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import Head from "next/head";

const Eq = ({ children }) => (
  <div className="my-4 p-4 rounded-lg bg-dark/5 dark:bg-light/5 font-mono text-center text-base md:text-sm overflow-x-auto">
    {children}
  </div>
);

const Theorem = ({ name, children }) => (
  <div className="my-6 p-5 rounded-lg border-l-4 border-primary dark:border-primaryDark bg-light dark:bg-dark">
    <div className="text-xs uppercase tracking-wider font-bold text-primary dark:text-primaryDark mb-2">
      Theorem — {name}
    </div>
    <div>{children}</div>
  </div>
);

export default function Framework() {
  return (
    <>
      <Head>
        <title>Framework — Lavoisier</title>
      </Head>
      <TransitionEffect />
      <article className="flex flex-col min-h-screen text-dark dark:text-light">
        <Layout className="!pt-8">
          <AnimatedText
            text="The framework."
            className="!text-6xl xl:!text-5xl lg:!text-5xl md:!text-4xl sm:!text-3xl mb-4"
          />
          <p className="text-lg text-dark/70 dark:text-light/70 mb-12 md:text-base">
            One axiom. Zero forces. Zero adjustable parameters.
          </p>

          <div className="max-w-4xl space-y-10">
            <section>
              <h2 className="text-3xl font-bold mb-4">1. The Axiom</h2>
              <p className="text-lg md:text-base leading-relaxed mb-4">
                All persistent dynamical systems occupy bounded regions of
                phase space with finite Liouville measure, and these bounded
                regions admit hierarchical partitioning into distinguishable
                subregions.
              </p>
              <Eq>μ(Ω) = ∫ dⁿq dⁿp / hⁿ &lt; ∞</Eq>
            </section>

            <section>
              <h2 className="text-3xl font-bold mb-4">2. Oscillatory Necessity</h2>
              <p className="leading-relaxed md:text-base">
                By Poincaré recurrence, bounded measure-preserving systems
                return to their initial region infinitely often. Static
                dynamics lacks evolution; monotonic dynamics diverges; chaotic
                dynamics destroys identity. Only oscillation survives.
              </p>
              <Theorem name="Mode Decomposition">
                Oscillatory dynamics in bounded systems admits discrete modes{" "}
                <code>{`{ωₖ}`}</code> from the Koopman operator spectrum.
                Energy-frequency identity: E = ℏω.
              </Theorem>
            </section>

            <section>
              <h2 className="text-3xl font-bold mb-4">3. The Triple Equivalence</h2>
              <p className="leading-relaxed md:text-base">
                For bounded systems with M degrees of freedom and n states
                each, three descriptions are mathematically identical:
              </p>
              <Eq>
                Ω<sub>osc</sub> = Ω<sub>cat</sub> = Z<sub>part</sub> = n<sup>M</sup>
                <br />
                S = k<sub>B</sub> M ln n
              </Eq>
              <Theorem name="Observation-Computation Equivalence">
                Rendering partition cells via a GPU fragment shader ≡ observing
                categorical states ≡ computing partition properties. The pixel
                value IS the observed state, not a picture of it.
              </Theorem>
            </section>

            <section>
              <h2 className="text-3xl font-bold mb-4">4. S-Entropy Coordinates</h2>
              <p className="leading-relaxed mb-3 md:text-base">
                Three independent quantities characterise any bounded
                oscillatory system:
              </p>
              <ul className="space-y-2 text-base pl-6 list-disc">
                <li>
                  <strong>S<sub>k</sub></strong> (knowledge): Shannon entropy of
                  the frequency distribution.
                </li>
                <li>
                  <strong>S<sub>t</sub></strong> (temporal): logarithmic ratio
                  of frequency extremes.
                </li>
                <li>
                  <strong>S<sub>e</sub></strong> (evolution): fraction of
                  harmonically proximate mode pairs.
                </li>
              </ul>
              <p className="mt-4 md:text-base">
                Three dimensions, base-3 ternary encoding. The address IS the
                trajectory (Position-Trajectory Duality).
              </p>
            </section>

            <section>
              <h2 className="text-3xl font-bold mb-4">5. The Partition Lagrangian</h2>
              <Eq>L_M = ½μ|ẋ|² + μẋ·A_M − M(x,t)</Eq>
              <p className="leading-relaxed md:text-base">
                All four mass analyser types emerge as field topology
                specialisations:
              </p>
              <div className="grid grid-cols-2 gap-3 mt-4 md:grid-cols-1">
                {[
                  ["TOF", "M = −κz", "T ∝ √(m/z)"],
                  ["Quadrupole", "M = κ₀(x²−y²)[U+Vcos Ωt]/2", "Mathieu stability"],
                  ["Orbitrap", "M = κ(z²−r²/2)/2", "ω ∝ √(z/m)"],
                  ["FT-ICR", "A_M = B(−y,x,0)/2", "ωc ∝ z/m"],
                ].map(([name, field, obs]) => (
                  <div
                    key={name}
                    className="p-4 rounded border border-dark/10 dark:border-light/10"
                  >
                    <div className="font-bold text-primary dark:text-primaryDark">
                      {name}
                    </div>
                    <div className="font-mono text-sm my-1">{field}</div>
                    <div className="text-xs text-dark/60 dark:text-light/60">
                      {obs}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h2 className="text-3xl font-bold mb-4">6. Elimination of Forces</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse md:text-sm">
                  <thead>
                    <tr className="border-b-2 border-dark dark:border-light">
                      <th className="py-2">&ldquo;Force&rdquo;</th>
                      <th className="py-2">Partition operation</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm">
                    {[
                      ["Gravity", "∇M — partition depth gradient"],
                      ["Strong nuclear", "Shared partition structure (M_bound < M_free)"],
                      ["Electromagnetic", "Categorical state propagation at c"],
                      ["Weak", "Partition coordinate transition (large τ_p)"],
                    ].map(([f, p]) => (
                      <tr key={f} className="border-b border-dark/10 dark:border-light/10">
                        <td className="py-2 font-semibold">{f}</td>
                        <td className="py-2 font-mono">{p}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section>
              <h2 className="text-3xl font-bold mb-4">7. The Six-Pass Ion Journey</h2>
              <ol className="space-y-3 list-decimal pl-6 md:text-base">
                <li>
                  <strong>Ionisation</strong> — partition creation: disrupt
                  categorical completeness, creating a malformation with M above
                  minimum.
                </li>
                <li>
                  <strong>Ion optics</strong> — gradient descent: ion follows
                  −∇M through electrode-shaped partition landscape.
                </li>
                <li>
                  <strong>Mass analysis</strong> — topology navigation: ion
                  traverses analyser-specific M(x,t) field.
                </li>
                <li>
                  <strong>Fragmentation</strong> — M redistribution: M_parent →
                  ΣM_fragment under selection rules.
                </li>
                <li>
                  <strong>Detection</strong> — partition completion: malformation
                  resolves at detector M-minimum.
                </li>
                <li>
                  <strong>Signal</strong> — categorical state flux through
                  phase-locked electronics at ~10⁸ m/s.
                </li>
              </ol>
            </section>

            <section>
              <h2 className="text-3xl font-bold mb-4">8. O(1) Memory</h2>
              <p className="leading-relaxed md:text-base">
                Observation is free to re-perform. Storing observations is
                redundant. The GPU holds the observation apparatus (shaders),
                not the data. Total memory: ~25 MB regardless of database size.
              </p>
              <Eq>
                Total GPU = shaders (10 MB) + 3 textures (3 MB) + working (12 MB)
              </Eq>
            </section>
          </div>
        </Layout>
      </article>
    </>
  );
}
