import Head from "next/head";
import dynamic from "next/dynamic";
import TransitionEffect from "@/components/TransitionEffect";

const ShapeshifterSandbox = dynamic(
  () => import("@/sandbox/ShapeshifterSandbox"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center min-h-[70vh] text-dark dark:text-light">
        <div className="text-center">
          <div className="animate-pulse text-2xl font-bold mb-2">
            loading shapeshifter…
          </div>
          <div className="text-sm text-dark/60 dark:text-light/60">
            initialising compiler and GPU pipeline
          </div>
        </div>
      </div>
    ),
  }
);

export default function SandboxPage() {
  return (
    <>
      <Head>
        <title>Shapeshifter Sandbox — Lavoisier</title>
        <meta name="description"
          content="Live compiler for the Shapeshifter mass spectrometry DSL. Write .ss scripts and execute them against the virtual instrument." />
      </Head>
      <TransitionEffect />
      <article className="flex flex-col min-h-screen text-dark dark:text-light">
        <div className="px-4 py-4 sm:px-8">
          <ShapeshifterSandbox />
        </div>
      </article>
    </>
  );
}
