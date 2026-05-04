import Head from "next/head";
import dynamic from "next/dynamic";
import TransitionEffect from "@/components/TransitionEffect";

// Load on the client only — D3 + Three relies on window.
const ExperimentDesigner = dynamic(
  () => import("@/components/experiment/ExperimentDesigner"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center min-h-[70vh] text-dark dark:text-light">
        <div className="text-center">
          <div className="animate-pulse text-2xl font-bold mb-2">
            booting virtual instrument…
          </div>
          <div className="text-sm text-dark/60 dark:text-light/60">
            assembling partition Lagrangian on this device
          </div>
        </div>
      </div>
    ),
  }
);

export default function ExperimentPage() {
  return (
    <>
      <Head>
        <title>Virtual Experiment — Lavoisier</title>
        <meta name="description"
          content="Design a mass spectrometry experiment and run it as a forward simulation on this device. Take the predicted library to your lab." />
      </Head>
      <TransitionEffect />
      <article className="flex flex-col min-h-screen text-dark dark:text-light">
        <ExperimentDesigner />
      </article>
    </>
  );
}
