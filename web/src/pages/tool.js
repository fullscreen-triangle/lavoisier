import Head from "next/head";
import dynamic from "next/dynamic";
import TransitionEffect from "@/components/TransitionEffect";

// Load the full workspace only on the client (WebGL2, File System Access API, etc.)
const Workspace = dynamic(() => import("@/components/tool/Workspace"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center min-h-[70vh] text-dark dark:text-light">
      <div className="text-center">
        <div className="animate-pulse text-2xl font-bold mb-2">Loading observation apparatus...</div>
        <div className="text-sm text-dark/60 dark:text-light/60">
          Initialising WebGL2 fragment shaders
        </div>
      </div>
    </div>
  ),
});

export default function ToolPage() {
  return (
    <>
      <Head>
        <title>Tool — Lavoisier</title>
      </Head>
      <TransitionEffect />
      <article className="flex flex-col min-h-screen text-dark dark:text-light">
        <Workspace />
      </article>
    </>
  );
}
