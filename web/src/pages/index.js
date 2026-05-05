import Head from "next/head";
import dynamic from "next/dynamic";
import TransitionEffect from "@/components/TransitionEffect";

// Three.js + GLTFLoader run client-only.
const AtomScene = dynamic(() => import("@/components/AtomScene"), {
  ssr: false,
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Lavoisier</title>
      </Head>

      <TransitionEffect />
      <article className="flex flex-col min-h-screen text-dark dark:text-light">
        <section className="flex-1 flex items-center justify-center
          min-h-[calc(100vh-160px)]">
          <div className="w-[80vmin] h-[80vmin] max-w-[760px] max-h-[760px]">
            <AtomScene />
          </div>
        </section>
      </article>
    </>
  );
}
