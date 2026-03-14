import dynamic from "next/dynamic";
import React, { useState } from "react";
import LandingPage from "../src/components/home/landing-page";
import Layout from "../src/layout/layout";
import ContactDefault from "../src/components/contact/contact-default";
import Header from "../src/layout/header";
import LeftRightBar from "../src/layout/left-right-bar";
import Mobilemenu from "../src/layout/mobilemenu";
import Modalbox from "../src/layout/modalbox";
import TopBar from "../src/layout/top-bar";

const GLBViewer = dynamic(
  () => import("../src/components/model/GLBViewer"),
  { ssr: false }
);

const Metabolomics = dynamic(
  () => import("../src/components/metabolomics/metabolomics"),
  { ssr: false }
);

const Proteomics = dynamic(
  () => import("../src/components/proteomics/proteomics"),
  { ssr: false }
);

const Chromatography = dynamic(
  () => import("../src/components/chromatography/chromatography"),
  { ssr: false }
);

const Charge = dynamic(
  () => import("../src/components/charge/charge"),
  { ssr: false }
);

const Mass = dynamic(
  () => import("../src/components/mass/mass"),
  { ssr: false }
);

const Union = dynamic(
  () => import("../src/components/union/union"),
  { ssr: false }
);

export default function Home() {
  const [ActiveIndex, setActiveIndex] = useState(0);
  const handleOnClick = (index) => {
    setActiveIndex(index);
  };

  const [isToggled, setToggled] = useState(false);
  const toggleTrueFalse = () => setToggled(!isToggled);

  return (
    <>
      <Layout>
        <Modalbox />
        <Header handleOnClick={handleOnClick} ActiveIndex={ActiveIndex} />
        <LeftRightBar />
        <TopBar toggleTrueFalse={toggleTrueFalse} isToggled={isToggled} />
        <Mobilemenu toggleTrueFalse={toggleTrueFalse} isToggled={isToggled} handleOnClick={handleOnClick} />

        {/* <!-- MAINPART --> */}
        <div className={ActiveIndex === 0 ? "cavani_tm_mainpart" : "cavani_tm_mainpart full-width"}>
          {ActiveIndex === 0 && <GLBViewer className="glb-viewer-container"/>}

          <div className="main_content">
            <LandingPage ActiveIndex={ActiveIndex} handleOnClick={handleOnClick} />

            <Metabolomics ActiveIndex={ActiveIndex} />

            <Proteomics ActiveIndex={ActiveIndex} />

            <Chromatography ActiveIndex={ActiveIndex} />

            <Charge ActiveIndex={ActiveIndex} />

            <Mass ActiveIndex={ActiveIndex} />

            <Union ActiveIndex={ActiveIndex} />

            <ContactDefault ActiveIndex={ActiveIndex} />
          </div>
        </div>
        {/* MAINPART */}
      </Layout>
    </>
  );
}
