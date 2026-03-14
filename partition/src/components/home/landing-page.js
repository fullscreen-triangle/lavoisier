// Landing.jsx
import Link from "next/link";
import React from "react";
import { RotateTextAnimation } from "../AnimationText";
import GLBViewer from "../model/GLBViewer"; // Import your GLB viewer

export default function LandingPage({ ActiveIndex, handleOnClick }) {
  return (
    <>
      {/* Landing Page Container */}
      <div
        className={
          ActiveIndex === 0
            ? "landing-page-container active"
            : "landing-page-container hidden"
        }
        id="home_"
      >
        {/* 3D Background */}
        {ActiveIndex === 0 && (
          <div className="landing-glb-background">
            <GLBViewer 
              modelPath="/glb/conformational_transition_of_troponin.glb"
              autoRotate={true}
              showControls={false}
              backgroundColor="transparent"
            />
          </div>
        )}

        {/* Content Overlay */}
        <div className="landing-content-wrapper">
          <div className="landing-hero">
            <div className="landing-text-content">
              <h1 className="landing-title">Partition</h1>
              <span className="landing-subtitle"></span>
              <h2 className="landing-job">
                <RotateTextAnimation />
              </h2>
              <div className="landing-cta">
                <Link href="#contact">
                  <a 
                    className="landing-button"
                    onClick={() => handleOnClick(5)}
                  >
                    Get in Touch
                  </a>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
