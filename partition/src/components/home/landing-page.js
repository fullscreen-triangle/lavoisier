import Link from "next/link";
import React from "react";
import { RotateTextAnimation } from "../AnimationText";

export default function LandingPage({ ActiveIndex, handleOnClick }) {
  return (
    <>
      {/* <!-- HOME --> */}
      <div
        className={
          ActiveIndex === 0
            ? "cavani_tm_section active animated fadeInUp"
            : "cavani_tm_section active hidden animated"
        }
        id="home_"
      >
        <div className="cavani_tm_home">
          <div className="content">
            <h3 className="name">Partition</h3>
            <span className="line"></span>
            <h3 className="job">
              <RotateTextAnimation />
            </h3>
            <div className="cavani_tm_button transition_link">
              <Link href="#contact">
                <a onClick={() => handleOnClick(4)}>Collaborate</a>
              </Link>
            </div>
          </div>
        </div>
      </div>
      {/* <!-- HOME --> */}
    </>
  );
}
