import { Fragment, useEffect, useState } from "react";
import Modal from "react-modal";
import { CloseButton } from "../plugin/svg";
const News = ({ ActiveIndex, animation }) => {
  const [isOpen4, setIsOpen4] = useState(false);
  const [modalContent, setModalContent] = useState({});

  useEffect(() => {
    var lists = document.querySelectorAll(".news_list > ul > li");
    let box = document.querySelector(".cavani_fn_moving_box");
    if (!box) {
      let body = document.querySelector("body");
      let div = document.createElement("div");
      div.classList.add("cavani_fn_moving_box");
      body.appendChild(div);
    }

    lists.forEach((list) => {
      list.addEventListener("mouseenter", (event) => {
        box.classList.add("opened");
        var imgURL = list.getAttribute("data-img");
        box.style.backgroundImage = `url(${imgURL})`;
        box.style.top = event.clientY - 50 + "px";
        if (imgURL === "") {
          box.classList.remove("opened");
          return false;
        }
      });
      list.addEventListener("mouseleave", () => {
        box.classList.remove("opened");
      });
    });
  }, []);

  function toggleModalFour(value) {
    setIsOpen4(!isOpen4);
    setModalContent(value);
  }
  const newsData = [
    {
      img: "img/news/1.jpg",
      tag: "Mass Spectrometry",
      date: "2025",
      comments: "Lavoisier Framework",
      title: "Categorical State Counting in Mass Spectrometry",
      text1:
        "This publication introduces a categorical framework for enumerating ion states in mass spectrometry. By treating the mass spectrum as a partition problem, we derive exact state counts for complex mixtures without combinatorial explosion.",
      text2:
        "The key insight is that ion populations in a mass spectrometer obey partition identities — the same mathematics that govern integer partitions in number theory. This connection allows us to import powerful counting techniques from combinatorics.",
      text3:
        "Validated against 20+ panels of experimental data from Waters qTOF and Thermo Orbitrap instruments, the categorical approach consistently matches or exceeds the accuracy of conventional peak-picking algorithms.",
    },
    {
      img: "img/news/2.jpg",
      tag: "Proteomics",
      date: "2025",
      comments: "Lavoisier Framework",
      title: "Derivation of Mass from First Principles",
      text1:
        "This work re-derives the mass spectrum from fundamental physical principles — ionization energetics, transport phenomena, and detector response functions — rather than treating it as an empirical signal to be processed.",
      text2:
        "By modelling the complete ion journey from source to detector, we obtain a forward model that predicts spectral features ab initio. Discrepancies between predicted and observed spectra reveal systematic biases in instrument calibration.",
      text3:
        "The derivation unifies fragmentation patterns, charge state distributions, and isotope envelopes under a single mathematical framework, enabling more accurate protein identification from tandem mass spectrometry data.",
    },
    {
      img: "img/news/3.jpg",
      tag: "Mathematics",
      date: "2024",
      comments: "Lavoisier Framework",
      title: "Bounded Phase Spaces and Ion Categories",
      text1:
        "We establish rigorous bounds on the phase space available to ions in a mass spectrometer, showing that physical constraints (energy conservation, angular momentum, detector geometry) reduce the effective state space dramatically.",
      text2:
        "These bounds are expressed as categorical limits — functors from the category of physical constraints to the category of admissible ion states. The resulting phase space is a proper subcategory of the naive combinatorial space.",
      text3:
        "Practical implications include faster spectral matching algorithms (searching a bounded space rather than an exponential one) and improved false discovery rate estimation in database search approaches.",
    },
    {
      img: "img/news/4.jpg",
      tag: "Computational",
      date: "2024",
      comments: "Lavoisier Framework",
      title: "S-Entropy and State Counting in Analytical Chemistry",
      text1:
        "We introduce S-entropy as a measure of informational content in mass spectra. Unlike Shannon entropy, S-entropy accounts for the structured nature of spectral data — peaks are not independent symbols but correlated measurements.",
      text2:
        "S-entropy provides a principled criterion for spectrum quality assessment, feature selection, and optimal binning. Spectra with high S-entropy contain more analytically useful information per unit of measurement time.",
      text3:
        "The state counting methodology connects to Boltzmann statistics through the Loschmidt number, providing a bridge between thermodynamic descriptions of molecular populations and information-theoretic descriptions of spectral data.",
    },
    {
      img: "img/news/5.jpg",
      tag: "Validation",
      date: "2025",
      comments: "Lavoisier Framework",
      title: "NIST Library Validation of the Partition Framework",
      text1:
        "Comprehensive validation of the Lavoisier partition framework against the NIST Mass Spectral Library. We processed standard reference compounds and compared identifications with the authoritative NIST database.",
      text2:
        "The dual-pipeline architecture — numerical decomposition and computer vision analysis operating independently — achieved concordant identifications in over 94% of test cases, with the remaining discrepancies traceable to known edge cases.",
      text3:
        "This validation establishes Lavoisier as a viable alternative to existing mass spectrometry software, with particular advantages in complex mixture analysis where traditional approaches struggle with spectral overlap.",
    },
    {
      img: "img/news/6.jpg",
      tag: "Ion Physics",
      date: "2024",
      comments: "Lavoisier Framework",
      title: "Ion Transport Phenomena in Mass Spectrometry",
      text1:
        "A detailed model of ion transport from electrospray source through quadrupole mass filter to time-of-flight detector. The model captures space-charge effects, collisional cooling, and RF confinement dynamics.",
      text2:
        "By solving the transport equations numerically, we predict ion transmission efficiency as a function of m/z, charge state, and instrument parameters. These predictions inform the forward model used in spectral interpretation.",
      text3:
        "The transport model reveals that ion losses are not uniform across the mass range — certain m/z windows experience enhanced or suppressed transmission due to resonance effects in the RF fields, explaining systematic biases in quantitative measurements.",
    },
  ];
  return (
    <Fragment>
      <div
        className={
          ActiveIndex === 3
            ? `cavani_tm_section active animated ${animation ? animation : "fadeInUp"
            }`
            : "cavani_tm_section hidden animated"
        }
        id="news__"
      >
        <div className="section_inner">
          <div className="cavani_tm_news">
            <div className="cavani_tm_title">
              <span>Publications &amp; Findings</span>
            </div>
            <div className="news_list">
              <ul>
                {newsData.map((news, i) => (
                  <li data-img={`img/news/${i + 1}.jpg`} key={i}>
                    <div className="list_inner">
                      <span className="number">{`${i <= 9 ? 0 : ""}${i + 1
                        }`}</span>
                      <div className="details">
                        <div className="extra_metas">
                          <ul>
                            <li>
                              <span>{news.date}</span>
                            </li>
                            <li>
                              <span>
                                <a
                                  href="#"
                                  onClick={() => toggleModalFour(news)}
                                >
                                  {news.tag}
                                </a>
                              </span>
                            </li>
                            <li>
                              <span>
                                <a
                                  href="#"
                                  onClick={() => toggleModalFour(news)}
                                >
                                  {news.comments}
                                </a>
                              </span>
                            </li>
                          </ul>
                        </div>
                        <div className="post_title">
                          <h3>
                            <a href="#" onClick={() => toggleModalFour(news)}>
                              {news.title}
                            </a>
                          </h3>
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
      {modalContent && (
        <Modal
          isOpen={isOpen4}
          onRequestClose={toggleModalFour}
          contentLabel="My dialog"
          className="mymodal"
          overlayClassName="myoverlay"
          closeTimeoutMS={300}
          openTimeoutMS={300}
        >
          <div className="cavani_tm_modalbox opened">
            <div className="box_inner">
              <div className="close" onClick={toggleModalFour}>
                <a href="#">
                  <i className="icon-cancel"></i>
                </a>
              </div>
              <div className="description_wrap">
                <div className="news_popup_informations">
                  <div className="image">
                    <img src="img/thumbs/4-2.jpg" alt="" />
                    <div
                      className="main"
                      data-img-url="img/news/1.jpg"
                      style={{ backgroundImage: `url(${modalContent.img})` }}
                    />
                  </div>
                  <div className="details">
                    <div className="meta">
                      <ul>
                        <li><span>{modalContent.date}</span></li>
                        <li><span><a href="#">{modalContent.tag}</a></span></li>
                        <li><span><a href="#">{modalContent.comments}</a></span></li>
                      </ul>
                    </div>
                    <div className="title">
                      <h3>{modalContent.title}</h3>
                    </div>
                  </div>
                  <div className="text">
                    <p>{modalContent.text1}</p>
                    <p>{modalContent.text2}</p>
                    <p>{modalContent.text3}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Modal>
      )}
    </Fragment>
  );
};
export default News;
