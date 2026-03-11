import React from 'react'
import Image from 'next/image'
import ProgressBar from '../progressBar';
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';

const circleProgressData = [
    {language: 'Python', progress: 95 },
    {language: 'Rust', progress: 85 },
    {language: 'LaTeX', progress: 90 },

];

const progressBarData = [
    { bgcolor: "#7d7789", completed: 95, title: 'Mass Spectrometry' },
    { bgcolor: "#7d7789", completed: 90, title: 'Proteomics' },
    { bgcolor: "#7d7789", completed: 85, title: 'Computational Chemistry' },
];

const testimonials = [
    {
        desc: "The Lavoisier framework provides a rigorous mathematical foundation for mass spectrometry analysis, bridging category theory with practical analytical chemistry.",
        img: "img/testimonials/1.jpg",
        info1: "Categorical State Counting",
        info2: "Publication"

    },
    {
        desc: "By treating ion transport as a partition problem, we achieve exact decomposition of complex mass spectra without relying on heuristic peak-picking algorithms.",
        img: "img/testimonials/2.jpg",
        info1: "Ion Partition Theory",
        info2: "Core Framework"

    },
    {
        desc: "The dual-pipeline architecture — combining traditional numerical methods with computer vision — validates results through independent analytical pathways.",
        img: "img/testimonials/3.jpg",
        info1: "Dual-Pipeline Validation",
        info2: "Architecture"

    },
]

export default function AboutDefault({ActiveIndex}) {
    return (
        <>
            {/* <!-- ABOUT --> */}
            <div className={ActiveIndex === 1 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section active hidden animated"} id="about_">
                <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="biography">
                            <div className="cavani_tm_title">
                                <span>About the Project</span>
                            </div>
                            <div className="wrapper">
                                <div className="left">
                                    <p><strong>Partition</strong> is the public-facing interface of the <strong>Lavoisier</strong> computational framework — a platform for rigorous mass spectrometry, proteomics, and molecular analysis.</p>
                                    <p>Built on category theory and partition mathematics, Lavoisier re-derives analytical chemistry from first principles, providing exact decompositions where traditional methods rely on approximation.</p>
                                </div>
                                <div className="right">
                                    <ul>
                                        <li><span className="first">Lead:</span><span className="second">Kundai Sachikonye</span></li>
                                        <li><span className="first">Domain:</span><span className="second">Computational Mass Spectrometry</span></li>
                                        <li><span className="first">Stack:</span><span className="second">Python, Rust, Next.js</span></li>
                                        <li><span className="first">License:</span><span className="second">MIT</span></li>
                                        <li><span className="first">Mail:</span><span className="second"><a href="#">contact@partition.dev</a></span></li>
                                        <li><span className="first">Status:</span><span className="second">Active Research</span></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div className="services">
                            <div className="wrapper">
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Core Capabilities</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Mass Spectrum Analysis</li>
                                            <li>Proteomics Pipeline</li>
                                            <li>Ion Transport Modelling</li>
                                            <li>Chromatographic Decomposition</li>
                                            <li>NIST Library Integration</li>
                                        </ul>
                                    </div>
                                </div>
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Research Areas</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Category Theory &amp; Partitions</li>
                                            <li>S-Entropy &amp; State Counting</li>
                                            <li>Computer Vision for MS</li>
                                            <li>Loschmidt Number Theory</li>
                                            <li>Bounded Phase Spaces</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="skills">
                            <div className="wrapper">
                                <div className="programming">
                                    <div className="cavani_tm_title">
                                        <span>Framework Coverage</span>
                                    </div>
                                    <div className="cavani_progress">
                                        {progressBarData.map((item, idx) => (
                                            <ProgressBar key={idx} bgcolor={item.bgcolor} completed={item.completed} title={item.title} />
                                        ))}
                                    </div>
                                </div>
                                <div className="language">
                                    <div className="cavani_tm_title">
                                        <span>Technology Stack</span>
                                    </div>
                                    <div className="circular_progress_bar">
                                        <div className='circle_holder'>
                                            {circleProgressData.map((item, idx) => (
                                                <div key={idx}>
                                                    <div className="list_inner">
                                                        <CircularProgressbar
                                                            value={item.progress}
                                                            text={`${item.progress}%`}
                                                            strokeWidth={3}
                                                            stroke='#7d7789'
                                                            Language={item.language}
                                                            className={"list_inner"}
                                                        />
                                                        <div className="title"><span>{item.language}</span></div>
                                                    </div>
                                                </div>
                                            ))}

                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="resume">
                            <div className="wrapper">
                                <div className="education">
                                    <div className="cavani_tm_title">
                                        <span>Framework Milestones</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2025 - Present</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Proteomics Module</h3>
                                                            <span>35+ specialized analysis modules, DDA linkage, ion journey modelling</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2024 - 2025</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Categorical State Counting</h3>
                                                            <span>Publication — category theory applied to mass spectrometry</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2023 - 2024</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Lavoisier Core Engine</h3>
                                                            <span>Rust + Python hybrid architecture with PyO3 bindings</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                                <div className="experience">
                                    <div className="cavani_tm_title">
                                        <span>Validation Milestones</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2025 - Present</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>NIST Library Validation</h3>
                                                            <span>Validated against NIST mass spectral databases</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2024 - 2025</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Multi-Instrument Testing</h3>
                                                            <span>Waters qTOF, Thermo Orbitrap — positive &amp; negative ionization</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2024</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Dual-Pipeline Architecture</h3>
                                                            <span>Numerical + computer vision independent validation</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="partners">
                            <div className="cavani_tm_title">
                                <span>Technology Partners</span>
                            </div>
                            <div className="list">
                                <ul>
                                    <li>
                                        <div className="list_inner">
                                            <img src="img/partners/1.png" alt="" />
                                            <a className="cavani_tm_full_link" href="#"></a>
                                        </div>
                                    </li>
                                    <li>
                                        <div className="list_inner">
                                            <img src="img/partners/2.png" alt="" />
                                            <a className="cavani_tm_full_link" href="#"></a>
                                        </div>
                                    </li>
                                    <li>
                                        <div className="list_inner">
                                            <img src="img/partners/3.png" alt="" />
                                            <a className="cavani_tm_full_link" href="#"></a>
                                        </div>
                                    </li>
                                    <li>
                                        <div className="list_inner">
                                            <img src="img/partners/4.png" alt="" />
                                            <a className="cavani_tm_full_link" href="#"></a>
                                        </div>
                                    </li>
                                    <li>
                                        <div className="list_inner">
                                            <img src="img/partners/5.png" alt="" />
                                            <a className="cavani_tm_full_link" href="#"></a>
                                        </div>
                                    </li>
                                </ul>
                            </div>
                        </div>
                        <div className="testimonials">
                            <div className="cavani_tm_title">
                                <span>Key Findings</span>
                            </div>
                            <div className="list">
                                <ul className="">
                                    <li>
                                        <Swiper
                                            slidesPerView={1}
                                            spaceBetween={30}
                                            loop={true}
                                            className="custom-class"
                                            breakpoints={{
                                                768: {
                                                    slidesPerView: 2,
                                                }
                                            }}
                                        >
                                            {testimonials.map((item, i) => (
                                                <SwiperSlide key={i}>
                                                    <div className="list_inner">
                                                        <div className="text">
                                                            <i className="icon-quote-left" />
                                                            <p>{item.desc}</p>
                                                        </div>
                                                        <div className="details">
                                                            <div className="image">
                                                                <div className="main" data-img-url={item.img} />
                                                            </div>
                                                            <div className="info">
                                                                <h3>{item.info1}</h3>
                                                                <span>{item.info2}</span>
                                                            </div>
                                                        </div>
                                                    </div>

                                                </SwiperSlide>
                                            ))}
                                        </Swiper>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- ABOUT --> */}
        </>
    )
}
