import React, { useState } from 'react'
import { dataImage } from '../../plugin/plugin'
import Modal from 'react-modal';
import {SVG_Custom1, SVG_Custom2, SVG_Custom3, SVG_Custom4, SVG_Custom5, SVG_Custom6 } from '../../plugin/svg';
export default function Service({ ActiveIndex }) {

    const [isOpen7, setIsOpen7] = useState(false);
    const [modalContent, setModalContent] = useState({});

    function toggleModalFour() {
        setIsOpen7(!isOpen7);
    }
    const service = [
        {
            img: "img/news/1.jpg",
            svg: <SVG_Custom1 />,
            text: "End-to-end mass spectrometry pipeline with automated data extraction, chromatographic decomposition, and ion identification.",
            date: "Core Module",
            title: "Mass Spectrum Analysis",
            text1: "The Lavoisier mass spectrum analysis engine processes raw mzML files through a multi-stage pipeline: data extraction, chromatographic separation, ionization modelling, and spectral identification against NIST databases.",
            text2: "Unlike traditional peak-picking approaches, Lavoisier uses partition mathematics to achieve exact decomposition of complex spectra. Each ion is traced through its complete journey from source to detector.",
            text3: "Validated against Waters qTOF and Thermo Orbitrap instruments in both positive and negative ionization modes, delivering reproducible results across instrument platforms."
        },
        {
            img: "img/news/2.jpg",
            svg: <SVG_Custom2 />,
            text: "Comprehensive protein analysis with 35+ specialized modules covering inference, normalisation, imputation, and database searching.",
            date: "Core Module",
            title: "Proteomics Pipeline",
            text1: "The proteomics module provides a complete workflow from raw mass spectrometry data to protein identification and quantification. It includes specialised components for DDA linkage, ion decomposition, and transport phenomena modelling.",
            text2: "Protein inference is handled through a rigorous mathematical framework that accounts for shared peptides, degenerate sequences, and multi-hit assignments — reducing false discovery rates compared to conventional approaches.",
            text3: "Integration with Comet search engine and custom ProteinDataFrame structures enables seamless data flow from spectral matching to statistical analysis and biological interpretation."
        },
        {
            img: "img/news/3.jpg",
            svg: <SVG_Custom3 />,
            text: "High-performance computational core built with Rust and PyO3, enabling parallel processing of large-scale mass spectrometry datasets.",
            date: "Infrastructure",
            title: "Rust Compute Engine",
            text1: "The Lavoisier computational core is implemented in Rust for maximum performance, with Python bindings via PyO3. Five specialised crates — core, io, buhera, computational, and mass-computing — handle different aspects of the analysis pipeline.",
            text2: "Parallel processing with Rayon and Crossbeam enables efficient utilisation of multi-core systems. The nalgebra and ndarray libraries provide optimised linear algebra operations for spectral decomposition and matrix factorisation.",
            text3: "Data serialization supports protobuf and multiple compression formats (LZ4, Zstd, FLATE2), enabling efficient storage and transmission of large spectral datasets across distributed computing environments."
        },
        {
            img: "img/news/4.jpg",
            svg: <SVG_Custom4 />,
            text: "Independent validation through dual numerical and computer vision pipelines, with comprehensive statistical testing and quality assessment.",
            date: "Validation",
            title: "Dual-Pipeline Validation",
            text1: "The validation framework implements a dual-pipeline architecture: traditional numerical methods and independent computer vision analysis operate in parallel, cross-validating results through agreement metrics.",
            text2: "Eight specialised validator classes cover statistical hypothesis testing, performance benchmarking, data quality assessment, completeness analysis, feature extraction comparison, and annotation performance evaluation.",
            text3: "This approach ensures that analytical results are robust against algorithmic bias — if both pipelines converge on the same identification, confidence is substantially higher than single-method approaches."
        },
        {
            img: "img/news/5.jpg",
            svg: <SVG_Custom5 />,
            text: "RESTful API endpoints enabling programmatic access to the Lavoisier analysis pipeline for external researchers and applications.",
            date: "Integration",
            title: "Collaboration API",
            text1: "The Partition API exposes Lavoisier's analytical capabilities through FastAPI endpoints. Researchers can submit mzML files for analysis, query the NIST library, run validation suites, and retrieve results programmatically.",
            text2: "Endpoints include /api/analyze for sample submission, /api/publications for accessing research outputs, /api/figures for retrieving generated visualisation panels, and /api/validate for running the validation framework.",
            text3: "Authentication and rate limiting ensure fair access. The API documentation is available through an interactive Swagger interface, making integration straightforward for computational biology workflows."
        },
        {
            img: "img/news/6.jpg",
            svg: <SVG_Custom6 />,
            text: "AI-powered domain expert routing with LLM integration for intelligent query handling and automated literature analysis.",
            date: "AI Module",
            title: "AI & LLM Integration",
            text1: "The Diadochi routing system directs analytical queries to specialised domain experts — each expert handles a specific aspect of mass spectrometry, proteomics, or computational chemistry with targeted knowledge.",
            text2: "LLM integration through LangChain, OpenAI, and Anthropic APIs enables natural language interaction with the framework. Researchers can query results, request explanations, and generate reports conversationally.",
            text3: "Vector search capabilities (FAISS, Annoy, HNSW) enable semantic similarity searches across spectral libraries and publication databases, connecting experimental results to relevant literature automatically."
        }
    ]
    return (
        <>
            {/* <!-- CAPABILITIES --> */}
            <div className={ActiveIndex === 7 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"} id="news_">
                <div className="section_inner">
                    <div className="cavani_tm_service">
                        <div className="cavani_tm_title">
                            <span>Capabilities</span>
                        </div>
                        <div className="service_list">
                            <ul>
                                {service.map((item, i) => (
                                    <li key={i}>
                                        <div className="list_inner" onClick={toggleModalFour}>
                                            {item.svg}
                                            <h3 className="title" onClick={toggleModalFour}>{item.title}</h3>
                                            <p className="text">{item.text}</p>
                                            <a className="cavani_tm_full_link" href="#" onClick={() => setModalContent(item)} />
                                            <img className="popup_service_image" src={item.img} alt="" />
                                            <div className="service_hidden_details">
                                                <div className="service_popup_informations">
                                                    <div className="descriptions">
                                                        <p>{item.text1}</p>
                                                        <p>{item.text2}</p>
                                                        <p>{item.text3}</p>
                                                    </div>
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
            {/* <!-- CAPABILITIES --> */}

            {modalContent && (
                <Modal
                    isOpen={isOpen7}
                    onRequestClose={toggleModalFour}
                    contentLabel="My dialog"
                    className="mymodal"
                    overlayClassName="myoverlay"
                    closeTimeoutMS={300}
                    openTimeoutMS={300}
                >
                    <div className="cavani_tm_modalbox opened">
                        <div className="box_inner">
                            <div className="close" onClick={toggleModalFour} >
                                <a href="#"><i className="icon-cancel"></i></a>
                            </div>
                            <div className="description_wrap">
                                <div className="service_popup_informations">
                                    <div className="image">
                                        <img src="img/thumbs/4-2.jpg" alt="" />
                                        <div className="main" data-img-url="img/news/1.jpg" style={{ backgroundImage: `url(${modalContent.img})` }} />
                                    </div>
                                    <div className="details">
                                        <span>{modalContent.date}</span>
                                        <h3>{modalContent.title}</h3>
                                    </div>
                                    <div className="descriptions">
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
        </>
    )
}
