import { useState, useEffect, useRef } from 'react'
import Isotope from 'isotope-layout'
import Image from 'next/image';
import { dataImage } from '../../plugin/plugin'
import Modal from 'react-modal';

export default function PortfolioDefault({ ActiveIndex, Animation }) {

    const [isOpen4, setIsOpen4] = useState(false);
    const [modalData, setModalData] = useState({});

    function openDetailModal(data) {
        setModalData(data);
        setIsOpen4(true);
    }

    function toggleModalFour() {
        setIsOpen4(!isOpen4);
    }

    const researchItems = [
        {
            filter: "proteomics",
            title: "Ion Journey Modelling",
            category: "Proteomics",
            img: "img/portfolio/1.jpg",
            description: "Complete ion trajectory simulation from electrospray source through mass analyser to detector, capturing transport phenomena and space-charge effects.",
            detail: "The ion journey model traces individual ions through the mass spectrometer, accounting for RF confinement, collisional cooling, and detector response. This forward model enables ab initio prediction of spectral features."
        },
        {
            filter: "spectrometry",
            title: "Chromatographic Decomposition",
            category: "Mass Spectrometry",
            img: "img/portfolio/2.jpg",
            description: "Mathematical decomposition of chromatographic peaks using partition theory, separating co-eluting compounds without requiring baseline resolution.",
            detail: "By treating chromatographic overlap as a partition problem, we achieve exact deconvolution of complex elution profiles. This approach is validated against known mixtures with complete recovery of individual component spectra."
        },
        {
            filter: "validation",
            title: "NIST Validation Panels",
            category: "Validation",
            img: "img/portfolio/3.jpg",
            description: "Comprehensive validation panels comparing Lavoisier identifications against the NIST Mass Spectral Library across multiple instrument platforms.",
            detail: "Over 20 validation panels demonstrate concordance between Lavoisier's partition-based identifications and NIST reference spectra. Agreement rates exceed 94% for standard reference compounds."
        },
        {
            filter: "mathematics",
            title: "Categorical State Counting",
            category: "Mathematics",
            img: "img/portfolio/4.jpg",
            description: "Category-theoretic framework for enumerating admissible ion states, connecting partition identities to mass spectrometry through functorial mappings.",
            detail: "The categorical approach reduces the combinatorial complexity of spectral interpretation by exploiting algebraic structure in the ion state space. Partition identities provide closed-form expressions for state counts."
        },
        {
            filter: "spectrometry",
            title: "Multimodal Detection",
            category: "Mass Spectrometry",
            img: "img/portfolio/5.jpg",
            description: "Simultaneous analysis using numerical and computer vision pipelines, providing independent validation of spectral identifications.",
            detail: "The dual-pipeline architecture processes spectra through two independent analytical pathways. Agreement between numerical decomposition and image-based classification provides a robust confidence metric."
        },
        {
            filter: "proteomics",
            title: "DDA Linkage Analysis",
            category: "Proteomics",
            img: "img/portfolio/6.jpg",
            description: "Data-dependent acquisition linkage connecting precursor ions to fragment spectra through transport-informed selection criteria.",
            detail: "DDA linkage maps the relationship between MS1 precursor selection and MS2 fragmentation, accounting for isolation window effects and co-fragmentation. This improves peptide identification rates in complex proteomics experiments."
        }
    ];

    const isotope = useRef()
    const [filterKey, setFilterKey] = useState('*')

    useEffect(() => {
        setTimeout(() => {
            isotope.current = new Isotope(".filter-container", {
                itemSelector: ".filter-item",
                layoutMode: "fitRows",
            });
        }, 500);
        return () => isotope.current.destroy();
    }, []);

    useEffect(() => {
        if (isotope.current) {
            filterKey === '*'
                ? isotope.current.arrange({ filter: '*' })
                : isotope.current.arrange({ filter: `.${filterKey}` })
        }
    }, [filterKey])

    const handleFilterKeyChange = key => () => setFilterKey(key)

    return (
        <>
            {/* <!-- RESEARCH --> */}

            <div className={ActiveIndex === 2 ? `cavani_tm_section active animated ${Animation ? Animation: "fadeInUp"}` : "cavani_tm_section hidden animated"} id="portfolio_">
                <div className="section_inner">
                    <div className="cavani_tm_portfolio">
                        <div className="cavani_tm_title">
                            <span>Research Outputs</span>
                        </div>

                        <div className="portfolio_filter">
                            <ul>
                                <li><a href='#' onClick={handleFilterKeyChange('*')} className="current">All</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('spectrometry')} data-filter=".spectrometry">Mass Spectrometry</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('proteomics')} data-filter=".proteomics">Proteomics</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('validation')} data-filter=".validation">Validation</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('mathematics')} data-filter=".mathematics">Mathematics</a></li>
                            </ul>
                        </div>
                        <div className="portfolio_list">
                            <div className="filter-container">
                                {researchItems.map((item, i) => (
                                    <div key={i} className={`filter-item ${item.filter} fadeInUp`}>
                                        <div className="list_inner">
                                            <div className="image">
                                                <img src="img/thumbs/1-1.jpg" alt="" />
                                                <div className="main" data-img-url={item.img} onClick={() => openDetailModal(item)}></div>
                                                <span className="icon"><i className="icon-doc-text-inv"></i></span>
                                                <div className="details">
                                                    <h3>{item.title}</h3>
                                                    <span>{item.category}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- /RESEARCH --> */}

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
                                <i className="icon-cancel" />
                            </a>
                        </div>
                        <div className="description_wrap">
                            <div className="popup_details">
                                <div className="top_image">
                                    <img src="img/thumbs/4-2.jpg" alt="" />
                                    <div className="main" data-img-url={modalData.img} style={{ backgroundImage: `url(${modalData.img})` }} />
                                </div>
                                <div className="portfolio_main_title">
                                    <h3>{modalData.title}</h3>
                                    <span>{modalData.category}</span>
                                    <div></div>
                                </div>
                                <div className="main_details">
                                    <div className="textbox">
                                        <p>{modalData.description}</p>
                                        <p>{modalData.detail}</p>
                                    </div>
                                    <div className="detailbox">
                                        <ul>
                                            <li>
                                                <span className="first">Framework</span>
                                                <span>Lavoisier</span>
                                            </li>
                                            <li>
                                                <span className="first">Category</span>
                                                <span><a href="#">{modalData.category}</a></span>
                                            </li>
                                            <li>
                                                <span className="first">Author</span>
                                                <span>Kundai Sachikonye</span>
                                            </li>
                                            <li>
                                                <span className="first">Status</span>
                                                <span>Active Research</span>
                                            </li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </Modal>

        </>
    )

}
