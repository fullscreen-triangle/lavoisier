import React,{useEffect} from 'react'
import { dataImage } from '../plugin/plugin'

export default function Mobilemenu({isToggled, handleOnClick}) {
  useEffect(() => {
    dataImage();
  });
    return (
        <>

            {/* MOBILE MENU */}
            <div className={!isToggled ? "cavani_tm_mobile_menu" :  "cavani_tm_mobile_menu opened"} >
                <div className="inner">
                    <div className="wrapper">
                        <div className="avatar">
                            <div className="image" data-img-url="img/about/1.jpg" />
                        </div>
                        <div className="menu_list">
                            <ul className="transition_link">
                                <li onClick={() => handleOnClick(0)}><a href="#home">Home</a></li>
                                <li onClick={() => handleOnClick(1)}><a href="#metabolomics">Metabolomics</a></li>
                                <li onClick={() => handleOnClick(2)}><a href="#proteomics">Proteomics</a></li>
                                <li onClick={() => handleOnClick(3)}><a href="#chromatography">Chromatography</a></li>
                                <li onClick={() => handleOnClick(4)}><a href="#charge">Charge</a></li>
                                <li onClick={() => handleOnClick(5)}><a href="#mass">Mass</a></li>
                                <li onClick={() => handleOnClick(6)}><a href="#framework">Framework</a></li>
                                <li onClick={() => handleOnClick(7)}><a href="#capabilities">Capabilities</a></li>
                                <li onClick={() => handleOnClick(8)}><a href="#collaborate">Collaborate</a></li>
                            </ul>
                        </div>
                        <div className="social">
                            <ul>
                                <li><a href="#"><img className="svg" src="img/svg/social/facebook.svg" alt="" /></a></li>
                                <li><a href="#"><img className="svg" src="img/svg/social/twitter.svg" alt="" /></a></li>
                                <li><a href="#"><img className="svg" src="img/svg/social/instagram.svg" alt="" /></a></li>
                                <li><a href="#"><img className="svg" src="img/svg/social/dribbble.svg" alt="" /></a></li>
                                <li><a href="#"><img className="svg" src="img/svg/social/tik-tok.svg" alt="" /></a></li>
                            </ul>
                        </div>
                        <div className="copyright">
                            <p>Partition © 2024 — Lavoisier Framework</p>
                        </div>
                    </div>
                </div>
            </div>
            {/* /MOBILE MENU */}


        </>
    )
}
