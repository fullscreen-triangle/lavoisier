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
                                <li onClick={() => handleOnClick(1)}><a href="#about">About</a></li>
                                <li onClick={() => handleOnClick(2)}><a href="#portfolio">Portfolio</a></li>
                                <li onClick={() => handleOnClick(3)}><a href="#news">News</a></li>
                                <li onClick={() => handleOnClick(4)}><a href="#contact">Contact</a></li>
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
                            <p>Copyright © 2022</p>
                        </div>
                    </div>
                </div>
            </div>
            {/* /MOBILE MENU */}


        </>
    )
}
