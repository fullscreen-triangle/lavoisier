import Link from "next/link";
import React from "react";
import Layout from "./Layout";

const Footer = () => {
  return (
    <footer
      className="w-full border-t-2 border-solid border-dark
    font-medium text-lg dark:text-light dark:border-light sm:text-base
    "
    >
      <Layout className="py-8 flex items-center justify-between lg:flex-col lg:py-6">
        <span>{new Date().getFullYear()} &copy; Lavoisier. MIT License.</span>

        <div className="flex items-center lg:py-2 text-base">
          Force-free mass spectrometry via
          <Link
            href="/framework"
            className="underline underline-offset-2 ml-1 hover:text-primary dark:hover:text-primaryDark"
          >
            partition depth minimisation
          </Link>
        </div>

        <Link
          href="https://github.com/fullscreen-triangle/lavoisier"
          target="_blank"
          className="underline underline-offset-2 hover:text-primary dark:hover:text-primaryDark"
        >
          Source on GitHub
        </Link>
      </Layout>
    </footer>
  );
};

export default Footer;
