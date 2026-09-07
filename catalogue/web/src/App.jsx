// =====================================================================
//  The shell: sticky nav, routed body, footer.
//
//  Routes are real URLs, so every page deep-links. That needs the SPA
//  rewrite in vercel.json --- without it a direct hit on
//  /paper/sink-detection is a 404 from the static host, not a routing
//  bug in here.
// =====================================================================

import { Routes, Route, NavLink, Link, useLocation } from 'react-router-dom'
import { useEffect } from 'react'
import { PAPERS } from './papers.js'

import Catalogue from './pages/Catalogue.jsx'
import PeptideMassInvariance from './pages/papers/PeptideMassInvariance.jsx'
import RuntimeGraph from './pages/papers/RuntimeGraph.jsx'
import SinkDetection from './pages/papers/SinkDetection.jsx'
import ObservationGroups from './pages/papers/ObservationGroups.jsx'
import CoordinateProvenance from './pages/papers/CoordinateProvenance.jsx'
import InstrumentProcessLadder from './pages/papers/InstrumentProcessLadder.jsx'
import UcDavisCasmi from './pages/papers/UcDavisCasmi.jsx'

const PAGES = {
  'peptide-mass-invariance': PeptideMassInvariance,
  'runtime-graph': RuntimeGraph,
  'sink-detection': SinkDetection,
  'observation-groups': ObservationGroups,
  'coordinate-provenance': CoordinateProvenance,
  'instrument-process-ladder': InstrumentProcessLadder,
  'uc-davis-casmi-catalogue': UcDavisCasmi,
}

// A route change should land at the top of the new page, not wherever
// the previous one was scrolled to.
function ScrollTop() {
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])
  return null
}

function NotFound() {
  return (
    <div className="wrap">
      <section>
        <div className="kicker">404</div>
        <h2>No such page</h2>
        <p className="sub">
          The catalogue has seven papers. <Link to="/">Start from the index.</Link>
        </p>
      </section>
    </div>
  )
}

export default function App() {
  return (
    <>
      <ScrollTop />
      <nav className="nav">
        <div className="nav-inner">
          <NavLink to="/" className="nav-brand">
            <span>Lavoisier</span> catalogue
          </NavLink>
          {PAPERS.map((p) => (
            <NavLink
              key={p.slug}
              to={'/paper/' + p.slug}
              className={({ isActive }) => (isActive ? 'on' : '')}
            >
              {p.short}
            </NavLink>
          ))}
        </div>
      </nav>

      <Routes>
        <Route path="/" element={<Catalogue />} />
        {PAPERS.map((p) => {
          const P = PAGES[p.slug]
          return <Route key={p.slug} path={'/paper/' + p.slug} element={<P />} />
        })}
        <Route path="*" element={<NotFound />} />
      </Routes>

      <footer>
        <div className="wrap">
          Seven papers, their validation experiments, and a shapeshifter runtime
          that recomputes the ladder results in your browser. Numbers that are
          read rather than recomputed say so where they appear; the full account
          is in <span className="mono">src/data/README.md</span>.
        </div>
      </footer>
    </>
  )
}
