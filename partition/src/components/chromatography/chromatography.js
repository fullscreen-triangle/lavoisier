import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { useScrollReveal } from '../../hooks/useScrollReveal'

// Generate synthetic TIC data with Gaussian peaks
function generateChromatogramData() {
  const points = []
  const peaks = [
    { center: 8, height: 0.85, width: 3 },
    { center: 22, height: 1.0, width: 4 },
    { center: 35, height: 0.65, width: 2.5 },
    { center: 48, height: 0.45, width: 3.5 },
  ]
  for (let i = 0; i <= 58; i += 1) {
    let intensity = 0.02 + Math.random() * 0.02
    for (const peak of peaks) {
      intensity += peak.height * Math.exp(-0.5 * Math.pow((i - peak.center) / peak.width, 2))
    }
    points.push({ time: i, intensity })
  }
  return points
}

// Generate S-entropy coordinate data
function generateEntropyData() {
  const clusters = [
    { cx: 0.8, cy: 0.67, se: 0.52, n: 6 },
    { cx: 0.35, cy: 0.45, se: 0.38, n: 5 },
    { cx: 0.6, cy: 0.82, se: 0.71, n: 5 },
    { cx: 0.2, cy: 0.25, se: 0.25, n: 4 },
    { cx: 0.9, cy: 0.3, se: 0.6, n: 5 },
  ]
  const points = []
  for (const c of clusters) {
    for (let i = 0; i < c.n; i++) {
      points.push({
        sk: Math.max(0, Math.min(1, c.cx + (Math.random() - 0.5) * 0.12)),
        st: Math.max(0, Math.min(1, c.cy + (Math.random() - 0.5) * 0.12)),
        se: Math.max(0.1, Math.min(1, c.se + (Math.random() - 0.5) * 0.15)),
      })
    }
  }
  return points
}

export default function Chromatography({ ActiveIndex }) {
  const sectionRef = useRef(null)
  const chromatogramRef = useRef(null)
  const entropyRef = useRef(null)

  useScrollReveal(sectionRef)

  // Chart 1: Chromatogram-Style Curve (TIC)
  useEffect(() => {
    if (ActiveIndex !== 3) return
    const svg = d3.select(chromatogramRef.current)
    svg.selectAll('*').remove()

    const width = 480
    const height = 280
    const margin = { top: 20, right: 20, bottom: 40, left: 50 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const data = generateChromatogramData()

    const x = d3.scaleLinear().domain([0, 58]).range([0, w])
    const y = d3.scaleLinear().domain([0, 1.15]).range([h, 0])

    // Gradient fill
    const defs = svg.append('defs')
    const gradient = defs.append('linearGradient')
      .attr('id', 'tic-gradient')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '0%').attr('y2', '100%')
    gradient.append('stop').attr('offset', '0%').attr('stop-color', 'rgba(153,128,250,0.15)')
    gradient.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(153,128,250,0.02)')

    // Area
    const area = d3.area()
      .x(d => x(d.time))
      .y0(h)
      .y1(d => y(d.intensity))
      .curve(d3.curveBasis)

    g.append('path')
      .datum(data)
      .attr('d', area)
      .attr('fill', 'url(#tic-gradient)')

    // Line
    const line = d3.line()
      .x(d => x(d.time))
      .y(d => y(d.intensity))
      .curve(d3.curveBasis)

    g.append('path')
      .datum(data)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#9980FA')
      .attr('stroke-width', 2)

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text').attr('fill', '#b0b0b0')
    g.selectAll('.domain, .tick line').attr('stroke', '#333333')

    g.append('g')
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.1f')))
      .selectAll('text').attr('fill', '#b0b0b0')
    g.selectAll('.domain, .tick line').attr('stroke', '#333333')

    // Axis labels
    g.append('text')
      .attr('x', w / 2).attr('y', h + 35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0').attr('font-size', 11)
      .text('Retention Time (min)')

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2).attr('y', -38)
      .attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0').attr('font-size', 11)
      .text('Relative Intensity')
  }, [ActiveIndex])

  // Chart 2: S-Entropy 3D Projection (Sk vs St)
  useEffect(() => {
    if (ActiveIndex !== 3) return
    const svg = d3.select(entropyRef.current)
    svg.selectAll('*').remove()

    const width = 480
    const height = 280
    const margin = { top: 20, right: 20, bottom: 40, left: 50 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const data = generateEntropyData()

    const x = d3.scaleLinear().domain([0, 1]).range([0, w])
    const y = d3.scaleLinear().domain([0, 1]).range([h, 0])
    const rScale = d3.scaleLinear().domain([0, 1]).range([4, 16])

    // Dashed grid lines
    const gridVals = [0.25, 0.5, 0.75]
    gridVals.forEach(v => {
      g.append('line')
        .attr('x1', x(v)).attr('x2', x(v))
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', '#333333').attr('stroke-dasharray', '4,4')
      g.append('line')
        .attr('x1', 0).attr('x2', w)
        .attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', '#333333').attr('stroke-dasharray', '4,4')
    })

    // Data points
    g.selectAll('circle')
      .data(data)
      .enter()
      .append('circle')
      .attr('cx', d => x(d.sk))
      .attr('cy', d => y(d.st))
      .attr('r', d => rScale(d.se))
      .attr('fill', '#f9d77e')
      .attr('fill-opacity', 0.5)
      .attr('stroke', '#f9d77e')
      .attr('stroke-opacity', 0.8)
      .attr('stroke-width', 1)

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(5))
      .selectAll('text').attr('fill', '#b0b0b0')
    g.selectAll('.domain, .tick line').attr('stroke', '#333333')

    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text').attr('fill', '#b0b0b0')
    g.selectAll('.domain, .tick line').attr('stroke', '#333333')

    // Axis labels
    g.append('text')
      .attr('x', w / 2).attr('y', h + 35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0').attr('font-size', 11)
      .text('Sk (kinetic)')

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2).attr('y', -38)
      .attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0').attr('font-size', 11)
      .text('St (thermodynamic)')
  }, [ActiveIndex])

  return (
    <>
      <div
        className={
          ActiveIndex === 3
            ? "cavani_tm_section active animated fadeInUp"
            : "cavani_tm_section hidden animated"
        }
        id="chromatography_"
      >
        <div className="section_inner" ref={sectionRef}>
          <div className="framework-section">
            <div className="section-header">
              <span className="section-label">Chromatography</span>
              <h2>Chromatography as Partition Lag</h2>
              <p style={{ color: '#b0b0b0', maxWidth: 640, marginTop: 8 }}>
                The column does not just separate molecules — it partitions phase space.
                Retention time is not an empirical parameter but the time required for
                phase-space partitioning to complete.
              </p>
            </div>

            <div className="section-grid">
              {/* Left: D3 Charts */}
              <div className="chart-column">
                <div className="chart-card">
                  <div className="chart-title">Total Ion Chromatogram (TIC)</div>
                  <svg
                    ref={chromatogramRef}
                    viewBox="0 0 480 280"
                    style={{ width: '100%', height: 'auto' }}
                  />
                </div>

                <div className="chart-card">
                  <div className="chart-title">S-Entropy Projection (Sk vs St)</div>
                  <svg
                    ref={entropyRef}
                    viewBox="0 0 480 280"
                    style={{ width: '100%', height: 'auto' }}
                  />
                </div>
              </div>

              {/* Right: Findings */}
              <div className="content-column">
                <div className="finding-block">
                  <h3>Chromatography as Partition Lag</h3>
                  <p style={{ color: '#b0b0b0' }}>
                    Retention time t<sub>R</sub> is not empirical — it is the partition lag
                    function: the time for phase-space partitioning to complete across all
                    accessible microstates. The column enforces a thermodynamic boundary
                    that each ion must traverse.
                  </p>
                  <div className="equation">
                    t<sub>R</sub> = &tau;<sub>p</sub>(S<sub>k</sub>, S<sub>t</sub>, S<sub>e</sub>)
                  </div>
                  <div className="stat-row">
                    <div className="stat-item">
                      <div className="stat-value">58 min</div>
                      <div className="stat-label">RT Range</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-value">0.34s</div>
                      <div className="stat-label">Processing</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-value">100%</div>
                      <div className="stat-label">Validation</div>
                    </div>
                  </div>
                </div>

                <div className="finding-block">
                  <h3>S-Entropy Coordinates</h3>
                  <p style={{ color: '#b0b0b0' }}>
                    Every ion is mapped to (S<sub>k</sub>, S<sub>t</sub>, S<sub>e</sub>) in [0,1]<sup>3</sup> —
                    a complete thermodynamic fingerprint. The kinetic, thermodynamic, and
                    electrospray entropy coordinates fully determine the ion journey through
                    the instrument.
                  </p>
                  <div className="equation">
                    (S<sub>k</sub>, S<sub>t</sub>, S<sub>e</sub>) &isin; [0,1]<sup>3</sup>
                  </div>
                  <div className="stat-row">
                    <div className="stat-item">
                      <div className="stat-value">3</div>
                      <div className="stat-label">Coordinates</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-value">0.8056</div>
                      <div className="stat-label">S<sub>k</sub> Example</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-value">0.6672</div>
                      <div className="stat-label">S<sub>t</sub> Example</div>
                    </div>
                  </div>
                </div>

                <div className="finding-block">
                  <h3>The Fundamental Identity</h3>
                  <p style={{ color: '#b0b0b0' }}>
                    Counting states IS time evolution. The rate of mass accumulation equals
                    the angular frequency divided by 2&pi;, which equals the inverse of the
                    mean partition lag. This identity closes the loop between mass, entropy,
                    and chromatographic time.
                  </p>
                  <div className="equation">
                    dM/dt = &omega;/(2&pi;) = 1/&langle;&tau;<sub>p</sub>&rangle;
                  </div>
                  <div className="equation" style={{ marginTop: 8 }}>
                    &Delta;<sub>M</sub> &middot; &tau;<sub>p</sub> &ge; &hbar;
                  </div>
                  <div className="stat-row">
                    <div className="stat-item">
                      <div className="stat-value">Se=0.52</div>
                      <div className="stat-label">Electrospray</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-value">100%</div>
                      <div className="stat-label">Pass Rate</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
