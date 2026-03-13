import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { useScrollReveal } from '../../hooks/useScrollReveal'

export default function Union({ ActiveIndex }) {
  const sectionRef = useRef(null)
  const sunburstRef = useRef(null)
  const barChartRef = useRef(null)

  useScrollReveal(sectionRef)

  // Derivation Tree / Sunburst Chart
  useEffect(() => {
    if (!sunburstRef.current) return

    const svg = d3.select(sunburstRef.current)
    svg.selectAll('*').remove()

    const width = 480
    const height = 400
    const cx = width / 2
    const cy = height / 2

    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const g = svg.append('g')
      .attr('transform', `translate(${cx},${cy})`)

    // Center circle - the single axiom
    const centerRadius = 45
    g.append('circle')
      .attr('r', 0)
      .attr('fill', '#f9d77e')
      .attr('opacity', 0.9)
      .transition()
      .duration(600)
      .attr('r', centerRadius)

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '-0.3em')
      .attr('fill', '#1a1a1a')
      .style('font-size', '9px')
      .style('font-weight', 'bold')
      .text('Bounded')

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.7em')
      .attr('fill', '#1a1a1a')
      .style('font-size', '9px')
      .style('font-weight', 'bold')
      .text('Phase Space')

    // Ring 1 - core consequences
    const ring1Data = [
      'Oscillation',
      'Partitions',
      '3D Space',
      'Charge',
      'Mass'
    ]
    const ring1Inner = 55
    const ring1Outer = 100
    const ring1Arc = d3.arc()
      .innerRadius(ring1Inner)
      .outerRadius(ring1Outer)

    const ring1Angle = (2 * Math.PI) / ring1Data.length
    const ring1Pad = 0.04

    ring1Data.forEach((label, i) => {
      const startAngle = i * ring1Angle + ring1Pad
      const endAngle = (i + 1) * ring1Angle - ring1Pad

      g.append('path')
        .attr('d', ring1Arc({ startAngle, endAngle }))
        .attr('fill', '#9980FA')
        .attr('opacity', 0)
        .attr('stroke', '#1a1a1a')
        .attr('stroke-width', 1)
        .transition()
        .duration(600)
        .delay(300 + i * 80)
        .attr('opacity', 0.85)

      // Label
      const midAngle = (startAngle + endAngle) / 2 - Math.PI / 2
      const labelR = (ring1Inner + ring1Outer) / 2
      g.append('text')
        .attr('x', Math.cos(midAngle) * labelR)
        .attr('y', Math.sin(midAngle) * labelR)
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
        .attr('fill', '#ffffff')
        .style('font-size', '9px')
        .style('font-weight', '600')
        .text(label)
    })

    // Ring 2 - derived theorems
    const ring2Data = [
      'E = mc\u00B2',
      'Equivalence',
      'Lorentz Inv.',
      'Selection Rules',
      'E = \u0127\u03C9',
      'Partition Coords',
      'Quantum Nos.',
      '\u0394l = \u00B11',
      'Lagrangian',
      'Landauer Limit',
      'Bijective Map',
      'Ion Journey'
    ]
    const ring2Inner = 108
    const ring2Outer = 155
    const ring2Arc = d3.arc()
      .innerRadius(ring2Inner)
      .outerRadius(ring2Outer)

    const ring2Angle = (2 * Math.PI) / ring2Data.length
    const ring2Pad = 0.03

    ring2Data.forEach((label, i) => {
      const startAngle = i * ring2Angle + ring2Pad
      const endAngle = (i + 1) * ring2Angle - ring2Pad
      const opacity = 0.45 + (i % 3) * 0.15

      g.append('path')
        .attr('d', ring2Arc({ startAngle, endAngle }))
        .attr('fill', '#9980FA')
        .attr('opacity', 0)
        .attr('stroke', '#1a1a1a')
        .attr('stroke-width', 1)
        .transition()
        .duration(600)
        .delay(700 + i * 60)
        .attr('opacity', opacity)

      // Label
      const midAngle = (startAngle + endAngle) / 2 - Math.PI / 2
      const labelR = (ring2Inner + ring2Outer) / 2
      g.append('text')
        .attr('x', Math.cos(midAngle) * labelR)
        .attr('y', Math.sin(midAngle) * labelR)
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
        .attr('fill', '#e0e0e0')
        .style('font-size', '7.5px')
        .text(label)
    })

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0')
      .style('font-size', '13px')
      .text('Derivation Tree: One Axiom \u2192 17 Results')
  }, [])

  // Validation Coverage Bar Chart (logarithmic)
  useEffect(() => {
    if (!barChartRef.current) return

    const svg = d3.select(barChartRef.current)
    svg.selectAll('*').remove()

    const width = 480
    const height = 280
    const margin = { top: 30, right: 60, bottom: 40, left: 120 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const data = [
      { category: 'Bijective Transforms', count: 127000, label: '127k' },
      { category: 'Source Libraries', count: 4545, label: '4,545' },
      { category: 'Lactoferrin Ions', count: 36, label: '36' },
      { category: 'NIST Spike', count: 34, label: '34' },
      { category: 'Glycan MS/MS', count: 20, label: '20' },
    ]

    const x = d3.scaleLog()
      .domain([10, 200000])
      .range([0, innerWidth])

    const y = d3.scaleBand()
      .domain(data.map(d => d.category))
      .range([0, innerHeight])
      .padding(0.35)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(4, '~s'))
      .selectAll('text')
      .attr('fill', '#b0b0b0')
      .style('font-size', '11px')

    g.selectAll('.domain, line')
      .attr('stroke', '#333333')

    // Y axis
    g.append('g')
      .call(d3.axisLeft(y))
      .selectAll('text')
      .attr('fill', '#b0b0b0')
      .style('font-size', '10px')

    g.selectAll('.domain, line')
      .attr('stroke', '#333333')

    // Bars
    g.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('y', d => y(d.category))
      .attr('height', y.bandwidth())
      .attr('x', 0)
      .attr('width', 0)
      .attr('fill', '#9980FA')
      .attr('opacity', (_, i) => 0.55 + i * 0.1)
      .attr('rx', 3)
      .transition()
      .duration(800)
      .delay((_, i) => i * 120)
      .attr('width', d => x(d.count))

    // Labels on bars
    g.selectAll('.bar-label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'bar-label')
      .attr('x', d => x(d.count) + 8)
      .attr('y', d => y(d.category) + y.bandwidth() / 2 + 4)
      .attr('fill', '#b0b0b0')
      .style('font-size', '11px')
      .text(d => d.label)

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0')
      .style('font-size', '13px')
      .text('Validation Coverage (log scale)')
  }, [])

  return (
    <div
      className={
        ActiveIndex === 6
          ? 'cavani_tm_section active animated fadeInUp'
          : 'cavani_tm_section hidden animated'
      }
      id="union_"
    >
      <div className="section_inner">
        <div className="framework-section" ref={sectionRef}>
          <div className="section-grid">
            {/* Left column: D3 charts */}
            <div className="chart-column">
              <div className="chart-card">
                <h3 className="chart-title">Derivation Tree</h3>
                <svg ref={sunburstRef} width="100%" />
              </div>
              <div className="chart-card">
                <h3 className="chart-title">Validation Coverage</h3>
                <svg ref={barChartRef} width="100%" />
              </div>
            </div>

            {/* Right column: findings and stats */}
            <div className="content-column">
              <div className="section-header">
                <span className="section-label">Union</span>
                <h2>Unified Framework</h2>
                <p>
                  A single axiom -- the Bounded Phase Space Law -- generates
                  all of classical mechanics, quantum numbers, charge, mass,
                  and relativity as theorems. Validated across 132,000+
                  partition operations from 11 independent laboratories.
                </p>
              </div>

              <div className="finding-block">
                <h4>One Axiom, Seventeen Results</h4>
                <p>
                  The Bounded Phase Space Law states that all persistent
                  dynamical systems occupy bounded regions of phase space
                  admitting partition and nesting. From this single postulate:
                  oscillatory dynamics, E = &#x127;&#x3C9;, partition
                  coordinates, three-dimensional space, charge, mass,
                  E = mc&#xB2;, the equivalence principle, and Lorentz
                  invariance all follow as derived theorems.
                </p>
              </div>

              <div className="finding-block">
                <h4>Cross-Laboratory Validation</h4>
                <p>
                  4,545 entries from 11 independent NIST labs using 2
                  instrument types (HCD, IT-FT) all conform to the partition
                  mathematics. 70 ion journeys across spike protein and
                  lactoferrin datasets show 100% conformance at every stage.
                </p>
              </div>

              <div className="finding-block">
                <h4>The Partition Lagrangian</h4>
                <p>
                  The complete dynamics of any ion is captured by the
                  Lagrangian where &#x3BC; = &#x3B1;(m/z). The gauge field
                  A_M encodes the partition memory, and the potential M(x,t)
                  gives the accumulated non-actualisations at each point.
                </p>
              </div>

              <div className="finding-block">
                <h4>Compression &amp; Information</h4>
                <p>
                  The partition framework achieves 10&#xB2;&#xB9;x volume
                  reduction at a cost of approximately 49 kBT, near the
                  Landauer limit. Each ion is described by 217 bits compared
                  to roughly 20 bits in conventional representations.
                </p>
              </div>

              <div className="stat-row">
                <div className="stat-item">
                  <span className="stat-value">1</span>
                  <span className="stat-label">Axiom</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">17</span>
                  <span className="stat-label">Results</span>
                </div>
              </div>
              <div className="stat-row">
                <div className="stat-item">
                  <span className="stat-value">132,000+</span>
                  <span className="stat-label">Operations</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">11</span>
                  <span className="stat-label">Labs</span>
                </div>
              </div>

              <div className="equation">
                L = (1/2)&#x3BC;|x&#x307;|&#xB2; + &#x3BC; x&#x307; &middot; A_M &minus; M(x,t)
              </div>

              <div className="equation" style={{ marginTop: '8px' }}>
                Axiom: All persistent dynamical systems occupy bounded regions
                of phase space admitting partition and nesting.
              </div>

              <div className="equation" style={{ marginTop: '8px' }}>
                Selection rules: &#x394;l = &#xB1;1, &nbsp; &#x394;m &#x2208; &#x7B;0, &#xB1;1&#x7D;, &nbsp; &#x394;s = 0
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
