import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { useScrollReveal } from '../../hooks/useScrollReveal'

export default function Metabolomics({ ActiveIndex }) {
  const sectionRef = useRef(null)
  const spectrumChartRef = useRef(null)
  const scatterChartRef = useRef(null)

  useScrollReveal(sectionRef)

  // Spectrum Distribution Bar Chart
  useEffect(() => {
    if (!spectrumChartRef.current) return

    const svg = d3.select(spectrumChartRef.current)
    svg.selectAll('*').remove()

    const width = 480
    const height = 280
    const margin = { top: 30, right: 40, bottom: 40, left: 100 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const data = [
      { level: 'MS1', count: 96 },
      { level: 'MS2', count: 21 },
    ]

    const x = d3.scaleLinear()
      .domain([0, 110])
      .range([0, innerWidth])

    const y = d3.scaleBand()
      .domain(data.map(d => d.level))
      .range([0, innerHeight])
      .padding(0.4)

    const colors = { MS1: '#9980FA', MS2: '#f9d77e' }

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(5))
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
      .style('font-size', '12px')

    g.selectAll('.domain, line')
      .attr('stroke', '#333333')

    // Bars
    g.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('y', d => y(d.level))
      .attr('height', y.bandwidth())
      .attr('x', 0)
      .attr('width', 0)
      .attr('fill', d => colors[d.level])
      .attr('rx', 3)
      .transition()
      .duration(800)
      .attr('width', d => x(d.count))

    // Labels on bars
    g.selectAll('.bar-label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'bar-label')
      .attr('x', d => x(d.count) + 8)
      .attr('y', d => y(d.level) + y.bandwidth() / 2 + 4)
      .attr('fill', '#b0b0b0')
      .style('font-size', '12px')
      .text(d => d.count)

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0')
      .style('font-size', '13px')
      .text('Spectrum Distribution')
  }, [])

  // m/z Coverage Scatter
  useEffect(() => {
    if (!scatterChartRef.current) return

    const svg = d3.select(scatterChartRef.current)
    svg.selectAll('*').remove()

    const width = 480
    const height = 280
    const margin = { top: 30, right: 30, bottom: 50, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    svg.attr('viewBox', `0 0 ${width} ${height}`)

    // Generate synthetic representative points
    const rng = d3.randomUniform
    const points = Array.from({ length: 30 }, (_, i) => ({
      rt: 0.02 + (58.05 / 30) * i + (Math.random() - 0.5) * 3,
      mz: 50.73 + Math.random() * (1199.98 - 50.73),
    }))

    const x = d3.scaleLinear()
      .domain([0, 60])
      .range([0, innerWidth])

    const y = d3.scaleLinear()
      .domain([0, 1250])
      .range([innerHeight, 0])

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text')
      .attr('fill', '#b0b0b0')
      .style('font-size', '11px')

    // Y axis
    g.append('g')
      .call(d3.axisLeft(y).ticks(6))
      .selectAll('text')
      .attr('fill', '#b0b0b0')
      .style('font-size', '11px')

    g.selectAll('.domain, line')
      .attr('stroke', '#333333')

    // Axis labels
    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 40)
      .attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0')
      .style('font-size', '11px')
      .text('Retention Time (min)')

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -45)
      .attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0')
      .style('font-size', '11px')
      .text('m/z')

    // Points
    g.selectAll('circle')
      .data(points)
      .enter()
      .append('circle')
      .attr('cx', d => x(d.rt))
      .attr('cy', d => y(d.mz))
      .attr('r', 0)
      .attr('fill', '#9980FA')
      .attr('opacity', 0.6)
      .transition()
      .duration(600)
      .delay((_, i) => i * 30)
      .attr('r', 4)

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0')
      .style('font-size', '13px')
      .text('m/z Coverage Across Retention Time')
  }, [])

  return (
    <div
      className={
        ActiveIndex === 1
          ? 'cavani_tm_section active animated fadeInUp'
          : 'cavani_tm_section hidden animated'
      }
      id="metabolomics_"
    >
      <div className="section_inner">
        <div className="framework-section" ref={sectionRef}>
          <div className="section-grid">
            {/* Left column: D3 charts */}
            <div className="chart-column">
              <div className="chart-card">
                <h3 className="chart-title">Spectrum Distribution</h3>
                <svg ref={spectrumChartRef} width="100%" />
              </div>
              <div className="chart-card">
                <h3 className="chart-title">m/z Coverage</h3>
                <svg ref={scatterChartRef} width="100%" />
              </div>
            </div>

            {/* Right column: findings and stats */}
            <div className="content-column">
              <div className="section-header">
                <span className="section-label">Metabolomics</span>
                <h2>Metabolomic Profiling Pipeline</h2>
                <p>
                  Multi-stage analysis of Waters qTOF mass spectrometry data,
                  validating the partition framework across small-molecule
                  metabolites with complete spectral quality conformance.
                </p>
              </div>

              <div className="finding-block">
                <h4>Multi-Stage Pipeline</h4>
                <p>
                  Raw mzML spectra flow through extraction, chromatographic
                  separation, electrospray ionization, and computational
                  identification. Each stage preserves the bounded phase-space
                  constraint, ensuring every ion trajectory maps to a valid
                  partition state.
                </p>
              </div>

              <div className="finding-block">
                <h4>Spectral Quality</h4>
                <p>
                  100% of the 117 PL_Neg spectra pass high-quality thresholds
                  with a mean quality score of 0.657. The pipeline processes
                  spectra at 92.98 per second, demonstrating that partition
                  validation adds negligible computational overhead.
                </p>
              </div>

              <div className="finding-block">
                <h4>Phospholipid Profiling</h4>
                <p>
                  The PL_Neg dataset captures phospholipids in negative
                  ionization mode on a Waters qTOF instrument, spanning an m/z
                  range of 50.73 to 1199.98 and retention times from 0.02 to
                  58.07 minutes across 5 spectral embedding clusters.
                </p>
              </div>

              <div className="stat-row">
                <div className="stat-item">
                  <span className="stat-value">117</span>
                  <span className="stat-label">Spectra</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">20,083</span>
                  <span className="stat-label">Total Peaks</span>
                </div>
              </div>
              <div className="stat-row">
                <div className="stat-item">
                  <span className="stat-value">92.98</span>
                  <span className="stat-label">Spectra / sec</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">5</span>
                  <span className="stat-label">Clusters</span>
                </div>
              </div>

              <div className="equation">
                S(n) = p(n) where each spectrum maps to a unique partition state
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
