import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { useScrollReveal } from '../../hooks/useScrollReveal'

export default function Charge({ ActiveIndex }) {
  const sectionRef = useRef(null)
  const capacityChartRef = useRef(null)
  const gaugeChartRef = useRef(null)

  useScrollReveal(sectionRef)

  // Partition Capacity Chart: C(n) = 2n^2
  useEffect(() => {
    if (!capacityChartRef.current) return
    const svg = d3.select(capacityChartRef.current)
    svg.selectAll('*').remove()

    const width = 480, height = 280
    const margin = { top: 30, right: 50, bottom: 40, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const data = [
      { n: 1, capacity: 2, cumulative: 2 },
      { n: 2, capacity: 8, cumulative: 10 },
      { n: 3, capacity: 18, cumulative: 28 },
      { n: 4, capacity: 32, cumulative: 60 },
      { n: 5, capacity: 50, cumulative: 110 },
      { n: 6, capacity: 72, cumulative: 182 },
      { n: 7, capacity: 98, cumulative: 280 },
    ]

    const x = d3.scaleBand().domain(data.map(d => d.n)).range([0, innerWidth]).padding(0.25)
    const yLeft = d3.scaleLinear().domain([0, 110]).range([innerHeight, 0])
    const yRight = d3.scaleLinear().domain([0, 300]).range([innerHeight, 0])

    const defs = svg.append('defs')
    const gradient = defs.append('linearGradient')
      .attr('id', 'capacityGradient').attr('x1', '0%').attr('y1', '100%').attr('x2', '0%').attr('y2', '0%')
    gradient.append('stop').attr('offset', '0%').attr('stop-color', '#9980FA')
    gradient.append('stop').attr('offset', '100%').attr('stop-color', '#f9d77e')

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    g.append('g').attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).tickFormat(d => `n=${d}`))
      .selectAll('text').attr('fill', '#b0b0b0').style('font-size', '11px')
    g.selectAll('.domain, line').attr('stroke', '#333333')

    g.append('g').call(d3.axisLeft(yLeft).ticks(6))
      .selectAll('text').attr('fill', '#b0b0b0').style('font-size', '11px')
    g.selectAll('.domain, line').attr('stroke', '#333333')

    g.append('g').attr('transform', `translate(${innerWidth},0)`)
      .call(d3.axisRight(yRight).ticks(6))
      .selectAll('text').attr('fill', '#f9d77e').style('font-size', '10px')

    g.append('text').attr('x', innerWidth / 2).attr('y', innerHeight + 35)
      .attr('text-anchor', 'middle').attr('fill', '#b0b0b0').style('font-size', '11px')
      .text('Principal Quantum Number n')

    g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerHeight / 2).attr('y', -45)
      .attr('text-anchor', 'middle').attr('fill', '#b0b0b0').style('font-size', '11px')
      .text('Capacity C(n)')

    g.selectAll('.bar').data(data).enter().append('rect')
      .attr('class', 'bar').attr('x', d => x(d.n)).attr('width', x.bandwidth())
      .attr('y', innerHeight).attr('height', 0).attr('fill', 'url(#capacityGradient)').attr('rx', 2)
      .transition().duration(800).delay((_, i) => i * 80)
      .attr('y', d => yLeft(d.capacity)).attr('height', d => innerHeight - yLeft(d.capacity))

    g.selectAll('.bar-label').data(data).enter().append('text')
      .attr('class', 'bar-label').attr('x', d => x(d.n) + x.bandwidth() / 2)
      .attr('y', d => yLeft(d.capacity) - 5).attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0').style('font-size', '10px').text(d => d.capacity)

    const line = d3.line().x(d => x(d.n) + x.bandwidth() / 2).y(d => yRight(d.cumulative)).curve(d3.curveMonotoneX)
    g.append('path').datum(data).attr('fill', 'none')
      .attr('stroke', '#f9d77e').attr('stroke-width', 2).attr('stroke-dasharray', '6,3').attr('d', line)

    g.selectAll('.cum-dot').data(data).enter().append('circle')
      .attr('cx', d => x(d.n) + x.bandwidth() / 2).attr('cy', d => yRight(d.cumulative))
      .attr('r', 3).attr('fill', '#f9d77e')

    svg.append('text').attr('x', width / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', '#b0b0b0').style('font-size', '13px')
      .text('Partition Capacity C(n) = 2n\u00B2')
  }, [])

  // Residue Ratio Gauge
  useEffect(() => {
    if (!gaugeChartRef.current) return
    const svg = d3.select(gaugeChartRef.current)
    svg.selectAll('*').remove()

    const width = 480, height = 280
    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const cx = width / 2, cy = height * 0.65, radius = 120
    const startAngle = -Math.PI, endAngle = 0

    const theoretical = 26 / 27
    const arcGenerator = d3.arc().innerRadius(radius - 18).outerRadius(radius).startAngle(startAngle).cornerRadius(2)

    const g = svg.append('g').attr('transform', `translate(${cx},${cy})`)

    g.append('path').attr('d', arcGenerator({ endAngle: endAngle })).attr('fill', '#333333')

    const theoreticalAngle = startAngle + theoretical * Math.PI
    g.append('path').attr('d', arcGenerator({ endAngle: startAngle })).attr('fill', '#f9d77e')
      .transition().duration(1200)
      .attrTween('d', function () {
        const interpolate = d3.interpolate(startAngle, theoreticalAngle)
        return function (t) { return arcGenerator({ endAngle: interpolate(t) }) }
      })

    const theoreticalX = -radius * Math.cos(theoretical * Math.PI)
    const theoreticalY = -radius * Math.sin(theoretical * Math.PI)
    g.append('circle').attr('cx', theoreticalX).attr('cy', theoreticalY)
      .attr('r', 5).attr('fill', '#f9d77e').attr('stroke', '#1a1a2e').attr('stroke-width', 2)
    g.append('text').attr('x', theoreticalX + 10).attr('y', theoreticalY - 8)
      .attr('fill', '#f9d77e').style('font-size', '10px').text('Theoretical: 0.963')

    const measuredAngle = 0.158 * Math.PI
    const measuredX = -radius * Math.cos(measuredAngle)
    const measuredY = -radius * Math.sin(measuredAngle)
    g.append('circle').attr('cx', measuredX).attr('cy', measuredY)
      .attr('r', 5).attr('fill', '#9980FA').attr('stroke', '#1a1a2e').attr('stroke-width', 2)
    g.append('text').attr('x', measuredX + 10).attr('y', measuredY + 15)
      .attr('fill', '#9980FA').style('font-size', '10px').text('Mean depth: 0.158')

    g.append('text').attr('x', 0).attr('y', -20).attr('text-anchor', 'middle')
      .attr('fill', '#f9d77e').style('font-size', '28px').style('font-weight', 'bold').text('26/27')
    g.append('text').attr('x', 0).attr('y', 5).attr('text-anchor', 'middle')
      .attr('fill', '#b0b0b0').style('font-size', '12px').text('Residue Ratio')
    g.append('text').attr('x', 0).attr('y', 22).attr('text-anchor', 'middle')
      .attr('fill', '#666').style('font-size', '10px').text('(b\u00B3\u2212 1) / b\u00B3 where b = 3')

    g.append('text').attr('x', -radius - 5).attr('y', 15).attr('text-anchor', 'end')
      .attr('fill', '#666').style('font-size', '10px').text('0')
    g.append('text').attr('x', radius + 5).attr('y', 15).attr('text-anchor', 'start')
      .attr('fill', '#666').style('font-size', '10px').text('1')

    svg.append('text').attr('x', width / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', '#b0b0b0').style('font-size', '13px')
      .text('Partition Residue Ratio Gauge')
  }, [])

  return (
    <div
      className={ActiveIndex === 4
        ? 'cavani_tm_section active animated fadeInUp'
        : 'cavani_tm_section hidden animated'}
      id="charge_"
    >
      <div className="section_inner">
        <div className="scrolly-section" ref={sectionRef}>
          {/* Left: pinned charts */}
          <div className="chart-wrapper">
            <div className="chart-card">
              <h3 className="chart-title">Partition Capacity</h3>
              <svg ref={capacityChartRef} width="100%" />
            </div>
            <div className="chart-card">
              <h3 className="chart-title">Residue Ratio</h3>
              <svg ref={gaugeChartRef} width="100%" />
            </div>
          </div>

          {/* Right: scrolling steps */}
          <div className="scroll-steps">
            <section className="step step-header">
              <span className="step-label">Charge</span>
              <h2>Charge Emergence from Partition</h2>
              <p>
                Charge is not a fundamental injection -- it emerges when matter
                is partitioned. Partition coordinates (n, l, m, s) assign each
                ion a unique state, and the observed charge states arise directly from
                the bounded phase-space structure.
              </p>
              <div className="stat-row">
                <div className="stat-item">
                  <span className="stat-value">4,545</span>
                  <span className="stat-label">NIST Entries</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">100%</span>
                  <span className="stat-label">Conformance</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">26/27</span>
                  <span className="stat-label">Residue Ratio</span>
                </div>
              </div>
            </section>

            <section className="step">
              <h4>Charge as Emergence</h4>
              <p>
                Charge is not injected; it appears when matter is partitioned.
                No partition means no charge. The act of confining an ion within
                a bounded phase space is what gives rise to its charge state --
                charge is a consequence of counting, not a separate postulate.
              </p>
            </section>

            <section className="step">
              <h4>Partition Coordinates</h4>
              <p>
                Each ion is assigned coordinates (n, l, m, s), exactly analogous
                to atomic orbitals. The capacity at each principal level follows
                C(n) = 2n&sup2;, and the cumulative state count grows as
                N_state(n) = n(n+1)(2n+1)/3, guaranteeing every ion maps to a
                unique, countable position.
              </p>
              <div className="equation">
                C(n) = 2n&sup2; &nbsp;&nbsp;|&nbsp;&nbsp; N_state(n) = n(n+1)(2n+1)/3
              </div>
            </section>

            <section className="step">
              <h4>The 26/27 Ratio</h4>
              <p>
                The partition residue ratio (b<sup>d</sup>&minus;1)/b<sup>d</sup> = 26/27
                for b=3, d=3 is the fraction of phase space that becomes
                &quot;mass.&quot; The mean partition depth of 0.166 and mean residue
                ratio of 0.158 quantify how deeply each ion is embedded in its
                partition tree.
              </p>
              <div className="equation">
                Residue ratio = (b<sup>d</sup> &minus; 1) / b<sup>d</sup> = 26/27
              </div>
            </section>

            <section className="step">
              <h4>Universal Conformance</h4>
              <p>
                100% charge emergence conformance across 4,545 NIST
                glycan/glycopeptide entries from 11 independent laboratories.
                Every observed charge state maps exactly to the partition
                prediction with zero exceptions.
              </p>
            </section>
          </div>
        </div>
      </div>
    </div>
  )
}
