import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { useScrollReveal } from '../../hooks/useScrollReveal'

export default function Mass({ ActiveIndex }) {
  const sectionRef = useRef(null)
  const timelineChartRef = useRef(null)
  const identityChartRef = useRef(null)

  useScrollReveal(sectionRef)

  // Mass Accumulation Timeline
  useEffect(() => {
    if (!timelineChartRef.current) return
    const svg = d3.select(timelineChartRef.current)
    svg.selectAll('*').remove()

    const width = 480, height = 280
    const margin = { top: 30, right: 40, bottom: 50, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const numPoints = 50, alpha = 0.8
    const data = Array.from({ length: numPoints }, (_, i) => {
      const t = (i / (numPoints - 1)) * 100
      const mass = alpha * 20 * Math.log(1 + t * 0.15)
      return { t, mass }
    })

    const x = d3.scaleLinear().domain([0, 100]).range([0, innerWidth])
    const y = d3.scaleLinear().domain([0, d3.max(data, d => d.mass) * 1.1]).range([innerHeight, 0])

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    g.append('g').attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(6)).selectAll('text').attr('fill', '#b0b0b0').style('font-size', '11px')
    g.selectAll('.domain, line').attr('stroke', '#333333')

    g.append('g').call(d3.axisLeft(y).ticks(6))
      .selectAll('text').attr('fill', '#b0b0b0').style('font-size', '11px')
    g.selectAll('.domain, line').attr('stroke', '#333333')

    g.append('text').attr('x', innerWidth / 2).attr('y', innerHeight + 40)
      .attr('text-anchor', 'middle').attr('fill', '#b0b0b0').style('font-size', '11px').text('State Count (time)')
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerHeight / 2).attr('y', -45)
      .attr('text-anchor', 'middle').attr('fill', '#b0b0b0').style('font-size', '11px').text('Accumulated Mass')

    g.append('path').datum(data).attr('fill', 'rgba(249, 215, 126, 0.1)')
      .attr('d', d3.area().x(d => x(d.t)).y0(innerHeight).y1(d => y(d.mass)).curve(d3.curveBasis))

    g.append('path').datum(data).attr('fill', 'none')
      .attr('stroke', '#f9d77e').attr('stroke-width', 2)
      .attr('d', d3.line().x(d => x(d.t)).y(d => y(d.mass)).curve(d3.curveBasis))

    g.append('text').attr('x', innerWidth * 0.55).attr('y', innerHeight * 0.25)
      .attr('fill', '#f9d77e').attr('text-anchor', 'middle')
      .style('font-size', '12px').style('font-style', 'italic')
      .text('m = \u03B1 \u00B7 \u222B |N(t)| dt')

    svg.append('text').attr('x', width / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', '#b0b0b0').style('font-size', '13px')
      .text('Mass Accumulation Timeline')
  }, [])

  // Identity Chain Visualization
  useEffect(() => {
    if (!identityChartRef.current) return
    const svg = d3.select(identityChartRef.current)
    svg.selectAll('*').remove()

    const width = 480, height = 350
    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const labels = ['Mass', 'Memory', 'Non-Actualisations', 'State Count', 'Entropy']
    const boxWidth = 220, boxHeight = 36, gap = 26, startY = 30
    const centerX = width / 2

    labels.forEach((label, i) => {
      const yPos = startY + i * (boxHeight + gap)

      svg.append('rect').attr('x', centerX - boxWidth / 2).attr('y', yPos)
        .attr('width', boxWidth).attr('height', boxHeight).attr('rx', 8).attr('ry', 8)
        .attr('fill', 'rgba(153, 128, 250, 0.1)').attr('stroke', '#9980FA').attr('stroke-width', 1.5)

      svg.append('text').attr('x', centerX).attr('y', yPos + boxHeight / 2 + 5)
        .attr('text-anchor', 'middle').attr('fill', '#b0b0b0')
        .style('font-size', '13px').style('font-weight', '500').text(label)

      if (i < labels.length - 1) {
        const arrowStartY = yPos + boxHeight
        const arrowEndY = yPos + boxHeight + gap
        svg.append('line').attr('x1', centerX).attr('y1', arrowStartY + 2)
          .attr('x2', centerX).attr('y2', arrowEndY - 6)
          .attr('stroke', '#f9d77e').attr('stroke-width', 1.5)
        svg.append('polygon')
          .attr('points', `${centerX - 5},${arrowEndY - 8} ${centerX + 5},${arrowEndY - 8} ${centerX},${arrowEndY - 2}`)
          .attr('fill', '#f9d77e')
      }
    })

    labels.slice(0, -1).forEach((_, i) => {
      const yPos = startY + i * (boxHeight + gap) + boxHeight + gap / 2
      svg.append('text').attr('x', centerX + boxWidth / 2 + 20).attr('y', yPos + 3)
        .attr('text-anchor', 'middle').attr('fill', '#f9d77e')
        .style('font-size', '16px').style('font-weight', 'bold').text('=')
    })
  }, [])

  return (
    <div
      className={ActiveIndex === 5
        ? 'cavani_tm_section active animated fadeInUp'
        : 'cavani_tm_section hidden animated'}
      id="mass_"
    >
      <div className="section_inner">
        <div className="scrolly-section" ref={sectionRef}>
          {/* Left: pinned charts */}
          <div className="chart-wrapper">
            <div className="chart-card">
              <h3 className="chart-title">Mass Accumulation</h3>
              <svg ref={timelineChartRef} width="100%" />
            </div>
            <div className="chart-card">
              <h3 className="chart-title">Identity Chain</h3>
              <svg ref={identityChartRef} width="100%" />
            </div>
          </div>

          {/* Right: scrolling steps */}
          <div className="scroll-steps">
            <section className="step step-header">
              <span className="step-label">Mass</span>
              <h2>Mass = Memory</h2>
              <p>
                Mass is not a property placed into particles. It is the
                accumulated record of non-actualisations — everything that
                did not happen. At each moment, one state actualises and all
                others do not. That accumulation IS mass.
              </p>
              <div className="stat-row">
                <div className="stat-item">
                  <span className="stat-value">132,000+</span>
                  <span className="stat-label">Validated Operations</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">100%</span>
                  <span className="stat-label">Conformance</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">5</span>
                  <span className="stat-label">Identity Chain Links</span>
                </div>
              </div>
            </section>

            <section className="step">
              <h4>Mass is Memory</h4>
              <p>
                Mass is the accumulated record of everything that DID NOT
                happen. At each moment one state actualises; all others
                contribute to N(t). This accumulation over the ion&#39;s journey
                is identically its mass: m = memory = state count = entropy.
              </p>
              <div className="equation">
                m = &#945; &middot; &int;<sub>0</sub><sup>t</sup> |N(&#964;)| d&#964;
              </div>
            </section>

            <section className="step">
              <h4>E = mc&sup2; as Theorem</h4>
              <p>
                Mass-energy equivalence follows as a mathematical consequence
                of bounded phase space. No postulate required — it is a
                theorem derivable from the single axiom.
              </p>
              <div className="equation">
                E = mc&sup2; (derived, not postulated)
              </div>
            </section>

            <section className="step">
              <h4>The Mass Spectrometer Reads Memory</h4>
              <p>
                The state-mass correspondence N<sub>state</sub> &#8596; m/z
                is an identity. Mass spectrometry does not just measure mass —
                it reads the accumulated history of the ion&#39;s journey through
                phase space.
              </p>
            </section>

            <section className="step">
              <h4>Heat-Entropy Decoupling</h4>
              <p>
                Energy exchange and memory accumulation are independent
                processes. This resolves Maxwell&#39;s Demon: the demon cannot
                erase memory without cost because memory IS mass.
              </p>
            </section>
          </div>
        </div>
      </div>
    </div>
  )
}
