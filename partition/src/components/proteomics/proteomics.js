import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import useScrollReveal from '../../hooks/useScrollReveal'

const stages = [
    'multimodal_detection',
    'bijective_validation',
    'fragmentation',
    'ms1_measurement',
    'ionization',
    'chromatography',
    'molecular_structure',
];

const radarMetrics = [
    { label: 'Selection Rules', value: 1.0 },
    { label: 'Containment', value: 1.0 },
    { label: 'Bijective', value: 1.0 },
    { label: 'Physics Quality', value: 1.0 },
    { label: 'DRIP Symmetry', value: 0.57 },
];

export default function Proteomics({ ActiveIndex }) {
    const barChartRef = useRef(null);
    const radarChartRef = useRef(null);
    useScrollReveal();

    // 7-Stage Validation Bar Chart
    useEffect(() => {
        if (!barChartRef.current) return;

        const svg = d3.select(barChartRef.current);
        svg.selectAll('*').remove();

        const width = 480;
        const height = 300;
        const margin = { top: 20, right: 60, bottom: 20, left: 170 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const xScale = d3.scaleLinear()
            .domain([0, 100])
            .range([0, innerWidth]);

        const yScale = d3.scaleBand()
            .domain(stages)
            .range([0, innerHeight])
            .padding(0.3);

        // X axis
        g.append('g')
            .attr('transform', `translate(0,${innerHeight})`)
            .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => d + '%'))
            .selectAll('text')
            .attr('fill', '#b0b0b0')
            .style('font-size', '10px');

        g.selectAll('.domain, .tick line')
            .attr('stroke', '#333333');

        // Y axis
        g.append('g')
            .call(d3.axisLeft(yScale).tickSize(0))
            .selectAll('text')
            .attr('fill', '#b0b0b0')
            .style('font-size', '10px');

        g.selectAll('.domain')
            .attr('stroke', '#333333');

        // Bars
        g.selectAll('.bar')
            .data(stages)
            .enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', 0)
            .attr('y', d => yScale(d))
            .attr('width', xScale(100))
            .attr('height', yScale.bandwidth())
            .attr('fill', '#9980FA')
            .attr('rx', 3);

        // Bar labels
        g.selectAll('.bar-label')
            .data(stages)
            .enter()
            .append('text')
            .attr('class', 'bar-label')
            .attr('x', xScale(100) + 5)
            .attr('y', d => yScale(d) + yScale.bandwidth() / 2)
            .attr('dy', '0.35em')
            .attr('fill', '#b0b0b0')
            .style('font-size', '11px')
            .text('100%');

    }, [ActiveIndex]);

    // Ion Journey Metrics Radar Chart
    useEffect(() => {
        if (!radarChartRef.current) return;

        const svg = d3.select(radarChartRef.current);
        svg.selectAll('*').remove();

        const width = 480;
        const height = 300;
        const cx = width / 2;
        const cy = height / 2;
        const radius = Math.min(cx, cy) - 50;
        const numPoints = radarMetrics.length;
        const angleSlice = (2 * Math.PI) / numPoints;

        const g = svg.append('g')
            .attr('transform', `translate(${cx},${cy})`);

        // Grid circles
        const gridLevels = [0.25, 0.5, 0.75, 1.0];
        gridLevels.forEach(level => {
            g.append('circle')
                .attr('r', radius * level)
                .attr('fill', 'none')
                .attr('stroke', '#333333')
                .attr('stroke-dasharray', level < 1.0 ? '2,3' : '0');
        });

        // Radial lines
        for (let i = 0; i < numPoints; i++) {
            const angle = angleSlice * i - Math.PI / 2;
            g.append('line')
                .attr('x1', 0)
                .attr('y1', 0)
                .attr('x2', radius * Math.cos(angle))
                .attr('y2', radius * Math.sin(angle))
                .attr('stroke', '#333333');
        }

        // Data polygon
        const points = radarMetrics.map((d, i) => {
            const angle = angleSlice * i - Math.PI / 2;
            return [
                radius * d.value * Math.cos(angle),
                radius * d.value * Math.sin(angle)
            ];
        });

        const lineGenerator = d3.line().curve(d3.curveLinearClosed);

        g.append('path')
            .datum(points)
            .attr('d', lineGenerator)
            .attr('fill', 'rgba(153, 128, 250, 0.2)')
            .attr('stroke', '#9980FA')
            .attr('stroke-width', 2);

        // Vertex dots
        points.forEach(([x, y]) => {
            g.append('circle')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', 4)
                .attr('fill', '#9980FA');
        });

        // Labels
        radarMetrics.forEach((d, i) => {
            const angle = angleSlice * i - Math.PI / 2;
            const labelRadius = radius + 25;
            const x = labelRadius * Math.cos(angle);
            const y = labelRadius * Math.sin(angle);

            g.append('text')
                .attr('x', x)
                .attr('y', y)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .attr('fill', '#b0b0b0')
                .style('font-size', '10px')
                .text(`${d.label} (${d.value})`);
        });

    }, [ActiveIndex]);

    return (
        <>
            <div
                className={ActiveIndex === 2
                    ? "cavani_tm_section active animated fadeInUp"
                    : "cavani_tm_section hidden animated"}
                id="proteomics_"
            >
                <div className="section_inner">
                    <div className="framework-section">
                        <div className="section-grid">
                            {/* Left side: D3 SVG charts */}
                            <div className="chart-column">
                                <div className="chart-card">
                                    <h3 className="chart-title">7-Stage Ion Journey Validation</h3>
                                    <svg
                                        ref={barChartRef}
                                        viewBox="0 0 480 300"
                                        width="100%"
                                        preserveAspectRatio="xMidYMid meet"
                                    />
                                </div>
                                <div className="chart-card">
                                    <h3 className="chart-title">Ion Journey Quality Metrics</h3>
                                    <svg
                                        ref={radarChartRef}
                                        viewBox="0 0 480 300"
                                        width="100%"
                                        preserveAspectRatio="xMidYMid meet"
                                    />
                                </div>
                            </div>

                            {/* Right side: Key findings */}
                            <div className="content-column">
                                <div className="section-header">
                                    <span className="section-label">Proteomics</span>
                                    <h2>Ion Journey Validation</h2>
                                    <p className="equation">
                                        70 ion journeys across 7 stages, 490 tests, 100% conformance
                                    </p>
                                </div>

                                <div className="finding-block">
                                    <h4>NIST Spike Protein Validation</h4>
                                    <p>
                                        34 spectra from SARS-CoV-2 spike protein dataset.
                                        6 unique peptides, 2 unique glycans. HCD fragmentation
                                        with 100% pass rate across all 7 validation stages.
                                    </p>
                                </div>

                                <div className="finding-block">
                                    <h4>Lactoferrin Glycopeptides</h4>
                                    <p>
                                        36 glycopeptide entries covering 29 unique glycan structures.
                                        m/z range 1216&ndash;2717. Complete partition conformance
                                        verified for every entry.
                                    </p>
                                </div>

                                <div className="finding-block">
                                    <h4>Bijective Transformation</h4>
                                    <p>
                                        Machine-precision reversibility with mean error ~10<sup>-16</sup> (1.49e-16).
                                        Weber number We = 2.37, Reynolds number Re = 0.83.
                                        Every droplet transformation is exactly invertible.
                                    </p>
                                </div>

                                <div className="finding-block">
                                    <h4>Information Density</h4>
                                    <p>
                                        217 bits/ion information content versus ~20 bits conventional.
                                        Compression cost 47.26 kBT per ion. The framework extracts
                                        an order of magnitude more information from each measurement.
                                    </p>
                                </div>

                                <div className="stat-row">
                                    <div className="stat-item">
                                        <span className="stat-value">70</span>
                                        <span className="stat-label">Ion Journeys</span>
                                    </div>
                                    <div className="stat-item">
                                        <span className="stat-value">490</span>
                                        <span className="stat-label">Stage Tests</span>
                                    </div>
                                    <div className="stat-item">
                                        <span className="stat-value">1,292</span>
                                        <span className="stat-label">Theorems</span>
                                    </div>
                                    <div className="stat-item">
                                        <span className="stat-value">100%</span>
                                        <span className="stat-label">Pass Rate</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
