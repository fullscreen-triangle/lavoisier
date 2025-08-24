`<svg xmlns="http://www.w3.org/2000/svg" width="980" height="560" viewBox="0 0 980 560">

  <title>Ecosystem Tool Integration (Expanded)</title>
  <defs>
    <marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8Z" fill="black"/>
    </marker>
    <style>text{font-size:11px;font-family:Arial,sans-serif}</style>
  </defs>

  <!-- Central orchestrator -->
  <circle cx="490" cy="280" r="95" stroke="black" fill="white"/>
  <text x="490" y="268" text-anchor="middle">Tool</text>
  <text x="490" y="284" text-anchor="middle">Orchestrator</text>
  <text x="490" y="300" text-anchor="middle">(Async Core)</text>

  <!-- External tools (hex ring + top/bottom) -->
  <rect x="445" y="40" width="90" height="50" stroke="black" fill="white"/>
  <text x="490" y="70" text-anchor="middle">Autobahn</text>

  <rect x="765" y="180" width="110" height="50" stroke="black" fill="white"/>
  <text x="820" y="210" text-anchor="middle">Hegel</text>

  <rect x="765" y="380" width="110" height="50" stroke="black" fill="white"/>
  <text x="820" y="410" text-anchor="middle">Nebuchadnezzar</text>

  <rect x="445" y="470" width="90" height="50" stroke="black" fill="white"/>
  <text x="490" y="500" text-anchor="middle">Lavoisier</text>

  <rect x="105" y="380" width="110" height="50" stroke="black" fill="white"/>
  <text x="160" y="410" text-anchor="middle">Borgia</text>

  <rect x="105" y="180" width="110" height="50" stroke="black" fill="white"/>
  <text x="160" y="210" text-anchor="middle">Bene Gesserit</text>

  <!-- Health / telemetry (dashed) -->
  <rect x="870" y="20" width="90" height="60" stroke="black" fill="none" stroke-dasharray="5 3"/>
  <text x="915" y="48" text-anchor="middle">Telemetry</text>
  <text x="915" y="60" text-anchor="middle">Monitoring</text>

  <!-- Arrows outward (requests) -->
  <g stroke="black" marker-end="url(#ar)">
    <line x1="490" y1="185" x2="490" y2="90"/>
    <line x1="560" y1="230" x2="760" y2="200"/>
    <line x1="560" y1="330" x2="760" y2="410"/>
    <line x1="490" y1="375" x2="490" y2="470"/>
    <line x1="420" y1="330" x2="215" y2="410"/>
    <line x1="420" y1="230" x2="215" y2="200"/>
  </g>

  <!-- Return (responses) dashed -->
  <g stroke="black" stroke-dasharray="5 3" marker-end="url(#ar)">
    <line x1="490" y1="90" x2="490" y2="170"/>
    <line x1="760" y1="200" x2="570" y2="250"/>
    <line x1="760" y1="410" x2="570" y2="310"/>
    <line x1="490" y1="470" x2="490" y2="390"/>
    <line x1="215" y1="410" x2="410" y2="310"/>
    <line x1="215" y1="200" x2="410" y2="250"/>
  </g>

  <!-- Telemetry link -->
  <g stroke="black" marker-end="url(#ar)">
    <line x1="820" y1="180" x2="900" y2="80"/>
    <line x1="820" y1="380" x2="900" y2="80"/>
    <line x1="490" y1="185" x2="900" y2="80"/>
    <line x1="490" y1="375" x2="900" y2="80"/>
    <line x1="160" y1="180" x2="900" y2="80"/>
    <line x1="160" y1="380" x2="900" y2="80"/>
  </g>

  <!-- Legend -->
  <rect x="40" y="30" width="300" height="70" stroke="black" fill="none"/>
  <text x="50" y="50">Solid: Request / Invocation</text>
  <text x="50" y="66">Dashed: Response / Result</text>
  <text x="50" y="82">Dashed Box: Monitoring System</text>
</svg>
`
