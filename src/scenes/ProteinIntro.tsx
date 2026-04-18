import React, { useState } from 'react'

// ── Section data ───────────────────────────────────────

interface Section {
  id: number
  title: string
  content: React.ReactNode
}

// ── Visual components ──────────────────────────────────

function AminoAcidChain() {
  const acids = [
    { letter: 'M', name: 'Met', color: '#f9a825', group: 'nonpolar' },
    { letter: 'V', name: 'Val', color: '#43a047', group: 'nonpolar' },
    { letter: 'L', name: 'Leu', color: '#43a047', group: 'nonpolar' },
    { letter: 'S', name: 'Ser', color: '#1e88e5', group: 'polar' },
    { letter: 'P', name: 'Pro', color: '#8e24aa', group: 'special' },
    { letter: 'A', name: 'Ala', color: '#757575', group: 'nonpolar' },
    { letter: 'D', name: 'Asp', color: '#e53935', group: 'negative' },
    { letter: 'K', name: 'Lys', color: '#1565c0', group: 'positive' },
    { letter: 'T', name: 'Thr', color: '#1e88e5', group: 'polar' },
    { letter: 'N', name: 'Asn', color: '#1e88e5', group: 'polar' },
    { letter: 'V', name: 'Val', color: '#43a047', group: 'nonpolar' },
    { letter: 'K', name: 'Lys', color: '#1565c0', group: 'positive' },
    { letter: 'A', name: 'Ala', color: '#757575', group: 'nonpolar' },
    { letter: '...', name: '', color: '#bbb', group: '' },
  ]

  return (
    <svg viewBox="0 0 720 100" style={{ width: '100%', maxWidth: 720, height: 'auto' }}>
      <defs>
        {acids.map((a, i) => a.name && (
          <radialGradient key={`g-${i}`} id={`acid-glow-${i}`}>
            <stop offset="0%" stopColor={a.color} stopOpacity={0.3} />
            <stop offset="100%" stopColor={a.color} stopOpacity={0} />
          </radialGradient>
        ))}
      </defs>

      {acids.map((a, i) => (
        <g key={i}>
          {/* Peptide bond line with animated pulse */}
          {i > 0 && (
            <g>
              <line x1={i * 50 - 8} y1={38} x2={i * 50 + 5} y2={38}
                stroke="#ccc" strokeWidth={2.5} strokeLinecap="round" />
              {/* Bond electron animation */}
              <circle r={2} fill={acids[i - 1].color} opacity={0.5}>
                <animateMotion dur="1.5s" begin={`${i * 0.12}s`} repeatCount="indefinite"
                  path={`M${i * 50 - 8},38 L${i * 50 + 5},38`} />
              </circle>
            </g>
          )}
          {/* Outer glow */}
          {a.name && (
            <circle cx={i * 50 + 22} cy={38} r={24} fill={`url(#acid-glow-${i})`}>
              <animate attributeName="r" values="22;26;22" dur={`${2 + i * 0.1}s`} repeatCount="indefinite" />
            </circle>
          )}
          {/* Main circle with entrance animation */}
          <circle cx={i * 50 + 22} cy={38} r={16} fill={a.color} opacity={0.9}>
            <animate attributeName="r" values="0;18;16" dur="0.5s" begin={`${i * 0.06}s`} fill="freeze" />
          </circle>
          <text x={i * 50 + 22} y={43} textAnchor="middle" fontSize={14}
            fill="#fff" fontWeight={700} fontFamily="monospace">{a.letter}</text>
          {a.name && (
            <text x={i * 50 + 22} y={68} textAnchor="middle" fontSize={9}
              fill="#999" fontFamily="Inter, sans-serif">{a.name}</text>
          )}
          {/* Position number */}
          {a.name && (
            <text x={i * 50 + 22} y={82} textAnchor="middle" fontSize={7.5}
              fill="#ccc" fontFamily="JetBrains Mono, monospace">{i + 1}</text>
          )}
        </g>
      ))}

      {/* Chemical property legend */}
      <g transform="translate(10, 2)">
        {[
          { color: '#43a047', label: 'hydrophobic' },
          { color: '#1e88e5', label: 'polar' },
          { color: '#e53935', label: 'negative' },
          { color: '#1565c0', label: 'positive' },
        ].map((item, i) => (
          <g key={i} transform={`translate(${i * 130}, 0)`}>
            <circle cx={4} cy={4} r={3.5} fill={item.color} opacity={0.7} />
            <text x={11} y={7} fontSize={8} fill="#aaa" fontFamily="Inter, sans-serif">{item.label}</text>
          </g>
        ))}
      </g>
    </svg>
  )
}

function FoldingLevels() {
  return (
    <svg viewBox="0 0 800 210" style={{ width: '100%', maxWidth: 800, height: 'auto' }}>
      <defs>
        <filter id="glow-primary">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="glow-secondary">
          <feGaussianBlur stdDeviation="2.5" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {/* Progression arrows between levels */}
      {[180, 390, 590].map((x, i) => (
        <g key={`arrow-${i}`}>
          <line x1={x} y1={60} x2={x + 25} y2={60} stroke="#ddd" strokeWidth={1.5} />
          <polygon points={`${x + 25},60 ${x + 18},55 ${x + 18},65`} fill="#ddd" />
          <circle r={2.5} fill={['#e65100', '#1565c0', '#2e7d32'][i]} opacity={0.6}>
            <animateMotion dur="1s" begin={`${i * 0.3}s`} repeatCount="indefinite"
              path={`M${x},60 L${x + 25},60`} />
          </circle>
        </g>
      ))}

      {/* Primary */}
      <g>
        <text x={90} y={20} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#e65100" fontFamily="Inter, sans-serif">Primary</text>
        <text x={90} y={34} textAnchor="middle" fontSize={10}
          fill="#999" fontFamily="Inter, sans-serif">sequence</text>
        {Array.from({ length: 8 }).map((_, i) => (
          <g key={i}>
            {i > 0 && <line x1={i * 22 + 18} y1={55} x2={i * 22 + 28} y2={55} stroke="#ddd" strokeWidth={1.5} />}
            <circle cx={i * 22 + 32} cy={55} r={7} fill="#e65100" opacity={0.8}>
              <animate attributeName="r" values="0;8;7" dur="0.4s" begin={`${i * 0.05}s`} fill="freeze" />
            </circle>
          </g>
        ))}
        <text x={90} y={80} textAnchor="middle" fontSize={8.5} fill="#bbb" fontFamily="Inter, sans-serif">
          MVLSPADK...
        </text>
      </g>

      {/* Secondary */}
      <g transform="translate(210, 0)">
        <text x={80} y={20} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#1565c0" fontFamily="Inter, sans-serif">Secondary</text>
        <text x={80} y={34} textAnchor="middle" fontSize={10}
          fill="#999" fontFamily="Inter, sans-serif">α-helix, β-sheet</text>
        {/* Animated helix with stroke-dashoffset */}
        <path d="M20,50 C35,35 50,65 65,50 C80,35 95,65 110,50 C125,35 140,65 155,50"
          fill="none" stroke="#1565c0" strokeWidth={4} strokeLinecap="round"
          filter="url(#glow-secondary)">
          <animate attributeName="stroke-dashoffset" values="200;0" dur="2s" fill="freeze" />
          <animate attributeName="stroke-dasharray" values="0 200;200 0" dur="2s" fill="freeze" />
        </path>
        {/* Animated particle along helix */}
        <circle r={3} fill="#42a5f5" opacity={0.7}>
          <animateMotion dur="3s" repeatCount="indefinite"
            path="M20,50 C35,35 50,65 65,50 C80,35 95,65 110,50 C125,35 140,65 155,50" />
        </circle>
        <text x={80} y={78} textAnchor="middle" fontSize={9} fill="#666" fontFamily="Inter, sans-serif">α-helix</text>
        {/* Beta sheet with hydrogen bond animation */}
        <g transform="translate(0, 85)">
          <line x1={20} y1={10} x2={155} y2={10} stroke="#1565c0" strokeWidth={3} />
          <line x1={20} y1={30} x2={155} y2={30} stroke="#1565c0" strokeWidth={3} />
          {[40, 80, 120].map((x, i) => (
            <g key={x}>
              <line x1={x} y1={13} x2={x} y2={27} stroke="#64b5f6" strokeWidth={1.2} strokeDasharray="2,2" />
              {/* Animated H-bond dot */}
              <circle r={2} fill="#64b5f6" opacity={0.7}>
                <animateMotion dur="1.5s" begin={`${i * 0.3}s`} repeatCount="indefinite"
                  path={`M${x},13 L${x},27`} />
              </circle>
            </g>
          ))}
          <text x={80} y={47} textAnchor="middle" fontSize={9} fill="#666" fontFamily="Inter, sans-serif">β-sheet (H-bonds)</text>
        </g>
      </g>

      {/* Tertiary */}
      <g transform="translate(420, 0)">
        <text x={80} y={20} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#2e7d32" fontFamily="Inter, sans-serif">Tertiary</text>
        <text x={80} y={34} textAnchor="middle" fontSize={10}
          fill="#999" fontFamily="Inter, sans-serif">3D fold</text>
        <path d="M30,55 C45,30 60,75 75,50 C90,25 80,80 100,55 C115,35 125,70 140,55"
          fill="none" stroke="#2e7d32" strokeWidth={4} strokeLinecap="round" />
        {/* Animated fold particle */}
        <circle r={3} fill="#66bb6a" opacity={0.7}>
          <animateMotion dur="4s" repeatCount="indefinite"
            path="M30,55 C45,30 60,75 75,50 C90,25 80,80 100,55 C115,35 125,70 140,55" />
        </circle>
        {/* Disulfide bond with pulsing */}
        <path d="M70,50 Q60,90 100,80 Q130,75 120,55"
          fill="none" stroke="#ffab00" strokeWidth={2} strokeDasharray="3,3">
          <animate attributeName="stroke-opacity" values="0.3;0.8;0.3" dur="2s" repeatCount="indefinite" />
        </path>
        <circle cx={70} cy={50} r={4} fill="#c62828" opacity={0.8}>
          <animate attributeName="r" values="3;5;3" dur="2s" repeatCount="indefinite" />
        </circle>
        <circle cx={120} cy={55} r={4} fill="#c62828" opacity={0.8}>
          <animate attributeName="r" values="3;5;3" dur="2s" begin="0.5s" repeatCount="indefinite" />
        </circle>
        <text x={95} y={100} textAnchor="middle" fontSize={8} fill="#999" fontFamily="Inter, sans-serif">
          disulfide bond (S-S)
        </text>
      </g>

      {/* Quaternary */}
      <g transform="translate(620, 0)">
        <text x={80} y={20} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#6a1b9a" fontFamily="Inter, sans-serif">Quaternary</text>
        <text x={80} y={34} textAnchor="middle" fontSize={10}
          fill="#999" fontFamily="Inter, sans-serif">multi-chain</text>
        {/* Animated subunits with gentle breathing */}
        {[
          { cx: 55, cy: 70, rx: 30, ry: 22, fill: '#ce93d8', delay: 0 },
          { cx: 95, cy: 65, rx: 28, ry: 20, fill: '#b39ddb', delay: 0.5 },
          { cx: 70, cy: 90, rx: 25, ry: 18, fill: '#e1bee7', delay: 1.0 },
          { cx: 100, cy: 88, rx: 27, ry: 19, fill: '#d1c4e9', delay: 1.5 },
        ].map((e, i) => (
          <ellipse key={i} cx={e.cx} cy={e.cy} rx={e.rx} ry={e.ry}
            fill={e.fill} opacity={0.45} stroke="#6a1b9a" strokeWidth={1.5}>
            <animate attributeName="rx" values={`${e.rx};${e.rx + 2};${e.rx}`}
              dur="3s" begin={`${e.delay}s`} repeatCount="indefinite" />
            <animate attributeName="ry" values={`${e.ry};${e.ry + 1.5};${e.ry}`}
              dur="3s" begin={`${e.delay}s`} repeatCount="indefinite" />
          </ellipse>
        ))}
        {/* Interface interaction particles */}
        <circle r={2} fill="#6a1b9a" opacity={0.5}>
          <animateMotion dur="2s" repeatCount="indefinite"
            path="M55,70 Q75,77 95,65" />
        </circle>
        <circle r={2} fill="#6a1b9a" opacity={0.5}>
          <animateMotion dur="2.5s" begin="0.7s" repeatCount="indefinite"
            path="M70,90 Q85,85 100,88" />
        </circle>
        <text x={75} y={125} textAnchor="middle" fontSize={8} fill="#999" fontFamily="Inter, sans-serif">
          e.g. hemoglobin (4 subunits)
        </text>
      </g>
    </svg>
  )
}

function LevinthalDiagram() {
  // Use seeded pseudo-random for consistent dot positions
  const dots = Array.from({ length: 50 }, (_, i) => ({
    x: 80 + Math.sin(i * 2.7) * (180 - i * 2) + i * 7,
    y: 25 + Math.sin(i * 1.3 + 0.5) * 15 + (i % 3) * 10,
    delay: (i * 0.07) % 3,
  }))

  return (
    <svg viewBox="0 0 520 270" style={{ width: '100%', maxWidth: 520, height: 'auto' }}>
      <defs>
        <linearGradient id="funnelGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#1976d2" stopOpacity={0.03} />
          <stop offset="50%" stopColor="#1976d2" stopOpacity={0.08} />
          <stop offset="100%" stopColor="#2e7d32" stopOpacity={0.15} />
        </linearGradient>
        <radialGradient id="nativeGlow">
          <stop offset="0%" stopColor="#2e7d32" stopOpacity={0.6} />
          <stop offset="50%" stopColor="#2e7d32" stopOpacity={0.2} />
          <stop offset="100%" stopColor="#2e7d32" stopOpacity={0} />
        </radialGradient>
      </defs>

      {/* Funnel with gradient fill */}
      <path d="M60,20 L260,230 L460,20 Z" fill="url(#funnelGrad)" stroke="#1976d2" strokeWidth={1.5} />

      {/* Animated conformational dots tumbling down */}
      {dots.map((d, i) => (
        <g key={i}>
          <circle cx={d.x} cy={d.y} r={2.5} fill="#1976d2" opacity={0.25}>
            <animate attributeName="opacity" values="0.15;0.4;0.15" dur={`${2 + d.delay}s`}
              begin={`${d.delay}s`} repeatCount="indefinite" />
          </circle>
          {/* Some dots "fall" down the funnel */}
          {i % 5 === 0 && (
            <circle r={2} fill="#42a5f5" opacity={0.4}>
              <animateMotion dur={`${3 + d.delay}s`} begin={`${d.delay * 2}s`} repeatCount="indefinite"
                path={`M${d.x},${d.y} Q260,${120 + i * 2} 260,230`} />
              <animate attributeName="opacity" values="0.5;0.1" dur={`${3 + d.delay}s`}
                begin={`${d.delay * 2}s`} repeatCount="indefinite" />
            </circle>
          )}
        </g>
      ))}

      {/* Main folding pathway arrow */}
      <path d="M260,50 Q240,120 260,220" fill="none" stroke="#e65100" strokeWidth={2.5} strokeDasharray="6,4" />
      <polygon points="260,225 254,213 266,213" fill="#e65100" />

      {/* Animated "correct pathway" particle */}
      <circle r={4} fill="#e65100" opacity={0.8}>
        <animateMotion dur="3s" repeatCount="indefinite"
          path="M260,50 Q240,120 260,220" />
      </circle>
      <circle r={8} fill="#e65100" opacity={0.15}>
        <animateMotion dur="3s" repeatCount="indefinite"
          path="M260,50 Q240,120 260,220" />
      </circle>

      {/* Native state with glow */}
      <circle cx={260} cy={240} r={20} fill="url(#nativeGlow)">
        <animate attributeName="r" values="18;24;18" dur="3s" repeatCount="indefinite" />
      </circle>
      <circle cx={260} cy={240} r={10} fill="#2e7d32" />
      <text x={260} y={244} textAnchor="middle" fontSize={8} fill="#fff" fontWeight={700}>Native</text>

      {/* Labels */}
      <text x={260} y={15} textAnchor="middle" fontSize={12} fontWeight={600}
        fill="#1565c0" fontFamily="Inter, sans-serif">~10³⁰⁰ possible conformations</text>
      <text x={260} y={265} textAnchor="middle" fontSize={11} fill="#2e7d32"
        fontWeight={600} fontFamily="Inter, sans-serif">1 native structure</text>

      {/* Time comparison */}
      <g transform="translate(320, 75)">
        <rect x={-8} y={-12} width={195} height={110} rx={8}
          fill="#fff5f5" stroke="#ffcdd2" strokeWidth={0.8} />
        <text x={0} y={0} fontSize={11} fill="#333" fontWeight={600} fontFamily="Inter, sans-serif">
          Brute-force search:
        </text>
        <text x={0} y={18} fontSize={11} fill="#c62828" fontWeight={600} fontFamily="Inter, sans-serif">
          &gt; age of the universe
        </text>
        <line x1={0} y1={30} x2={170} y2={30} stroke="#eee" strokeWidth={1} />
        <text x={0} y={48} fontSize={11} fill="#333" fontWeight={600} fontFamily="Inter, sans-serif">
          Actual protein folding:
        </text>
        <text x={0} y={66} fontSize={11} fill="#2e7d32" fontWeight={600} fontFamily="Inter, sans-serif">
          μs to seconds
        </text>
        <text x={0} y={90} fontSize={10} fill="#999" fontStyle="italic" fontFamily="Inter, sans-serif">
          — Levinthal's Paradox (1969)
        </text>
      </g>
    </svg>
  )
}

function ProteinExamples() {
  const examples = [
    {
      name: 'Hemoglobin',
      residues: '~574 (4 chains)',
      function: 'Carries oxygen in blood',
      difficulty: 'Quaternary structure — 4 subunits must assemble correctly',
      color: '#c62828',
    },
    {
      name: 'Insulin',
      residues: '51 (2 chains)',
      function: 'Regulates blood sugar',
      difficulty: 'Disulfide bonds between chains — must predict cross-chain contacts',
      color: '#1565c0',
    },
    {
      name: 'SARS-CoV-2 Spike',
      residues: '~1273',
      function: 'Viral entry into cells',
      difficulty: 'Very large, flexible, heavily glycosylated — multiple conformational states',
      color: '#2e7d32',
    },
    {
      name: 'p53 (tumor suppressor)',
      residues: '393',
      function: 'Prevents cancer',
      difficulty: 'Intrinsically disordered regions — no single stable structure',
      color: '#6a1b9a',
    },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, maxWidth: 720 }}>
      {examples.map(ex => (
        <div key={ex.name} style={{
          padding: '14px 16px', borderRadius: 8,
          border: `1px solid ${ex.color}22`,
          background: `${ex.color}08`,
        }}>
          <div style={{ fontSize: 15, fontWeight: 700, color: ex.color, marginBottom: 4 }}>
            {ex.name}
          </div>
          <div style={{ fontSize: 12, color: '#666', marginBottom: 2 }}>
            {ex.residues} residues — {ex.function}
          </div>
          <div style={{ fontSize: 11, color: '#999', fontStyle: 'italic' }}>
            Challenge: {ex.difficulty}
          </div>
        </div>
      ))}
    </div>
  )
}

function TimelineBar() {
  const events = [
    { year: '1972', label: 'Anfinsen wins Nobel:\nsequence → structure', x: 40 },
    { year: '1994', label: 'CASP1 competition\nlaunches', x: 160 },
    { year: '2018', label: 'AlphaFold1\nwins CASP13', x: 380 },
    { year: '2020', label: 'AlphaFold2\nCASP14: GDT ~92', x: 520 },
    { year: '2022', label: '200M structures\npredicted', x: 660 },
  ]

  return (
    <svg viewBox="0 0 750 110" style={{ width: '100%', maxWidth: 750, height: 'auto' }}>
      {/* Timeline line */}
      <line x1={30} y1={40} x2={720} y2={40} stroke="#ccc" strokeWidth={2} />

      {events.map((ev, i) => (
        <g key={i}>
          <circle cx={ev.x} cy={40} r={5}
            fill={i === 3 ? '#e65100' : i === 4 ? '#2e7d32' : '#1976d2'} />
          <text x={ev.x} y={28} textAnchor="middle" fontSize={11}
            fontWeight={700} fill={i === 3 ? '#e65100' : '#333'} fontFamily="Inter, sans-serif">
            {ev.year}
          </text>
          {ev.label.split('\n').map((line, li) => (
            <text key={li} x={ev.x} y={60 + li * 14} textAnchor="middle"
              fontSize={10} fill="#666" fontFamily="Inter, sans-serif">{line}</text>
          ))}
        </g>
      ))}
    </svg>
  )
}

function WhyUniqueComparison() {
  const approaches = [
    {
      name: 'Physics-based\n(Rosetta, ~2000s)',
      color: '#90a4ae',
      items: ['Energy minimization', 'Monte Carlo sampling', 'Fragment assembly'],
      limit: 'Too slow for large proteins;\ngets stuck in local minima',
      accuracy: 35,
    },
    {
      name: 'Co-evolution\n(DCA, ~2010s)',
      color: '#42a5f5',
      items: ['MSA statistics', 'Contact prediction', 'Separate folding step'],
      limit: 'Predicts contacts, not\nfull 3D structure',
      accuracy: 50,
    },
    {
      name: 'AlphaFold2\n(2020)',
      color: '#ff6d00',
      items: [
        'End-to-end differentiable',
        'Triangle geometry constraint',
        'Iterative recycling',
        'Invariant Point Attention',
        'Direct 3D prediction',
      ],
      limit: '',
      accuracy: 92,
    },
  ]

  return (
    <svg viewBox="0 0 820 420" style={{ width: '100%', maxWidth: 820, height: 'auto' }}>
      <text x={410} y={24} textAnchor="middle" fontSize={15} fontWeight={700}
        fill="#1a237e" fontFamily="Inter, sans-serif">
        Why AlphaFold2 is Different: Three Generations of Approaches
      </text>

      {approaches.map((ap, col) => {
        const x = 30 + col * 270
        const isAF2 = col === 2
        return (
          <g key={ap.name}>
            {/* Column background */}
            <rect x={x} y={40} width={240} height={360} rx={12}
              fill={isAF2 ? `${ap.color}08` : '#fafafa'}
              stroke={isAF2 ? ap.color : '#e0e0e0'}
              strokeWidth={isAF2 ? 2.5 : 1}
            />
            {isAF2 && (
              <rect x={x} y={40} width={240} height={360} rx={12}
                fill="none" stroke={ap.color} strokeWidth={2.5}>
                <animate attributeName="stroke-opacity" values="1;0.4;1" dur="2.5s" repeatCount="indefinite" />
              </rect>
            )}

            {/* Title */}
            {ap.name.split('\n').map((line, li) => (
              <text key={li} x={x + 120} y={70 + li * 16} textAnchor="middle"
                fontSize={li === 0 ? 14 : 11} fontWeight={li === 0 ? 700 : 400}
                fill={isAF2 ? ap.color : '#555'} fontFamily="Inter, sans-serif">{line}</text>
            ))}

            {/* Method items */}
            {ap.items.map((item, i) => (
              <g key={i}>
                <circle cx={x + 20} cy={110 + i * 26} r={4}
                  fill={ap.color} opacity={0.7} />
                <text x={x + 30} y={114 + i * 26} fontSize={11.5}
                  fill="#444" fontFamily="Inter, sans-serif">{item}</text>
              </g>
            ))}

            {/* Accuracy bar */}
            <text x={x + 120} y={270} textAnchor="middle" fontSize={11}
              fill="#999" fontFamily="Inter, sans-serif">Accuracy (GDT-TS)</text>
            <rect x={x + 30} y={278} width={180} height={14} rx={7}
              fill="#eee" />
            <rect x={x + 30} y={278} width={180 * ap.accuracy / 100} height={14} rx={7}
              fill={ap.color} opacity={0.75}>
              {isAF2 && (
                <animate attributeName="width" from="0" to={`${180 * ap.accuracy / 100}`}
                  dur="1.5s" fill="freeze" />
              )}
            </rect>
            <text x={x + 30 + 180 * ap.accuracy / 100 - 5} y={289}
              textAnchor="end" fontSize={10} fontWeight={700}
              fill="#fff" fontFamily="Inter, sans-serif">{ap.accuracy}</text>

            {/* Limitation / triumph */}
            {ap.limit ? (
              <g>
                <text x={x + 20} y={315} fontSize={10} fill="#c62828"
                  fontFamily="Inter, sans-serif" fontWeight={600}>Limitation:</text>
                {ap.limit.split('\n').map((line, li) => (
                  <text key={li} x={x + 20} y={330 + li * 14} fontSize={10.5}
                    fill="#999" fontStyle="italic" fontFamily="Inter, sans-serif">{line}</text>
                ))}
              </g>
            ) : (
              <g>
                <text x={x + 120} y={320} textAnchor="middle" fontSize={12}
                  fill="#2e7d32" fontWeight={700} fontFamily="Inter, sans-serif">
                  Reached experimental accuracy
                </text>
                <text x={x + 120} y={340} textAnchor="middle" fontSize={11}
                  fill="#2e7d32" fontFamily="Inter, sans-serif">
                  200M+ structures predicted
                </text>
              </g>
            )}
          </g>
        )
      })}

      {/* Evolution arrows between columns */}
      {[0, 1].map(i => {
        const x1 = 30 + i * 270 + 240
        const x2 = 30 + (i + 1) * 270
        return (
          <g key={`arrow-${i}`}>
            <line x1={x1 + 2} y1={200} x2={x2 - 2} y2={200}
              stroke="#bbb" strokeWidth={2} />
            <polygon points={`${x2 - 2},200 ${x2 - 10},195 ${x2 - 10},205`} fill="#bbb" />
            {/* Animated particle along arrow */}
            <circle r={3} fill="#1976d2" opacity={0.8}>
              <animateMotion dur="1.5s" begin={`${i * 0.5}s`} repeatCount="indefinite"
                path={`M${x1 + 2},200 L${x2 - 2},200`} />
            </circle>
          </g>
        )
      })}
    </svg>
  )
}

function DataFlowInsight() {
  // Show the key AF2 insight: evolution → co-evolution → geometry → structure
  return (
    <svg viewBox="0 0 760 180" style={{ width: '100%', maxWidth: 760, height: 'auto' }}>
      <defs>
        <linearGradient id="flowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#1565c0" />
          <stop offset="33%" stopColor="#7b1fa2" />
          <stop offset="66%" stopColor="#e65100" />
          <stop offset="100%" stopColor="#2e7d32" />
        </linearGradient>
      </defs>

      {/* Flow line */}
      <line x1={60} y1={90} x2={700} y2={90} stroke="url(#flowGrad)" strokeWidth={3} opacity={0.3} />

      {/* Animated particle along the flow */}
      <circle r={5} fill="url(#flowGrad)" opacity={0.9}>
        <animateMotion dur="4s" repeatCount="indefinite" path="M60,90 L700,90" />
      </circle>
      <circle r={12} fill="url(#flowGrad)" opacity={0.15}>
        <animateMotion dur="4s" repeatCount="indefinite" path="M60,90 L700,90" />
      </circle>

      {[
        { x: 60, label: 'Sequences\nacross species', icon: 'MSA', color: '#1565c0', sub: 'Billions of years\nof evolution' },
        { x: 250, label: 'Co-evolution\nsignal', icon: 'Pair', color: '#7b1fa2', sub: 'Correlated mutations\nreveal contacts' },
        { x: 450, label: 'Geometric\nconstraint', icon: '△', color: '#e65100', sub: 'Triangle inequality\nenforces consistency' },
        { x: 650, label: '3D\nstructure', icon: '🧬', color: '#2e7d32', sub: 'Atomic-level\naccuracy' },
      ].map((step, i) => (
        <g key={i}>
          {/* Node */}
          <circle cx={step.x} cy={90} r={22} fill={step.color} opacity={0.12} />
          <circle cx={step.x} cy={90} r={14} fill={step.color} opacity={0.85} />
          <text x={step.x} y={94} textAnchor="middle" fontSize={step.icon.length > 2 ? 10 : 11}
            fill="#fff" fontWeight={700} fontFamily="Inter, sans-serif">{step.icon}</text>

          {/* Label above */}
          {step.label.split('\n').map((line, li) => (
            <text key={li} x={step.x} y={48 + li * 14} textAnchor="middle"
              fontSize={12} fontWeight={600} fill={step.color}
              fontFamily="Inter, sans-serif">{line}</text>
          ))}

          {/* Sub-label below */}
          {step.sub.split('\n').map((line, li) => (
            <text key={li} x={step.x} y={125 + li * 14} textAnchor="middle"
              fontSize={10} fill="#999" fontFamily="Inter, sans-serif">{line}</text>
          ))}

          {/* Arrow to next */}
          {i < 3 && (
            <g>
              <line x1={step.x + 25} y1={90} x2={[250, 450, 650][i] - 25} y2={90}
                stroke={step.color} strokeWidth={2} opacity={0.4}
                strokeDasharray="4,4" />
            </g>
          )}
        </g>
      ))}
    </svg>
  )
}

function CASPScoreChart() {
  const data = [
    { year: 'CASP10\n2012', score: 40, color: '#90a4ae' },
    { year: 'CASP11\n2014', score: 45, color: '#90a4ae' },
    { year: 'CASP12\n2016', score: 50, color: '#90a4ae' },
    { year: 'CASP13\n2018', score: 60, color: '#1976d2' },
    { year: 'CASP14\n2020', score: 92, color: '#e65100' },
  ]
  const barW = 60
  const maxH = 160

  return (
    <svg viewBox="0 0 420 240" style={{ width: '100%', maxWidth: 420, height: 'auto' }}>
      <text x={210} y={18} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="#333" fontFamily="Inter, sans-serif">CASP Free-Modeling GDT-TS Scores</text>

      {/* Experimental threshold */}
      <line x1={40} y1={30 + maxH * (1 - 90 / 100)} x2={380} y2={30 + maxH * (1 - 90 / 100)}
        stroke="#2e7d32" strokeWidth={1} strokeDasharray="4,3" />
      <text x={385} y={30 + maxH * (1 - 90 / 100) + 4} fontSize={9} fill="#2e7d32"
        fontFamily="Inter, sans-serif">~experimental accuracy</text>

      {data.map((d, i) => {
        const h = maxH * d.score / 100
        const x = 55 + i * 72
        const isAF2 = i === data.length - 1
        return (
          <g key={i}>
            {/* Animated bar growth */}
            <rect x={x} y={30 + maxH} width={barW} height={0}
              fill={d.color} opacity={0.8} rx={4}>
              <animate attributeName="height" from="0" to={`${h}`} dur="0.8s"
                begin={`${i * 0.15}s`} fill="freeze" />
              <animate attributeName="y" from={`${30 + maxH}`} to={`${30 + maxH - h}`} dur="0.8s"
                begin={`${i * 0.15}s`} fill="freeze" />
            </rect>
            {/* Score label */}
            <text x={x + barW / 2} y={30 + maxH - h - 6} textAnchor="middle"
              fontSize={isAF2 ? 14 : 12} fontWeight={700} fill={d.color} fontFamily="Inter, sans-serif">
              {d.score}
            </text>
            {/* AF2 glow effect */}
            {isAF2 && (
              <rect x={x - 3} y={30 + maxH - h - 3} width={barW + 6} height={h + 6} rx={6}
                fill="none" stroke="#e65100" strokeWidth={2}>
                <animate attributeName="stroke-opacity" values="0.8;0.2;0.8" dur="2s" repeatCount="indefinite" />
              </rect>
            )}
            {d.year.split('\n').map((line, li) => (
              <text key={li} x={x + barW / 2} y={30 + maxH + 16 + li * 13}
                textAnchor="middle" fontSize={10} fill={isAF2 ? '#e65100' : '#666'}
                fontWeight={isAF2 ? 600 : 400}
                fontFamily="Inter, sans-serif">
                {line}
              </text>
            ))}
          </g>
        )
      })}

      {/* AF2 callout */}
      <text x={55 + 4 * 72 + barW / 2} y={225} textAnchor="middle" fontSize={10}
        fontWeight={700} fill="#e65100" fontFamily="Inter, sans-serif">AlphaFold2</text>
    </svg>
  )
}

// ── Sections ───────────────────────────────────────────

function makeSections(): Section[] {
  return [
    {
      id: 0,
      title: 'What are Proteins?',
      content: (
        <div>
          <p style={pStyle}>
            Proteins are the molecular machines of life. They are chains of <strong>amino acids</strong> —
            20 different types, each with unique chemical properties — linked together like beads on a string.
            A typical protein is 100–1000 amino acids long.
          </p>
          <AminoAcidChain />
          <p style={{ ...pStyle, marginTop: 16, fontSize: 13, color: '#666' }}>
            ↑ The first 13 residues of <strong>human hemoglobin</strong> β-chain: M-V-L-S-P-A-D-K-T-N-V-K-A...
          </p>
          <p style={pStyle}>
            This linear chain folds into a specific <strong>3D shape</strong> — and that shape determines
            everything about what the protein does. A misfolded protein can cause diseases like Alzheimer's,
            Parkinson's, and cystic fibrosis.
          </p>
          <div style={{ marginTop: 16 }}>
            <h4 style={h4Style}>Four levels of protein structure</h4>
            <FoldingLevels />
          </div>
        </div>
      ),
    },
    {
      id: 1,
      title: 'Why Is Structure Prediction So Hard?',
      content: (
        <div>
          <p style={pStyle}>
            Each amino acid has several rotatable bonds. For a protein with just 100 residues, the number
            of possible 3D arrangements is roughly <strong>10<sup>300</sup></strong> — more than the number
            of atoms in the observable universe (10<sup>80</sup>).
          </p>
          <LevinthalDiagram />
          <p style={pStyle}>
            Yet real proteins fold in milliseconds to seconds. This <strong>Levinthal's Paradox</strong> tells
            us that proteins don't randomly search — they follow an energy landscape "funnel" guided by physics.
            But simulating this from first principles requires enormous computing power.
          </p>
          <h4 style={h4Style}>Why not just use experiments?</h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, maxWidth: 700 }}>
            {[
              { method: 'X-ray Crystallography', time: 'Months–years', cost: '$50K–200K', note: 'Needs crystal (many proteins won\'t crystallize)' },
              { method: 'Cryo-EM', time: 'Weeks–months', cost: '$10K–100K', note: 'Limited resolution for small proteins' },
              { method: 'NMR', time: 'Months', cost: '$50K+', note: 'Only works for small proteins (<40 kDa)' },
            ].map(m => (
              <div key={m.method} style={{
                padding: 12, borderRadius: 8, background: '#f5f5f5', border: '1px solid #e0e0e0',
              }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#333', marginBottom: 4 }}>{m.method}</div>
                <div style={{ fontSize: 11, color: '#666' }}>Time: {m.time}</div>
                <div style={{ fontSize: 11, color: '#666' }}>Cost: {m.cost}</div>
                <div style={{ fontSize: 11, color: '#999', fontStyle: 'italic', marginTop: 4 }}>{m.note}</div>
              </div>
            ))}
          </div>
          <p style={{ ...pStyle, marginTop: 12 }}>
            As of 2020, only ~170,000 structures were experimentally determined — out of billions of known
            protein sequences. We desperately needed a computational solution.
          </p>
        </div>
      ),
    },
    {
      id: 2,
      title: 'Real Examples: Why Each Protein is a Puzzle',
      content: (
        <div>
          <p style={pStyle}>
            Every protein presents unique challenges. Here are four examples that illustrate why a general
            solution was considered one of biology's grand challenges:
          </p>
          <ProteinExamples />
          <p style={{ ...pStyle, marginTop: 16 }}>
            These challenges — multi-chain assembly, disulfide bonds, conformational flexibility, intrinsic
            disorder — mean that no single rule or template can predict all protein structures. The solution
            needed to learn from <strong>evolution itself</strong>.
          </p>
        </div>
      ),
    },
    {
      id: 3,
      title: 'The AlphaFold2 Breakthrough',
      content: (
        <div>
          <p style={pStyle}>
            In November 2020, AlphaFold2 achieved a median GDT-TS score of <strong>92.4</strong> at CASP14 —
            reaching experimental-level accuracy for the first time. The protein structure prediction problem,
            a 50-year grand challenge, was effectively solved.
          </p>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <CASPScoreChart />
            <div style={{ flex: 1, minWidth: 250 }}>
              <h4 style={h4Style}>Key ideas that made it work</h4>
              <ol style={{ margin: 0, padding: '0 0 0 18px', fontSize: 13, color: '#555', lineHeight: 1.8 }}>
                <li><strong>Evolution as a teacher</strong> — co-evolving residues in MSAs reveal which positions are in spatial contact</li>
                <li><strong>Triangle geometry</strong> — if A is near B, and B is near C, then A-C distance is constrained (triangle inequality)</li>
                <li><strong>End-to-end learning</strong> — directly predict 3D coordinates, not intermediate distance maps</li>
                <li><strong>Iterative refinement</strong> — recycle predictions 3× to self-correct errors</li>
                <li><strong>Invariant Point Attention</strong> — attention in 3D space that respects rotational symmetry</li>
              </ol>
            </div>
          </div>
          <TimelineBar />
        </div>
      ),
    },
    {
      id: 4,
      title: 'Why AlphaFold2 is Unique',
      content: (
        <div>
          <p style={pStyle}>
            AlphaFold2 didn't just improve on prior methods — it <strong>rethought the entire approach</strong>.
            Previous methods either simulated physics (too slow) or predicted contacts from statistics (incomplete).
            AlphaFold2 fused evolutionary information with geometric reasoning in an end-to-end differentiable system.
          </p>
          <WhyUniqueComparison />
          <h4 style={h4Style}>The Core Data Flow: Evolution → Geometry → Structure</h4>
          <p style={{ ...pStyle, fontSize: 13, color: '#666' }}>
            The key insight is a pipeline that transforms billions of years of evolutionary data into precise 3D coordinates:
          </p>
          <DataFlowInsight />
          <div style={{
            marginTop: 20, padding: '16px 20px', borderRadius: 10,
            background: 'linear-gradient(135deg, #e8eaf6, #f3e5f5)',
            border: '1px solid #c5cae9', maxWidth: 720,
          }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#1a237e', marginBottom: 6 }}>
              The Triangle Insight — AlphaFold2's Secret Weapon
            </div>
            <p style={{ margin: 0, fontSize: 13, color: '#555', lineHeight: 1.6 }}>
              If residue A is close to residue B, and B is close to C, then the distance between A and C
              is <strong>constrained by the triangle inequality</strong>. This simple geometric fact is
              encoded directly into the architecture through <em>triangle attention</em> and
              <em> triangle multiplicative updates</em> — letting the network reason about 3D geometry
              without ever explicitly computing distances. No prior method had this.
            </p>
          </div>
        </div>
      ),
    },
  ]
}

const pStyle: React.CSSProperties = {
  margin: '0 0 12px', fontSize: 14, color: '#333', lineHeight: 1.65,
  maxWidth: 750, fontFamily: 'Inter, system-ui, sans-serif',
}

const h4Style: React.CSSProperties = {
  margin: '16px 0 10px', fontSize: 14, fontWeight: 700, color: '#1a237e',
  fontFamily: 'Inter, system-ui, sans-serif',
}

// ── Main Component ─────────────────────────────────────

export function ProteinIntro({ onContinue }: { onContinue: () => void }) {
  const sections = makeSections()
  const [activeSection, setActiveSection] = useState(0)
  const section = sections[activeSection]
  const isLast = activeSection === sections.length - 1

  return (
    <div style={{
      width: '100%', height: '100%', background: '#fff',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      fontFamily: 'Inter, system-ui, sans-serif',
    }}>
      {/* Top bar */}
      <div style={{
        padding: '14px 32px 10px',
        borderBottom: '1px solid #eee',
        display: 'flex', alignItems: 'center', gap: 16,
        flexShrink: 0,
      }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#1a237e' }}>
          AlphaFold2 Visually
        </h1>
        <span style={{ fontSize: 14, color: '#78909c' }}>
          Understanding the protein folding revolution
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          {sections.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setActiveSection(i)}
              style={{
                border: 'none', borderRadius: 16,
                padding: '4px 14px', cursor: 'pointer', fontSize: 12,
                background: i === activeSection ? '#1a237e' : '#f0f0f0',
                color: i === activeSection ? '#fff' : '#666',
                fontWeight: i === activeSection ? 600 : 400,
                transition: 'all 0.2s',
              }}
            >
              {i + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Content area */}
      <div style={{ flex: 1, overflow: 'auto', padding: '28px 40px 40px' }}>
        <h2 style={{
          margin: '0 0 20px', fontSize: 26, fontWeight: 800,
          color: '#1a237e',
        }}>
          {section.title}
        </h2>
        {section.content}
      </div>

      {/* Bottom nav */}
      <div style={{
        padding: '12px 32px',
        borderTop: '1px solid #eee',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <button
          onClick={() => setActiveSection(s => Math.max(0, s - 1))}
          disabled={activeSection === 0}
          style={{
            ...navBtnStyle,
            opacity: activeSection === 0 ? 0.4 : 1,
          }}
        >
          ← Previous
        </button>

        <div style={{ display: 'flex', gap: 6 }}>
          {sections.map((_, i) => (
            <div key={i} style={{
              width: 8, height: 8, borderRadius: '50%',
              background: i === activeSection ? '#1a237e' : i < activeSection ? '#90a4ae' : '#ddd',
              transition: 'background 0.2s',
            }} />
          ))}
        </div>

        {isLast ? (
          <button onClick={onContinue} style={{
            ...navBtnStyle,
            background: '#1a237e', color: '#fff',
            fontWeight: 700, fontSize: 14, padding: '10px 24px',
          }}>
            Explore the Architecture →
          </button>
        ) : (
          <button
            onClick={() => setActiveSection(s => s + 1)}
            style={navBtnStyle}
          >
            Next →
          </button>
        )}
      </div>
    </div>
  )
}

const navBtnStyle: React.CSSProperties = {
  border: '1px solid #ccc', borderRadius: 8, background: '#fff',
  padding: '8px 18px', cursor: 'pointer', fontSize: 13, color: '#555',
  fontFamily: 'Inter, system-ui, sans-serif',
}
