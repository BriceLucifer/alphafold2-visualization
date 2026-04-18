import { useState, useRef, useEffect, useCallback } from 'react'

// ── Algorithm stage descriptions ───────────────────────

interface StageInfo {
  id: number
  title: string
  subtitle: string
  description: string
  details: string[]
  highlightRect?: { x: number; y: number; w: number; h: number }
}

const STAGES: StageInfo[] = [
  {
    id: 0,
    title: 'Input Sequence',
    subtitle: 'The starting point',
    description:
      'AlphaFold2 takes a single amino acid sequence as input — a string of letters like MVLSPADKTNVKA... representing the protein to fold. For example, human hemoglobin β-chain is 147 residues; the SARS-CoV-2 spike protein is 1,273.',
    details: [
      'Each letter represents one of 20 amino acids (e.g. M = methionine, V = valine)',
      'The sequence alone determines the 3D structure — this is Anfinsen\'s dogma (Nobel Prize, 1972)',
      'Example: insulin is just 51 residues but forms a complex 2-chain structure with 3 disulfide bonds',
      'The challenge: a 100-residue protein has ~10³⁰⁰ possible conformations to search through',
    ],
    highlightRect: { x: 5, y: 90, w: 120, h: 60 },
  },
  {
    id: 1,
    title: 'Genetic Database Search → MSA',
    subtitle: 'Finding evolutionary relatives',
    description:
      'Search genetic databases (UniRef90, BFD, MGnify) with JackHMMER/HHblits to find sequences from organisms across the tree of life that evolved from the same ancestor protein.',
    details: [
      'Example: searching human hemoglobin finds hemoglobin from fish, birds, reptiles — all share the same fold',
      'Co-evolution signal: if position 23 (buried) and position 87 (buried) are always in contact, when one mutates the other compensates',
      'A concrete case: in trypsin, positions 102 (Asp) and 195 (Ser) always co-evolve — they form the catalytic triad',
      'Deep MSAs (>1000 sequences) dramatically improve accuracy; orphan proteins with few homologs are harder',
      'This is the key insight: 3.8 billion years of evolution encode structural constraints',
    ],
    highlightRect: { x: 145, y: 25, w: 265, h: 200 },
  },
  {
    id: 2,
    title: 'Template Search',
    subtitle: 'Known structural homologs',
    description:
      'Search the PDB for proteins with similar sequences that already have experimentally determined 3D structures. For well-studied protein families, close templates exist; for novel folds, this step provides weaker signal.',
    details: [
      'Example: to predict a new kinase, templates from >500 known kinase structures provide excellent starting geometry',
      'Even 20–30% sequence identity templates can provide useful fold-level information',
      'Template distances directly initialize the pair representation: "residues i and j are ~8Å apart in the template"',
      'AlphaFold2 achieved its CASP14 breakthrough partly on "free modeling" targets with NO templates',
      'For the CASP14 target T1049 (ORF8 of SARS-CoV-2), AlphaFold2 predicted a novel fold with no known homologs',
    ],
    highlightRect: { x: 145, y: 310, w: 210, h: 130 },
  },
  {
    id: 3,
    title: 'Initial Representations',
    subtitle: 'Encoding the data as tensors',
    description:
      'The MSA, paired sequences, and templates are combined into two key tensors. Think of the MSA representation as a "stack of annotated sequences" and the pair representation as a "relationship matrix" between all residue pairs.',
    details: [
      'MSA representation: shape (s, r, c) — e.g. (512 sequences, 200 residues, 256 channels)',
      'Pair representation: shape (r, r, c) — e.g. (200×200 residue pairs, 128 channels)',
      'For a 200-residue protein: MSA repr = 26M values, Pair repr = 5.1M values',
      'MSA repr is initialized from amino acid embeddings + positional encoding',
      'Pair repr is initialized from: relative position encoding + template distances + outer sum of MSA features',
      'These are the two "working memories" that the Evoformer iteratively refines',
    ],
    highlightRect: { x: 430, y: 100, w: 190, h: 340 },
  },
  {
    id: 4,
    title: 'Evoformer',
    subtitle: 'The core — 48 blocks of iterative refinement',
    description:
      'The Evoformer is the heart of AlphaFold2. Each of its 48 blocks applies attention and update operations that let the MSA and pair representations communicate and refine each other. This is where co-evolution signals become geometric constraints.',
    details: [
      '① Row-wise MSA attention: within sequence "MVLSPA...", each residue attends to every other',
      '② Column-wise MSA attention: position 42 across human, mouse, fish sequences → "this position is always hydrophobic"',
      '③ Outer product mean: "positions 23 and 87 always co-evolve" → "they are probably in contact in 3D"',
      '④ Triangle multiplicative update: "if res A is near B, and B is near C, then constrain A-C distance" (triangle inequality)',
      '⑤ Triangle self-attention: learned attention patterns over triangle edges — more expressive than multiplicative rules',
      'Example: after 48 blocks, the pair repr for a β-sheet correctly shows alternating contact patterns between strands',
      '→ Click the Evoformer box in the diagram to explore each operation with interactive visualizations',
    ],
    highlightRect: { x: 688, y: 152, w: 144, h: 152 },
  },
  {
    id: 5,
    title: 'Evoformer Output',
    subtitle: 'Refined representations',
    description:
      'After 48 Evoformer blocks, the model has built a rich internal model of the protein. The pair representation now effectively encodes a predicted distance/contact map, and the single representation encodes per-residue structural features.',
    details: [
      'Single repr (r, c): extracted from the first row of the MSA — now a "structure-aware" per-residue embedding',
      'Pair repr (r, r, c): at this point, residue pairs that are in 3D contact have strong, distinctive pair features',
      'Example: for an α-helix, the pair repr shows strong signals at (i, i+4) offsets — the characteristic hydrogen bond pattern',
      'For a β-sheet, it shows the inter-strand contact pattern — alternating (i, j), (i+2, j-2) pairs',
      'These two tensors now contain enough information to reconstruct full 3D coordinates',
    ],
    highlightRect: { x: 875, y: 80, w: 120, h: 340 },
  },
  {
    id: 6,
    title: 'Structure Module',
    subtitle: '8 blocks of Invariant Point Attention (IPA)',
    description:
      'The Structure Module converts abstract features into actual 3D coordinates. Its key innovation is Invariant Point Attention — attention that uses both learned features and geometric distances in 3D space, while being invariant to global rotation/translation.',
    details: [
      'Each residue gets a rigid body frame: a rotation matrix R (3×3) + translation t (3D) — like a local coordinate system',
      'IPA: each residue projects "query points" into 3D space using its frame, computes distances to other residues\' "key points"',
      'Example: the backbone of residue 42 is placed at position (12.3, 5.7, -3.1) Å with orientation facing the helix axis',
      '"Invariant" means: if you rotate the whole protein 90°, IPA produces the same output — it only cares about relative geometry',
      'After 8 refinement blocks: predicts N, Cα, C backbone atoms + side-chain torsion angles (χ₁–χ₄)',
      'Trained with FAPE loss: compares each residue\'s local frame against the true structure — captures both position and orientation',
    ],
    highlightRect: { x: 1060, y: 152, w: 145, h: 152 },
  },
  {
    id: 7,
    title: '3D Structure Output',
    subtitle: 'Predicted protein structure with confidence',
    description:
      'The final output is a full atomic-resolution 3D structure with per-residue confidence scores. At CASP14, AlphaFold2 achieved GDT-TS ~92 — reaching experimental accuracy for the first time in history.',
    details: [
      'pLDDT (predicted LDDT): 0–100 confidence per residue — trained to predict its own accuracy',
      'Blue (>90): "I am very confident" — e.g. residues in the hydrophobic core of a well-folded domain',
      'Orange (<50): "I am uncertain" — e.g. flexible loops, disordered termini, or intrinsically disordered regions',
      'Example: for T1049 (SARS-CoV-2 ORF8), AlphaFold2 correctly predicted the immunoglobulin-like fold with pLDDT >80',
      'PAE (Predicted Aligned Error): shows which domains are confidently positioned relative to each other',
      'The AlphaFold Protein Structure Database now contains >200 million predicted structures',
    ],
    highlightRect: { x: 1250, y: 118, w: 155, h: 110 },
  },
  {
    id: 8,
    title: 'Recycling',
    subtitle: 'Iterative self-refinement — 3 rounds',
    description:
      'The entire Evoformer + Structure Module is run 3 times. Each round, the predicted structure feeds back to update the pair representation with actual 3D distances, allowing the model to self-correct — like a sculptor progressively refining a statue.',
    details: [
      'Round 1: predicts from MSA/templates alone — often gets the overall fold right but details wrong',
      'Round 2: "I predicted residues 45 and 112 are 6Å apart — let me feed that back and refine"',
      'Round 3: final refinement — typically improves GDT by 2–5 points over round 1',
      'Example: for multi-domain proteins, round 1 may get individual domains right but wrong relative orientation; recycling fixes this',
      'Gradients only flow through the final round (stop-gradient on rounds 1–2) — prevents training instability',
      'This mirrors iterative refinement in experimental crystallography: solve → refine → solve again',
    ],
    highlightRect: { x: 480, y: 450, w: 720, h: 55 },
  },
]

// ── Flow particle ──────────────────────────────────────

function FlowParticle({
  path, duration, delay, color = '#1976d2', size = 5, id,
}: {
  path: string; duration: number; delay: number; color?: string; size?: number; id: string
}) {
  return (
    <g>
      <defs>
        <path id={id} d={path} fill="none" />
        <radialGradient id={`glow-${id}`}>
          <stop offset="0%" stopColor={color} stopOpacity={0.9} />
          <stop offset="60%" stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </radialGradient>
      </defs>
      <circle r={size * 2.5} fill={`url(#glow-${id})`}>
        <animateMotion dur={`${duration}s`} begin={`${delay}s`} repeatCount="indefinite" fill="freeze">
          <mpath href={`#${id}`} />
        </animateMotion>
      </circle>
      <circle r={size} fill={color} opacity={0.95}>
        <animateMotion dur={`${duration}s`} begin={`${delay}s`} repeatCount="indefinite" fill="freeze">
          <mpath href={`#${id}`} />
        </animateMotion>
      </circle>
    </g>
  )
}

// ── Small SVG building blocks ──────────────────────────

function SequenceIcons({ x, y, scale = 1 }: { x: number; y: number; scale?: number }) {
  const items: { type: 'dot' | 'up'; color: string; dx: number }[] = [
    { type: 'dot', color: '#c62828', dx: 0 },
    { type: 'up', color: '#1565c0', dx: 16 },
    { type: 'up', color: '#e65100', dx: 32 },
    { type: 'dot', color: '#c62828', dx: 48 },
    { type: 'up', color: '#6a1b9a', dx: 64 },
    { type: 'up', color: '#f9a825', dx: 80 },
  ]
  return (
    <g transform={`translate(${x}, ${y}) scale(${scale})`}>
      {items.map((it, i) =>
        it.type === 'dot' ? (
          <circle key={i} cx={it.dx + 5} cy={0} r={4.5} fill={it.color} />
        ) : (
          <polygon key={i} points={`${it.dx},7 ${it.dx + 5},-7 ${it.dx + 10},7`} fill={it.color} />
        )
      )}
    </g>
  )
}

function DatabaseCylinder({ x, y, label1, label2, label3 }: {
  x: number; y: number; label1: string; label2: string; label3: string
}) {
  return (
    <g transform={`translate(${x}, ${y})`}>
      <ellipse cx={30} cy={6} rx={28} ry={8} fill="#e8e8e8" stroke="#aaa" strokeWidth={1} />
      <rect x={2} y={6} width={56} height={28} fill="#e8e8e8" stroke="none" />
      <line x1={2} y1={6} x2={2} y2={34} stroke="#aaa" strokeWidth={1} />
      <line x1={58} y1={6} x2={58} y2={34} stroke="#aaa" strokeWidth={1} />
      <ellipse cx={30} cy={34} rx={28} ry={8} fill="#ddd" stroke="#aaa" strokeWidth={1} />
      <text x={30} y={60} textAnchor="middle" fontSize={10.5} fill="#333" fontFamily="Inter, sans-serif">{label1}</text>
      <text x={30} y={72} textAnchor="middle" fontSize={10.5} fill="#333" fontFamily="Inter, sans-serif">{label2}</text>
      <text x={30} y={84} textAnchor="middle" fontSize={10.5} fill="#333" fontFamily="Inter, sans-serif">{label3}</text>
    </g>
  )
}

function ArrowLine({ points, color = '#1a237e', strokeWidth = 2, dashed = false, headSize = 7 }: {
  points: [number, number][]; color?: string; strokeWidth?: number; dashed?: boolean; headSize?: number
}) {
  if (points.length < 2) return null
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ')
  const last = points[points.length - 1]
  const prev = points[points.length - 2]
  const angle = Math.atan2(last[1] - prev[1], last[0] - prev[0])
  return (
    <g>
      <path d={d} fill="none" stroke={color} strokeWidth={strokeWidth}
        strokeDasharray={dashed ? '6,4' : undefined} />
      <polygon points={`${last[0]},${last[1]} ${last[0] - headSize * Math.cos(angle - 0.35)},${last[1] - headSize * Math.sin(angle - 0.35)} ${last[0] - headSize * Math.cos(angle + 0.35)},${last[1] - headSize * Math.sin(angle + 0.35)}`} fill={color} />
    </g>
  )
}

function PlusSign({ x, y }: { x: number; y: number }) {
  return (
    <g transform={`translate(${x}, ${y})`}>
      <circle cx={0} cy={0} r={13} fill="#fff" stroke="#1565c0" strokeWidth={1.5} />
      <line x1={-6} y1={0} x2={6} y2={0} stroke="#1565c0" strokeWidth={2.5} />
      <line x1={0} y1={-6} x2={0} y2={6} stroke="#1565c0" strokeWidth={2.5} />
    </g>
  )
}

function RoundedPill({ x, y, text }: { x: number; y: number; text: string }) {
  const w = text.length * 8 + 20
  return (
    <g transform={`translate(${x - w / 2}, ${y - 13})`}>
      <rect x={0} y={0} width={w} height={26} rx={13} fill="#fff" stroke="#b0bec5" strokeWidth={1.2} />
      <text x={w / 2} y={17} textAnchor="middle" fontSize={12} fill="#444" fontFamily="Inter, sans-serif">{text}</text>
    </g>
  )
}

function SeqIconRow({ x, y, scale = 0.55 }: { x: number; y: number; scale?: number }) {
  const items: { type: 'dot' | 'up'; color: string }[] = [
    { type: 'dot', color: '#c62828' }, { type: 'up', color: '#1565c0' },
    { type: 'up', color: '#e65100' }, { type: 'dot', color: '#c62828' },
    { type: 'up', color: '#6a1b9a' }, { type: 'up', color: '#f9a825' },
  ]
  return (
    <g transform={`translate(${x}, ${y}) scale(${scale})`}>
      {items.map((it, i) =>
        it.type === 'dot' ? (
          <circle key={i} cx={i * 15 + 5} cy={0} r={4} fill={it.color} />
        ) : (
          <polygon key={i} points={`${i * 15},6 ${i * 15 + 5},-6 ${i * 15 + 10},6`} fill={it.color} />
        )
      )}
    </g>
  )
}

function MSABox({ x, y, w, h, rows = 4, cols = 7 }: {
  x: number; y: number; w: number; h: number; rows?: number; cols?: number
}) {
  const colors = ['#c62828', '#1565c0', '#e65100', '#6a1b9a', '#f9a825', '#388e3c']
  const cellW = (w - 16) / cols
  const cellH = (h - 16) / rows
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect x={0} y={0} width={w} height={h} rx={3} fill="#fff" stroke="#1976d2" strokeWidth={1.5} />
      {Array.from({ length: rows }).map((_, r) =>
        Array.from({ length: cols }).map((_, c) => {
          const ci = (r * 3 + c * 7 + r * c) % colors.length
          const isSymbol = (r + c) % 3 !== 0
          return isSymbol ? (
            <polygon key={`${r}-${c}`}
              points={`${c * cellW + 8},${r * cellH + cellH + 4} ${c * cellW + cellW / 2 + 8},${r * cellH + 6} ${c * cellW + cellW + 4},${r * cellH + cellH + 4}`}
              fill={colors[ci]} opacity={0.8} />
          ) : (
            <circle key={`${r}-${c}`}
              cx={c * cellW + cellW / 2 + 6} cy={r * cellH + cellH / 2 + 6}
              r={Math.min(cellW, cellH) / 3} fill={colors[ci]} opacity={0.8} />
          )
        })
      )}
    </g>
  )
}

function RepresentationBlock({ x, y, w, h, color, opacity: baseOp = 0.6 }: {
  x: number; y: number; w: number; h: number; color: string; opacity?: number
}) {
  const cells = 5; const cw = w / cells; const ch = h / cells
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect x={0} y={0} width={w} height={h} rx={4} fill="#f5f8fc" stroke="#1976d2" strokeWidth={1.5} />
      {Array.from({ length: cells }).map((_, r) =>
        Array.from({ length: cells }).map((_, c) => (
          <rect key={`${r}-${c}`}
            x={c * cw + 1.5} y={r * ch + 1.5} width={cw - 3} height={ch - 3}
            fill={color} opacity={Math.max(0.08, baseOp - Math.abs(r - c) * 0.12)} rx={2} />
        ))
      )}
    </g>
  )
}

function MSAReprBlock({ x, y, w, h }: { x: number; y: number; w: number; h: number }) {
  const rows = 4; const cols = 5
  const cw = (w - 10) / cols; const ch = (h - 10) / rows
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect x={0} y={0} width={w} height={h} rx={4} fill="#fff8f0" stroke="#1976d2" strokeWidth={1.5} />
      {Array.from({ length: rows }).map((_, r) =>
        Array.from({ length: cols }).map((_, c) => (
          <rect key={`${r}-${c}`}
            x={c * cw + 5} y={r * ch + 5} width={cw - 2} height={ch - 2}
            fill="#e8a87c" opacity={Math.max(0.15, 0.9 - (r * 0.15 + c * 0.08))} rx={1} />
        ))
      )}
    </g>
  )
}

function BigModule({ x, y, w, h, line1, line2 }: {
  x: number; y: number; w: number; h: number; line1: string; line2: string
}) {
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect x={0} y={0} width={w} height={h} rx={10} fill="rgba(200,220,240,0.25)" stroke="#90a4ae" strokeWidth={1.5} />
      <text x={w / 2} y={h / 2 - 4} textAnchor="middle" fontSize={15} fill="#333" fontWeight="600" fontFamily="Inter, sans-serif">{line1}</text>
      <text x={w / 2} y={h / 2 + 14} textAnchor="middle" fontSize={12} fill="#666" fontFamily="Inter, sans-serif">{line2}</text>
    </g>
  )
}

function Protein3D({ x, y }: { x: number; y: number }) {
  return (
    <g transform={`translate(${x}, ${y})`}>
      <path d="M5,10 C15,-5 30,25 45,10" fill="none" stroke="#1976d2" strokeWidth={7} strokeLinecap="round" opacity={0.85} />
      <path d="M10,30 C25,15 40,45 55,30" fill="none" stroke="#1e88e5" strokeWidth={6} strokeLinecap="round" opacity={0.75} />
      <path d="M0,50 C15,35 35,60 50,48" fill="none" stroke="#42a5f5" strokeWidth={5.5} strokeLinecap="round" opacity={0.7} />
      <path d="M8,65 C20,55 40,75 55,62" fill="none" stroke="#64b5f6" strokeWidth={5} strokeLinecap="round" opacity={0.6} />
      <path d="M45,10 Q60,20 55,30" fill="none" stroke="#a5d6a7" strokeWidth={2.5} strokeLinecap="round" />
      <path d="M55,30 Q65,40 50,48" fill="none" stroke="#c8e6c9" strokeWidth={2.5} strokeLinecap="round" />
      <path d="M50,48 Q62,56 55,62" fill="none" stroke="#e8f5e9" strokeWidth={2} strokeLinecap="round" />
      <line x1={75} y1={0} x2={75} y2={75} stroke="#e65100" strokeWidth={1.5} opacity={0.5} />
      <text x={85} y={8} fontSize={10} fill="#1565c0" fontWeight="600" fontFamily="Inter, sans-serif">High</text>
      <text x={85} y={19} fontSize={9} fill="#1565c0" fontFamily="Inter, sans-serif">confidence</text>
      <text x={85} y={62} fontSize={10} fill="#e65100" fontWeight="600" fontFamily="Inter, sans-serif">Low</text>
      <text x={85} y={73} fontSize={9} fill="#e65100" fontFamily="Inter, sans-serif">confidence</text>
      <text x={30} y={95} textAnchor="middle" fontSize={12} fill="#333" fontFamily="Inter, sans-serif">3D structure</text>
    </g>
  )
}

function TemplateBox({ x, y, w, h }: { x: number; y: number; w: number; h: number }) {
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect x={0} y={0} width={w} height={h} rx={3} fill="#1a1a1a" stroke="#555" strokeWidth={1} />
      {[12, 28, 44].map(px =>
        [12, 28, 44].map(py => (
          <g key={`${px}-${py}`} opacity={0.5}>
            <line x1={px - 3} y1={py - 3} x2={px + 3} y2={py + 3} stroke="#c62828" strokeWidth={1.2} />
            <line x1={px + 3} y1={py - 3} x2={px - 3} y2={py + 3} stroke="#c62828" strokeWidth={1.2} />
          </g>
        ))
      )}
    </g>
  )
}

function TensorShapeLabel({ x, y, shape, example, color }: {
  x: number; y: number; shape: string; example: string; color: string
}) {
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect x={-40} y={-10} width={80} height={20} rx={10}
        fill={`${color}15`} stroke={color} strokeWidth={0.8} />
      <text x={0} y={3} textAnchor="middle" fontSize={9} fontWeight={600}
        fill={color} fontFamily="JetBrains Mono, monospace">{shape}</text>
      <text x={0} y={20} textAnchor="middle" fontSize={8}
        fill="#999" fontFamily="JetBrains Mono, monospace">{example}</text>
    </g>
  )
}

function DataTransformLabel({ x, y, text }: { x: number; y: number; text: string }) {
  return (
    <g transform={`translate(${x}, ${y})`}>
      <text x={0} y={0} textAnchor="middle" fontSize={8.5}
        fill="#78909c" fontStyle="italic" fontFamily="Inter, sans-serif">{text}</text>
    </g>
  )
}

function SingleRepr({ x, y }: { x: number; y: number }) {
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect x={0} y={0} width={110} height={22} rx={3} fill="#fff" stroke="#9c27b0" strokeWidth={1.5} strokeDasharray="4,2" />
      <text x={55} y={15} textAnchor="middle" fontSize={10.5} fill="#444" fontWeight="500" fontFamily="Inter, sans-serif">
        Single repr. (r,c)
      </text>
    </g>
  )
}

// ── Highlight overlay for active stage ─────────────────

function StageHighlight({ rect, active }: { rect: { x: number; y: number; w: number; h: number }; active: boolean }) {
  if (!active) return null
  return (
    <g>
      <rect
        x={rect.x} y={rect.y} width={rect.w} height={rect.h}
        rx={8} fill="rgba(25, 118, 210, 0.07)" stroke="#1976d2" strokeWidth={2.5}
        strokeDasharray="8,4"
      >
        <animate attributeName="stroke-opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite" />
      </rect>
    </g>
  )
}

// ── Explanation panel (HTML overlay) ───────────────────

function ExplanationPanel({ stage, onPrev, onNext, onStageClick, playing, onTogglePlay }: {
  stage: StageInfo
  onPrev: () => void
  onNext: () => void
  onStageClick: (id: number) => void
  playing: boolean
  onTogglePlay: () => void
}) {
  return (
    <div style={{
      background: '#f8fafc',
      borderTop: '2px solid #e0e7ee',
      padding: '20px 32px',
      display: 'flex',
      gap: 28,
      alignItems: 'flex-start',
      minHeight: 160,
      fontFamily: 'Inter, system-ui, sans-serif',
    }}>
      {/* Stage nav dots */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, paddingTop: 4, minWidth: 60 }}>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'center', maxWidth: 50 }}>
          {STAGES.map(s => (
            <div
              key={s.id}
              onClick={() => onStageClick(s.id)}
              style={{
                width: 10, height: 10, borderRadius: '50%', cursor: 'pointer',
                background: s.id === stage.id ? '#1976d2' : '#ccc',
                transition: 'background 0.2s',
              }}
            />
          ))}
        </div>
        <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
          <button onClick={onPrev} style={navBtnStyle}>◀</button>
          <button onClick={onTogglePlay} style={{ ...navBtnStyle, width: 36, fontSize: 13 }}>
            {playing ? '⏸' : '▶'}
          </button>
          <button onClick={onNext} style={navBtnStyle}>▶</button>
        </div>
        <span style={{ fontSize: 11, color: '#999' }}>{stage.id + 1}/{STAGES.length}</span>
      </div>

      {/* Main content */}
      <div style={{ flex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 6 }}>
          <h3 style={{ margin: 0, fontSize: 17, color: '#1a237e', fontWeight: 700 }}>{stage.title}</h3>
          <span style={{ fontSize: 13, color: '#78909c', fontStyle: 'italic' }}>{stage.subtitle}</span>
        </div>
        <p style={{ margin: '4px 0 10px', fontSize: 14, color: '#333', lineHeight: 1.55, maxWidth: 700 }}>
          {stage.description}
        </p>
        <ul style={{ margin: 0, padding: '0 0 0 18px', fontSize: 13, color: '#555', lineHeight: 1.7, columns: stage.details.length > 4 ? 2 : 1, columnGap: 32 }}>
          {stage.details.map((d, i) => (
            <li key={i} style={{ breakInside: 'avoid' }}>{d}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}

const navBtnStyle: React.CSSProperties = {
  width: 28, height: 28, border: '1px solid #ccc', borderRadius: 4,
  background: '#fff', cursor: 'pointer', fontSize: 11, display: 'flex',
  alignItems: 'center', justifyContent: 'center', color: '#555',
}

// ── Main Component ─────────────────────────────────────

export function ArchitectureOverview({ onDrillIn, onBack }: { onDrillIn?: (target: string) => void; onBack?: () => void }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [activeStage, setActiveStage] = useState(0)
  const [playing, setPlaying] = useState(true)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stage = STAGES[activeStage]

  const goNext = useCallback(() => {
    setActiveStage(s => (s + 1) % STAGES.length)
  }, [])

  const goPrev = useCallback(() => {
    setActiveStage(s => (s - 1 + STAGES.length) % STAGES.length)
  }, [])

  // Auto-advance
  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(goNext, 5000)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [playing, goNext])

  const handleStageClick = (id: number) => {
    setActiveStage(id)
    setPlaying(false)
  }

  const Y_MSA = 60
  const Y_PAIR = 240
  const Y_STRUCT = 355
  const Y_RECYCLE = 460
  const CYCLE = 12

  const flowPaths = [
    { id: 'f1a', path: `M105,${Y_MSA + 52} L155,${Y_MSA + 20}`, duration: 1.2, delay: 0, color: '#1976d2', size: 4 },
    { id: 'f1b', path: `M105,${Y_MSA + 58} L168,${Y_PAIR + 10}`, duration: 1.5, delay: 0.1, color: '#1976d2', size: 4 },
    { id: 'f1c', path: `M105,${Y_MSA + 62} L155,${Y_STRUCT + 10}`, duration: 1.8, delay: 0.2, color: '#1976d2', size: 4 },
    { id: 'f2a', path: `M215,${Y_MSA + 20} L265,${Y_MSA + 20}`, duration: 1, delay: 1.5, color: '#1976d2', size: 4 },
    { id: 'f2b', path: `M245,${Y_PAIR + 10} L275,${Y_PAIR + 10}`, duration: 1, delay: 1.7, color: '#1976d2', size: 4 },
    { id: 'f2c', path: `M215,${Y_STRUCT + 20} L275,${Y_STRUCT + 20}`, duration: 1, delay: 1.9, color: '#1976d2', size: 4 },
    { id: 'f3a', path: `M400,${Y_MSA + 40} L435,${Y_MSA + 90}`, duration: 1, delay: 3, color: '#e65100', size: 4 },
    { id: 'f3b', path: `M380,${Y_PAIR - 5} L435,${Y_MSA + 85}`, duration: 1, delay: 3.2, color: '#e65100', size: 4 },
    { id: 'f3c', path: `M458,${Y_MSA + 90} L510,${Y_MSA + 90}`, duration: 0.8, delay: 4, color: '#e65100', size: 4.5 },
    { id: 'f3d', path: `M380,${Y_PAIR + 30} L435,${Y_PAIR + 80}`, duration: 1, delay: 3.1, color: '#42a5f5', size: 4 },
    { id: 'f3e', path: `M340,${Y_STRUCT + 20} L435,${Y_PAIR + 82}`, duration: 1.2, delay: 3.3, color: '#42a5f5', size: 4 },
    { id: 'f3f', path: `M458,${Y_PAIR + 82} L510,${Y_PAIR + 82}`, duration: 0.8, delay: 4.2, color: '#42a5f5', size: 4.5 },
    { id: 'f4a', path: `M615,${Y_MSA + 95} L690,${Y_MSA + 140}`, duration: 1.2, delay: 5, color: '#e65100', size: 5 },
    { id: 'f4b', path: `M605,${Y_PAIR + 95} L690,${Y_PAIR + 50}`, duration: 1.2, delay: 5.2, color: '#42a5f5', size: 5 },
    { id: 'f5a', path: `M825,${Y_MSA + 140} L885,${Y_MSA + 95}`, duration: 1, delay: 7, color: '#e65100', size: 5 },
    { id: 'f5b', path: `M825,${Y_PAIR + 50} L885,${Y_PAIR + 85}`, duration: 1, delay: 7.2, color: '#42a5f5', size: 5 },
    { id: 'f6a', path: `M990,${Y_MSA + 95} L1070,${Y_MSA + 140}`, duration: 1, delay: 8.5, color: '#e65100', size: 5 },
    { id: 'f6b', path: `M980,${Y_PAIR + 95} L1070,${Y_PAIR + 50}`, duration: 1, delay: 8.7, color: '#42a5f5', size: 5 },
    { id: 'f7', path: `M1200,${Y_MSA + 170} L1255,${Y_MSA + 170}`, duration: 0.8, delay: 10, color: '#1976d2', size: 5 },
    { id: 'f8a', path: `M760,${Y_MSA + 240} L760,${Y_RECYCLE - 5}`, duration: 1.2, delay: 10.5, color: '#78909c', size: 4 },
    { id: 'f8b', path: `M1135,${Y_MSA + 240} L1135,${Y_RECYCLE - 5}`, duration: 1.2, delay: 10.7, color: '#78909c', size: 4 },
    { id: 'f8c', path: `M540,${Y_RECYCLE + 21} L490,${Y_RECYCLE + 21} L490,${Y_MSA + 95} L510,${Y_MSA + 93}`, duration: 2, delay: 11, color: '#78909c', size: 4 },
    { id: 'f8d', path: `M540,${Y_RECYCLE + 21} L490,${Y_RECYCLE + 21} L490,${Y_PAIR + 90} L510,${Y_PAIR + 88}`, duration: 2, delay: 11.2, color: '#78909c', size: 4 },
  ]

  return (
    <div style={{
      width: '100%',
      height: '100%',
      background: '#fff',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Title bar */}
      <div style={{
        padding: '12px 32px 8px',
        borderBottom: '1px solid #eee',
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        fontFamily: 'Inter, system-ui, sans-serif',
        flexShrink: 0,
      }}>
        {onBack && (
          <button onClick={onBack} style={{
            border: '1px solid #ccc', borderRadius: 6, background: '#fff',
            padding: '5px 12px', cursor: 'pointer', fontSize: 12, color: '#555',
          }}>
            ← Intro
          </button>
        )}
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#1a237e' }}>
          AlphaFold2 Visually
        </h1>
        <span style={{ fontSize: 14, color: '#78909c' }}>
          Interactive Architecture Overview — click any stage to learn how it works
        </span>
      </div>

      {/* SVG diagram area */}
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'auto',
        minHeight: 0,
      }}>
        <svg
          ref={svgRef}
          viewBox="0 0 1420 530"
          style={{ width: '95%', maxWidth: 1420, height: 'auto', padding: '10px 20px' }}
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* ── Clickable hit areas (invisible rects for click targets) ── */}
          {STAGES.map(s => s.highlightRect && (
            <rect
              key={`hit-${s.id}`}
              x={s.highlightRect.x} y={s.highlightRect.y}
              width={s.highlightRect.w} height={s.highlightRect.h}
              fill="transparent" cursor="pointer"
              onClick={() => handleStageClick(s.id)}
            />
          ))}

          {/* ── Stage highlight ── */}
          {stage.highlightRect && <StageHighlight rect={stage.highlightRect} active={true} />}

          {/* ── Static diagram elements ── */}

          {/* Input sequence */}
          <SequenceIcons x={20} y={Y_MSA + 55} scale={1} />
          <text x={60} y={Y_MSA + 85} textAnchor="middle" fontSize={12} fill="#e65100" fontFamily="Inter, sans-serif">Input sequence</text>

          <ArrowLine points={[[105, Y_MSA + 52], [155, Y_MSA + 20]]} />
          <ArrowLine points={[[105, Y_MSA + 58], [168, Y_PAIR + 10]]} />
          <ArrowLine points={[[105, Y_MSA + 62], [155, Y_STRUCT + 10]]} />

          {/* Genetic DB + MSA */}
          <DatabaseCylinder x={155} y={Y_MSA - 15} label1="Genetic" label2="database" label3="search" />
          <ArrowLine points={[[215, Y_MSA + 20], [265, Y_MSA + 20]]} />

          <rect x={255} y={Y_MSA - 20} width={145} height={160} rx={6} fill="rgba(230,235,240,0.5)" stroke="#cfd8dc" strokeWidth={1} />
          <SeqIconRow x={270} y={Y_MSA - 8} scale={0.6} />
          <MSABox x={270} y={Y_MSA + 2} w={115} h={80} rows={4} cols={7} />
          <text x={327} y={Y_MSA + 145} textAnchor="middle" fontSize={13} fill="#1976d2" fontFamily="Inter, sans-serif">MSA</text>

          {/* Pairing */}
          <RoundedPill x={210} y={Y_PAIR + 10} text="Pairing" />
          <ArrowLine points={[[245, Y_PAIR + 10], [275, Y_PAIR + 10]]} />
          <MSABox x={280} y={Y_PAIR - 20} w={100} h={65} rows={3} cols={6} />

          {/* Structure DB + Templates */}
          <DatabaseCylinder x={155} y={Y_STRUCT - 15} label1="Structure" label2="database" label3="search" />
          <ArrowLine points={[[215, Y_STRUCT + 20], [275, Y_STRUCT + 20]]} />
          <TemplateBox x={280} y={Y_STRUCT - 5} w={60} h={60} />
          <text x={310} y={Y_STRUCT + 70} textAnchor="middle" fontSize={11} fill="#e65100" fontFamily="Inter, sans-serif">Templates</text>

          {/* Plus → MSA Repr */}
          <ArrowLine points={[[400, Y_MSA + 40], [435, Y_MSA + 90]]} />
          <ArrowLine points={[[380, Y_PAIR - 5], [435, Y_MSA + 85]]} />
          <PlusSign x={445} y={Y_MSA + 90} />
          <ArrowLine points={[[458, Y_MSA + 90], [510, Y_MSA + 90]]} />

          <SeqIconRow x={520} y={Y_MSA + 50} scale={0.55} />
          <MSAReprBlock x={515} y={Y_MSA + 58} w={95} h={70} />
          <text x={562} y={Y_MSA + 146} textAnchor="middle" fontSize={11} fill="#333" fontFamily="Inter, sans-serif">MSA</text>
          <text x={562} y={Y_MSA + 158} textAnchor="middle" fontSize={11} fill="#333" fontFamily="Inter, sans-serif">representation</text>
          <text x={562} y={Y_MSA + 172} textAnchor="middle" fontSize={11} fill="#e65100" fontFamily="Inter, sans-serif">(s,r,c)</text>

          {/* Plus → Pair Repr */}
          <ArrowLine points={[[380, Y_PAIR + 30], [435, Y_PAIR + 80]]} />
          <ArrowLine points={[[340, Y_STRUCT + 20], [435, Y_PAIR + 82]]} />
          <PlusSign x={445} y={Y_PAIR + 82} />
          <ArrowLine points={[[458, Y_PAIR + 82], [510, Y_PAIR + 82]]} />

          <SeqIconRow x={520} y={Y_PAIR + 44} scale={0.55} />
          <g transform={`translate(${610}, ${Y_PAIR + 52}) rotate(90)`}>
            <SeqIconRow x={0} y={0} scale={0.55} />
          </g>
          <RepresentationBlock x={515} y={Y_PAIR + 52} w={85} h={85} color="#64b5f6" />
          <text x={557} y={Y_PAIR + 152} textAnchor="middle" fontSize={11} fill="#333" fontFamily="Inter, sans-serif">Pair</text>
          <text x={557} y={Y_PAIR + 164} textAnchor="middle" fontSize={11} fill="#333" fontFamily="Inter, sans-serif">representation</text>
          <text x={557} y={Y_PAIR + 178} textAnchor="middle" fontSize={11} fill="#e65100" fontFamily="Inter, sans-serif">(r,r,c)</text>

          {/* → Evoformer */}
          <ArrowLine points={[[615, Y_MSA + 95], [690, Y_MSA + 140]]} />
          <ArrowLine points={[[605, Y_PAIR + 95], [690, Y_PAIR + 50]]} />
          <g onClick={() => onDrillIn?.('evoformer')} cursor="pointer">
            {/* Pulsing glow behind Evoformer to show it's the core */}
            <rect x={690} y={Y_MSA + 95} width={140} height={150} rx={14}
              fill="none" stroke="#1976d2" strokeWidth={2}>
              <animate attributeName="stroke-opacity" values="0.1;0.4;0.1" dur="3s" repeatCount="indefinite" />
            </rect>
            <BigModule x={695} y={Y_MSA + 100} w={130} h={140} line1="Evoformer" line2="(48 blocks)" />
            {/* Internal detail hint — tiny triangle icon */}
            <g transform={`translate(760, ${Y_MSA + 195})`}>
              <polygon points="0,-8 7,6 -7,6" fill="none" stroke="#1976d2" strokeWidth={1.2} opacity={0.5} />
              <circle cx={0} cy={-8} r={2} fill="#1976d2" opacity={0.5} />
              <circle cx={7} cy={6} r={2} fill="#1976d2" opacity={0.5} />
              <circle cx={-7} cy={6} r={2} fill="#1976d2" opacity={0.5} />
            </g>
            <text x={760} y={Y_MSA + 215} textAnchor="middle" fontSize={9} fill="#1976d2" fontFamily="Inter, sans-serif" opacity={0.6}>
              triangle attention
            </text>
            <text x={760} y={Y_MSA + 250} textAnchor="middle" fontSize={10} fill="#1976d2" fontFamily="Inter, sans-serif" opacity={0.7}>
              click to explore →
            </text>
          </g>

          {/* → Output Reprs */}
          <ArrowLine points={[[825, Y_MSA + 140], [885, Y_MSA + 95]]} />
          <ArrowLine points={[[825, Y_PAIR + 50], [885, Y_PAIR + 85]]} />

          <SeqIconRow x={895} y={Y_MSA + 50} scale={0.55} />
          <MSAReprBlock x={890} y={Y_MSA + 58} w={95} h={70} />
          <SingleRepr x={885} y={Y_MSA + 33} />

          <SeqIconRow x={895} y={Y_PAIR + 44} scale={0.55} />
          <g transform={`translate(${985}, ${Y_PAIR + 52}) rotate(90)`}>
            <SeqIconRow x={0} y={0} scale={0.55} />
          </g>
          <RepresentationBlock x={890} y={Y_PAIR + 52} w={85} h={85} color="#64b5f6" />
          <text x={932} y={Y_PAIR + 152} textAnchor="middle" fontSize={11} fill="#333" fontFamily="Inter, sans-serif">Pair</text>
          <text x={932} y={Y_PAIR + 164} textAnchor="middle" fontSize={11} fill="#333" fontFamily="Inter, sans-serif">representation</text>
          <text x={932} y={Y_PAIR + 178} textAnchor="middle" fontSize={11} fill="#e65100" fontFamily="Inter, sans-serif">(r,r,c)</text>

          {/* → Structure module */}
          <ArrowLine points={[[990, Y_MSA + 95], [1070, Y_MSA + 140]]} />
          <ArrowLine points={[[980, Y_PAIR + 95], [1070, Y_PAIR + 50]]} />
          <g onClick={() => onDrillIn?.('structure')} cursor="pointer">
            <BigModule x={1070} y={Y_MSA + 100} w={130} h={140} line1="Structure" line2="module" />
            <text x={1135} y={Y_MSA + 190} textAnchor="middle" fontSize={11} fill="#666" fontFamily="Inter, sans-serif">(8 blocks)</text>
            <text x={1135} y={Y_MSA + 252} textAnchor="middle" fontSize={10} fill="#1976d2" fontFamily="Inter, sans-serif" opacity={0.7}>
              click to explore →
            </text>
          </g>

          {/* → 3D structure */}
          <ArrowLine points={[[1200, Y_MSA + 170], [1255, Y_MSA + 170]]} />
          <Protein3D x={1260} y={Y_MSA + 125} />

          {/* Recycling */}
          <ArrowLine points={[[760, Y_MSA + 240], [760, Y_RECYCLE - 5]]} color="#78909c" />
          <ArrowLine points={[[1135, Y_MSA + 240], [1135, Y_RECYCLE - 5]]} color="#78909c" />
          <rect x={540} y={Y_RECYCLE} width={650} height={42} rx={6} fill="rgba(200,220,240,0.2)" stroke="#90a4ae" strokeWidth={1.5} />
          <text x={865} y={Y_RECYCLE + 27} textAnchor="middle" fontSize={14} fill="#333" fontFamily="Inter, sans-serif">← Recycling (three times)</text>
          <ArrowLine points={[[540, Y_RECYCLE + 21], [490, Y_RECYCLE + 21], [490, Y_MSA + 95], [510, Y_MSA + 93]]} color="#78909c" dashed />
          <ArrowLine points={[[540, Y_RECYCLE + 21], [490, Y_RECYCLE + 21], [490, Y_PAIR + 90], [510, Y_PAIR + 88]]} color="#78909c" dashed />

          {/* ── Tensor shape annotations ── */}
          <TensorShapeLabel x={562} y={Y_MSA + 40} shape="(s, r, c)" example="512×200×256" color="#e65100" />
          <TensorShapeLabel x={557} y={Y_PAIR + 38} shape="(r, r, c)" example="200×200×128" color="#1565c0" />
          <TensorShapeLabel x={932} y={Y_PAIR + 38} shape="(r, r, c)" example="200×200×128" color="#1565c0" />

          {/* ── Data transformation annotations along arrows ── */}
          <DataTransformLabel x={448} y={Y_MSA + 70} text="embed + fuse" />
          <DataTransformLabel x={740} y={Y_MSA + 95} text="48× refine" />
          <DataTransformLabel x={1135} y={Y_MSA + 95} text="→ xyz coords" />

          {/* ── Flow particles ── */}
          {flowPaths.map(fp => (
            <FlowParticle key={fp.id} id={fp.id} path={fp.path} duration={fp.duration} delay={fp.delay} color={fp.color} size={fp.size} />
          ))}
          {flowPaths.map(fp => (
            <FlowParticle key={`${fp.id}-b`} id={`${fp.id}-b`} path={fp.path} duration={fp.duration} delay={fp.delay + CYCLE / 2} color={fp.color} size={fp.size * 0.8} />
          ))}
        </svg>
      </div>

      {/* Explanation panel */}
      <ExplanationPanel
        stage={stage}
        onPrev={goPrev}
        onNext={goNext}
        onStageClick={handleStageClick}
        playing={playing}
        onTogglePlay={() => setPlaying(p => !p)}
      />
    </div>
  )
}
