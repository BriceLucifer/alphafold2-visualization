import { useState, useEffect, useRef, useCallback } from 'react'
import { KaTeXFormula } from '../ui/KaTeXFormula'

// ── Input Embedding operations (Algorithms 3-5, 32) ──

interface EmbOp {
  id: number
  name: string
  short: string
  category: 'target' | 'msa' | 'pair' | 'template'
  description: string
  details: string[]
  formula: string
  algRef: string
}

const EMBEDDING_OPS: EmbOp[] = [
  {
    id: 0,
    name: 'Target Feature Embedding',
    short: 'Target Feat',
    category: 'target',
    description:
      'The input sequence is converted to one-hot amino acid features and profile features (from MSA statistics). Two independent linear projections produce per-residue embeddings a_i and b_i. These seed both the MSA representation (first row) and the pair representation (via outer sum a_i + b_j).',
    details: [
      'Target features: one-hot(21 classes) + MSA profile(22) + deletion mean(1) = 44 features',
      'a_i = Linear(target_feat_i) ∈ ℝ^c_z — projects to pair channel dimension (c_z=128)',
      'b_j = Linear(target_feat_j) ∈ ℝ^c_z — second independent projection',
      'Pair init: z_ij = a_i + b_j — outer sum seeds the pair representation',
      'This asymmetric pair initialization breaks the symmetry z_ij ≠ z_ji initially',
      'The two projections learn different "roles": a captures "what I offer", b captures "what I need"',
    ],
    formula: '\\mathbf{z}_{ij}^{\\text{init}} = \\text{Linear}_a(\\text{target\\_feat}_i) + \\text{Linear}_b(\\text{target\\_feat}_j)',
    algRef: 'Algorithm 3, lines 1-3',
  },
  {
    id: 1,
    name: 'Relative Position Encoding',
    short: 'Rel Pos',
    category: 'pair',
    description:
      'Encodes the sequence distance between residue pairs as a learned embedding. The offset d_ij = i - j is clamped to [-32, 32] and converted to a one-hot vector of 65 bins, then linearly projected. This gives the pair representation a strong prior: nearby residues in sequence are more likely to be in contact in 3D.',
    details: [
      'Offset: d_ij = residue_index_i - residue_index_j',
      'Clamp: d_ij = clip(d_ij, -32, 32) — beyond ±32, all treated equally',
      'One-hot: 65 bins for values in [-32, 32]',
      'Projection: p_ij = Linear(one_hot(d_ij + 32)) ∈ ℝ^c_z',
      'Added to pair: z_ij += p_ij',
      'Captures the "sequence locality" prior: i,i+1 are bonded neighbors, i,i+4 is α-helix pattern',
    ],
    formula: 'p_{ij} = \\text{Linear}\\!\\left(\\text{one\\_hot}\\!\\left(\\text{clip}(i - j,\\, -32,\\, 32) + 32\\right)\\right) \\in \\mathbb{R}^{c_z}',
    algRef: 'Algorithm 4',
  },
  {
    id: 2,
    name: 'MSA Feature Embedding',
    short: 'MSA Feat',
    category: 'msa',
    description:
      'Each row of the MSA (a homologous sequence) is embedded independently. For the first row (the target sequence), target features are added. For extra MSA sequences beyond the cluster limit, a separate ExtraMSAEmbedder is used with reduced channels (c_e=64 vs c_m=256). The MSA representation captures per-sequence, per-position information.',
    details: [
      'MSA features per (s,i): amino acid one-hot(23) + has_deletion(1) + deletion_value(1) = 25 features',
      'MSA embed: m_si = Linear(msa_feat_si) ∈ ℝ^c_m, where c_m = 256',
      'First row special: m_1i += Linear(target_feat_i) — target sequence gets extra features',
      'Cluster MSA: up to N_clust = 512 sequences, c_m = 256 channels',
      'Extra MSA stack: additional sequences with c_e = 64 channels (4 blocks)',
      'Cluster averaging: each cluster center carries the average profile of its cluster members',
    ],
    formula: '\\mathbf{m}_{si} = \\text{Linear}(\\text{msa\\_feat}_{si}) + \\begin{cases} \\text{Linear}(\\text{target\\_feat}_i) & s = 1 \\\\ 0 & s > 1 \\end{cases}',
    algRef: 'Algorithm 3, lines 4-5',
  },
  {
    id: 3,
    name: 'Template Embedding',
    short: 'Templates',
    category: 'template',
    description:
      'Known structural templates provide direct geometric information. For each template, pairwise features (distances, orientation angles, unit vectors) between aligned residues are computed, processed by a 2-block template pair stack (with triangle operations!), then attention-weighted across templates. This gives the pair representation a strong structural prior.',
    details: [
      'Template features per (i,j): pseudo-β distance + 6 orientation features + backbone mask',
      'Pseudo-β: virtual Cβ position estimated from backbone (N, Cα, C)',
      'Orientation: 3 unit vectors × 2 pairings → 6 features encoding relative frame orientation',
      'Template pair stack: 2 blocks of triangle mult + triangle attn (same as Evoformer pair track!)',
      'Template pointwise attention: v_ij = Σ_t softmax_t(q·k/√c) · v_t — weighted across N_templ templates',
      'Added to pair: z_ij += template_embedding_ij',
    ],
    formula: '\\mathbf{z}_{ij} \\mathrel{+}= \\sum_t \\text{softmax}_t\\!\\left(\\frac{\\mathbf{q}_{ij}^\\top \\mathbf{k}_{tij}}{\\sqrt{c}}\\right) \\mathbf{v}_{tij}',
    algRef: 'Algorithm 16-17',
  },
  {
    id: 4,
    name: 'Extra MSA Stack',
    short: 'Extra MSA',
    category: 'msa',
    description:
      'Beyond the N_clust=512 clustered sequences, thousands of additional MSA sequences are processed in 4 lightweight blocks. These use column-wise global attention (not full self-attention) for efficiency. The pair representation is updated via outer product mean, then the extra MSA embedding is discarded — only the pair update is kept.',
    details: [
      'Extra MSA sequences: up to N_extra_seq additional homologs',
      'Reduced channels: c_e = 64 (vs c_m = 256 for cluster MSA)',
      '4 blocks, each with: MSA row attn, MSA column global attn, outer product mean, pair triangle ops',
      'Column global attention: compute global mean query, attend with full keys — O(N_seq · N_res) instead of O(N_seq²)',
      'After 4 blocks: extra MSA embedding is discarded, only pair updates remain',
      'This allows leveraging deep MSAs (>10,000 seqs) without quadratic memory',
    ],
    formula: '\\mathbf{e}_{si} = \\text{ExtraMSABlock}^{(4)}(\\mathbf{e}_{si}, \\mathbf{z}_{ij}), \\quad \\mathbf{z}_{ij} \\mathrel{+}= \\text{OPM}(\\mathbf{e})',
    algRef: 'Algorithm 18-19',
  },
]

// ── Visualization components ──

function TargetFeatureViz() {
  const [phase, setPhase] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval>>(null!)

  useEffect(() => {
    timerRef.current = setInterval(() => setPhase(p => (p + 1) % 4), 1800)
    return () => clearInterval(timerRef.current)
  }, [])

  const residues = 'MVLSPADKTN'.split('')
  const cz = 128

  return (
    <svg viewBox="0 0 700 420" style={{ width: '100%', maxWidth: 700 }}>
      {/* Background */}
      <rect width="700" height="420" fill="#0d0d1a" rx={8} />

      {/* Title */}
      <text x={350} y={28} textAnchor="middle" fontSize={14} fill="#8ab4f8" fontFamily="JetBrains Mono, monospace" fontWeight={600}>
        Target Feature → Pair Representation Init
      </text>

      {/* Input sequence */}
      <text x={40} y={65} fontSize={11} fill="#78909c" fontFamily="Inter, sans-serif">Input sequence:</text>
      {residues.map((r, i) => (
        <g key={i}>
          <rect x={40 + i * 42} y={75} width={36} height={36} rx={4}
            fill={phase >= 1 && i < 5 ? 'rgba(129,212,250,0.15)' : 'rgba(255,255,255,0.05)'}
            stroke={phase >= 1 && i < 5 ? '#81d4fa' : '#333'} strokeWidth={1} />
          <text x={40 + i * 42 + 18} y={98} textAnchor="middle" fontSize={14}
            fill={phase >= 1 && i < 5 ? '#81d4fa' : '#aaa'} fontFamily="JetBrains Mono, monospace" fontWeight={600}>
            {r}
          </text>
        </g>
      ))}

      {/* Arrow down to Linear_a and Linear_b */}
      {phase >= 1 && (
        <g opacity={phase >= 1 ? 1 : 0} style={{ transition: 'opacity 0.5s' }}>
          <line x1={220} y1={118} x2={150} y2={150} stroke="#e0a040" strokeWidth={1.5} markerEnd="url(#arrowY)" />
          <line x1={220} y1={118} x2={290} y2={150} stroke="#40a0e0" strokeWidth={1.5} markerEnd="url(#arrowB)" />

          {/* Linear_a box */}
          <rect x={100} y={155} width={100} height={32} rx={6} fill="rgba(224,160,64,0.15)" stroke="#e0a040" strokeWidth={1.2} />
          <text x={150} y={176} textAnchor="middle" fontSize={12} fill="#e0a040" fontFamily="JetBrains Mono, monospace">Linear_a</text>

          {/* Linear_b box */}
          <rect x={240} y={155} width={100} height={32} rx={6} fill="rgba(64,160,224,0.15)" stroke="#40a0e0" strokeWidth={1.2} />
          <text x={290} y={176} textAnchor="middle" fontSize={12} fill="#40a0e0" fontFamily="JetBrains Mono, monospace">Linear_b</text>

          {/* a_i vector */}
          <line x1={150} y1={192} x2={150} y2={220} stroke="#e0a040" strokeWidth={1.5} markerEnd="url(#arrowY)" />
          <text x={150} y={240} textAnchor="middle" fontSize={12} fill="#e0a040" fontFamily="JetBrains Mono, monospace">
            a_i ∈ ℝ^{cz}
          </text>

          {/* b_j vector */}
          <line x1={290} y1={192} x2={290} y2={220} stroke="#40a0e0" strokeWidth={1.5} markerEnd="url(#arrowB)" />
          <text x={290} y={240} textAnchor="middle" fontSize={12} fill="#40a0e0" fontFamily="JetBrains Mono, monospace">
            b_j ∈ ℝ^{cz}
          </text>
        </g>
      )}

      {/* Outer sum → pair matrix */}
      {phase >= 2 && (
        <g opacity={1} style={{ transition: 'opacity 0.5s' }}>
          <line x1={150} y1={248} x2={220} y2={280} stroke="#e0a040" strokeWidth={1.5} strokeDasharray="4,3" />
          <line x1={290} y1={248} x2={220} y2={280} stroke="#40a0e0" strokeWidth={1.5} strokeDasharray="4,3" />

          {/* Plus circle */}
          <circle cx={220} cy={268} r={12} fill="rgba(255,255,255,0.08)" stroke="#aaa" strokeWidth={1} />
          <line x1={214} y1={268} x2={226} y2={268} stroke="#fff" strokeWidth={2} />
          <line x1={220} y1={262} x2={220} y2={274} stroke="#fff" strokeWidth={2} />

          {/* Pair matrix visualization */}
          <text x={220} y={300} textAnchor="middle" fontSize={11} fill="#aaa" fontFamily="Inter, sans-serif">
            z_ij = a_i + b_j
          </text>

          {/* Small matrix grid */}
          {Array.from({ length: 8 }).map((_, r) =>
            Array.from({ length: 8 }).map((_, c) => {
              const val = Math.sin(r * 0.7 + c * 0.5) * 0.5 + 0.5
              const highlighted = phase >= 3 && r < 4 && c < 4
              return (
                <rect key={`${r}-${c}`}
                  x={140 + c * 20} y={310 + r * 12} width={18} height={10} rx={1.5}
                  fill={highlighted
                    ? `rgba(129,212,250,${0.3 + val * 0.5})`
                    : `rgba(100,181,246,${0.1 + val * 0.25})`}
                  stroke={highlighted ? '#81d4fa' : 'rgba(255,255,255,0.05)'} strokeWidth={0.5}>
                  {highlighted && (
                    <animate attributeName="opacity" values="0.6;1;0.6" dur="1.5s" repeatCount="indefinite" />
                  )}
                </rect>
              )
            })
          )}

          {/* Labels */}
          <text x={135} y={318} textAnchor="end" fontSize={9} fill="#e0a040" fontFamily="JetBrains Mono, monospace">i→</text>
          <text x={138} y={310} textAnchor="end" fontSize={9} fill="#40a0e0" fontFamily="JetBrains Mono, monospace">j↓</text>
          <text x={220} y={415} textAnchor="middle" fontSize={12} fill="#64b5f6" fontFamily="Inter, sans-serif">
            Pair representation (r×r×{cz})
          </text>
        </g>
      )}

      {/* Right side: summary */}
      <g transform="translate(420, 80)">
        <rect x={0} y={0} width={250} height={300} rx={8} fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
        <text x={125} y={28} textAnchor="middle" fontSize={13} fill="#8ab4f8" fontFamily="Inter, sans-serif" fontWeight={600}>
          How it works
        </text>
        <line x1={20} y1={38} x2={230} y2={38} stroke="rgba(255,255,255,0.1)" strokeWidth={1} />

        {[
          { step: '1', text: 'One-hot encode each amino acid', color: '#aaa', active: phase >= 0 },
          { step: '2', text: 'Linear_a → row embedding a_i', color: '#e0a040', active: phase >= 1 },
          { step: '3', text: 'Linear_b → col embedding b_j', color: '#40a0e0', active: phase >= 1 },
          { step: '4', text: 'Outer sum: z_ij = a_i + b_j', color: '#81d4fa', active: phase >= 2 },
          { step: '5', text: 'Add relative position encoding', color: '#a5d6a7', active: phase >= 3 },
          { step: '6', text: 'Add template features', color: '#ef9a9a', active: phase >= 3 },
        ].map((s, i) => (
          <g key={i} opacity={s.active ? 1 : 0.3}>
            <circle cx={30} cy={62 + i * 40} r={10} fill={s.active ? `${s.color}30` : 'transparent'} stroke={s.color} strokeWidth={1.2} />
            <text x={30} y={66 + i * 40} textAnchor="middle" fontSize={10} fill={s.color} fontFamily="JetBrains Mono, monospace" fontWeight={600}>
              {s.step}
            </text>
            <text x={50} y={66 + i * 40} fontSize={11} fill={s.active ? '#ddd' : '#555'} fontFamily="Inter, sans-serif">
              {s.text}
            </text>
          </g>
        ))}
      </g>

      {/* Arrow markers */}
      <defs>
        <marker id="arrowY" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0,0 8,3 0,6" fill="#e0a040" />
        </marker>
        <marker id="arrowB" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0,0 8,3 0,6" fill="#40a0e0" />
        </marker>
      </defs>
    </svg>
  )
}

function RelPosViz() {
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number } | null>(null)
  const N = 12

  return (
    <svg viewBox="0 0 700 400" style={{ width: '100%', maxWidth: 700 }}>
      <rect width="700" height="400" fill="#0d0d1a" rx={8} />
      <text x={350} y={28} textAnchor="middle" fontSize={14} fill="#a5d6a7" fontFamily="JetBrains Mono, monospace" fontWeight={600}>
        Relative Position Encoding: d_ij = clip(i - j, -32, 32)
      </text>

      {/* Matrix showing d_ij values */}
      <g transform="translate(60, 50)">
        {/* Column labels */}
        {Array.from({ length: N }).map((_, j) => (
          <text key={`col-${j}`} x={55 + j * 32} y={18} textAnchor="middle" fontSize={9}
            fill={hoveredCell?.j === j ? '#a5d6a7' : '#666'} fontFamily="JetBrains Mono, monospace">
            {j}
          </text>
        ))}
        <text x={55 + N * 16} y={-2} textAnchor="middle" fontSize={10} fill="#78909c" fontFamily="Inter, sans-serif">j →</text>

        {/* Row labels + cells */}
        {Array.from({ length: N }).map((_, i) => (
          <g key={`row-${i}`}>
            <text x={28} y={40 + i * 26} textAnchor="middle" fontSize={9}
              fill={hoveredCell?.i === i ? '#a5d6a7' : '#666'} fontFamily="JetBrains Mono, monospace">
              {i}
            </text>
            {Array.from({ length: N }).map((_, j) => {
              const d = i - j
              const clamped = Math.max(-32, Math.min(32, d))
              const norm = (clamped + 32) / 64
              const isHovered = hoveredCell?.i === i && hoveredCell?.j === j
              const isOnDiag = i === j
              const r = Math.round(norm < 0.5 ? 40 + (0.5 - norm) * 2 * 180 : 40)
              const g = Math.round(norm < 0.5 ? 40 : 40)
              const b = Math.round(norm > 0.5 ? 40 + (norm - 0.5) * 2 * 180 : 40)
              return (
                <g key={`${i}-${j}`}
                  onMouseEnter={() => setHoveredCell({ i, j })}
                  onMouseLeave={() => setHoveredCell(null)}>
                  <rect x={40 + j * 32} y={26 + i * 26} width={28} height={22} rx={3}
                    fill={isOnDiag ? 'rgba(165,214,167,0.25)' : `rgba(${r},${g},${b},0.5)`}
                    stroke={isHovered ? '#a5d6a7' : 'rgba(255,255,255,0.06)'} strokeWidth={isHovered ? 1.5 : 0.5} />
                  <text x={40 + j * 32 + 14} y={26 + i * 26 + 15} textAnchor="middle" fontSize={8.5}
                    fill={isHovered ? '#fff' : '#999'} fontFamily="JetBrains Mono, monospace">
                    {d > 0 ? `+${d}` : d}
                  </text>
                </g>
              )
            })}
          </g>
        ))}
        <text x={0} y={26 + N * 13} textAnchor="middle" fontSize={10} fill="#78909c" fontFamily="Inter, sans-serif" transform={`rotate(-90, 0, ${26 + N * 13})`}>
          i →
        </text>
      </g>

      {/* Right side: explanation */}
      <g transform="translate(460, 60)">
        <rect x={0} y={0} width={220} height={250} rx={8} fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
        <text x={110} y={28} textAnchor="middle" fontSize={12} fill="#a5d6a7" fontFamily="Inter, sans-serif" fontWeight={600}>
          Encoding pipeline
        </text>
        <line x1={15} y1={38} x2={205} y2={38} stroke="rgba(255,255,255,0.1)" strokeWidth={1} />

        <text x={20} y={65} fontSize={11} fill="#ccc" fontFamily="Inter, sans-serif">d_ij = i - j</text>
        <text x={35} y={85} fontSize={10} fill="#888" fontFamily="Inter, sans-serif">↓ clip to [-32, 32]</text>
        <text x={20} y={110} fontSize={11} fill="#ccc" fontFamily="Inter, sans-serif">one_hot(d + 32)</text>
        <text x={35} y={130} fontSize={10} fill="#888" fontFamily="Inter, sans-serif">↓ 65 bins → Linear</text>
        <text x={20} y={155} fontSize={11} fill="#ccc" fontFamily="Inter, sans-serif">p_ij ∈ ℝ^128</text>
        <text x={35} y={175} fontSize={10} fill="#888" fontFamily="Inter, sans-serif">↓ add to pair repr</text>
        <text x={20} y={200} fontSize={11} fill="#a5d6a7" fontFamily="Inter, sans-serif">z_ij += p_ij</text>

        {hoveredCell && (
          <g>
            <rect x={15} y={215} width={190} height={28} rx={4} fill="rgba(165,214,167,0.1)" />
            <text x={110} y={234} textAnchor="middle" fontSize={11} fill="#a5d6a7" fontFamily="JetBrains Mono, monospace">
              d({hoveredCell.i},{hoveredCell.j}) = {hoveredCell.i - hoveredCell.j} → bin {hoveredCell.i - hoveredCell.j + 32}
            </text>
          </g>
        )}
      </g>

      {/* Color legend */}
      <g transform="translate(60, 370)">
        <text x={0} y={12} fontSize={10} fill="#888" fontFamily="Inter, sans-serif">nearby (i≈j)</text>
        <rect x={100} y={2} width={200} height={12} rx={3}>
          <defs>
            <linearGradient id="relposGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgb(220,40,40)" />
              <stop offset="50%" stopColor="rgb(40,40,40)" />
              <stop offset="100%" stopColor="rgb(40,40,220)" />
            </linearGradient>
          </defs>
        </rect>
        <rect x={100} y={2} width={200} height={12} rx={3} fill="url(#relposGrad)" />
        <text x={310} y={12} fontSize={10} fill="#888" fontFamily="Inter, sans-serif">far apart</text>
        <text x={200} y={28} textAnchor="middle" fontSize={9} fill="#666" fontFamily="JetBrains Mono, monospace">d = -N ... 0 ... +N</text>
      </g>
    </svg>
  )
}

function MSAEmbedViz() {
  const [activeRow, setActiveRow] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval>>(null!)

  useEffect(() => {
    timerRef.current = setInterval(() => setActiveRow(r => (r + 1) % 6), 1200)
    return () => clearInterval(timerRef.current)
  }, [])

  const seqs = [
    { label: 'Target', seq: 'MVLSPADKTN', special: true },
    { label: 'Homolog 1', seq: 'MVLSAADKTN', special: false },
    { label: 'Homolog 2', seq: 'MLLSPADKAN', special: false },
    { label: 'Homolog 3', seq: 'MVLTPADKTN', special: false },
    { label: 'Homolog 4', seq: 'MALSPADKTN', special: false },
    { label: 'Homolog 5', seq: 'MVLSPADKAN', special: false },
  ]

  const aaColors: Record<string, string> = {
    M: '#e57373', V: '#64b5f6', L: '#81c784', S: '#ffb74d',
    P: '#ce93d8', A: '#4dd0e1', D: '#ff8a65', K: '#aed581',
    T: '#90a4ae', N: '#f06292',
  }

  return (
    <svg viewBox="0 0 700 420" style={{ width: '100%', maxWidth: 700 }}>
      <rect width="700" height="420" fill="#0d0d1a" rx={8} />
      <text x={350} y={28} textAnchor="middle" fontSize={14} fill="#e8a87c" fontFamily="JetBrains Mono, monospace" fontWeight={600}>
        MSA Feature Embedding → m_si ∈ ℝ^256
      </text>

      {/* MSA alignment visualization */}
      <g transform="translate(30, 50)">
        {seqs.map((s, si) => {
          const isActive = si === activeRow
          return (
            <g key={si}>
              {/* Row label */}
              <text x={0} y={si * 48 + 24} fontSize={10}
                fill={isActive ? (s.special ? '#81d4fa' : '#e8a87c') : '#666'}
                fontFamily="Inter, sans-serif" fontWeight={s.special ? 600 : 400}>
                {s.label}
              </text>
              {/* Amino acid cells */}
              {s.seq.split('').map((aa, ai) => (
                <g key={ai}>
                  <rect x={80 + ai * 34} y={si * 48 + 8} width={30} height={28} rx={3}
                    fill={isActive ? `${aaColors[aa]}30` : 'rgba(255,255,255,0.03)'}
                    stroke={isActive ? aaColors[aa] : 'rgba(255,255,255,0.06)'} strokeWidth={isActive ? 1.2 : 0.5} />
                  <text x={80 + ai * 34 + 15} y={si * 48 + 27} textAnchor="middle" fontSize={12}
                    fill={isActive ? aaColors[aa] : '#555'} fontFamily="JetBrains Mono, monospace" fontWeight={600}>
                    {aa}
                  </text>
                </g>
              ))}
              {/* Arrow to embedding */}
              {isActive && (
                <g>
                  <line x1={422} y1={si * 48 + 22} x2={460} y2={si * 48 + 22}
                    stroke="#e8a87c" strokeWidth={1.5} markerEnd="url(#arrowO)" />
                  {/* Linear box */}
                  <rect x={465} y={si * 48 + 6} width={70} height={30} rx={5}
                    fill="rgba(232,168,124,0.15)" stroke="#e8a87c" strokeWidth={1.2} />
                  <text x={500} y={si * 48 + 26} textAnchor="middle" fontSize={10}
                    fill="#e8a87c" fontFamily="JetBrains Mono, monospace">
                    Linear
                  </text>
                  {/* Output vector */}
                  <line x1={540} y1={si * 48 + 22} x2={570} y2={si * 48 + 22}
                    stroke="#e8a87c" strokeWidth={1.5} markerEnd="url(#arrowO)" />
                  {/* Embedding vector visualization */}
                  {Array.from({ length: 16 }).map((_, k) => (
                    <rect key={k} x={575 + k * 6} y={si * 48 + 10} width={5} height={24} rx={1}
                      fill="#e8a87c" opacity={0.2 + Math.sin(k * 0.8 + si) * 0.3 + 0.3}>
                      <animate attributeName="opacity"
                        values={`${0.2 + Math.sin(k * 0.8) * 0.3};${0.5 + Math.sin(k * 0.8 + 1) * 0.3};${0.2 + Math.sin(k * 0.8) * 0.3}`}
                        dur="2s" repeatCount="indefinite" />
                    </rect>
                  ))}
                </g>
              )}
              {/* Special indicator for target */}
              {s.special && isActive && (
                <g>
                  <text x={500} y={si * 48 + 50} textAnchor="middle" fontSize={9} fill="#81d4fa" fontFamily="Inter, sans-serif">
                    + target_feat
                  </text>
                </g>
              )}
            </g>
          )
        })}
      </g>

      {/* Output description */}
      <g transform="translate(30, 350)">
        <rect x={0} y={0} width={640} height={55} rx={6} fill="rgba(232,168,124,0.08)" stroke="rgba(232,168,124,0.2)" strokeWidth={1} />
        <text x={320} y={22} textAnchor="middle" fontSize={12} fill="#e8a87c" fontFamily="Inter, sans-serif">
          Output: MSA representation m ∈ ℝ^(N_seq × N_res × c_m) = (512 × 200 × 256) — 26M values
        </text>
        <text x={320} y={42} textAnchor="middle" fontSize={11} fill="#888" fontFamily="Inter, sans-serif">
          Each cell encodes amino acid identity + evolutionary context + position information
        </text>
      </g>

      <defs>
        <marker id="arrowO" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0,0 8,3 0,6" fill="#e8a87c" />
        </marker>
      </defs>
    </svg>
  )
}

function TemplateViz() {
  const [activeTemplate, setActiveTemplate] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval>>(null!)

  useEffect(() => {
    timerRef.current = setInterval(() => setActiveTemplate(t => (t + 1) % 3), 2500)
    return () => clearInterval(timerRef.current)
  }, [])

  return (
    <svg viewBox="0 0 700 400" style={{ width: '100%', maxWidth: 700 }}>
      <rect width="700" height="400" fill="#0d0d1a" rx={8} />
      <text x={350} y={28} textAnchor="middle" fontSize={14} fill="#ef9a9a" fontFamily="JetBrains Mono, monospace" fontWeight={600}>
        Template Integration: Known Structures → Pair Bias
      </text>

      {/* Template structures */}
      {[0, 1, 2].map(t => {
        const isActive = t === activeTemplate
        const xOff = 40 + t * 210
        return (
          <g key={t} opacity={isActive ? 1 : 0.4} style={{ transition: 'opacity 0.5s' }}>
            <rect x={xOff} y={50} width={180} height={140} rx={8}
              fill={isActive ? 'rgba(239,154,154,0.08)' : 'rgba(255,255,255,0.02)'}
              stroke={isActive ? '#ef9a9a' : '#333'} strokeWidth={isActive ? 1.5 : 0.8} />
            <text x={xOff + 90} y={72} textAnchor="middle" fontSize={11}
              fill={isActive ? '#ef9a9a' : '#666'} fontFamily="Inter, sans-serif" fontWeight={600}>
              Template {t + 1} ({[82, 45, 23][t]}% seq id)
            </text>
            {/* Simplified backbone cartoon */}
            <path d={[
              'M60,100 C80,80 110,120 140,95 S170,110 175,100',
              'M60,105 C90,85 100,125 130,100 S160,115 175,105',
              'M60,98 C75,118 120,78 145,108 S165,90 175,108',
            ][t]}
              fill="none" stroke={isActive ? '#ef9a9a' : '#555'}
              strokeWidth={3} strokeLinecap="round"
              transform={`translate(${xOff - 20}, 0)`} />
            {/* Distance pairs */}
            {isActive && (
              <g>
                {[[70, 130], [90, 150], [110, 160]].map(([a, b], pi) => (
                  <g key={pi}>
                    <line x1={xOff + a - 20} y1={[100, 105, 98][t] + (pi * 3)}
                      x2={xOff + b - 20} y2={[95, 100, 108][t] - (pi * 2)}
                      stroke="#ef9a9a" strokeWidth={0.8} strokeDasharray="3,2" opacity={0.6}>
                      <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2s"
                        begin={`${pi * 0.3}s`} repeatCount="indefinite" />
                    </line>
                  </g>
                ))}
              </g>
            )}
            <text x={xOff + 90} y={175} textAnchor="middle" fontSize={10}
              fill={isActive ? '#ccc' : '#555'} fontFamily="Inter, sans-serif">
              d_ij, angles, masks
            </text>
          </g>
        )
      })}

      {/* Arrow down to attention */}
      <line x1={350} y1={200} x2={350} y2={230} stroke="#ef9a9a" strokeWidth={1.5} markerEnd="url(#arrowR)" />

      {/* Template Pair Stack */}
      <rect x={220} y={235} width={260} height={40} rx={6} fill="rgba(239,154,154,0.1)" stroke="#ef9a9a" strokeWidth={1.2} />
      <text x={350} y={260} textAnchor="middle" fontSize={12} fill="#ef9a9a" fontFamily="JetBrains Mono, monospace">
        Template Pair Stack (2 blocks)
      </text>

      {/* Arrow down to attention pooling */}
      <line x1={350} y1={280} x2={350} y2={300} stroke="#ef9a9a" strokeWidth={1.5} markerEnd="url(#arrowR)" />

      {/* Pointwise attention */}
      <rect x={200} y={305} width={300} height={40} rx={6} fill="rgba(239,154,154,0.15)" stroke="#ef9a9a" strokeWidth={1.2} />
      <text x={350} y={330} textAnchor="middle" fontSize={12} fill="#ef9a9a" fontFamily="JetBrains Mono, monospace">
        Pointwise Attention (across templates)
      </text>

      {/* Arrow to pair */}
      <line x1={350} y1={350} x2={350} y2={375} stroke="#64b5f6" strokeWidth={1.5} markerEnd="url(#arrowPB)" />
      <text x={350} y={395} textAnchor="middle" fontSize={12} fill="#64b5f6" fontFamily="Inter, sans-serif">
        z_ij += template_embedding
      </text>

      <defs>
        <marker id="arrowR" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0,0 8,3 0,6" fill="#ef9a9a" />
        </marker>
        <marker id="arrowPB" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0,0 8,3 0,6" fill="#64b5f6" />
        </marker>
      </defs>
    </svg>
  )
}

function ExtraMSAViz() {
  const [blockIdx, setBlockIdx] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval>>(null!)

  useEffect(() => {
    timerRef.current = setInterval(() => setBlockIdx(b => (b + 1) % 5), 2000)
    return () => clearInterval(timerRef.current)
  }, [])

  return (
    <svg viewBox="0 0 700 400" style={{ width: '100%', maxWidth: 700 }}>
      <rect width="700" height="400" fill="#0d0d1a" rx={8} />
      <text x={350} y={28} textAnchor="middle" fontSize={14} fill="#ce93d8" fontFamily="JetBrains Mono, monospace" fontWeight={600}>
        Extra MSA Stack: Deep MSA → Pair Updates
      </text>

      {/* Large MSA block */}
      <g transform="translate(30, 50)">
        <text x={0} y={16} fontSize={11} fill="#888" fontFamily="Inter, sans-serif">Extra sequences (N_extra &gt; 5000)</text>
        <rect x={0} y={25} width={160} height={120} rx={6} fill="rgba(206,147,216,0.08)" stroke="#ce93d8" strokeWidth={1.2} />
        {Array.from({ length: 10 }).map((_, r) =>
          Array.from({ length: 12 }).map((_, c) => (
            <rect key={`${r}-${c}`} x={6 + c * 12.5} y={30 + r * 11} width={10} height={9} rx={1}
              fill="#ce93d8" opacity={0.1 + Math.sin(r * 0.5 + c * 0.7) * 0.15 + 0.15} />
          ))
        )}
        <text x={80} y={165} textAnchor="middle" fontSize={10} fill="#ce93d8" fontFamily="JetBrains Mono, monospace">
          e_si ∈ ℝ^64
        </text>
      </g>

      {/* Arrow */}
      <line x1={200} y1={130} x2={230} y2={130} stroke="#ce93d8" strokeWidth={1.5} markerEnd="url(#arrowP)" />

      {/* 4 blocks */}
      {[0, 1, 2, 3].map(b => {
        const isActive = b === blockIdx && blockIdx < 4
        return (
          <g key={b} transform={`translate(${240 + b * 95}, 70)`}>
            <rect x={0} y={0} width={80} height={120} rx={6}
              fill={isActive ? 'rgba(206,147,216,0.15)' : 'rgba(255,255,255,0.03)'}
              stroke={isActive ? '#ce93d8' : '#444'} strokeWidth={isActive ? 1.5 : 0.8}>
              {isActive && <animate attributeName="stroke-opacity" values="0.5;1;0.5" dur="1.5s" repeatCount="indefinite" />}
            </rect>
            <text x={40} y={20} textAnchor="middle" fontSize={10} fill={isActive ? '#ce93d8' : '#666'}
              fontFamily="Inter, sans-serif" fontWeight={600}>Block {b + 1}</text>

            {['Row attn', 'Col global', 'OPM', 'Tri ops'].map((op, oi) => (
              <text key={oi} x={40} y={40 + oi * 22} textAnchor="middle" fontSize={8.5}
                fill={isActive ? '#bbb' : '#555'} fontFamily="Inter, sans-serif">{op}</text>
            ))}
          </g>
        )
      })}

      {/* Arrow to pair */}
      <g transform="translate(250, 210)">
        <line x1={170} y1={0} x2={170} y2={40} stroke="#64b5f6" strokeWidth={1.5} markerEnd="url(#arrowPB2)" />
        <text x={170} y={60} textAnchor="middle" fontSize={12} fill="#64b5f6" fontFamily="Inter, sans-serif">
          Pair representation updated
        </text>

        <rect x={60} y={75} width={220} height={40} rx={6} fill="rgba(100,181,246,0.08)" stroke="#64b5f6" strokeWidth={1} />
        <text x={170} y={100} textAnchor="middle" fontSize={11} fill="#888" fontFamily="Inter, sans-serif">
          Extra MSA embedding discarded after this
        </text>
      </g>

      {/* Key insight */}
      <g transform="translate(30, 320)">
        <rect x={0} y={0} width={640} height={55} rx={6} fill="rgba(206,147,216,0.06)" stroke="rgba(206,147,216,0.2)" strokeWidth={1} />
        <text x={320} y={20} textAnchor="middle" fontSize={11} fill="#ce93d8" fontFamily="Inter, sans-serif" fontWeight={600}>
          Key efficiency trick: Column Global Attention
        </text>
        <text x={320} y={42} textAnchor="middle" fontSize={10} fill="#888" fontFamily="Inter, sans-serif">
          Global mean query → O(N_seq × N_res) vs full attention O(N_seq² × N_res) — enables &gt;10K sequences
        </text>
      </g>

      <defs>
        <marker id="arrowP" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0,0 8,3 0,6" fill="#ce93d8" />
        </marker>
        <marker id="arrowPB2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0,0 8,3 0,6" fill="#64b5f6" />
        </marker>
      </defs>
    </svg>
  )
}

// ── Visualization selector ──

function VizForOp({ opId }: { opId: number }) {
  switch (opId) {
    case 0: return <TargetFeatureViz />
    case 1: return <RelPosViz />
    case 2: return <MSAEmbedViz />
    case 3: return <TemplateViz />
    case 4: return <ExtraMSAViz />
    default: return null
  }
}

// ── Category colors ──

const CAT_COLORS: Record<string, string> = {
  target: '#8ab4f8',
  msa: '#e8a87c',
  pair: '#a5d6a7',
  template: '#ef9a9a',
}

// ── Main Component ──

export function InputEmbeddingDetail({ onBack }: { onBack?: () => void }) {
  const [activeOpId, setActiveOpId] = useState(0)
  const [playing, setPlaying] = useState(true)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const activeOp = EMBEDDING_OPS[activeOpId]

  const goNext = useCallback(() => {
    setActiveOpId(id => (id + 1) % EMBEDDING_OPS.length)
  }, [])
  const goPrev = useCallback(() => {
    setActiveOpId(id => (id - 1 + EMBEDDING_OPS.length) % EMBEDDING_OPS.length)
  }, [])

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(goNext, 8000)
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [playing, goNext])

  return (
    <div style={{
      width: '100%', height: '100%', background: '#0a0a15',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      fontFamily: 'Inter, system-ui, sans-serif', color: '#e0e0e0',
    }}>
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .emb-fade { animation: fadeIn 0.4s ease-out; }
      `}</style>

      {/* Header */}
      <div style={{
        padding: '14px 28px 10px', borderBottom: '1px solid rgba(255,255,255,0.08)',
        display: 'flex', alignItems: 'center', gap: 16, flexShrink: 0,
      }}>
        {onBack && (
          <button onClick={onBack} style={{
            border: '1px solid rgba(255,255,255,0.15)', borderRadius: 6,
            background: 'rgba(255,255,255,0.05)', padding: '5px 14px',
            cursor: 'pointer', fontSize: 12, color: '#aaa',
          }}>
            ← Overview
          </button>
        )}
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#8ab4f8' }}>
          Input Embedding
        </h1>
        <span style={{ fontSize: 13, color: '#78909c' }}>
          Algorithm 3-5 — How sequence, MSA, and templates become tensor representations
        </span>
      </div>

      {/* Main content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        {/* Sidebar — operation list */}
        <div style={{
          width: 220, borderRight: '1px solid rgba(255,255,255,0.08)',
          overflowY: 'auto', padding: '12px 0', flexShrink: 0,
        }}>
          {EMBEDDING_OPS.map(op => {
            const isActive = op.id === activeOpId
            const color = CAT_COLORS[op.category]
            return (
              <div key={op.id}
                onClick={() => { setActiveOpId(op.id); setPlaying(false) }}
                style={{
                  padding: '10px 16px', cursor: 'pointer',
                  background: isActive ? 'rgba(255,255,255,0.06)' : 'transparent',
                  borderLeft: isActive ? `3px solid ${color}` : '3px solid transparent',
                  transition: 'all 0.2s',
                }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: isActive ? color : '#999', marginBottom: 2 }}>
                  {op.short}
                </div>
                <div style={{ fontSize: 10, color: '#666' }}>{op.algRef}</div>
              </div>
            )
          })}

          {/* Play controls */}
          <div style={{ padding: '16px', borderTop: '1px solid rgba(255,255,255,0.06)', marginTop: 8 }}>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
              <button onClick={goPrev} style={btnStyle}>◀</button>
              <button onClick={() => setPlaying(p => !p)} style={{ ...btnStyle, width: 40 }}>
                {playing ? '⏸' : '▶'}
              </button>
              <button onClick={goNext} style={btnStyle}>▶</button>
            </div>
            <div style={{ textAlign: 'center', fontSize: 10, color: '#666', marginTop: 6 }}>
              {activeOpId + 1} / {EMBEDDING_OPS.length}
            </div>
          </div>
        </div>

        {/* Right content area */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 28px' }}>
          <div key={`viz-${activeOpId}`} className="emb-fade">
            {/* Operation title */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
                <span style={{
                  fontSize: 10, padding: '2px 8px', borderRadius: 10,
                  background: `${CAT_COLORS[activeOp.category]}20`,
                  color: CAT_COLORS[activeOp.category], fontWeight: 600,
                  textTransform: 'uppercase', letterSpacing: 1,
                }}>
                  {activeOp.category}
                </span>
                <span style={{ fontSize: 11, color: '#78909c' }}>{activeOp.algRef}</span>
              </div>
              <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: '#e0e0e0' }}>
                {activeOp.name}
              </h2>
            </div>

            {/* Visualization */}
            <div style={{ marginBottom: 20 }}>
              <VizForOp opId={activeOpId} />
            </div>

            {/* Formula */}
            <div style={{ marginBottom: 16 }}>
              <KaTeXFormula formula={activeOp.formula} />
            </div>

            {/* Description */}
            <p style={{ fontSize: 14, color: '#bbb', lineHeight: 1.65, maxWidth: 700, margin: '0 0 14px' }}>
              {activeOp.description}
            </p>

            {/* Details */}
            <ul style={{
              margin: 0, padding: '0 0 0 18px', fontSize: 13,
              color: '#999', lineHeight: 1.75,
            }}>
              {activeOp.details.map((d, i) => (
                <li key={i}>{d}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

const btnStyle: React.CSSProperties = {
  width: 32, height: 28, border: '1px solid rgba(255,255,255,0.15)',
  borderRadius: 4, background: 'rgba(255,255,255,0.05)', cursor: 'pointer',
  fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#aaa',
}
