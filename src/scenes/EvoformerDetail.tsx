import { useState, useEffect, useRef, useCallback } from 'react'
import { KaTeXFormula } from '../ui/KaTeXFormula'

// ── Evoformer block operations (accurate to Supplementary Algorithms 6-15) ──

interface BlockOp {
  id: number
  name: string
  short: string
  category: 'msa' | 'pair' | 'cross'
  description: string
  details: string[]
  formula: string
  algRef: string
}

const EVOFORMER_OPS: BlockOp[] = [
  {
    id: 0,
    name: 'MSA Row-wise Gated Self-Attention with Pair Bias',
    short: 'Row Attn',
    category: 'msa',
    description:
      'Within each MSA sequence, every residue attends to every other residue. The pair representation injects structural knowledge by biasing the attention logits — if pair(i,j) says "these residues are close", they attend more strongly. A sigmoid gate controls which information passes through.',
    details: [
      'Input projections: q, k, v = LinearNoBias(LayerNorm(m_si)), plus gating g = sigmoid(Linear(m_si))',
      'Pair bias: b_ij = LinearNoBias(LayerNorm(z_ij)) — structure guides MSA attention',
      'Attention: a_sij = softmax_j(q_si · k_sj / √c + b_ij)',
      'Gated output: o_si = g_si ⊙ Σ_j a_sij · v_sj',
      'N_head = 8, c = 32 per head, with DropoutRowwise(0.15)',
    ],
    formula: '\\mathbf{o}_{si}^h = \\mathbf{g}_{si}^h \\odot \\sum_j \\text{softmax}_j\\!\\left(\\frac{1}{\\sqrt{c}}\\,\\mathbf{q}_{si}^{h\\top}\\mathbf{k}_{sj}^h + b_{ij}^h\\right) \\mathbf{v}_{sj}^h',
    algRef: 'Algorithm 7',
  },
  {
    id: 1,
    name: 'MSA Column-wise Gated Self-Attention',
    short: 'Col Attn',
    category: 'msa',
    description:
      'At each residue position, attention is computed across all MSA sequences. This detects co-evolution: if position i is always arginine when position j is glutamate across many species, this column-wise pattern captures that correlation. No pair bias here — pure MSA signal.',
    details: [
      'Attention axis: softmax over sequences t at fixed position i',
      'a_sti = softmax_t(q_si · k_ti / √c) — no pair bias, unlike row-wise',
      'Captures co-evolutionary signals: correlated mutations across species',
      'Much cheaper than full (s×r) attention — O(s²) per position',
      'Same gating mechanism: g = sigmoid(Linear(m_si))',
    ],
    formula: '\\mathbf{o}_{si}^h = \\mathbf{g}_{si}^h \\odot \\sum_t a_{sti}^h \\, \\mathbf{v}_{ti}^h, \\quad a_{sti}^h = \\text{softmax}_t\\!\\left(\\frac{1}{\\sqrt{c}}\\,\\mathbf{q}_{si}^{h\\top}\\mathbf{k}_{ti}^h\\right)',
    algRef: 'Algorithm 8',
  },
  {
    id: 2,
    name: 'Outer Product Mean',
    short: 'Outer Prod',
    category: 'cross',
    description:
      'The bridge from MSA → Pair representation. For each pair (i,j), project MSA features at positions i and j with two independent Linear layers, compute their outer product, average across all sequences, then project to pair dimension. Co-evolving positions produce strong signals.',
    details: [
      'Project: a_si = Linear(LayerNorm(m_si)), b_si = Linear(m_si), both ∈ ℝ^c, c=32',
      'Outer product per sequence: a_si ⊗ b_sj ∈ ℝ^(c×c)',
      'Average over sequences: o_ij = flatten(mean_s(a_si ⊗ b_sj))',
      'Final projection: z_ij = Linear(o_ij) ∈ ℝ^c_z',
      'Key insight: co-evolution ↔ spatial proximity in 3D',
    ],
    formula: '\\mathbf{z}_{ij} = \\text{Linear}\\!\\left(\\text{flatten}\\!\\left(\\text{mean}_s\\left(\\mathbf{a}_{si} \\otimes \\mathbf{b}_{sj}\\right)\\right)\\right)',
    algRef: 'Algorithm 10',
  },
  {
    id: 3,
    name: 'Triangle Multiplicative Update (Outgoing)',
    short: '△ Mult Out',
    category: 'pair',
    description:
      'Updates pair(i,j) by combining information from all triangles with outgoing edges. For every k, multiplies gated projections of pair(i,k) and pair(j,k). The "outgoing" name means edges go from i→k and j→k. Uses sigmoid gating on both the projections and the final output.',
    details: [
      'Project & gate: a_ij = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij), similarly for b_ij',
      'Aggregate triangles: Σ_k a_ik ⊙ b_jk — outgoing edges from i and j',
      'Output gating: z̃_ij = g_ij ⊙ Linear(LayerNorm(Σ_k a_ik ⊙ b_jk))',
      'g_ij = sigmoid(Linear(z_ij)) — learned gate controls information flow',
      'Key geometric bias: if dist(i,k) and dist(j,k) both small → dist(i,j) likely small',
    ],
    formula: '\\tilde{\\mathbf{z}}_{ij} = \\mathbf{g}_{ij} \\odot \\text{Linear}\\!\\left(\\text{LayerNorm}\\!\\left(\\sum_k \\mathbf{a}_{ik} \\odot \\mathbf{b}_{jk}\\right)\\right)',
    algRef: 'Algorithm 11',
  },
  {
    id: 4,
    name: 'Triangle Multiplicative Update (Incoming)',
    short: '△ Mult In',
    category: 'pair',
    description:
      'The mirror of outgoing: edges come into i and j from a common source k. Multiplies pair(k,i) × pair(k,j). The subscript order flips: a_ki ⊙ b_kj instead of a_ik ⊙ b_jk. Together with outgoing, this enforces full triangle consistency — "friends of friends are friends".',
    details: [
      'Same projection & gating as outgoing, but different edge direction',
      'Aggregate: Σ_k a_ki ⊙ b_kj — incoming edges to i and j from k',
      'Subscript difference is crucial: outgoing sums over columns, incoming sums over rows',
      'With DropoutRowwise(0.25)',
    ],
    formula: '\\tilde{\\mathbf{z}}_{ij} = \\mathbf{g}_{ij} \\odot \\text{Linear}\\!\\left(\\text{LayerNorm}\\!\\left(\\sum_k \\mathbf{a}_{ki} \\odot \\mathbf{b}_{kj}\\right)\\right)',
    algRef: 'Algorithm 12',
  },
  {
    id: 5,
    name: 'Triangle Self-Attention (Starting Node)',
    short: '△ Attn Start',
    category: 'pair',
    description:
      'Attention over the pair matrix: fix starting node i, let ending node j attend to all k via pair(i,k). The attention is biased by the third triangle edge pair(j,k), providing geometric context. More expressive than multiplicative updates — learns complex relational patterns.',
    details: [
      'Fix row i: pair[i,j] attends to pair[i,k] for all k',
      'Attention: a_ijk = softmax_k(q_ij · k_ik / √c + b_jk)',
      'Triangle bias b_jk = LinearNoBias(z_jk) — third edge of triangle',
      'Gated: g_ij = sigmoid(Linear(z_ij))',
      'N_head = 4, c = 32, with DropoutRowwise(0.25)',
    ],
    formula: '\\mathbf{o}_{ij}^h = \\mathbf{g}_{ij}^h \\odot \\sum_k \\text{softmax}_k\\!\\left(\\frac{1}{\\sqrt{c}}\\,\\mathbf{q}_{ij}^{h\\top}\\mathbf{k}_{ik}^h + b_{jk}^h\\right) \\mathbf{v}_{ik}^h',
    algRef: 'Algorithm 13',
  },
  {
    id: 6,
    name: 'Triangle Self-Attention (Ending Node)',
    short: '△ Attn End',
    category: 'pair',
    description:
      'The transpose: fix ending node j, let starting node i attend to all k via pair(k,j). Bias comes from pair(i,k). Together with Starting, this provides full bidirectional attention flow through the pair matrix with geometric triangle constraints.',
    details: [
      'Fix col j: pair[i,j] attends to pair[k,j] for all k',
      'Attention: a_ijk = softmax_k(q_ij · k_kj / √c + b_ik)',
      'Transpose of Starting: swaps row/column roles',
      'With DropoutColumnwise(0.25) — note: columnwise, not rowwise',
    ],
    formula: '\\mathbf{o}_{ij}^h = \\mathbf{g}_{ij}^h \\odot \\sum_k \\text{softmax}_k\\!\\left(\\frac{1}{\\sqrt{c}}\\,\\mathbf{q}_{ij}^{h\\top}\\mathbf{k}_{kj}^h + b_{ik}^h\\right) \\mathbf{v}_{kj}^h',
    algRef: 'Algorithm 14',
  },
  {
    id: 7,
    name: 'Pair Transition (Feed-Forward)',
    short: 'Pair FFN',
    category: 'pair',
    description:
      'A 2-layer MLP applied independently to each pair(i,j). Expands channels by 4× (128 → 512), applies ReLU, then compresses back. Adds non-linear transformation capacity after the attention and multiplicative updates. Followed by residual connection.',
    details: [
      'LayerNorm → Linear (c_z → 4·c_z) → ReLU → Linear (4·c_z → c_z)',
      'Applied per-position: no cross-(i,j) interaction',
      'Residual: z_ij += PairTransition(z_ij)',
      'The 4× expansion is standard "bottleneck" design from transformers',
    ],
    formula: '\\mathbf{m}_{si} \\leftarrow \\text{Linear}\\!\\left(\\text{relu}\\!\\left(\\text{Linear}\\!\\left(\\text{LayerNorm}(\\mathbf{m}_{si})\\right)\\right)\\right)',
    algRef: 'Algorithm 15',
  },
]

// ── Dark theme colors ─────────────────────────────────

const DARK = {
  bg: '#0a0a15',
  surface: 'rgba(20, 20, 40, 0.95)',
  surfaceLight: 'rgba(30, 30, 55, 0.8)',
  border: 'rgba(255, 255, 255, 0.08)',
  borderLight: 'rgba(255, 255, 255, 0.12)',
  text: '#e0e0e0',
  textMuted: '#8899aa',
  textDim: '#556677',
}

const CAT_COLORS = {
  msa: { bg: 'rgba(230, 81, 0, 0.12)', border: '#e65100', text: '#ffab40', glow: '#ff6d00', accent: '#ff9100' },
  pair: { bg: 'rgba(21, 101, 192, 0.12)', border: '#1565c0', text: '#64b5f6', glow: '#2196f3', accent: '#42a5f5' },
  cross: { bg: 'rgba(123, 31, 162, 0.12)', border: '#7b1fa2', text: '#ce93d8', glow: '#9c27b0', accent: '#ab47bc' },
}

// ── Pair matrix visualization ──────────────────────────

const N = 8

function PairMatrixViz({
  activeOp, highlightI, highlightJ, highlightK,
}: {
  activeOp: BlockOp
  highlightI: number
  highlightJ: number
  highlightK: number
}) {
  const cellSize = 38
  const matrixSize = N * cellSize
  const ox = 35
  const oy = 35

  const residueLabels = ['A', 'G', 'V', 'L', 'I', 'P', 'F', 'W']

  const pairValues: number[][] = []
  for (let i = 0; i < N; i++) {
    pairValues.push([])
    for (let j = 0; j < N; j++) {
      const dist = Math.abs(i - j)
      pairValues[i][j] = Math.max(0.05, Math.exp(-dist * 0.4) + Math.sin(i * 0.7 + j * 0.5) * 0.2)
    }
  }

  const isTriangleOp = activeOp.id >= 3 && activeOp.id <= 6
  const isOutgoing = activeOp.id === 3
  const isIncoming = activeOp.id === 4
  const isAttnStart = activeOp.id === 5
  const isAttnEnd = activeOp.id === 6

  const getTriangleEdges = (i: number, j: number, k: number) => {
    if (isOutgoing) return [[i, k], [j, k]]
    if (isIncoming) return [[k, i], [k, j]]
    if (isAttnStart) return [[i, k], [j, k]]
    if (isAttnEnd) return [[k, j], [i, k]]
    return []
  }

  const triangleEdges = isTriangleOp ? getTriangleEdges(highlightI, highlightJ, highlightK) : []

  return (
    <svg viewBox={`0 0 ${matrixSize + ox + 65} ${matrixSize + oy + 90}`}
      style={{ width: '100%', maxWidth: 560, height: 'auto' }}>
      <defs>
        <filter id="cellGlow">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="triGlow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {/* Column labels */}
      {residueLabels.map((l, i) => (
        <text key={`cl-${i}`} x={ox + i * cellSize + cellSize / 2} y={oy - 10}
          textAnchor="middle" fontSize={11} fill={i === highlightJ ? '#ff9100' : '#667788'}
          fontWeight={i === highlightJ ? 700 : 400} fontFamily="JetBrains Mono, monospace">
          {l}<tspan fontSize={8} dy={2}>{i}</tspan>
        </text>
      ))}

      {/* Row labels */}
      {residueLabels.map((l, i) => (
        <text key={`rl-${i}`} x={ox - 10} y={oy + i * cellSize + cellSize / 2 + 4}
          textAnchor="end" fontSize={11} fill={i === highlightI ? '#ff9100' : '#667788'}
          fontWeight={i === highlightI ? 700 : 400} fontFamily="JetBrains Mono, monospace">
          {l}<tspan fontSize={8} dy={2}>{i}</tspan>
        </text>
      ))}

      {/* Axis labels */}
      <text x={ox + matrixSize / 2} y={oy - 24} textAnchor="middle"
        fontSize={10} fill="#556677" fontFamily="Inter, sans-serif">j →</text>
      <text x={ox - 28} y={oy + matrixSize / 2} textAnchor="middle"
        fontSize={10} fill="#556677" fontFamily="Inter, sans-serif"
        transform={`rotate(-90, ${ox - 28}, ${oy + matrixSize / 2})`}>i →</text>

      {/* Matrix cells */}
      {Array.from({ length: N }).map((_, r) =>
        Array.from({ length: N }).map((_, c) => {
          const isTarget = r === highlightI && c === highlightJ
          const isTriEdge = triangleEdges.some(e => e[0] === r && e[1] === c)
          const val = pairValues[r][c]

          // Cyan-to-magenta colormap on dark background
          const intensity = val * 0.6
          let fill = `rgba(0, 188, 212, ${intensity})`
          if (val > 0.5) {
            const t = (val - 0.5) * 2
            fill = `rgba(${Math.round(224 * t)}, ${Math.round(64 + 124 * (1 - t))}, ${Math.round(251 * t + 212 * (1 - t))}, ${intensity + 0.1})`
          }

          let strokeColor = 'rgba(255,255,255,0.06)'
          let strokeW = 0.5

          if (isTarget) {
            fill = 'rgba(255, 145, 0, 0.6)'
            strokeColor = '#ff9100'
            strokeW = 2
          } else if (isTriEdge) {
            fill = 'rgba(76, 175, 80, 0.45)'
            strokeColor = '#4caf50'
            strokeW = 1.5
          }

          return (
            <g key={`c-${r}-${c}`}>
              <rect
                x={ox + c * cellSize + 1} y={oy + r * cellSize + 1}
                width={cellSize - 2} height={cellSize - 2}
                fill={fill} stroke={strokeColor} strokeWidth={strokeW} rx={3}
              />
              {isTarget && (
                <rect
                  x={ox + c * cellSize - 1} y={oy + r * cellSize - 1}
                  width={cellSize + 2} height={cellSize + 2}
                  fill="none" stroke="#ff9100" strokeWidth={2} rx={4}
                  filter="url(#cellGlow)"
                >
                  <animate attributeName="stroke-opacity" values="1;0.3;1" dur="1.5s" repeatCount="indefinite" />
                </rect>
              )}
            </g>
          )
        })
      )}

      {/* Triangle lines */}
      {isTriangleOp && triangleEdges.length === 2 && (() => {
        const tx = ox + highlightJ * cellSize + cellSize / 2
        const ty = oy + highlightI * cellSize + cellSize / 2
        const e1x = ox + triangleEdges[0][1] * cellSize + cellSize / 2
        const e1y = oy + triangleEdges[0][0] * cellSize + cellSize / 2
        const e2x = ox + triangleEdges[1][1] * cellSize + cellSize / 2
        const e2y = oy + triangleEdges[1][0] * cellSize + cellSize / 2
        return (
          <g filter="url(#triGlow)">
            <polygon
              points={`${tx},${ty} ${e1x},${e1y} ${e2x},${e2y}`}
              fill="#4caf50" opacity={0.05}
            >
              <animate attributeName="opacity" values="0.03;0.1;0.03" dur="2s" repeatCount="indefinite" />
            </polygon>

            {[[e1x, e1y, tx, ty], [e2x, e2y, tx, ty], [e1x, e1y, e2x, e2y]].map(([x1, y1, x2, y2], i) => (
              <g key={`edge-${i}`}>
                <line x1={x1} y1={y1} x2={x2} y2={y2}
                  stroke="#4caf50" strokeWidth={i < 2 ? 2 : 1.5}
                  strokeDasharray={i < 2 ? undefined : '3,3'}
                  opacity={i < 2 ? 0.7 : 0.4} />
              </g>
            ))}

            {/* Particles flowing to target */}
            {[0, 0.5, 1.0].map((delay, pi) => (
              <circle key={`p1-${pi}`} r={3.5} fill="#4caf50" opacity={0.85}>
                <animateMotion dur="1.2s" begin={`${delay}s`} repeatCount="indefinite"
                  path={`M${e1x},${e1y} L${tx},${ty}`} />
              </circle>
            ))}
            {[0.2, 0.7, 1.2].map((delay, pi) => (
              <circle key={`p2-${pi}`} r={3.5} fill="#66bb6a" opacity={0.85}>
                <animateMotion dur="1.2s" begin={`${delay}s`} repeatCount="indefinite"
                  path={`M${e2x},${e2y} L${tx},${ty}`} />
              </circle>
            ))}

            {/* Impact burst */}
            <circle cx={tx} cy={ty} r={6} fill="none" stroke="#ffab00" strokeWidth={1.5}>
              <animate attributeName="r" values="4;14;4" dur="1.5s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.7;0;0.7" dur="1.5s" repeatCount="indefinite" />
            </circle>
          </g>
        )
      })()}

      {/* Gating indicator */}
      {isTriangleOp && (
        <g transform={`translate(${ox + matrixSize + 8}, ${oy + highlightI * cellSize})`}>
          <rect x={0} y={0} width={50} height={cellSize - 2} rx={4}
            fill="rgba(255, 145, 0, 0.1)" stroke="#ff9100" strokeWidth={1} strokeDasharray="3,2" />
          <text x={25} y={14} textAnchor="middle" fontSize={8} fill="#ffab40" fontFamily="Inter, sans-serif">
            σ gate
          </text>
          <text x={25} y={28} textAnchor="middle" fontSize={10} fill="#ff9100" fontWeight={700} fontFamily="Inter, sans-serif">
            g_ij
          </text>
        </g>
      )}

      {/* Legend */}
      <g transform={`translate(${ox}, ${oy + matrixSize + 14})`}>
        <rect x={0} y={0} width={12} height={12} fill="rgba(255, 145, 0, 0.6)" rx={2} />
        <text x={16} y={10} fontSize={10} fill="#8899aa" fontFamily="Inter, sans-serif">Target (i,j)</text>

        <rect x={110} y={0} width={12} height={12} fill="rgba(76, 175, 80, 0.45)" rx={2} />
        <text x={126} y={10} fontSize={10} fill="#8899aa" fontFamily="Inter, sans-serif">Triangle edges</text>

        <rect x={240} y={0} width={12} height={12} fill="rgba(0, 188, 212, 0.35)" rx={2} />
        <text x={256} y={10} fontSize={10} fill="#8899aa" fontFamily="Inter, sans-serif">Pair value</text>
      </g>

      {/* Edge direction label */}
      {isTriangleOp && (
        <g transform={`translate(${ox}, ${oy + matrixSize + 34})`}>
          <text x={0} y={12} fontSize={10} fill="#4caf50" fontWeight={600} fontFamily="JetBrains Mono, monospace">
            {isOutgoing ? 'Σ_k a_ik ⊙ b_jk  (outgoing)' :
             isIncoming ? 'Σ_k a_ki ⊙ b_kj  (incoming)' :
             isAttnStart ? 'softmax_k(q_ij·k_ik + b_jk)  (start)' :
             'softmax_k(q_ij·k_kj + b_ik)  (end)'}
          </text>
        </g>
      )}

      {/* k indicator */}
      {isTriangleOp && (
        <text x={ox + matrixSize + 10} y={oy + highlightK * cellSize + cellSize / 2 + 4}
          fontSize={12} fill="#4caf50" fontWeight={700} fontFamily="Inter, sans-serif">
          ← k={highlightK}
        </text>
      )}
    </svg>
  )
}

// ── MSA matrix visualization ───────────────────────────

function MSAMatrixViz({ activeOp }: { activeOp: BlockOp }) {
  const nSeq = 5
  const nRes = 8
  const cellW = 38
  const cellH = 26
  const ox = 45
  const oy = 32

  const isRow = activeOp.id === 0
  const residueLabels = ['A', 'G', 'V', 'L', 'I', 'P', 'F', 'W']

  return (
    <svg viewBox={`0 0 ${nRes * cellW + ox + 40} ${nSeq * cellH + oy + 80}`}
      style={{ width: '100%', maxWidth: 560, height: 'auto' }}>

      <text x={ox + nRes * cellW / 2} y={14} textAnchor="middle" fontSize={10} fill="#667788" fontFamily="Inter, sans-serif">
        Residue position →
      </text>
      <text x={14} y={oy + nSeq * cellH / 2} textAnchor="middle" fontSize={10} fill="#667788"
        fontFamily="Inter, sans-serif" transform={`rotate(-90, 14, ${oy + nSeq * cellH / 2})`}>
        Sequences →
      </text>

      {residueLabels.map((l, i) => (
        <text key={`c-${i}`} x={ox + i * cellW + cellW / 2} y={oy - 6}
          textAnchor="middle" fontSize={10} fill="#667788" fontFamily="JetBrains Mono, monospace">{l}{i}</text>
      ))}

      {Array.from({ length: nSeq }).map((_, i) => (
        <text key={`r-${i}`} x={ox - 6} y={oy + i * cellH + cellH / 2 + 4}
          textAnchor="end" fontSize={10} fill="#667788" fontFamily="JetBrains Mono, monospace">s{i}</text>
      ))}

      {Array.from({ length: nSeq }).map((_, s) =>
        Array.from({ length: nRes }).map((_, r) => {
          const val = 0.25 + Math.sin(s * 1.2 + r * 0.8) * 0.25
          const colors = ['#c62828', '#1565c0', '#e65100', '#6a1b9a', '#f9a825']
          const color = colors[(s * 3 + r * 7) % colors.length]

          return (
            <rect key={`${s}-${r}`}
              x={ox + r * cellW + 1} y={oy + s * cellH + 1}
              width={cellW - 2} height={cellH - 2}
              fill={color} opacity={val} rx={2}
            />
          )
        })
      )}

      {/* Attention pattern overlay */}
      {isRow ? (
        <g>
          <rect x={ox - 2} y={oy + 2 * cellH - 2} width={nRes * cellW + 4} height={cellH + 4}
            fill="none" stroke="#ff9100" strokeWidth={2} rx={4}>
            <animate attributeName="stroke-opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite" />
          </rect>
          {[0, 1, 3, 5, 7].map(r => (
            <line key={`a-${r}`}
              x1={ox + 4 * cellW + cellW / 2} y1={oy + 2 * cellH + cellH / 2}
              x2={ox + r * cellW + cellW / 2} y2={oy + 2 * cellH + cellH / 2}
              stroke="#ff9100" strokeWidth={1.5} opacity={0.5} markerEnd="url(#arrowOrangeDk)" />
          ))}

          {/* Pair bias indicator */}
          <g transform={`translate(${ox + nRes * cellW + 6}, ${oy + 2 * cellH})`}>
            <rect x={0} y={-4} width={28} height={cellH + 8} rx={4}
              fill="rgba(33, 150, 243, 0.15)" stroke="#42a5f5" strokeWidth={1} />
            <text x={14} y={10} textAnchor="middle" fontSize={7} fill="#64b5f6" fontFamily="Inter, sans-serif">pair</text>
            <text x={14} y={22} textAnchor="middle" fontSize={7} fill="#64b5f6" fontFamily="Inter, sans-serif">bias</text>
          </g>
        </g>
      ) : (
        <g>
          <rect x={ox + 3 * cellW - 2} y={oy - 2} width={cellW + 4} height={nSeq * cellH + 4}
            fill="none" stroke="#42a5f5" strokeWidth={2} rx={4}>
            <animate attributeName="stroke-opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite" />
          </rect>
          {[0, 1, 3, 4].map(s => (
            <line key={`a-${s}`}
              x1={ox + 3 * cellW + cellW / 2} y1={oy + 2 * cellH + cellH / 2}
              x2={ox + 3 * cellW + cellW / 2} y2={oy + s * cellH + cellH / 2}
              stroke="#42a5f5" strokeWidth={1.5} opacity={0.5} />
          ))}
          <text x={ox + 3 * cellW + cellW / 2} y={oy + nSeq * cellH + 16}
            textAnchor="middle" fontSize={10} fill="#42a5f5" fontWeight={600} fontFamily="Inter, sans-serif">
            ↕ co-evolution
          </text>
        </g>
      )}

      {/* Gating annotation */}
      <g transform={`translate(${ox}, ${oy + nSeq * cellH + 28})`}>
        <rect x={0} y={0} width={nRes * cellW} height={24} rx={4}
          fill="rgba(255, 255, 255, 0.03)" stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
        <text x={nRes * cellW / 2} y={16} textAnchor="middle" fontSize={10} fill="#8899aa" fontFamily="Inter, sans-serif">
          g = σ(Linear(m)) ⊙ output — sigmoid gating controls info flow
        </text>
      </g>

      <defs>
        <marker id="arrowOrangeDk" markerWidth={6} markerHeight={6} refX={5} refY={3} orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#ff9100" opacity={0.7} />
        </marker>
      </defs>
    </svg>
  )
}

// ── Outer product visualization ────────────────────────

function OuterProductViz() {
  return (
    <svg viewBox="0 0 440 310" style={{ width: '100%', maxWidth: 560, height: 'auto' }}>
      <text x={50} y={18} textAnchor="middle" fontSize={11} fill="#ce93d8" fontWeight={600} fontFamily="Inter, sans-serif">
        MSA[:,i]
      </text>
      {Array.from({ length: 5 }).map((_, s) => (
        <rect key={`i-${s}`} x={30} y={26 + s * 22} width={40} height={18}
          fill="#7b1fa2" opacity={0.3 + s * 0.08} rx={3} stroke="#9c27b0" strokeWidth={0.5} />
      ))}

      <text x={160} y={18} textAnchor="middle" fontSize={11} fill="#ce93d8" fontWeight={600} fontFamily="Inter, sans-serif">
        MSA[:,j]
      </text>
      {Array.from({ length: 5 }).map((_, s) => (
        <rect key={`j-${s}`} x={140} y={26 + s * 22} width={40} height={18}
          fill="#7b1fa2" opacity={0.3 + s * 0.08} rx={3} stroke="#9c27b0" strokeWidth={0.5} />
      ))}

      <text x={105} y={82} textAnchor="middle" fontSize={18} fill="#ce93d8" fontFamily="Inter, sans-serif">⊗</text>
      <line x1={105} y1={92} x2={105} y2={125} stroke="#9c27b0" strokeWidth={1.5} />
      <text x={130} y={112} fontSize={9} fill="#667788" fontFamily="Inter, sans-serif">mean over s</text>

      <rect x={75} y={130} width={60} height={55} fill="rgba(33, 150, 243, 0.15)" stroke="#42a5f5" strokeWidth={1.5} rx={4} />
      <text x={105} y={162} textAnchor="middle" fontSize={10} fill="#64b5f6" fontWeight={600} fontFamily="JetBrains Mono, monospace">
        z_ij
      </text>

      {Array.from({ length: 5 }).map((_, s) => (
        <g key={`p-${s}`}>
          <circle r={3} fill="#9c27b0" opacity={0.7}>
            <animateMotion dur="1.6s" begin={`${s * 0.25}s`} repeatCount="indefinite"
              path={`M50,${35 + s * 22} L105,${140}`} />
          </circle>
          <circle r={3} fill="#9c27b0" opacity={0.7}>
            <animateMotion dur="1.6s" begin={`${s * 0.25 + 0.12}s`} repeatCount="indefinite"
              path={`M160,${35 + s * 22} L105,${140}`} />
          </circle>
        </g>
      ))}

      <circle cx={105} cy={140} r={6} fill="none" stroke="#9c27b0" strokeWidth={1.5}>
        <animate attributeName="r" values="6;16;6" dur="2s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.5;0;0.5" dur="2s" repeatCount="indefinite" />
      </circle>

      <text x={105} y={210} textAnchor="middle" fontSize={11} fill="#667788" fontFamily="Inter, sans-serif">
        Co-evolution signal → Pair feature
      </text>

      {/* Pipeline diagram */}
      <g transform="translate(220, 30)">
        <text x={90} y={0} textAnchor="middle" fontSize={10} fill="#8899aa" fontWeight={600} fontFamily="Inter, sans-serif">
          Data flow (Alg 10)
        </text>
        {[
          { y: 14, label: 'LayerNorm(m_si)', color: '#7b1fa2' },
          { y: 44, label: 'a = Linear(m), b = Linear(m)', color: '#9c27b0' },
          { y: 74, label: 'a_si ⊗ b_sj (outer product)', color: '#ab47bc' },
          { y: 104, label: 'mean over sequences s', color: '#ce93d8' },
          { y: 134, label: 'flatten → Linear → z_ij', color: '#42a5f5' },
        ].map((step, i) => (
          <g key={i}>
            <rect x={0} y={step.y} width={180} height={22} rx={4}
              fill={`${step.color}15`} stroke={step.color} strokeWidth={0.8} />
            <text x={90} y={step.y + 15} textAnchor="middle" fontSize={8.5}
              fill={step.color} fontFamily="JetBrains Mono, monospace">{step.label}</text>
            {i < 4 && (
              <line x1={90} y1={step.y + 24} x2={90} y2={step.y + 28}
                stroke="#556677" strokeWidth={1} />
            )}
          </g>
        ))}
      </g>
    </svg>
  )
}

// ── Triangle 3D intuition diagram ─────────────────────

function TriangleIntuition({ highlightK, activeOp }: { highlightK: number; activeOp: BlockOp }) {
  const isOutgoing = activeOp.id === 3
  const isIncoming = activeOp.id === 4
  const points = {
    i: { x: 80, y: 200, label: 'i', color: '#ff9100' },
    j: { x: 280, y: 200, label: 'j', color: '#42a5f5' },
    k: { x: 180, y: 60, label: `k=${highlightK}`, color: '#4caf50' },
  }

  // Arrow direction depends on outgoing vs incoming
  const edgeLabel1 = isOutgoing ? 'a(i,k)' : isIncoming ? 'a(k,i)' : 'q(i,j)→k(i,k)'
  const edgeLabel2 = isOutgoing ? 'b(j,k)' : isIncoming ? 'b(k,j)' : 'bias: b(j,k)'

  return (
    <svg viewBox="0 0 360 290" style={{ width: '100%', maxWidth: 420, height: 'auto' }}>
      <text x={180} y={18} textAnchor="middle" fontSize={12} fill="#c0c8d0" fontWeight={600} fontFamily="Inter, sans-serif">
        Triangle in residue space
      </text>

      {/* Edges */}
      <line x1={points.i.x} y1={points.i.y} x2={points.j.x} y2={points.j.y}
        stroke="#ff9100" strokeWidth={2.5} opacity={0.6} />
      <line x1={points.i.x} y1={points.i.y} x2={points.k.x} y2={points.k.y}
        stroke="#4caf50" strokeWidth={2} strokeDasharray="5,3" opacity={0.6} />
      <line x1={points.j.x} y1={points.j.y} x2={points.k.x} y2={points.k.y}
        stroke="#4caf50" strokeWidth={2} strokeDasharray="5,3" opacity={0.6} />

      {/* Edge labels */}
      <text x={180} y={230} textAnchor="middle" fontSize={9} fill="#ff9100" fontFamily="JetBrains Mono, monospace">
        target: z(i,j)
      </text>
      <text x={108} y={125} textAnchor="middle" fontSize={8} fill="#4caf50" fontFamily="JetBrains Mono, monospace"
        transform="rotate(-50, 108, 125)">{edgeLabel1}</text>
      <text x={252} y={125} textAnchor="middle" fontSize={8} fill="#4caf50" fontFamily="JetBrains Mono, monospace"
        transform="rotate(50, 252, 125)">{edgeLabel2}</text>

      {/* Particles */}
      <circle r={3.5} fill="#4caf50" opacity={0.8}>
        <animateMotion dur="2s" repeatCount="indefinite"
          path={`M${points.k.x},${points.k.y} L${points.i.x},${points.i.y} L${points.j.x},${points.j.y}`} />
      </circle>
      <circle r={3.5} fill="#66bb6a" opacity={0.8}>
        <animateMotion dur="2s" repeatCount="indefinite" begin="1s"
          path={`M${points.k.x},${points.k.y} L${points.j.x},${points.j.y} L${points.i.x},${points.i.y}`} />
      </circle>

      {/* Nodes */}
      {Object.values(points).map(p => (
        <g key={p.label}>
          <circle cx={p.x} cy={p.y} r={18} fill={p.color} opacity={0.1} />
          <circle cx={p.x} cy={p.y} r={11} fill={p.color} opacity={0.75} />
          <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={10} fill="#fff"
            fontWeight={700} fontFamily="Inter, sans-serif">{p.label}</text>
        </g>
      ))}

      <text x={180} y={265} textAnchor="middle" fontSize={10} fill="#556677" fontStyle="italic" fontFamily="Inter, sans-serif">
        "if i↔k close and j↔k close → i↔j likely close"
      </text>
    </svg>
  )
}

// ── FFN visualization ─────────────────────────────────

function FFNVisualization() {
  return (
    <div style={{ display: 'flex', gap: 28, alignItems: 'center', padding: 16 }}>
      <svg viewBox="0 0 280 440" style={{ width: 280, height: 'auto' }}>
        <defs>
          <marker id="arrCyan" markerWidth={6} markerHeight={6} refX={5} refY={3} orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#42a5f5" />
          </marker>
        </defs>

        {/* Input */}
        <text x={140} y={16} textAnchor="middle" fontSize={12} fontWeight={600}
          fill="#64b5f6" fontFamily="JetBrains Mono, monospace">z_ij</text>
        <text x={140} y={30} textAnchor="middle" fontSize={9}
          fill="#556677" fontFamily="Inter, sans-serif">128 channels</text>

        {/* LayerNorm */}
        <rect x={60} y={38} width={160} height={28} rx={6}
          fill="rgba(126, 87, 194, 0.15)" stroke="#7e57c2" strokeWidth={1} />
        <text x={140} y={56} textAnchor="middle" fontSize={11} fontWeight={600}
          fill="#b39ddb" fontFamily="Inter, sans-serif">LayerNorm</text>

        <line x1={140} y1={68} x2={140} y2={86} stroke="#42a5f5" strokeWidth={1.5} markerEnd="url(#arrCyan)" />

        {/* Linear expand */}
        <rect x={30} y={90} width={220} height={38} rx={6}
          fill="rgba(33, 150, 243, 0.1)" stroke="#42a5f5" strokeWidth={1.5} />
        <text x={140} y={108} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#64b5f6" fontFamily="Inter, sans-serif">Linear</text>
        <text x={140} y={122} textAnchor="middle" fontSize={9}
          fill="#42a5f5" fontFamily="Inter, sans-serif">128 → 512 (4× expand)</text>

        <line x1={140} y1={130} x2={140} y2={148} stroke="#42a5f5" strokeWidth={1.5} markerEnd="url(#arrCyan)" />

        {/* ReLU */}
        <rect x={50} y={152} width={180} height={52} rx={6}
          fill="rgba(249, 168, 37, 0.08)" stroke="#f9a825" strokeWidth={1.5} />
        <text x={140} y={170} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#fbc02d" fontFamily="Inter, sans-serif">ReLU</text>
        <g transform="translate(80, 178)">
          <line x1={0} y1={16} x2={60} y2={16} stroke="rgba(255,255,255,0.1)" strokeWidth={0.5} />
          <line x1={30} y1={0} x2={30} y2={22} stroke="rgba(255,255,255,0.1)" strokeWidth={0.5} />
          <line x1={5} y1={16} x2={30} y2={16} stroke="#ef5350" strokeWidth={2} strokeLinecap="round" />
          <line x1={30} y1={16} x2={55} y2={2} stroke="#4caf50" strokeWidth={2} strokeLinecap="round" />
          <circle r={3} fill="#fbc02d" opacity={0.8}>
            <animateMotion dur="2.5s" repeatCount="indefinite" path="M5,16 L30,16 L55,2" />
          </circle>
        </g>

        <line x1={140} y1={206} x2={140} y2={224} stroke="#42a5f5" strokeWidth={1.5} markerEnd="url(#arrCyan)" />

        {/* Linear compress */}
        <rect x={30} y={228} width={220} height={38} rx={6}
          fill="rgba(33, 150, 243, 0.1)" stroke="#42a5f5" strokeWidth={1.5} />
        <text x={140} y={246} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#64b5f6" fontFamily="Inter, sans-serif">Linear</text>
        <text x={140} y={260} textAnchor="middle" fontSize={9}
          fill="#42a5f5" fontFamily="Inter, sans-serif">512 → 128 (compress)</text>

        <line x1={140} y1={268} x2={140} y2={296} stroke="#42a5f5" strokeWidth={1.5} markerEnd="url(#arrCyan)" />

        {/* Residual skip */}
        <path d="M28,52 Q8,52 8,195 Q8,308 130,308"
          fill="none" stroke="#556677" strokeWidth={1.5} strokeDasharray="5,3" />
        <circle r={2.5} fill="#8899aa" opacity={0.6}>
          <animateMotion dur="3s" repeatCount="indefinite"
            path="M28,52 Q8,52 8,195 Q8,308 130,308" />
        </circle>

        {/* Plus */}
        <circle cx={140} cy={310} r={12} fill="rgba(33, 150, 243, 0.15)" stroke="#42a5f5" strokeWidth={1.5} />
        <line x1={134} y1={310} x2={146} y2={310} stroke="#42a5f5" strokeWidth={2} />
        <line x1={140} y1={304} x2={140} y2={316} stroke="#42a5f5" strokeWidth={2} />
        <text x={165} y={314} fontSize={9} fill="#556677" fontFamily="JetBrains Mono, monospace">
          x + FFN(x)
        </text>

        <line x1={140} y1={324} x2={140} y2={342} stroke="#42a5f5" strokeWidth={1.5} markerEnd="url(#arrCyan)" />

        {/* Output */}
        <text x={140} y={358} textAnchor="middle" fontSize={12} fontWeight={600}
          fill="#64b5f6" fontFamily="JetBrains Mono, monospace">z_ij'</text>
        <text x={140} y={372} textAnchor="middle" fontSize={9}
          fill="#556677" fontFamily="Inter, sans-serif">updated</text>

        {/* Flow particle */}
        <circle r={2.5} fill="#42a5f5" opacity={0.5}>
          <animateMotion dur="3.5s" repeatCount="indefinite"
            path="M140,38 L140,90 L140,152 L140,228 L140,268 L140,296 L140,340" />
        </circle>
      </svg>

      <div style={{ maxWidth: 200 }}>
        <h4 style={{ margin: '0 0 10px', fontSize: 13, fontWeight: 700, color: '#c0c8d0', fontFamily: 'Inter, sans-serif' }}>
          Why expand then compress?
        </h4>
        {[
          { bg: 'rgba(33, 150, 243, 0.08)', border: '#1565c0', color: '#64b5f6',
            text: 'The "bottleneck" lets the network learn complex feature interactions in expanded 512-dim space, then distill back to 128.' },
          { bg: 'rgba(249, 168, 37, 0.06)', border: '#f9a825', color: '#fbc02d',
            text: 'ReLU creates sparse activations — each pair feature only activates a subset of hidden units.' },
          { bg: 'rgba(126, 87, 194, 0.08)', border: '#7e57c2', color: '#b39ddb',
            text: 'LayerNorm stabilizes training by normalizing to zero mean, unit variance.' },
        ].map((item, i) => (
          <div key={i} style={{
            padding: '8px 10px', borderRadius: 6, marginBottom: 8,
            background: item.bg, border: `1px solid ${item.border}40`,
            fontSize: 11, color: item.color, lineHeight: 1.5,
            fontFamily: 'Inter, sans-serif',
          }}>
            {item.text}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Computational intensity ───────────────────────────

const HEAT_VALUES: Record<number, number> = {
  0: 0.7, 1: 0.5, 2: 0.6, 3: 0.95, 4: 0.9, 5: 0.85, 6: 0.85, 7: 0.3,
}

function heatColor(heat: number): string {
  if (heat < 0.33) {
    const t = heat / 0.33
    return `rgb(${Math.round(66 + t * 100)},${Math.round(165 - t * 40)},${Math.round(245 - t * 130)})`
  } else if (heat < 0.66) {
    const t = (heat - 0.33) / 0.33
    return `rgb(${Math.round(166 + t * 64)},${Math.round(125 - t * 50)},${Math.round(115 - t * 85)})`
  }
  const t = (heat - 0.66) / 0.34
  return `rgb(${Math.round(230 + t * 15)},${Math.round(75 - t * 40)},${Math.round(30 - t * 20)})`
}

// ── Step flow sidebar ──────────────────────────────────

function StepConnector({ done, active }: { done: boolean; active: boolean }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'center', padding: '0 0 0 22px', height: 14,
    }}>
      <div style={{
        width: 1.5, height: '100%',
        background: done ? '#42a5f5' : active ? 'rgba(66, 165, 245, 0.4)' : 'rgba(255,255,255,0.06)',
        transition: 'background 0.3s',
        position: 'relative' as const,
      }}>
        {active && (
          <div style={{
            position: 'absolute', left: -3, top: 2,
            width: 7, height: 7, borderRadius: '50%',
            background: '#42a5f5',
            animation: 'evoPulse 1.5s infinite',
          }} />
        )}
      </div>
    </div>
  )
}

function BlockList({ ops, activeId, onSelect }: {
  ops: BlockOp[]; activeId: number; onSelect: (id: number) => void
}) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', padding: '8px 0',
    }}>
      <style>{`
        @keyframes evoPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(0.6); }
        }
        @keyframes evoNext {
          0%, 100% { transform: translateX(0); opacity: 0.6; }
          50% { transform: translateX(3px); opacity: 1; }
        }
        @keyframes evoFadeIn {
          from { opacity: 0; transform: translateY(8px) scale(0.97); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
      `}</style>

      <div style={{
        fontSize: 10, color: '#556677', textTransform: 'uppercase', letterSpacing: 1.5,
        padding: '0 12px 8px', fontFamily: 'Inter, sans-serif',
      }}>
        One Evoformer Block
      </div>

      <div style={{
        padding: '4px 12px', fontSize: 9, color: '#556677',
        display: 'flex', alignItems: 'center', gap: 5,
        fontFamily: 'Inter, sans-serif',
      }}>
        <span style={{ fontSize: 10, opacity: 0.6 }}>→</span> m_si + z_ij in
      </div>
      <StepConnector done={activeId >= 0} active={activeId === 0} />

      {ops.map((op, idx) => {
        const cat = CAT_COLORS[op.category]
        const active = op.id === activeId
        const done = op.id < activeId
        const isNext = op.id === activeId + 1
        const heat = HEAT_VALUES[op.id] ?? 0.5
        const hc = heatColor(heat)

        return (
          <div key={op.id}>
            <div
              onClick={() => onSelect(op.id)}
              style={{
                padding: '5px 8px 5px 5px',
                borderRadius: 6,
                cursor: 'pointer',
                background: active ? cat.bg : done ? 'rgba(66, 165, 245, 0.04)' : 'transparent',
                border: active
                  ? `1.5px solid ${cat.border}80`
                  : isNext ? `1.5px dashed rgba(66, 165, 245, 0.3)` : '1.5px solid transparent',
                transition: 'all 0.25s',
                display: 'flex', alignItems: 'center', gap: 5,
                position: 'relative' as const, overflow: 'hidden',
                margin: '0 4px',
              }}
            >
              {/* Heat bar */}
              <div style={{
                position: 'absolute', left: 0, top: 0, bottom: 0,
                width: `${heat * 100}%`,
                background: active
                  ? `linear-gradient(90deg, ${cat.glow}10, transparent)`
                  : `linear-gradient(90deg, ${hc}08, transparent)`,
                transition: 'all 0.3s',
              }} />

              {/* Step circle */}
              <div style={{
                width: 24, height: 24, borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 700, fontFamily: 'Inter, sans-serif',
                flexShrink: 0, position: 'relative' as const, zIndex: 1,
                background: active ? cat.border : done ? '#42a5f5' : 'rgba(255,255,255,0.08)',
                color: (active || done) ? '#fff' : '#667788',
                boxShadow: active ? `0 0 10px ${cat.glow}44` : 'none',
                transition: 'all 0.3s',
              }}>
                {done ? '✓' : idx + 1}
              </div>

              {/* Label */}
              <div style={{ position: 'relative' as const, zIndex: 1, flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: 11, fontWeight: active ? 700 : done ? 600 : 400,
                  color: active ? cat.text : done ? '#64b5f6' : '#8899aa',
                  fontFamily: 'Inter, sans-serif',
                  display: 'flex', alignItems: 'center', gap: 4,
                  whiteSpace: 'nowrap' as const,
                }}>
                  {op.short}
                  <span style={{
                    fontSize: 7, padding: '0px 3px', borderRadius: 4,
                    background: `${hc}18`, color: hc, fontWeight: 700,
                    letterSpacing: 0.5, flexShrink: 0,
                  }}>
                    {heat >= 0.8 ? 'HOT' : heat >= 0.5 ? 'MED' : 'LOW'}
                  </span>
                </div>
                <div style={{
                  fontSize: 9, color: active ? `${cat.accent}99` : '#445566',
                  fontFamily: 'Inter, sans-serif',
                }}>
                  {op.category === 'msa' ? 'MSA stack' : op.category === 'pair' ? 'Pair stack' : 'MSA → Pair'}
                </div>
              </div>

              {isNext && (
                <div style={{
                  position: 'relative' as const, zIndex: 1,
                  fontSize: 9, color: '#42a5f5', fontWeight: 600,
                  fontFamily: 'Inter, sans-serif',
                  animation: 'evoNext 1.5s infinite', flexShrink: 0,
                }}>
                  NEXT →
                </div>
              )}
            </div>

            {idx < ops.length - 1 && (
              <StepConnector done={done} active={op.id === activeId} />
            )}
          </div>
        )
      })}

      <StepConnector done={activeId >= ops.length - 1} active={false} />
      <div style={{
        padding: '4px 12px', fontSize: 9, color: '#556677',
        display: 'flex', alignItems: 'center', gap: 5,
        fontFamily: 'Inter, sans-serif',
      }}>
        <span style={{ fontSize: 10, opacity: 0.6 }}>→</span> Updated m_si + z_ij out
      </div>

      <div style={{
        margin: '10px 6px 0', padding: '6px 8px',
        background: 'rgba(66, 165, 245, 0.06)',
        borderRadius: 6, fontSize: 9, color: '#42a5f5',
        textAlign: 'center', fontFamily: 'Inter, sans-serif',
        border: '1px solid rgba(66, 165, 245, 0.15)', fontWeight: 600,
      }}>
        ↻ Repeat × N_block = 48
      </div>

      {/* Heat legend */}
      <div style={{
        margin: '8px 6px 0', padding: '5px 6px',
        background: 'rgba(255,255,255,0.03)', borderRadius: 4,
        fontSize: 8, color: '#556677', fontFamily: 'Inter, sans-serif',
      }}>
        <div style={{
          height: 4, borderRadius: 2,
          background: 'linear-gradient(90deg, #42a5f5, #ffa726, #e53935)',
          marginBottom: 2, opacity: 0.6,
        }} />
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>LOW</span><span>MED</span><span>HOT</span>
        </div>
      </div>
    </div>
  )
}

// ── Main Component ────────────────────────────────────

export function EvoformerDetail({ onBack }: { onBack: () => void }) {
  const [activeOpId, setActiveOpId] = useState(3)
  const [playing, setPlaying] = useState(true)
  const [highlightK, setHighlightK] = useState(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const kIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const activeOp = EVOFORMER_OPS[activeOpId]
  const highlightI = 2
  const highlightJ = 6

  useEffect(() => {
    const isTriangle = activeOpId >= 3 && activeOpId <= 6
    if (isTriangle && playing) {
      kIntervalRef.current = setInterval(() => {
        setHighlightK(k => (k + 1) % N)
      }, 1200)
    }
    return () => { if (kIntervalRef.current) clearInterval(kIntervalRef.current) }
  }, [activeOpId, playing])

  const goNext = useCallback(() => {
    setActiveOpId(id => (id + 1) % EVOFORMER_OPS.length)
    setHighlightK(0)
  }, [])

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(goNext, 8000)
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [playing, goNext])

  const handleSelect = (id: number) => {
    setActiveOpId(id)
    setHighlightK(0)
    setPlaying(false)
  }

  const cat = CAT_COLORS[activeOp.category]
  const isTriangleOp = activeOp.id >= 3 && activeOp.id <= 6
  const isMSAOp = activeOp.id <= 1
  const isOuterProduct = activeOp.id === 2

  return (
    <div style={{
      width: '100%', height: '100%', background: DARK.bg,
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      fontFamily: 'Inter, system-ui, sans-serif',
    }}>
      {/* Top bar */}
      <div style={{
        padding: '10px 20px',
        borderBottom: `1px solid ${DARK.border}`,
        background: DARK.surface,
        display: 'flex', alignItems: 'center', gap: 14, flexShrink: 0,
      }}>
        <button onClick={onBack} style={{
          border: `1px solid ${DARK.borderLight}`, borderRadius: 6, background: 'transparent',
          padding: '5px 12px', cursor: 'pointer', fontSize: 12, color: DARK.textMuted,
          display: 'flex', alignItems: 'center', gap: 5,
        }}>
          ← Back
        </button>
        <h1 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: '#c0c8d0' }}>
          Evoformer Deep Dive
        </h1>
        <span style={{ fontSize: 12, color: DARK.textDim }}>
          48 blocks × 8 operations per block
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
          <button onClick={() => { setActiveOpId(id => (id - 1 + EVOFORMER_OPS.length) % EVOFORMER_OPS.length); setPlaying(false) }}
            style={darkBtnStyle}>◀</button>
          <button onClick={() => setPlaying(p => !p)}
            style={{ ...darkBtnStyle, width: 50 }}>{playing ? '⏸' : '▶'}</button>
          <button onClick={() => { goNext(); setPlaying(false) }}
            style={darkBtnStyle}>▶</button>
        </div>
      </div>

      {/* Main content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        {/* Left sidebar */}
        <div style={{
          width: 175, borderRight: `1px solid ${DARK.border}`, overflowY: 'auto',
          flexShrink: 0, background: DARK.surface,
        }}>
          <BlockList ops={EVOFORMER_OPS} activeId={activeOpId} onSelect={handleSelect} />
        </div>

        {/* Center visualization */}
        <div key={`viz-${activeOpId}`} style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: 24, padding: '20px', overflow: 'auto', minWidth: 0,
          animation: 'evoFadeIn 0.4s ease-out',
        }}>
          {isTriangleOp && (
            <>
              <PairMatrixViz
                activeOp={activeOp}
                highlightI={highlightI}
                highlightJ={highlightJ}
                highlightK={highlightK}
              />
              <TriangleIntuition highlightK={highlightK} activeOp={activeOp} />
            </>
          )}
          {isMSAOp && <MSAMatrixViz activeOp={activeOp} />}
          {isOuterProduct && <OuterProductViz />}
          {activeOp.id === 7 && <FFNVisualization />}
        </div>

        {/* Right panel */}
        <div style={{
          width: 320, borderLeft: `1px solid ${DARK.border}`, overflowY: 'auto',
          padding: '16px 16px', flexShrink: 0, background: DARK.surface,
        }}>
          {/* Category badge */}
          <div style={{
            display: 'inline-block', padding: '2px 8px', borderRadius: 10,
            background: cat.bg, border: `1px solid ${cat.border}60`,
            fontSize: 10, color: cat.text, fontWeight: 600, marginBottom: 8,
          }}>
            {activeOp.category === 'msa' ? 'MSA Stack' :
             activeOp.category === 'pair' ? 'Pair Stack' : 'Cross-Repr'}
          </div>

          {/* Algorithm reference */}
          <span style={{
            fontSize: 9, color: DARK.textDim, marginLeft: 8,
            fontFamily: 'JetBrains Mono, monospace',
          }}>
            {activeOp.algRef}
          </span>

          <h2 style={{ margin: '6px 0', fontSize: 14, color: '#c0c8d0', fontWeight: 700 }}>
            {activeOp.name}
          </h2>

          <p style={{ margin: '0 0 12px', fontSize: 12, color: DARK.textMuted, lineHeight: 1.6 }}>
            {activeOp.description}
          </p>

          {/* Formula box — rendered with KaTeX */}
          <KaTeXFormula
            formula={activeOp.formula}
            style={{
              marginBottom: 12,
              borderLeft: `3px solid ${cat.border}60`,
              fontSize: 13,
            }}
          />

          <ul style={{
            margin: 0, padding: '0 0 0 14px', fontSize: 11.5,
            color: DARK.textMuted, lineHeight: 1.7,
          }}>
            {activeOp.details.map((d, i) => (
              <li key={i} style={{ marginBottom: 3 }}>{d}</li>
            ))}
          </ul>

          {/* Triangle k control */}
          {isTriangleOp && (
            <div style={{
              marginTop: 16, padding: '10px 12px',
              background: 'rgba(76, 175, 80, 0.06)',
              borderRadius: 6, border: '1px solid rgba(76, 175, 80, 0.15)',
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#4caf50', marginBottom: 4 }}>
                Third residue k = {highlightK}
              </div>
              <input
                type="range" min={0} max={N - 1} value={highlightK}
                onChange={e => { setHighlightK(Number(e.target.value)); setPlaying(false) }}
                style={{ width: '100%', accentColor: '#4caf50' }}
              />
              <div style={{ fontSize: 10, color: '#556677', marginTop: 3 }}>
                Scanning triangle: (i={highlightI}, j={highlightJ}, k={highlightK})
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const darkBtnStyle: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, background: 'rgba(255,255,255,0.05)',
  padding: '4px 10px', cursor: 'pointer', fontSize: 12, color: '#8899aa',
}
