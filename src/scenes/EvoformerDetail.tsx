import { useState, useEffect, useRef, useCallback } from 'react'

// ── Evoformer block operations ─────────────────────────

interface BlockOp {
  id: number
  name: string
  short: string
  category: 'msa' | 'pair' | 'cross'
  description: string
  details: string[]
  formula?: string
}

const EVOFORMER_OPS: BlockOp[] = [
  {
    id: 0,
    name: 'MSA Row-wise Self-Attention',
    short: 'Row Attn',
    category: 'msa',
    description:
      'Within each sequence in the MSA, every residue attends to every other residue. This captures which residues interact within one protein sequence. The pair representation biases the attention weights — if pair(i,j) says "these residues are close", they attend more to each other.',
    details: [
      'Input: MSA repr (s, r, c) — operates along the r (residue) axis for each sequence s',
      'Pair bias: attention logits += pair_repr[i, j] — structure knowledge guides MSA attention',
      'Each head learns different interaction patterns',
      'Output: updated MSA repr with residue-residue context within each sequence',
    ],
    formula: 'Attn(Q,K,V) with bias b_ij from pair repr',
  },
  {
    id: 1,
    name: 'MSA Column-wise Self-Attention',
    short: 'Col Attn',
    category: 'msa',
    description:
      'At each residue position, we attend across all sequences in the MSA. This is how AlphaFold2 detects co-evolution — if position i is always arginine when position j is glutamate across many species, this attention pattern captures that correlation.',
    details: [
      'Input: MSA repr (s, r, c) — operates along the s (sequence) axis for each position r',
      'Captures co-evolutionary signals: correlated mutations across species',
      'Much cheaper than full (s×r) attention — only O(s²) per position',
      'This is where "evolutionary information" gets aggregated',
    ],
    formula: 'Attn across sequences at each position',
  },
  {
    id: 2,
    name: 'Outer Product Mean',
    short: 'Outer Prod',
    category: 'cross',
    description:
      'This is the bridge from MSA → Pair representation. For each pair of positions (i,j), take the outer product of MSA features at position i and j, then average across all sequences. Co-evolving positions produce strong outer-product signals → strong pair features.',
    details: [
      'For each sequence s: compute outer product of MSA[s,i,:] and MSA[s,j,:]',
      'Average the outer products across all sequences s',
      'Result shape: (r, r, c×c) → projected to (r, r, c)',
      'This converts co-evolution signals into pairwise distance/contact predictions',
      'Key insight: co-evolution ↔ spatial proximity',
    ],
    formula: 'pair[i,j] += mean_s(MSA[s,i] ⊗ MSA[s,j])',
  },
  {
    id: 3,
    name: 'Triangle Multiplicative Update (Outgoing)',
    short: '△ Mult Out',
    category: 'pair',
    description:
      'The first triangle operation. To update edge (i,j) in the pair matrix, we look at all "outgoing" triangles: for every third residue k, multiply pair(i,k) × pair(j,k). If residues i and j both have strong signals to the same residue k, then i and j are likely close to each other.',
    details: [
      'For each edge (i,j): aggregate over all k of pair(i,k) ⊙ pair(j,k)',
      '"Outgoing" = edges leaving from i and j toward common third node k',
      'Enforces triangle inequality: if dist(i,k) and dist(j,k) are small → dist(i,j) is small',
      'Uses gating: elementwise sigmoid gates control information flow',
      'This is AlphaFold2\'s key geometric inductive bias',
    ],
    formula: 'pair[i,j] += Σ_k gate(pair[i,k]) ⊙ gate(pair[j,k])',
  },
  {
    id: 4,
    name: 'Triangle Multiplicative Update (Incoming)',
    short: '△ Mult In',
    category: 'pair',
    description:
      'The mirror of outgoing: to update edge (i,j), look at "incoming" triangles. For every k, multiply pair(k,i) × pair(k,j). This captures the reverse direction — if some residue k is close to both i and j, it constrains the i-j relationship.',
    details: [
      'For each edge (i,j): aggregate over all k of pair(k,i) ⊙ pair(k,j)',
      '"Incoming" = edges arriving at i and j from common source k',
      'Complements outgoing update for full triangle consistency',
      'Together, outgoing + incoming enforce: "friends of friends are friends"',
    ],
    formula: 'pair[i,j] += Σ_k gate(pair[k,i]) ⊙ gate(pair[k,j])',
  },
  {
    id: 5,
    name: 'Triangle Self-Attention (Starting)',
    short: '△ Attn Start',
    category: 'pair',
    description:
      'Attention over the pair representation where, to update pair(i,j), position j attends to all positions k. The attention weights are biased by pair(i,k→j) — the triangle completion. This lets the network learn complex relational patterns beyond simple multiplicative updates.',
    details: [
      'Fix row i, let j attend to all k: pair[i,j] attends to pair[i,k] for all k',
      'Attention bias from pair(j,k) — the third triangle edge',
      '"Starting" = the fixed node i is the start of edges (i,j) and (i,k)',
      'More expressive than multiplicative update — learned attention patterns',
    ],
    formula: 'pair[i,j] = Σ_k softmax(q_j · k_k + bias(pair[j,k])) · v_k',
  },
  {
    id: 6,
    name: 'Triangle Self-Attention (Ending)',
    short: '△ Attn End',
    category: 'pair',
    description:
      'The transpose of starting attention. Fix column j, let row i attend to all rows k. This ensures that information flows in both directions through the pair matrix — both "who does j relate to" and "who relates to j" get captured.',
    details: [
      'Fix col j, let i attend to all k: pair[i,j] attends to pair[k,j] for all k',
      'Attention bias from pair(i,k) — the third triangle edge',
      '"Ending" = the fixed node j is the end of edges (i,j) and (k,j)',
      'Combines with Starting attention for full bidirectional information flow',
    ],
    formula: 'pair[i,j] = Σ_k softmax(q_i · k_k + bias(pair[i,k])) · v_k',
  },
  {
    id: 7,
    name: 'Pair Transition (Feed-Forward)',
    short: 'Pair FFN',
    category: 'pair',
    description:
      'A standard feed-forward network applied independently to each (i,j) position. This gives the network non-linear capacity to process and transform the pair features after all the attention and multiplicative updates.',
    details: [
      'Two-layer MLP: Linear → ReLU → Linear for each pair(i,j)',
      'Applied independently per position (no cross-position interaction)',
      'Adds non-linear transformation capacity',
      'Followed by residual connection + LayerNorm',
    ],
    formula: 'pair[i,j] = FFN(pair[i,j]) + pair[i,j]',
  },
]

// ── Colors for categories ──────────────────────────────

const CAT_COLORS = {
  msa: { bg: '#fff3e0', border: '#e65100', text: '#bf360c', glow: '#ff6d00' },
  pair: { bg: '#e3f2fd', border: '#1565c0', text: '#0d47a1', glow: '#2196f3' },
  cross: { bg: '#f3e5f5', border: '#7b1fa2', text: '#4a148c', glow: '#9c27b0' },
}

// ── Pair matrix visualization ──────────────────────────

const N = 8 // small matrix for visualization

function PairMatrixViz({
  activeOp, animStep, highlightI, highlightJ, highlightK,
}: {
  activeOp: BlockOp
  animStep: number
  highlightI: number
  highlightJ: number
  highlightK: number
}) {
  const cellSize = 40
  const matrixSize = N * cellSize
  const ox = 30 // offset x for labels
  const oy = 30 // offset y for labels

  const residueLabels = ['A', 'G', 'V', 'L', 'I', 'P', 'F', 'W']

  // Generate synthetic pair values
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

  // Determine which cells to highlight for triangle edges
  const getTriangleEdges = (i: number, j: number, k: number) => {
    if (isOutgoing) return [[i, k], [j, k]] // outgoing: (i,k) and (j,k)
    if (isIncoming) return [[k, i], [k, j]] // incoming: (k,i) and (k,j)
    if (isAttnStart) return [[i, k], [j, k]] // start: fix i, j attends to k
    if (isAttnEnd) return [[k, j], [i, k]] // end: fix j, i attends to k
    return []
  }

  const triangleEdges = isTriangleOp ? getTriangleEdges(highlightI, highlightJ, highlightK) : []

  return (
    <svg viewBox={`0 0 ${matrixSize + ox + 60} ${matrixSize + oy + 80}`}
      style={{ width: '100%', maxWidth: 440, height: 'auto' }}>

      {/* Column labels */}
      {residueLabels.map((l, i) => (
        <text key={`cl-${i}`} x={ox + i * cellSize + cellSize / 2} y={oy - 8}
          textAnchor="middle" fontSize={12} fill={i === highlightJ ? '#1565c0' : '#666'}
          fontWeight={i === highlightJ ? 700 : 400} fontFamily="monospace">
          {l}<tspan fontSize={9} dy={2}>{i}</tspan>
        </text>
      ))}

      {/* Row labels */}
      {residueLabels.map((l, i) => (
        <text key={`rl-${i}`} x={ox - 8} y={oy + i * cellSize + cellSize / 2 + 4}
          textAnchor="end" fontSize={12} fill={i === highlightI ? '#e65100' : '#666'}
          fontWeight={i === highlightI ? 700 : 400} fontFamily="monospace">
          {l}<tspan fontSize={9} dy={2}>{i}</tspan>
        </text>
      ))}

      {/* Matrix cells */}
      {Array.from({ length: N }).map((_, r) =>
        Array.from({ length: N }).map((_, c) => {
          const isTarget = r === highlightI && c === highlightJ
          const isTriEdge = triangleEdges.some(e => e[0] === r && e[1] === c)
          const isKRow = isTriangleOp && (
            (isOutgoing && c === highlightK) ||
            (isIncoming && r === highlightK) ||
            (isAttnStart && c === highlightK && r === highlightI) ||
            (isAttnEnd && r === highlightK && c === highlightJ)
          )
          const val = pairValues[r][c]
          const baseColor = `rgba(25, 118, 210, ${val * 0.7})`

          let fill = baseColor
          let strokeColor = 'rgba(200,200,200,0.4)'
          let strokeW = 0.5

          if (isTarget) {
            fill = 'rgba(230, 81, 0, 0.6)'
            strokeColor = '#e65100'
            strokeW = 2.5
          } else if (isTriEdge) {
            fill = 'rgba(76, 175, 80, 0.5)'
            strokeColor = '#2e7d32'
            strokeW = 2
          } else if (isKRow) {
            fill = `rgba(25, 118, 210, ${val * 0.7 + 0.15})`
            strokeColor = '#1976d2'
            strokeW = 1
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
                  fill="none" stroke="#e65100" strokeWidth={2} rx={4}
                >
                  <animate attributeName="stroke-opacity" values="1;0.3;1" dur="1.5s" repeatCount="indefinite" />
                </rect>
              )}
            </g>
          )
        })
      )}

      {/* Triangle lines for triangle ops */}
      {isTriangleOp && triangleEdges.length === 2 && (
        <g>
          {/* Line from edge1 to target */}
          <line
            x1={ox + triangleEdges[0][1] * cellSize + cellSize / 2}
            y1={oy + triangleEdges[0][0] * cellSize + cellSize / 2}
            x2={ox + highlightJ * cellSize + cellSize / 2}
            y2={oy + highlightI * cellSize + cellSize / 2}
            stroke="#4caf50" strokeWidth={2} strokeDasharray="4,3" opacity={0.7}
          />
          {/* Line from edge2 to target */}
          <line
            x1={ox + triangleEdges[1][1] * cellSize + cellSize / 2}
            y1={oy + triangleEdges[1][0] * cellSize + cellSize / 2}
            x2={ox + highlightJ * cellSize + cellSize / 2}
            y2={oy + highlightI * cellSize + cellSize / 2}
            stroke="#4caf50" strokeWidth={2} strokeDasharray="4,3" opacity={0.7}
          />
          {/* Line between the two edges */}
          <line
            x1={ox + triangleEdges[0][1] * cellSize + cellSize / 2}
            y1={oy + triangleEdges[0][0] * cellSize + cellSize / 2}
            x2={ox + triangleEdges[1][1] * cellSize + cellSize / 2}
            y2={oy + triangleEdges[1][0] * cellSize + cellSize / 2}
            stroke="#81c784" strokeWidth={1.5} strokeDasharray="3,3" opacity={0.5}
          />

          {/* Animated particle: edge1 → target */}
          <circle r={4} fill="#4caf50" opacity={0.9}>
            <animateMotion
              dur="1.5s" repeatCount="indefinite"
              path={`M${ox + triangleEdges[0][1] * cellSize + cellSize / 2},${oy + triangleEdges[0][0] * cellSize + cellSize / 2} L${ox + highlightJ * cellSize + cellSize / 2},${oy + highlightI * cellSize + cellSize / 2}`}
            />
          </circle>
          {/* Animated particle: edge2 → target */}
          <circle r={4} fill="#4caf50" opacity={0.9}>
            <animateMotion
              dur="1.5s" repeatCount="indefinite" begin="0.3s"
              path={`M${ox + triangleEdges[1][1] * cellSize + cellSize / 2},${oy + triangleEdges[1][0] * cellSize + cellSize / 2} L${ox + highlightJ * cellSize + cellSize / 2},${oy + highlightI * cellSize + cellSize / 2}`}
            />
          </circle>
        </g>
      )}

      {/* Legend */}
      <g transform={`translate(${ox}, ${oy + matrixSize + 12})`}>
        <rect x={0} y={0} width={12} height={12} fill="rgba(230, 81, 0, 0.6)" rx={2} />
        <text x={16} y={10} fontSize={10} fill="#666" fontFamily="Inter, sans-serif">Target (i,j)</text>

        <rect x={100} y={0} width={12} height={12} fill="rgba(76, 175, 80, 0.5)" rx={2} />
        <text x={116} y={10} fontSize={10} fill="#666" fontFamily="Inter, sans-serif">Triangle edges</text>

        <rect x={220} y={0} width={12} height={12} fill="rgba(25, 118, 210, 0.5)" rx={2} />
        <text x={236} y={10} fontSize={10} fill="#666" fontFamily="Inter, sans-serif">Pair value</text>
      </g>

      {/* k label */}
      {isTriangleOp && (
        <text x={ox + matrixSize + 10} y={oy + highlightK * cellSize + cellSize / 2 + 4}
          fontSize={13} fill="#2e7d32" fontWeight={700} fontFamily="Inter, sans-serif">
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
  const cellW = 40
  const cellH = 28
  const ox = 40
  const oy = 30

  const isRow = activeOp.id === 0
  const residueLabels = ['A', 'G', 'V', 'L', 'I', 'P', 'F', 'W']

  return (
    <svg viewBox={`0 0 ${nRes * cellW + ox + 30} ${nSeq * cellH + oy + 60}`}
      style={{ width: '100%', maxWidth: 420, height: 'auto' }}>

      {/* Axis labels */}
      <text x={ox + nRes * cellW / 2} y={15} textAnchor="middle" fontSize={11} fill="#999" fontFamily="Inter, sans-serif">
        Residue position →
      </text>
      <text x={12} y={oy + nSeq * cellH / 2} textAnchor="middle" fontSize={11} fill="#999"
        fontFamily="Inter, sans-serif" transform={`rotate(-90, 12, ${oy + nSeq * cellH / 2})`}>
        Sequences →
      </text>

      {/* Column labels */}
      {residueLabels.map((l, i) => (
        <text key={`c-${i}`} x={ox + i * cellW + cellW / 2} y={oy - 5}
          textAnchor="middle" fontSize={10} fill="#666" fontFamily="monospace">{l}{i}</text>
      ))}

      {/* Row labels */}
      {Array.from({ length: nSeq }).map((_, i) => (
        <text key={`r-${i}`} x={ox - 5} y={oy + i * cellH + cellH / 2 + 4}
          textAnchor="end" fontSize={10} fill="#666" fontFamily="monospace">seq{i}</text>
      ))}

      {/* Cells */}
      {Array.from({ length: nSeq }).map((_, s) =>
        Array.from({ length: nRes }).map((_, r) => {
          const val = 0.3 + Math.sin(s * 1.2 + r * 0.8) * 0.3
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
        // Row-wise: highlight one row, show horizontal arrows
        <g>
          <rect x={ox - 2} y={oy + 2 * cellH - 2} width={nRes * cellW + 4} height={cellH + 4}
            fill="none" stroke="#e65100" strokeWidth={2} rx={4}>
            <animate attributeName="stroke-opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite" />
          </rect>
          {/* Arrows between residues in the highlighted row */}
          {[0, 1, 3, 5, 7].map(r => (
            <line key={`a-${r}`}
              x1={ox + 4 * cellW + cellW / 2} y1={oy + 2 * cellH + cellH / 2}
              x2={ox + r * cellW + cellW / 2} y2={oy + 2 * cellH + cellH / 2}
              stroke="#e65100" strokeWidth={1.5} opacity={0.5} markerEnd="url(#arrowOrange)" />
          ))}
          <text x={ox + nRes * cellW + 8} y={oy + 2 * cellH + cellH / 2 + 4}
            fontSize={11} fill="#e65100" fontWeight={600} fontFamily="Inter, sans-serif">
            ← residues attend
          </text>
          <text x={ox + nRes * cellW + 8} y={oy + 2 * cellH + cellH / 2 + 18}
            fontSize={10} fill="#e65100" fontFamily="Inter, sans-serif">
            to each other
          </text>
        </g>
      ) : (
        // Column-wise: highlight one column, show vertical arrows
        <g>
          <rect x={ox + 3 * cellW - 2} y={oy - 2} width={cellW + 4} height={nSeq * cellH + 4}
            fill="none" stroke="#1565c0" strokeWidth={2} rx={4}>
            <animate attributeName="stroke-opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite" />
          </rect>
          {[0, 1, 3, 4].map(s => (
            <line key={`a-${s}`}
              x1={ox + 3 * cellW + cellW / 2} y1={oy + 2 * cellH + cellH / 2}
              x2={ox + 3 * cellW + cellW / 2} y2={oy + s * cellH + cellH / 2}
              stroke="#1565c0" strokeWidth={1.5} opacity={0.5} />
          ))}
          <text x={ox + 3 * cellW + cellW / 2} y={oy + nSeq * cellH + 18}
            textAnchor="middle" fontSize={11} fill="#1565c0" fontWeight={600} fontFamily="Inter, sans-serif">
            ↕ sequences share info
          </text>
        </g>
      )}

      {/* Arrow marker */}
      <defs>
        <marker id="arrowOrange" markerWidth={6} markerHeight={6} refX={5} refY={3} orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#e65100" opacity={0.7} />
        </marker>
      </defs>
    </svg>
  )
}

// ── Outer product visualization ────────────────────────

function OuterProductViz() {
  return (
    <svg viewBox="0 0 440 380" style={{ width: '100%', maxWidth: 440, height: 'auto' }}>
      {/* MSA column i */}
      <text x={50} y={20} textAnchor="middle" fontSize={12} fill="#7b1fa2" fontWeight={600} fontFamily="Inter, sans-serif">
        MSA[:,i]
      </text>
      {Array.from({ length: 5 }).map((_, s) => (
        <rect key={`i-${s}`} x={30} y={28 + s * 24} width={40} height={20}
          fill="#ce93d8" opacity={0.5 + s * 0.1} rx={3} stroke="#7b1fa2" strokeWidth={1} />
      ))}

      {/* MSA column j */}
      <text x={160} y={20} textAnchor="middle" fontSize={12} fill="#7b1fa2" fontWeight={600} fontFamily="Inter, sans-serif">
        MSA[:,j]
      </text>
      {Array.from({ length: 5 }).map((_, s) => (
        <rect key={`j-${s}`} x={140} y={28 + s * 24} width={40} height={20}
          fill="#ce93d8" opacity={0.5 + s * 0.1} rx={3} stroke="#7b1fa2" strokeWidth={1} />
      ))}

      {/* Outer product symbol */}
      <text x={105} y={90} textAnchor="middle" fontSize={20} fill="#7b1fa2" fontFamily="Inter, sans-serif">⊗</text>

      {/* Arrow down */}
      <line x1={105} y1={100} x2={105} y2={140} stroke="#7b1fa2" strokeWidth={2} markerEnd="url(#arrowPurple)" />
      <text x={130} y={125} fontSize={10} fill="#999" fontFamily="Inter, sans-serif">mean over s</text>

      {/* Resulting pair cell */}
      <rect x={75} y={150} width={60} height={60} fill="rgba(25,118,210,0.3)" stroke="#1565c0" strokeWidth={2} rx={4} />
      <text x={105} y={185} textAnchor="middle" fontSize={11} fill="#1565c0" fontWeight={600} fontFamily="Inter, sans-serif">
        pair[i,j]
      </text>

      {/* Animated particles from MSA columns to pair cell */}
      {Array.from({ length: 5 }).map((_, s) => (
        <g key={`p-${s}`}>
          <circle r={3} fill="#7b1fa2" opacity={0.8}>
            <animateMotion dur="2s" begin={`${s * 0.3}s`} repeatCount="indefinite"
              path={`M50,${38 + s * 24} L105,${155}`} />
          </circle>
          <circle r={3} fill="#7b1fa2" opacity={0.8}>
            <animateMotion dur="2s" begin={`${s * 0.3 + 0.15}s`} repeatCount="indefinite"
              path={`M160,${38 + s * 24} L105,${155}`} />
          </circle>
        </g>
      ))}

      <text x={105} y={235} textAnchor="middle" fontSize={12} fill="#555" fontFamily="Inter, sans-serif">
        Co-evolution signal → Pair feature
      </text>

      <defs>
        <marker id="arrowPurple" markerWidth={6} markerHeight={6} refX={5} refY={3} orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#7b1fa2" />
        </marker>
      </defs>
    </svg>
  )
}

// ── Residue triangle diagram (3D intuition) ────────────

function TriangleIntuition({ highlightK }: { highlightK: number }) {
  // Show 3 residues as balls in "3D space" forming a triangle
  const points = {
    i: { x: 80, y: 220, label: 'i', color: '#e65100' },
    j: { x: 280, y: 220, label: 'j', color: '#1565c0' },
    k: { x: 180, y: 60, label: `k=${highlightK}`, color: '#2e7d32' },
  }

  return (
    <svg viewBox="0 0 360 300" style={{ width: '100%', maxWidth: 360, height: 'auto' }}>
      <text x={180} y={20} textAnchor="middle" fontSize={13} fill="#333" fontWeight={600} fontFamily="Inter, sans-serif">
        Triangle Intuition in 3D Space
      </text>

      {/* Edges */}
      <line x1={points.i.x} y1={points.i.y} x2={points.j.x} y2={points.j.y}
        stroke="#e65100" strokeWidth={3} opacity={0.7} />
      <line x1={points.i.x} y1={points.i.y} x2={points.k.x} y2={points.k.y}
        stroke="#4caf50" strokeWidth={2.5} strokeDasharray="6,3" opacity={0.7} />
      <line x1={points.j.x} y1={points.j.y} x2={points.k.x} y2={points.k.y}
        stroke="#4caf50" strokeWidth={2.5} strokeDasharray="6,3" opacity={0.7} />

      {/* Edge labels */}
      <text x={180} y={245} textAnchor="middle" fontSize={10} fill="#e65100" fontFamily="Inter, sans-serif">
        pair(i,j) — target to update
      </text>
      <text x={110} y={130} textAnchor="middle" fontSize={10} fill="#4caf50" fontFamily="Inter, sans-serif"
        transform="rotate(-50, 110, 130)">pair(i,k)</text>
      <text x={250} y={130} textAnchor="middle" fontSize={10} fill="#4caf50" fontFamily="Inter, sans-serif"
        transform="rotate(50, 250, 130)">pair(j,k)</text>

      {/* Particles flowing along triangle edges */}
      <circle r={4} fill="#4caf50" opacity={0.85}>
        <animateMotion dur="2s" repeatCount="indefinite"
          path={`M${points.k.x},${points.k.y} L${points.i.x},${points.i.y} L${points.j.x},${points.j.y}`} />
      </circle>
      <circle r={4} fill="#4caf50" opacity={0.85}>
        <animateMotion dur="2s" repeatCount="indefinite" begin="1s"
          path={`M${points.k.x},${points.k.y} L${points.j.x},${points.j.y} L${points.i.x},${points.i.y}`} />
      </circle>

      {/* Nodes */}
      {Object.values(points).map(p => (
        <g key={p.label}>
          <circle cx={p.x} cy={p.y} r={20} fill={p.color} opacity={0.15} />
          <circle cx={p.x} cy={p.y} r={12} fill={p.color} opacity={0.8} />
          <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={11} fill="#fff"
            fontWeight={700} fontFamily="Inter, sans-serif">{p.label}</text>
        </g>
      ))}

      <text x={180} y={280} textAnchor="middle" fontSize={11} fill="#666" fontStyle="italic" fontFamily="Inter, sans-serif">
        "If i↔k close and j↔k close → i↔j likely close"
      </text>
    </svg>
  )
}

// ── Computational intensity heat values (normalized 0-1) ──
// Reflects relative FLOPs / importance of each operation
const HEAT_VALUES: Record<number, number> = {
  0: 0.7,   // Row attn — O(r²) per sequence
  1: 0.5,   // Col attn — O(s²) per position (cheaper)
  2: 0.6,   // Outer product — bridge operation
  3: 0.95,  // Triangle mult out — O(r²·k), the expensive one
  4: 0.9,   // Triangle mult in — similarly expensive
  5: 0.85,  // Triangle attn start — attention + triangle bias
  6: 0.85,  // Triangle attn end
  7: 0.3,   // FFN — simple per-position MLP
}

function heatColor(heat: number): string {
  // Cool blue → warm orange → hot red
  if (heat < 0.33) {
    const t = heat / 0.33
    const r = Math.round(66 + t * 100)
    const g = Math.round(165 - t * 40)
    const b = Math.round(245 - t * 130)
    return `rgb(${r},${g},${b})`
  } else if (heat < 0.66) {
    const t = (heat - 0.33) / 0.33
    const r = Math.round(166 + t * 64)
    const g = Math.round(125 - t * 50)
    const b = Math.round(115 - t * 85)
    return `rgb(${r},${g},${b})`
  } else {
    const t = (heat - 0.66) / 0.34
    const r = Math.round(230 + t * 15)
    const g = Math.round(75 - t * 40)
    const b = Math.round(30 - t * 20)
    return `rgb(${r},${g},${b})`
  }
}

// ── Step flow sidebar ──────────────────────────────────

function StepConnector({ done, active }: { done: boolean; active: boolean }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'center', padding: '0 0 0 22px',
      height: 16,
    }}>
      <div style={{
        width: 2,
        height: '100%',
        background: done ? '#1976d2' : active ? '#90caf9' : '#e0e0e0',
        transition: 'background 0.3s',
        position: 'relative' as const,
      }}>
        {active && (
          <div style={{
            position: 'absolute', left: -3, top: 2,
            width: 8, height: 8, borderRadius: '50%',
            background: '#1976d2',
            animation: 'pulse 1.5s infinite',
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
      display: 'flex', flexDirection: 'column',
      padding: '8px 0',
    }}>
      {/* CSS for pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(0.6); }
        }
        @keyframes nextArrow {
          0%, 100% { transform: translateX(0); opacity: 0.7; }
          50% { transform: translateX(3px); opacity: 1; }
        }
      `}</style>

      <div style={{
        fontSize: 11, color: '#999', textTransform: 'uppercase', letterSpacing: 1,
        padding: '0 12px 8px', fontFamily: 'Inter, sans-serif',
      }}>
        One Evoformer Block
      </div>

      {/* Input indicator */}
      <div style={{
        padding: '4px 12px', fontSize: 10, color: '#78909c',
        display: 'flex', alignItems: 'center', gap: 6,
        fontFamily: 'Inter, sans-serif',
      }}>
        <span style={{ fontSize: 12 }}>📥</span> MSA repr + Pair repr in
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
                padding: '6px 10px 6px 6px',
                borderRadius: 8,
                cursor: 'pointer',
                background: active
                  ? `linear-gradient(135deg, ${cat.bg}, ${cat.bg}ee)`
                  : done
                    ? 'rgba(25, 118, 210, 0.04)'
                    : 'transparent',
                border: active
                  ? `2px solid ${cat.border}`
                  : isNext
                    ? '2px dashed #90caf9'
                    : '2px solid transparent',
                transition: 'all 0.25s',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                position: 'relative' as const,
                overflow: 'hidden',
                margin: '0 4px',
              }}
            >
              {/* Heat bar background */}
              <div style={{
                position: 'absolute', left: 0, top: 0, bottom: 0,
                width: `${heat * 100}%`,
                background: active
                  ? `linear-gradient(90deg, ${cat.glow}12, transparent)`
                  : `linear-gradient(90deg, ${hc}06, transparent)`,
                transition: 'all 0.3s',
              }} />

              {/* Step number circle */}
              <div style={{
                width: 26, height: 26, borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 12, fontWeight: 700,
                fontFamily: 'Inter, sans-serif',
                flexShrink: 0,
                position: 'relative' as const, zIndex: 1,
                background: active ? cat.border
                  : done ? '#1976d2'
                  : '#e0e0e0',
                color: (active || done) ? '#fff' : '#999',
                boxShadow: active ? `0 0 12px ${cat.glow}66` : 'none',
                transition: 'all 0.3s',
              }}>
                {done ? '✓' : idx + 1}
              </div>

              {/* Label */}
              <div style={{ position: 'relative' as const, zIndex: 1, flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: 11.5, fontWeight: active ? 700 : done ? 600 : 500,
                  color: active ? cat.text : done ? '#1565c0' : '#666',
                  fontFamily: 'Inter, sans-serif',
                  display: 'flex', alignItems: 'center', gap: 5,
                  whiteSpace: 'nowrap' as const,
                }}>
                  {op.short}
                  <span style={{
                    fontSize: 7, padding: '1px 4px', borderRadius: 6,
                    background: `${hc}20`, color: hc, fontWeight: 700,
                    letterSpacing: 0.5, flexShrink: 0,
                  }}>
                    {heat >= 0.8 ? 'HOT' : heat >= 0.5 ? 'MED' : 'LOW'}
                  </span>
                </div>
                <div style={{
                  fontSize: 9.5, color: active ? cat.border : done ? '#64b5f6' : '#bbb',
                  fontFamily: 'Inter, sans-serif',
                }}>
                  {op.category === 'msa' ? 'MSA' : op.category === 'pair' ? 'Pair' : 'MSA → Pair'}
                </div>
              </div>

              {/* "Next" indicator */}
              {isNext && (
                <div style={{
                  position: 'relative' as const, zIndex: 1,
                  fontSize: 10, color: '#1976d2', fontWeight: 600,
                  fontFamily: 'Inter, sans-serif',
                  animation: 'nextArrow 1.5s infinite',
                  flexShrink: 0,
                }}>
                  NEXT →
                </div>
              )}
            </div>

            {/* Connector arrow between steps */}
            {idx < ops.length - 1 && (
              <StepConnector done={done} active={op.id === activeId} />
            )}
          </div>
        )
      })}

      {/* Output indicator */}
      <StepConnector done={activeId >= ops.length - 1} active={false} />
      <div style={{
        padding: '4px 12px', fontSize: 10, color: '#78909c',
        display: 'flex', alignItems: 'center', gap: 6,
        fontFamily: 'Inter, sans-serif',
      }}>
        <span style={{ fontSize: 12 }}>📤</span> Updated MSA + Pair repr out
      </div>

      {/* Repeat indicator */}
      <div style={{
        margin: '10px 8px 0',
        padding: '8px 10px',
        background: 'linear-gradient(135deg, #e8eaf6, #f3e5f5)',
        borderRadius: 8,
        fontSize: 10,
        color: '#5c6bc0',
        textAlign: 'center',
        fontFamily: 'Inter, sans-serif',
        border: '1px solid #c5cae9',
        fontWeight: 600,
      }}>
        ↻ Repeat all 8 steps × 48 blocks
      </div>

      {/* Heat legend */}
      <div style={{
        margin: '8px 8px 0', padding: '6px 8px',
        background: '#fafafa', borderRadius: 6,
        fontSize: 8, color: '#999', fontFamily: 'Inter, sans-serif',
      }}>
        <div style={{
          height: 5, borderRadius: 3,
          background: 'linear-gradient(90deg, #42a5f5, #ffa726, #e53935)',
          marginBottom: 2,
        }} />
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>LOW</span><span>MED</span><span>HOT</span>
        </div>
      </div>
    </div>
  )
}

// ── Main Evoformer Detail Component ────────────────────

export function EvoformerDetail({ onBack }: { onBack: () => void }) {
  const [activeOpId, setActiveOpId] = useState(3) // start on triangle mult
  const [playing, setPlaying] = useState(true)
  const [animStep, setAnimStep] = useState(0)
  const [highlightK, setHighlightK] = useState(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const kIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const activeOp = EVOFORMER_OPS[activeOpId]
  const highlightI = 2
  const highlightJ = 6

  // Auto-advance k for triangle operations
  useEffect(() => {
    const isTriangle = activeOpId >= 3 && activeOpId <= 6
    if (isTriangle && playing) {
      kIntervalRef.current = setInterval(() => {
        setHighlightK(k => (k + 1) % N)
      }, 1200)
    }
    return () => {
      if (kIntervalRef.current) clearInterval(kIntervalRef.current)
    }
  }, [activeOpId, playing])

  // Auto-advance operation
  const goNext = useCallback(() => {
    setActiveOpId(id => (id + 1) % EVOFORMER_OPS.length)
    setHighlightK(0)
  }, [])

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(goNext, 8000)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
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
      width: '100%', height: '100%', background: '#fff',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      fontFamily: 'Inter, system-ui, sans-serif',
    }}>
      {/* Top bar */}
      <div style={{
        padding: '10px 24px',
        borderBottom: '1px solid #eee',
        display: 'flex', alignItems: 'center', gap: 16,
        flexShrink: 0,
      }}>
        <button onClick={onBack} style={{
          border: '1px solid #ccc', borderRadius: 6, background: '#fff',
          padding: '6px 14px', cursor: 'pointer', fontSize: 13, color: '#555',
          display: 'flex', alignItems: 'center', gap: 6,
        }}>
          ← Back to Overview
        </button>
        <h1 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: '#1a237e' }}>
          Evoformer Deep Dive
        </h1>
        <span style={{ fontSize: 13, color: '#78909c' }}>
          48 blocks × 8 operations — click each operation to see how it works
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <button onClick={() => setActiveOpId(id => (id - 1 + EVOFORMER_OPS.length) % EVOFORMER_OPS.length)}
            style={smallBtnStyle}>◀ Prev</button>
          <button onClick={() => setPlaying(p => !p)}
            style={{ ...smallBtnStyle, width: 60 }}>{playing ? '⏸ Pause' : '▶ Play'}</button>
          <button onClick={() => { goNext(); setPlaying(false) }}
            style={smallBtnStyle}>Next ▶</button>
        </div>
      </div>

      {/* Main content */}
      <div style={{
        flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0,
      }}>
        {/* Left: Block list */}
        <div style={{
          width: 180, borderRight: '1px solid #eee', overflowY: 'auto',
          flexShrink: 0,
        }}>
          <BlockList ops={EVOFORMER_OPS} activeId={activeOpId} onSelect={handleSelect} />
        </div>

        {/* Center: Visualization */}
        <div style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: 20, padding: '20px', overflow: 'auto', minWidth: 0,
        }}>
          {isTriangleOp && (
            <>
              <PairMatrixViz
                activeOp={activeOp}
                animStep={animStep}
                highlightI={highlightI}
                highlightJ={highlightJ}
                highlightK={highlightK}
              />
              <TriangleIntuition highlightK={highlightK} />
            </>
          )}
          {isMSAOp && (
            <MSAMatrixViz activeOp={activeOp} />
          )}
          {isOuterProduct && (
            <OuterProductViz />
          )}
          {activeOp.id === 7 && (
            <div style={{
              textAlign: 'center', padding: 40,
            }}>
              <svg viewBox="0 0 300 200" style={{ width: 300 }}>
                <rect x={20} y={30} width={260} height={50} rx={6} fill="#e3f2fd" stroke="#1565c0" strokeWidth={1.5} />
                <text x={150} y={60} textAnchor="middle" fontSize={13} fill="#1565c0" fontFamily="Inter, sans-serif">
                  Linear → ReLU → Linear
                </text>

                {/* Input arrow */}
                <line x1={150} y1={10} x2={150} y2={28} stroke="#1565c0" strokeWidth={2} markerEnd="url(#arrBlue)" />
                <text x={150} y={8} textAnchor="middle" fontSize={10} fill="#999" fontFamily="Inter, sans-serif">pair[i,j]</text>

                {/* Output arrow */}
                <line x1={150} y1={82} x2={150} y2={110} stroke="#1565c0" strokeWidth={2} markerEnd="url(#arrBlue)" />

                {/* Residual */}
                <path d="M30,55 Q10,55 10,90 Q10,120 150,120" fill="none" stroke="#999" strokeWidth={1.5} strokeDasharray="4,3" />
                <circle cx={150} cy={120} r={10} fill="#fff" stroke="#1565c0" strokeWidth={1.5} />
                <text x={150} y={124} textAnchor="middle" fontSize={12} fill="#1565c0" fontFamily="Inter, sans-serif">+</text>
                <text x={40} y={96} fontSize={9} fill="#999" fontFamily="Inter, sans-serif">residual</text>

                {/* LayerNorm */}
                <line x1={150} y1={132} x2={150} y2={155} stroke="#1565c0" strokeWidth={2} markerEnd="url(#arrBlue)" />
                <rect x={100} y={155} width={100} height={25} rx={4} fill="#e8eaf6" stroke="#5c6bc0" strokeWidth={1} />
                <text x={150} y={172} textAnchor="middle" fontSize={11} fill="#5c6bc0" fontFamily="Inter, sans-serif">LayerNorm</text>

                <defs>
                  <marker id="arrBlue" markerWidth={6} markerHeight={6} refX={5} refY={3} orient="auto">
                    <path d="M0,0 L6,3 L0,6 Z" fill="#1565c0" />
                  </marker>
                </defs>
              </svg>
            </div>
          )}
        </div>

        {/* Right: Explanation */}
        <div style={{
          width: 340, borderLeft: '1px solid #eee', overflowY: 'auto',
          padding: '20px 20px', flexShrink: 0,
        }}>
          {/* Category badge */}
          <div style={{
            display: 'inline-block', padding: '3px 10px', borderRadius: 12,
            background: cat.bg, border: `1px solid ${cat.border}`,
            fontSize: 11, color: cat.text, fontWeight: 600, marginBottom: 10,
          }}>
            {activeOp.category === 'msa' ? 'MSA Operation' :
             activeOp.category === 'pair' ? 'Pair Operation' : 'Cross-Representation'}
          </div>

          <h2 style={{ margin: '0 0 6px', fontSize: 16, color: '#1a237e', fontWeight: 700 }}>
            {activeOp.name}
          </h2>

          <p style={{ margin: '0 0 14px', fontSize: 13, color: '#333', lineHeight: 1.6 }}>
            {activeOp.description}
          </p>

          {activeOp.formula && (
            <div style={{
              background: '#f5f5f5', padding: '8px 12px', borderRadius: 6,
              fontFamily: 'monospace', fontSize: 12, color: '#555',
              marginBottom: 14, borderLeft: `3px solid ${cat.border}`,
            }}>
              {activeOp.formula}
            </div>
          )}

          <ul style={{
            margin: 0, padding: '0 0 0 16px', fontSize: 12.5,
            color: '#555', lineHeight: 1.8,
          }}>
            {activeOp.details.map((d, i) => (
              <li key={i}>{d}</li>
            ))}
          </ul>

          {/* Triangle k control for triangle ops */}
          {isTriangleOp && (
            <div style={{
              marginTop: 20, padding: '12px', background: '#f1f8e9',
              borderRadius: 8, border: '1px solid #c5e1a5',
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#33691e', marginBottom: 6 }}>
                Third residue k = {highlightK}
              </div>
              <input
                type="range" min={0} max={N - 1} value={highlightK}
                onChange={e => { setHighlightK(Number(e.target.value)); setPlaying(false) }}
                style={{ width: '100%' }}
              />
              <div style={{ fontSize: 11, color: '#689f38', marginTop: 4 }}>
                Drag to see different triangles. Currently scanning: (i={highlightI}, j={highlightJ}, k={highlightK})
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const smallBtnStyle: React.CSSProperties = {
  border: '1px solid #ccc', borderRadius: 4, background: '#fff',
  padding: '4px 10px', cursor: 'pointer', fontSize: 12, color: '#555',
}
