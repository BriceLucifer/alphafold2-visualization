import { useState, useEffect, useRef, useCallback } from 'react'
import { KaTeXFormula } from '../ui/KaTeXFormula'

// ── Structure Module operations (accurate to Algorithms 20-28) ──

interface SMOp {
  id: number
  name: string
  short: string
  category: 'ipa' | 'backbone' | 'sidechain' | 'loss'
  description: string
  details: string[]
  formula: string
  algRef: string
}

const SM_OPS: SMOp[] = [
  {
    id: 0,
    name: 'Invariant Point Attention (IPA)',
    short: 'IPA',
    category: 'ipa',
    description:
      'The core innovation of the Structure Module. IPA combines three attention signals: (1) standard scalar q·k attention on the single representation, (2) pair bias from the pair representation, and (3) 3D point attention using query/key points projected into global space via each residue\'s rigid frame T_i. The result is invariant under global Euclidean transformations.',
    details: [
      '12 heads, c=16 per head, N_query_points=4, N_point_values=8',
      'Scalar: w_L · (q_i · k_j / √c)',
      'Pair bias: b_ij = LinearNoBias(z_ij)',
      'Point: −(γ^h · w_C / 2) · Σ_p ||T_i ∘ q̃_i^p − T_j ∘ k̃_j^p||²',
      'w_L = √(1/3), w_C = √(2/(9·N_query_points)) — equal variance initialization',
      'γ^h = softplus(learnable) — per-head weight for point component',
      'Output: concat(scalar_out, pair_out, point_out, ||point_out||) → Linear',
      'Invariance proof: ||T_global ∘ T_i ∘ q̃ − T_global ∘ T_j ∘ k̃||² = ||T_i ∘ q̃ − T_j ∘ k̃||²',
    ],
    formula: 'a_{ij}^h = \\text{softmax}_j\\!\\left(w_L\\frac{\\mathbf{q}_i^{h\\top}\\mathbf{k}_j^h}{\\sqrt{c}} + b_{ij}^h - \\frac{\\gamma^h w_C}{2}\\sum_p \\left\\|T_i \\circ \\vec{\\mathbf{q}}_i^{hp} - T_j \\circ \\vec{\\mathbf{k}}_j^{hp}\\right\\|^2\\right)',
    algRef: 'Algorithm 22',
  },
  {
    id: 1,
    name: 'Backbone Frame Update',
    short: 'Backbone',
    category: 'backbone',
    description:
      'After IPA + transition, a small network predicts a quaternion (b,c,d) and translation Δt from the single representation s_i. The quaternion is converted to a rotation matrix R_i with first component fixed to 1: (1,b,c,d)/||(1,b,c,d)||. The frame is updated by composition: T_i ← T_i ∘ ΔT_i. Between iterations, rotation gradients are stopped to stabilize training.',
    details: [
      'Predict: b_i, c_i, d_i, Δt_i = Linear(s_i)',
      'Quaternion: (1, b, c, d) / √(1+b²+c²+d²) — forces near-identity initialization',
      'Frame composition: T_i^(l+1) = T_i^(l) ∘ (R_i, Δt_i)',
      'No rotation gradients between iterations: T ← (stopgrad(R), t) for stability',
      '"Black hole" init: all T_i = (I, 0⃗) — all residues start at origin',
      '8 layers with shared weights progressively unfold the structure',
    ],
    formula: 'T_i \\leftarrow T_i \\circ \\text{BackboneUpdate}(\\mathbf{s}_i), \\quad (1,b_i,c_i,d_i) / \\sqrt{1+b_i^2+c_i^2+d_i^2} \\to R_i',
    algRef: 'Algorithm 23',
  },
  {
    id: 2,
    name: 'Compute All Atom Coordinates',
    short: 'All Atoms',
    category: 'backbone',
    description:
      'From backbone frames T_i and predicted torsion angles, compute all atom positions. Atoms are organized into "rigid groups" — the backbone group (N, Cα, C, Cβ, O) uses the frame directly; side-chain atoms use nested compositions of torsion angle frames. Ideal bond geometry (lengths, angles) from literature is used — only rotations are predicted.',
    details: [
      'Backbone: N, Cα, C positions from ideal offsets rotated by frame T_i',
      'Side-chain: χ₁ frame composed on backbone, χ₂ on χ₁, etc.',
      'Torsion angles predicted as (sin χ, cos χ) ∈ ℝ² — avoids angle wrapping',
      'Each amino acid type has different rigid groups (Table 2 in paper)',
      'Example: Lys has 4 χ angles → 5 rigid groups',
      'Bond lengths: N-Cα = 1.458Å, Cα-C = 1.523Å, C-N = 1.329Å (fixed, not predicted)',
    ],
    formula: 'T_i^f, \\vec{x}_i^a = \\text{computeAllAtomCoordinates}(T_i, \\vec{\\alpha}_i^f)',
    algRef: 'Algorithm 24',
  },
  {
    id: 3,
    name: 'Side-chain Torsion Angles',
    short: 'Torsions χ₁-χ₄',
    category: 'sidechain',
    description:
      'A shallow ResNet predicts torsion angles as 2D vectors (sin χ, cos χ) from combined single + initial representations. This avoids discontinuity at ±180°. An auxiliary anglenorm loss encourages unit norm. Some side chains have 180°-rotation symmetry (ASP, GLU, PHE, TYR) — the network can predict either orientation.',
    details: [
      'Input: a_i = Linear(s_i) + Linear(s_i^initial)',
      'Network: two residual blocks of Linear(relu(Linear(relu(a)))) + a',
      'Output: α_i^f = Linear(relu(a_i)) ∈ ℝ², normalized to unit circle',
      'Torsion types: ω, φ, ψ (backbone) + χ₁, χ₂, χ₃, χ₄ (side-chain)',
      'L_anglenorm = 0.02 · mean(|||α|| - 1|) — regularizer for unit norm',
      '180° symmetric groups (ASP O^δ, GLU O^ε, PHE/TYR C^δ) can use alternate truth',
    ],
    formula: '\\vec{\\alpha}_i^f = \\text{Linear}(\\text{relu}(\\mathbf{a}_i)),\\quad \\tilde{\\alpha}_i^f = \\vec{\\alpha}_i^f / \\|\\vec{\\alpha}_i^f\\|',
    algRef: 'Algorithm 27',
  },
  {
    id: 4,
    name: 'FAPE Loss',
    short: 'FAPE',
    category: 'loss',
    description:
      'Frame Aligned Point Error — the primary training loss. For each frame-atom pair (i,j), transform atom j into frame i\'s local coordinates, compare to ground truth. This captures both position AND orientation accuracy without global alignment. Clamped at 10Å for robustness. FAPE is a pseudometric satisfying non-negativity, identity, symmetry, and triangle inequality.',
    details: [
      'Local coords: x_ij = T_i⁻¹ ∘ x_j (transform j\'s position into i\'s frame)',
      'Ground truth: x_ij^true = T_i^true⁻¹ ∘ x_j^true',
      'Distance: d_ij = √(||x_ij - x_ij^true||² + ε), ε = 10⁻⁴ Å²',
      'Clamping: min(d_clamp=10Å, d_ij) — robustness to outliers',
      'L_FAPE = (1/Z) · mean_{i,j}(min(d_clamp, d_ij)), Z = 10Å',
      'Auxiliary FAPE (Cα only) computed at each of the 8 layers',
      'Total loss: 0.5·L_FAPE + 0.5·L_aux + 0.3·L_dist + 2.0·L_msa + 0.01·L_conf',
    ],
    formula: '\\mathcal{L}_{\\text{FAPE}} = \\frac{1}{Z} \\text{mean}_{i,j} \\min\\!\\left(d_{\\text{clamp}},\\, \\sqrt{\\|T_i^{-1} \\circ \\vec{x}_j - T_i^{\\text{true}\\,-1} \\circ \\vec{x}_j^{\\text{true}}\\|^2 + \\epsilon}\\right)',
    algRef: 'Algorithm 28',
  },
]

// ── Dark theme ────────────────────────────────────────

const DARK = {
  bg: '#0a0a15',
  surface: 'rgba(20, 20, 40, 0.95)',
  border: 'rgba(255, 255, 255, 0.08)',
  borderLight: 'rgba(255, 255, 255, 0.12)',
  text: '#e0e0e0',
  textMuted: '#8899aa',
  textDim: '#556677',
}

const CAT_COLORS = {
  ipa: { bg: 'rgba(63, 81, 181, 0.12)', border: '#3f51b5', text: '#7986cb', glow: '#5c6bc0', accent: '#9fa8da' },
  backbone: { bg: 'rgba(0, 137, 123, 0.12)', border: '#00897b', text: '#4db6ac', glow: '#26a69a', accent: '#80cbc4' },
  sidechain: { bg: 'rgba(239, 108, 0, 0.12)', border: '#ef6c00', text: '#ffb74d', glow: '#ff9800', accent: '#ffcc02' },
  loss: { bg: 'rgba(198, 40, 40, 0.12)', border: '#c62828', text: '#ef9a9a', glow: '#ef5350', accent: '#e57373' },
}

// ── IPA Visualization ──────────────────────────────────

function IPAVisualization() {
  const residues = [
    { id: 0, x: 70, y: 180, angle: -30, label: 'Ala₁₂' },
    { id: 1, x: 185, y: 110, angle: 15, label: 'Gly₁₃' },
    { id: 2, x: 310, y: 145, angle: -10, label: 'Val₁₄' },
    { id: 3, x: 425, y: 90, angle: 25, label: 'Leu₁₅' },
    { id: 4, x: 530, y: 190, angle: -20, label: 'Ser₁₆' },
  ]

  const pointOffset = 18

  return (
    <svg viewBox="0 0 640 380" style={{ width: '100%', maxWidth: 780, height: 'auto' }}>
      <defs>
        <filter id="ipaGlow">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      <text x={320} y={18} textAnchor="middle" fontSize={12} fontWeight={600}
        fill="#c0c8d0" fontFamily="Inter, sans-serif">
        IPA: Three attention components combined
      </text>

      {/* Three component labels */}
      <g transform="translate(40, 30)">
        {[
          { x: 0, label: '① Scalar q·k', color: '#ef5350', desc: 'standard' },
          { x: 190, label: '② Pair bias b_ij', color: '#42a5f5', desc: 'from z_ij' },
          { x: 380, label: '③ Point distance', color: '#4caf50', desc: '3D invariant' },
        ].map((c, i) => (
          <g key={i}>
            <rect x={c.x} y={0} width={150} height={30} rx={4}
              fill={`${c.color}12`} stroke={`${c.color}40`} strokeWidth={1} />
            <text x={c.x + 75} y={14} textAnchor="middle" fontSize={10} fontWeight={600}
              fill={c.color} fontFamily="Inter, sans-serif">{c.label}</text>
            <text x={c.x + 75} y={25} textAnchor="middle" fontSize={8}
              fill={`${c.color}88`} fontFamily="Inter, sans-serif">{c.desc}</text>
          </g>
        ))}
      </g>

      {/* Attention lines */}
      {residues.map((r1, i) =>
        residues.map((r2, j) => {
          if (i >= j) return null
          const dist = Math.sqrt((r1.x - r2.x) ** 2 + (r1.y - r2.y) ** 2)
          const attnWeight = Math.max(0.05, Math.exp(-dist / 250))
          const isHighlight = (i === 0 && j === 3) || (i === 1 && j === 4)
          return (
            <g key={`a-${i}-${j}`}>
              <line x1={r1.x} y1={r1.y} x2={r2.x} y2={r2.y}
                stroke={isHighlight ? '#ff9100' : 'rgba(255,255,255,0.08)'}
                strokeWidth={isHighlight ? 2 : attnWeight * 2}
                opacity={isHighlight ? 0.6 : attnWeight * 0.4}
                strokeDasharray={isHighlight ? undefined : '3,3'}
              />
              {isHighlight && (
                <circle r={3} fill="#ff9100" opacity={0.7}>
                  <animateMotion dur="2s" repeatCount="indefinite"
                    path={`M${r1.x},${r1.y} L${r2.x},${r2.y}`} />
                </circle>
              )}
            </g>
          )
        })
      )}

      {/* Residue frames */}
      {residues.map(r => {
        const rad = (r.angle * Math.PI) / 180
        const axisLen = 24
        const xEnd = { x: r.x + Math.cos(rad) * axisLen, y: r.y + Math.sin(rad) * axisLen }
        const yEnd = { x: r.x + Math.cos(rad + Math.PI / 2) * axisLen * 0.7, y: r.y + Math.sin(rad + Math.PI / 2) * axisLen * 0.7 }
        const qx = r.x + Math.cos(rad + 0.5) * pointOffset
        const qy = r.y + Math.sin(rad + 0.5) * pointOffset

        return (
          <g key={r.id}>
            <circle cx={r.x} cy={r.y} r={22} fill="#3f51b5" opacity={0.06} />

            {/* Frame axes — R_i */}
            <line x1={r.x} y1={r.y} x2={xEnd.x} y2={xEnd.y}
              stroke="#ef5350" strokeWidth={1.5} opacity={0.6} />
            <line x1={r.x} y1={r.y} x2={yEnd.x} y2={yEnd.y}
              stroke="#4caf50" strokeWidth={1.5} opacity={0.6} />

            {/* Cα */}
            <circle cx={r.x} cy={r.y} r={7} fill="#3f51b5" opacity={0.8} />
            <text x={r.x} y={r.y + 2.5} textAnchor="middle" fontSize={6}
              fill="#fff" fontWeight={700} fontFamily="monospace">Cα</text>

            {/* Query/key point — T_i ∘ q̃ */}
            <circle cx={qx} cy={qy} r={3.5} fill="#4caf50" opacity={0.5}>
              <animate attributeName="opacity" values="0.3;0.7;0.3" dur="2s" repeatCount="indefinite" />
            </circle>
            <line x1={r.x} y1={r.y} x2={qx} y2={qy}
              stroke="#4caf50" strokeWidth={0.8} strokeDasharray="2,2" opacity={0.3} />

            <text x={r.x} y={r.y + 34} textAnchor="middle" fontSize={9}
              fill="#667788" fontFamily="Inter, sans-serif">{r.label}</text>
          </g>
        )
      })}

      {/* Distance annotation */}
      {[[0, 3], [1, 4]].map(([i, j]) => {
        const r1 = residues[i], r2 = residues[j]
        const mx = (r1.x + r2.x) / 2, my = (r1.y + r2.y) / 2
        return (
          <g key={`dist-${i}-${j}`}>
            <text x={mx} y={my - 10} textAnchor="middle" fontSize={9}
              fill="#ff9100" fontWeight={600} fontFamily="JetBrains Mono, monospace">
              {i === 0 ? '6.2Å' : '7.1Å'}
            </text>
            <text x={mx} y={my + 2} textAnchor="middle" fontSize={7}
              fill="#556677" fontFamily="Inter, sans-serif">
              ||T_i∘q̃ − T_j∘k̃||² → high attn
            </text>
          </g>
        )
      })}

      {/* Invariance note */}
      <g transform="translate(30, 260)">
        <rect x={0} y={0} width={580} height={46} rx={6}
          fill="rgba(76, 175, 80, 0.06)" stroke="rgba(76, 175, 80, 0.15)" strokeWidth={1} />
        <text x={290} y={16} textAnchor="middle" fontSize={10} fill="#4caf50" fontWeight={600} fontFamily="Inter, sans-serif">
          SE(3) Invariance — rotating the whole protein doesn't change attention
        </text>
        <text x={290} y={32} textAnchor="middle" fontSize={9} fill="#556677" fontFamily="JetBrains Mono, monospace">
          ||T_global ∘ T_i ∘ q̃ − T_global ∘ T_j ∘ k̃||² = ||T_i ∘ q̃ − T_j ∘ k̃||²
        </text>
      </g>

      {/* Legend */}
      <g transform="translate(30, 320)">
        <line x1={0} y1={5} x2={12} y2={5} stroke="#ef5350" strokeWidth={2} />
        <text x={16} y={9} fontSize={9} fill="#667788" fontFamily="Inter, sans-serif">x-axis (R)</text>
        <line x1={90} y1={5} x2={102} y2={5} stroke="#4caf50" strokeWidth={2} />
        <text x={106} y={9} fontSize={9} fill="#667788" fontFamily="Inter, sans-serif">y-axis (R)</text>
        <circle cx={195} cy={5} r={3.5} fill="#4caf50" opacity={0.5} />
        <text x={203} y={9} fontSize={9} fill="#667788" fontFamily="Inter, sans-serif">query/key point (T∘q̃)</text>
        <line x1={340} y1={5} x2={365} y2={5} stroke="#ff9100" strokeWidth={2} />
        <text x={370} y={9} fontSize={9} fill="#667788" fontFamily="Inter, sans-serif">high attention (3D close)</text>
      </g>
    </svg>
  )
}

// ── Backbone refinement ──────────────────────────────

function BackboneRefinement({ block }: { block: number }) {
  const nRes = 8
  const t = block / 7

  const flatPositions = Array.from({ length: nRes }, (_, i) => ({
    x: 75 + i * 60, y: 150,
  }))

  const foldedPositions = [
    { x: 75, y: 170 }, { x: 130, y: 110 }, { x: 190, y: 150 },
    { x: 250, y: 95 }, { x: 310, y: 130 }, { x: 370, y: 75 },
    { x: 430, y: 120 }, { x: 490, y: 160 },
  ]

  const positions = flatPositions.map((flat, i) => ({
    x: flat.x + (foldedPositions[i].x - flat.x) * t,
    y: flat.y + (foldedPositions[i].y - flat.y) * t,
  }))

  const labels = ['Ala', 'Gly', 'Val', 'Leu', 'Ile', 'Pro', 'Phe', 'Trp']

  return (
    <svg viewBox="0 0 580 310" style={{ width: '100%', maxWidth: 700, height: 'auto' }}>
      <text x={290} y={20} textAnchor="middle" fontSize={12} fontWeight={600}
        fill="#c0c8d0" fontFamily="Inter, sans-serif">
        {block === 0 ? 'Black Hole Initialization — all frames at identity' :
         `Backbone Refinement — Layer ${block + 1} of 8`}
      </text>

      {/* Block progress */}
      <g transform="translate(115, 34)">
        {Array.from({ length: 8 }).map((_, i) => (
          <g key={i}>
            <rect x={i * 44} y={0} width={38} height={15} rx={3}
              fill={i <= block ? '#00897b' : 'rgba(255,255,255,0.05)'}
              opacity={i === block ? 1 : (i < block ? 0.4 : 0.3)}
              stroke={i === block ? '#4db6ac' : 'transparent'} strokeWidth={1}
            />
            <text x={i * 44 + 19} y={11} textAnchor="middle" fontSize={8}
              fill={i <= block ? '#fff' : '#556677'} fontFamily="Inter, sans-serif">
              {i + 1}
            </text>
          </g>
        ))}
      </g>

      {/* Black hole indicator at block 0 */}
      {block === 0 && (
        <g>
          <circle cx={290} cy={150} r={30} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={1} strokeDasharray="4,3">
            <animate attributeName="r" values="25;35;25" dur="3s" repeatCount="indefinite" />
          </circle>
          <text x={290} y={200} textAnchor="middle" fontSize={9} fill="#556677" fontFamily="JetBrains Mono, monospace">
            T_i = (I, 0⃗) for all i
          </text>
        </g>
      )}

      {/* Backbone bonds */}
      {positions.map((p, i) => {
        if (i === 0) return null
        const prev = positions[i - 1]
        return (
          <line key={`b-${i}`} x1={prev.x} y1={prev.y} x2={p.x} y2={p.y}
            stroke="#00897b" strokeWidth={2.5} opacity={0.5 + t * 0.3} />
        )
      })}

      {/* Residues with frames */}
      {positions.map((p, i) => {
        const angle = i > 0
          ? Math.atan2(p.y - positions[i - 1].y, p.x - positions[i - 1].x) + t * (i * 0.3 - 0.8)
          : 0
        const axisLen = 12 * t + 5

        return (
          <g key={i}>
            {t > 0.15 && (
              <>
                <line x1={p.x} y1={p.y}
                  x2={p.x + Math.cos(angle) * axisLen}
                  y2={p.y + Math.sin(angle) * axisLen}
                  stroke="#ef5350" strokeWidth={1.2} opacity={t * 0.5} />
                <line x1={p.x} y1={p.y}
                  x2={p.x + Math.cos(angle + Math.PI / 2) * axisLen * 0.6}
                  y2={p.y + Math.sin(angle + Math.PI / 2) * axisLen * 0.6}
                  stroke="#4caf50" strokeWidth={1.2} opacity={t * 0.5} />
              </>
            )}

            {t > 0.35 && (
              <>
                <circle cx={p.x - 5 * Math.cos(angle)} cy={p.y - 5 * Math.sin(angle)}
                  r={2.5} fill="#42a5f5" opacity={t * 0.7} />
                <circle cx={p.x + 5 * Math.cos(angle)} cy={p.y + 5 * Math.sin(angle)}
                  r={2.5} fill="#78909c" opacity={t * 0.7} />
              </>
            )}

            <circle cx={p.x} cy={p.y} r={8} fill="#00897b" opacity={0.8} />
            <text x={p.x} y={p.y + 3} textAnchor="middle" fontSize={6}
              fill="#fff" fontWeight={700} fontFamily="monospace">Cα</text>
            <text x={p.x} y={p.y + 22} textAnchor="middle" fontSize={8}
              fill="#556677" fontFamily="Inter, sans-serif">{labels[i]}</text>
          </g>
        )
      })}

      {/* Stage label */}
      <text x={290} y={250} textAnchor="middle" fontSize={11} fill="#4db6ac" fontWeight={600}
        fontFamily="Inter, sans-serif">
        {block === 0 ? '"Black hole" — all residues stacked at origin' :
         block <= 2 ? 'Early: rough secondary structure emerges' :
         block <= 5 ? 'Middle: domains arrange, tertiary contacts form' :
         'Final: sub-Å refinement, side-chain ready'}
      </text>

      {/* Quaternion note */}
      <g transform="translate(100, 270)">
        <rect x={0} y={0} width={380} height={24} rx={4}
          fill="rgba(0, 137, 123, 0.06)" stroke="rgba(0, 137, 123, 0.15)" strokeWidth={1} />
        <text x={190} y={16} textAnchor="middle" fontSize={9} fill="#556677" fontFamily="JetBrains Mono, monospace">
          ΔT = ((1,b,c,d)/||·|| → R, Δt) — quaternion favors small rotations
        </text>
      </g>
    </svg>
  )
}

// ── Torsion angles ────────────────────────────────────

function TorsionAngles() {
  const angles = [
    { name: 'χ₁', bond: 'Cα-Cβ', value: -60, color: '#ef6c00' },
    { name: 'χ₂', bond: 'Cβ-Cγ', value: 180, color: '#f57c00' },
    { name: 'χ₃', bond: 'Cγ-Cδ', value: 65, color: '#ff9800' },
    { name: 'χ₄', bond: 'Cδ-Cε', value: -170, color: '#ffa726' },
  ]

  return (
    <svg viewBox="0 0 580 310" style={{ width: '100%', maxWidth: 700, height: 'auto' }}>
      <text x={290} y={20} textAnchor="middle" fontSize={12} fontWeight={600}
        fill="#c0c8d0" fontFamily="Inter, sans-serif">
        Side-chain Torsion Angles — Lysine (K)
      </text>

      <g transform="translate(30, 65)">
        {/* Backbone */}
        <line x1={0} y1={0} x2={45} y2={0} stroke="rgba(255,255,255,0.2)" strokeWidth={2.5} />
        <circle cx={0} cy={0} r={5} fill="#42a5f5" />
        <text x={0} y={-10} textAnchor="middle" fontSize={8} fill="#42a5f5" fontFamily="monospace">N</text>
        <circle cx={45} cy={0} r={7} fill="#00897b" />
        <text x={45} y={-10} textAnchor="middle" fontSize={8} fill="#00897b" fontFamily="monospace">Cα</text>
        <line x1={45} y1={0} x2={90} y2={0} stroke="rgba(255,255,255,0.2)" strokeWidth={2.5} />
        <circle cx={90} cy={0} r={5} fill="#78909c" />
        <text x={90} y={-10} textAnchor="middle" fontSize={8} fill="#78909c" fontFamily="monospace">C</text>

        {/* Side chain */}
        {angles.map((a, i) => {
          const baseY = 28 + i * 48
          const rad = (a.value * Math.PI) / 180
          const endX = 45 + Math.sin(rad) * 30
          const endY = baseY + Math.cos(rad) * 12

          return (
            <g key={a.name}>
              <line x1={45} y1={i === 0 ? 7 : baseY - 20} x2={45} y2={baseY}
                stroke={a.color} strokeWidth={2} opacity={0.7} />
              <circle cx={45} cy={baseY} r={15} fill="none" stroke={a.color}
                strokeWidth={1} strokeDasharray="2,2" opacity={0.25} />
              <line x1={45} y1={baseY} x2={endX} y2={endY + baseY * 0.04}
                stroke={a.color} strokeWidth={2} opacity={0.7} />
              <circle cx={endX} cy={endY + baseY * 0.04} r={4} fill={a.color} opacity={0.7}>
                <animate attributeName="opacity" values="0.4;0.9;0.4" dur="2s" repeatCount="indefinite"
                  begin={`${i * 0.4}s`} />
              </circle>
              <text x={95} y={baseY + 4} fontSize={11} fontWeight={700}
                fill={a.color} fontFamily="Inter, sans-serif">{a.name}</text>
              <text x={120} y={baseY + 4} fontSize={9}
                fill="#667788" fontFamily="Inter, sans-serif">
                {a.bond} = {a.value}°
              </text>
            </g>
          )
        })}

        {['Cβ', 'Cγ', 'Cδ', 'Cε', 'Nζ'].map((atom, i) => (
          <g key={atom}>
            <circle cx={45} cy={28 + i * 48} r={4} fill={i === 4 ? '#42a5f5' : '#ef6c00'} opacity={0.6} />
            <text x={32} y={28 + i * 48 + 3.5} textAnchor="end" fontSize={7}
              fill="#667788" fontFamily="monospace">{atom}</text>
          </g>
        ))}
      </g>

      {/* sin/cos representation */}
      <g transform="translate(290, 70)">
        <text x={0} y={0} fontSize={11} fontWeight={600} fill="#c0c8d0" fontFamily="Inter, sans-serif">
          Predicted as (sin χ, cos χ):
        </text>
        <text x={0} y={20} fontSize={10} fill="#667788" fontFamily="JetBrains Mono, monospace">
          → no discontinuity at ±180°
        </text>
        <text x={0} y={36} fontSize={10} fill="#667788" fontFamily="JetBrains Mono, monospace">
          → L_torsion ≡ cosine distance
        </text>

        {/* Unit circle diagram */}
        <g transform="translate(100, 100)">
          <circle cx={0} cy={0} r={40} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
          <line x1={-45} y1={0} x2={45} y2={0} stroke="rgba(255,255,255,0.05)" strokeWidth={0.5} />
          <line x1={0} y1={-45} x2={0} y2={45} stroke="rgba(255,255,255,0.05)" strokeWidth={0.5} />
          <text x={48} y={4} fontSize={7} fill="#556677" fontFamily="monospace">cos χ</text>
          <text x={3} y={-46} fontSize={7} fill="#556677" fontFamily="monospace">sin χ</text>

          {/* Point on circle */}
          <circle cx={0} cy={0} r={0} fill="#ff9800">
            <animate attributeName="cx" values="40;0;-40;0;40" dur="4s" repeatCount="indefinite" />
            <animate attributeName="cy" values="0;-40;0;40;0" dur="4s" repeatCount="indefinite" />
            <animate attributeName="r" values="4;4;4;4;4" dur="4s" repeatCount="indefinite" />
          </circle>
        </g>

        <text x={0} y={170} fontSize={10} fontWeight={600} fill="#c0c8d0" fontFamily="Inter, sans-serif">
          180° symmetric groups:
        </text>
        {['ASP: O^δ1 ↔ O^δ2', 'GLU: O^ε1 ↔ O^ε2', 'PHE/TYR: C^δ1↔C^δ2, C^ε1↔C^ε2'].map((s, i) => (
          <text key={i} x={0} y={188 + i * 15} fontSize={9} fill="#556677"
            fontFamily="JetBrains Mono, monospace">{s}</text>
        ))}
      </g>
    </svg>
  )
}

// ── FAPE Loss Visualization ───────────────────────────

function FAPEVisualization() {
  return (
    <svg viewBox="0 0 600 310" style={{ width: '100%', maxWidth: 740, height: 'auto' }}>
      <text x={300} y={18} textAnchor="middle" fontSize={12} fontWeight={600}
        fill="#c0c8d0" fontFamily="Inter, sans-serif">
        FAPE: Frame-Aligned Point Error
      </text>

      {/* Predicted */}
      <g transform="translate(30, 45)">
        <text x={100} y={0} textAnchor="middle" fontSize={11} fontWeight={600}
          fill="#7986cb" fontFamily="Inter, sans-serif">Predicted</text>

        {[
          { x: 35, y: 55, angle: -20 },
          { x: 110, y: 35, angle: 10 },
          { x: 185, y: 60, angle: -5 },
        ].map((f, i) => {
          const rad = (f.angle * Math.PI) / 180
          return (
            <g key={i}>
              <line x1={f.x} y1={f.y} x2={f.x + Math.cos(rad) * 18} y2={f.y + Math.sin(rad) * 18}
                stroke="#ef5350" strokeWidth={1.5} opacity={0.5} />
              <line x1={f.x} y1={f.y} x2={f.x + Math.cos(rad + Math.PI / 2) * 13}
                y2={f.y + Math.sin(rad + Math.PI / 2) * 13}
                stroke="#4caf50" strokeWidth={1.5} opacity={0.5} />
              <circle cx={f.x} cy={f.y} r={7} fill="#3f51b5" opacity={0.7} />
              <text x={f.x} y={f.y + 3} textAnchor="middle" fontSize={7}
                fill="#fff" fontWeight={700} fontFamily="monospace">{i + 1}</text>
            </g>
          )
        })}
        <line x1={35} y1={55} x2={110} y2={35} stroke="#3f51b5" strokeWidth={1.5} opacity={0.3} />
        <line x1={110} y1={35} x2={185} y2={60} stroke="#3f51b5" strokeWidth={1.5} opacity={0.3} />
      </g>

      {/* Ground truth */}
      <g transform="translate(320, 45)">
        <text x={100} y={0} textAnchor="middle" fontSize={11} fontWeight={600}
          fill="#4db6ac" fontFamily="Inter, sans-serif">Ground Truth</text>

        {[
          { x: 40, y: 50, angle: -25 },
          { x: 115, y: 33, angle: 15 },
          { x: 185, y: 58, angle: -8 },
        ].map((f, i) => {
          const rad = (f.angle * Math.PI) / 180
          return (
            <g key={i}>
              <line x1={f.x} y1={f.y} x2={f.x + Math.cos(rad) * 18} y2={f.y + Math.sin(rad) * 18}
                stroke="#ef5350" strokeWidth={1.5} opacity={0.5} />
              <line x1={f.x} y1={f.y} x2={f.x + Math.cos(rad + Math.PI / 2) * 13}
                y2={f.y + Math.sin(rad + Math.PI / 2) * 13}
                stroke="#4caf50" strokeWidth={1.5} opacity={0.5} />
              <circle cx={f.x} cy={f.y} r={7} fill="#00897b" opacity={0.7} />
              <text x={f.x} y={f.y + 3} textAnchor="middle" fontSize={7}
                fill="#fff" fontWeight={700} fontFamily="monospace">{i + 1}</text>
            </g>
          )
        })}
        <line x1={40} y1={50} x2={115} y2={33} stroke="#00897b" strokeWidth={1.5} opacity={0.3} />
        <line x1={115} y1={33} x2={185} y2={58} stroke="#00897b" strokeWidth={1.5} opacity={0.3} />
      </g>

      {/* FAPE explanation */}
      <g transform="translate(30, 160)">
        <rect x={0} y={0} width={540} height={80} rx={6}
          fill="rgba(198, 40, 40, 0.05)" stroke="rgba(198, 40, 40, 0.15)" strokeWidth={1} />

        <text x={270} y={20} textAnchor="middle" fontSize={10} fill="#ef5350" fontWeight={600} fontFamily="Inter, sans-serif">
          Key: align locally in each residue's frame, then measure
        </text>
        <text x={270} y={38} textAnchor="middle" fontSize={9} fill="#667788" fontFamily="JetBrains Mono, monospace">
          x_ij = T_i⁻¹ ∘ x_j (local coords)   →   d_ij = √(||x_ij − x_ij^true||² + ε)
        </text>
        <text x={270} y={54} textAnchor="middle" fontSize={9} fill="#667788" fontFamily="JetBrains Mono, monospace">
          L_FAPE = (1/Z) · mean_ij min(d_clamp=10Å, d_ij)
        </text>
        <text x={270} y={70} textAnchor="middle" fontSize={9} fill="#556677" fontFamily="Inter, sans-serif">
          Captures position AND orientation accuracy — no global alignment needed
        </text>
      </g>

      {/* Total loss breakdown */}
      <g transform="translate(30, 255)">
        <text x={0} y={12} fontSize={10} fontWeight={600} fill="#c0c8d0" fontFamily="Inter, sans-serif">
          Total loss:
        </text>
        <text x={75} y={12} fontSize={9} fill="#667788" fontFamily="JetBrains Mono, monospace">
          0.5·L_FAPE + 0.5·L_aux + 0.3·L_dist + 2.0·L_msa + 0.01·L_conf
        </text>

        {/* Pseudometric properties */}
        <text x={0} y={34} fontSize={9} fill="#556677" fontFamily="Inter, sans-serif">
          FAPE is a pseudometric: FAPE(X,X)=0, FAPE(X,Y)=FAPE(Y,X), triangle inequality ✓
        </text>
      </g>
    </svg>
  )
}

// ── Heat values ───────────────────────────────────────

const SM_HEAT: Record<number, number> = {
  0: 0.95, 1: 0.7, 2: 0.4, 3: 0.6, 4: 0.85,
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

// ── Step connector ────────────────────────────────────

function StepConnector({ done, active }: { done: boolean; active: boolean }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'center', padding: '0 0 0 22px', height: 14,
    }}>
      <div style={{
        width: 1.5, height: '100%',
        background: done ? '#26a69a' : active ? 'rgba(38, 166, 154, 0.4)' : 'rgba(255,255,255,0.06)',
        transition: 'background 0.3s',
        position: 'relative' as const,
      }}>
        {active && (
          <div style={{
            position: 'absolute', left: -3, top: 2,
            width: 7, height: 7, borderRadius: '50%', background: '#26a69a',
            animation: 'smPulse 1.5s infinite',
          }} />
        )}
      </div>
    </div>
  )
}

// ── Block list sidebar ────────────────────────────────

function BlockList({ ops, activeId, onSelect }: {
  ops: SMOp[]; activeId: number; onSelect: (id: number) => void
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', padding: '8px 0' }}>
      <style>{`
        @keyframes smPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(0.6); }
        }
        @keyframes smNext {
          0%, 100% { transform: translateX(0); opacity: 0.6; }
          50% { transform: translateX(3px); opacity: 1; }
        }
        @keyframes smFadeIn {
          from { opacity: 0; transform: translateY(8px) scale(0.97); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
      `}</style>

      <div style={{
        fontSize: 10, color: '#556677', textTransform: 'uppercase', letterSpacing: 1.5,
        padding: '0 12px 8px', fontFamily: 'Inter, sans-serif',
      }}>
        Structure Module
      </div>

      <div style={{
        padding: '4px 12px', fontSize: 9, color: '#556677',
        display: 'flex', alignItems: 'center', gap: 5,
        fontFamily: 'Inter, sans-serif',
      }}>
        <span style={{ fontSize: 10, opacity: 0.6 }}>→</span> s_i + z_ij in
      </div>
      <StepConnector done={activeId >= 0} active={activeId === 0} />

      {ops.map((op, idx) => {
        const cat = CAT_COLORS[op.category]
        const active = op.id === activeId
        const done = op.id < activeId
        const isNext = op.id === activeId + 1
        const heat = SM_HEAT[op.id] ?? 0.5
        const hc = heatColor(heat)

        return (
          <div key={op.id}>
            <div onClick={() => onSelect(op.id)}
              style={{
                padding: '5px 8px 5px 5px', borderRadius: 6, cursor: 'pointer',
                background: active ? cat.bg : done ? 'rgba(38, 166, 154, 0.04)' : 'transparent',
                border: active
                  ? `1.5px solid ${cat.border}80`
                  : isNext ? '1.5px dashed rgba(38, 166, 154, 0.3)' : '1.5px solid transparent',
                transition: 'all 0.25s',
                display: 'flex', alignItems: 'center', gap: 5,
                position: 'relative' as const, overflow: 'hidden', margin: '0 4px',
              }}>
              <div style={{
                position: 'absolute', left: 0, top: 0, bottom: 0,
                width: `${heat * 100}%`,
                background: active
                  ? `linear-gradient(90deg, ${cat.glow}10, transparent)`
                  : `linear-gradient(90deg, ${hc}08, transparent)`,
              }} />

              <div style={{
                width: 24, height: 24, borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 700, fontFamily: 'Inter, sans-serif',
                flexShrink: 0, position: 'relative' as const, zIndex: 1,
                background: active ? cat.border : done ? '#26a69a' : 'rgba(255,255,255,0.08)',
                color: (active || done) ? '#fff' : '#667788',
                boxShadow: active ? `0 0 10px ${cat.glow}44` : 'none',
              }}>
                {done ? '✓' : idx + 1}
              </div>

              <div style={{ position: 'relative' as const, zIndex: 1, flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: 11, fontWeight: active ? 700 : done ? 600 : 400,
                  color: active ? cat.text : done ? '#4db6ac' : '#8899aa',
                  fontFamily: 'Inter, sans-serif',
                  display: 'flex', alignItems: 'center', gap: 4,
                  whiteSpace: 'nowrap' as const,
                }}>
                  {op.short}
                  <span style={{
                    fontSize: 7, padding: '0px 3px', borderRadius: 4,
                    background: `${hc}18`, color: hc, fontWeight: 700,
                  }}>
                    {heat >= 0.8 ? 'HOT' : heat >= 0.5 ? 'MED' : 'LOW'}
                  </span>
                </div>
                <div style={{
                  fontSize: 9, color: active ? `${cat.accent}99` : '#445566',
                  fontFamily: 'Inter, sans-serif',
                }}>
                  {op.category === 'ipa' ? '3D Attention' : op.category === 'backbone' ? 'Geometry' :
                   op.category === 'sidechain' ? 'Side-chain' : 'Loss'}
                </div>
              </div>

              {isNext && (
                <div style={{
                  position: 'relative' as const, zIndex: 1,
                  fontSize: 9, color: '#26a69a', fontWeight: 600,
                  animation: 'smNext 1.5s infinite', flexShrink: 0,
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
        <span style={{ fontSize: 10, opacity: 0.6 }}>→</span> 3D coords + pLDDT
      </div>

      <div style={{
        margin: '10px 6px 0', padding: '6px 8px',
        background: 'rgba(38, 166, 154, 0.06)',
        borderRadius: 6, fontSize: 9, color: '#26a69a',
        textAlign: 'center', fontFamily: 'Inter, sans-serif',
        border: '1px solid rgba(38, 166, 154, 0.15)', fontWeight: 600,
      }}>
        ↻ IPA + Backbone × N_layer = 8 (shared weights)
      </div>
    </div>
  )
}

// ── Main Component ────────────────────────────────────

export function StructureModuleDetail({ onBack }: { onBack: () => void }) {
  const [activeOpId, setActiveOpId] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [backboneBlock, setBackboneBlock] = useState(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const blockIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const activeOp = SM_OPS[activeOpId]
  const cat = CAT_COLORS[activeOp.category]

  const goNext = useCallback(() => {
    setActiveOpId(id => (id + 1) % SM_OPS.length)
  }, [])

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(goNext, 8000)
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [playing, goNext])

  useEffect(() => {
    if ((activeOpId === 1 || activeOpId === 2) && playing) {
      blockIntervalRef.current = setInterval(() => {
        setBackboneBlock(b => (b + 1) % 8)
      }, 1500)
    }
    return () => { if (blockIntervalRef.current) clearInterval(blockIntervalRef.current) }
  }, [activeOpId, playing])

  const handleSelect = (id: number) => {
    setActiveOpId(id)
    setPlaying(false)
    if (id === 1 || id === 2) setBackboneBlock(0)
  }

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
        }}>← Back</button>
        <h1 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: '#c0c8d0' }}>
          Structure Module Deep Dive
        </h1>
        <span style={{ fontSize: 12, color: DARK.textDim }}>
          8 layers — abstract features → 3D atomic coordinates
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <button onClick={() => { setActiveOpId(id => (id - 1 + SM_OPS.length) % SM_OPS.length); setPlaying(false) }}
            style={darkBtn}>◀</button>
          <button onClick={() => setPlaying(p => !p)}
            style={{ ...darkBtn, width: 50 }}>{playing ? '⏸' : '▶'}</button>
          <button onClick={() => { goNext(); setPlaying(false) }}
            style={darkBtn}>▶</button>
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        {/* Left sidebar */}
        <div style={{
          width: 175, borderRight: `1px solid ${DARK.border}`, overflowY: 'auto',
          flexShrink: 0, background: DARK.surface,
        }}>
          <BlockList ops={SM_OPS} activeId={activeOpId} onSelect={handleSelect} />
        </div>

        {/* Center */}
        <div key={`smviz-${activeOpId}`} style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: 20, overflow: 'auto', minWidth: 0,
          animation: 'smFadeIn 0.4s ease-out',
        }}>
          {activeOpId === 0 && <IPAVisualization />}
          {activeOpId === 1 && <BackboneRefinement block={backboneBlock} />}
          {activeOpId === 2 && <BackboneRefinement block={7} />}
          {activeOpId === 3 && <TorsionAngles />}
          {activeOpId === 4 && <FAPEVisualization />}
        </div>

        {/* Right panel */}
        <div style={{
          width: 320, borderLeft: `1px solid ${DARK.border}`, overflowY: 'auto',
          padding: 16, flexShrink: 0, background: DARK.surface,
        }}>
          <div style={{
            display: 'inline-block', padding: '2px 8px', borderRadius: 10,
            background: cat.bg, border: `1px solid ${cat.border}60`,
            fontSize: 10, color: cat.text, fontWeight: 600, marginBottom: 8,
          }}>
            {activeOp.category === 'ipa' ? '3D Attention' :
             activeOp.category === 'backbone' ? 'Geometry' :
             activeOp.category === 'sidechain' ? 'Side-chain' : 'Loss'}
          </div>
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

          {(activeOpId === 1 || activeOpId === 2) && (
            <div style={{
              marginTop: 16, padding: '10px 12px',
              background: 'rgba(0, 137, 123, 0.06)',
              borderRadius: 6, border: '1px solid rgba(0, 137, 123, 0.15)',
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#4db6ac', marginBottom: 4 }}>
                Refinement block: {backboneBlock + 1} / 8
              </div>
              <input type="range" min={0} max={7} value={backboneBlock}
                onChange={e => { setBackboneBlock(Number(e.target.value)); setPlaying(false) }}
                style={{ width: '100%', accentColor: '#26a69a' }} />
              <div style={{ fontSize: 10, color: '#556677', marginTop: 3 }}>
                Drag to see progressive unfolding from "black hole" → folded
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const darkBtn: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, background: 'rgba(255,255,255,0.05)',
  padding: '4px 10px', cursor: 'pointer', fontSize: 12, color: '#8899aa',
}
