import { useState, useEffect, useRef, useCallback } from 'react'

// ── Structure Module operations ────────────────────────

interface SMOp {
  id: number
  name: string
  short: string
  category: 'ipa' | 'backbone' | 'sidechain' | 'loss'
  description: string
  details: string[]
  formula?: string
}

const SM_OPS: SMOp[] = [
  {
    id: 0,
    name: 'Invariant Point Attention (IPA)',
    short: 'IPA',
    category: 'ipa',
    description:
      'The core innovation of the Structure Module. Unlike standard attention that only uses sequence features, IPA also projects 3D "query points" and "key points" into space using each residue\'s rigid body frame, then computes distances between them. This lets the model reason directly in 3D.',
    details: [
      'Each residue has a frame T = (R, t): rotation matrix R ∈ SO(3) and translation t ∈ ℝ³',
      'Standard attention component: Q·K from single representation (like a normal transformer)',
      'Pair bias component: attention logits += pair_repr[i,j] (structural context)',
      '3D point component: project learned query/key points into global frame, compute squared distances',
      'Final attention = softmax(standard + pair_bias + point_distance)',
      'Example: residue 42 in an α-helix projects points along its helix axis; residue 87 in a nearby β-strand projects points toward it — their 3D distance is small → high attention',
      '"Invariant" = rotating the entire protein doesn\'t change the output, because only relative distances matter',
    ],
    formula: 'α_ij = softmax(q_i·k_j + b_ij + Σ_p ||T_i·q_p - T_j·k_p||²)',
  },
  {
    id: 1,
    name: 'Backbone Frame Update',
    short: 'Backbone',
    category: 'backbone',
    description:
      'After IPA computes updated features for each residue, a small network predicts a rotation and translation update to each residue\'s frame. This incrementally moves the backbone atoms (N, Cα, C) toward their correct positions over 8 iterative blocks.',
    details: [
      'Each residue\'s frame T_i is updated: T_i ← T_i ∘ ΔT_i (compose with predicted delta)',
      'ΔT_i = (ΔR_i, Δt_i) predicted from the IPA output features via a small MLP',
      'Block 1: rough fold — gets secondary structure approximately right',
      'Block 4: domain-level arrangement improves significantly',
      'Block 8: fine-grained refinement — sub-angstrom corrections',
      'The frames start at identity (all residues at origin) and gradually "unfold" into the 3D structure',
      'Example: a β-hairpin starts flat, then block by block the two strands separate and twist into position',
    ],
    formula: 'T_i^(l+1) = T_i^(l) ∘ MLP(s_i^(l))',
  },
  {
    id: 2,
    name: 'Backbone Atom Positions',
    short: 'N-Cα-C',
    category: 'backbone',
    description:
      'From each residue\'s rigid body frame, the three backbone atom positions (N, Cα, C) are computed using fixed ideal bond lengths and angles. The frame\'s rotation determines the peptide plane orientation; the translation places it in space.',
    details: [
      'Cα position = the translation component of the frame: t_i',
      'N position = Cα + R_i · offset_N (offset_N is a fixed vector based on ideal bond geometry)',
      'C position = Cα + R_i · offset_C (similarly fixed)',
      'Bond lengths: N-Cα = 1.458Å, Cα-C = 1.523Å, C-N = 1.329Å (peptide bond)',
      'These are NOT predicted — they use ideal geometry. Only the frame (R, t) is learned.',
      'Example: for residue 100 with frame at position (23.1, 15.7, -8.3)Å and rotation facing "up", N is placed 1.458Å in the N-direction, C is placed 1.523Å in the C-direction',
    ],
    formula: 'x_N = t_i + R_i · [−0.526, 1.362, 0.000]',
  },
  {
    id: 3,
    name: 'Side-chain Torsion Angles',
    short: 'Torsions χ₁-χ₄',
    category: 'sidechain',
    description:
      'After placing the backbone, AlphaFold2 predicts side-chain torsion angles (χ₁, χ₂, χ₃, χ₄) for each amino acid. These angles describe rotations around the side-chain bonds and determine where each atom of the side-chain ends up in 3D space.',
    details: [
      'Each amino acid type has a different number of χ angles: Ala has 0, Leu has 2, Lys has 4, Trp has 2',
      'Predicted as (sin χ, cos χ) pairs to avoid angle wrapping issues at ±180°',
      'Uses the single representation features as input, processed through a small network',
      'The full atomic structure is built by applying χ angles to an ideal side-chain template (rotamer library)',
      'Example: Leucine (L) — χ₁ rotates around Cα-Cβ bond (~-60°, 180°, or 60°), χ₂ rotates around Cβ-Cγ',
      'Getting χ₁ wrong by 120° can misplace the side-chain tip by >5Å — critical for protein-ligand binding',
    ],
    formula: 'χ angles predicted as (sin χ, cos χ) = MLP(s_i)',
  },
  {
    id: 4,
    name: 'FAPE Loss',
    short: 'FAPE',
    category: 'loss',
    description:
      'Frame-Aligned Point Error — AlphaFold2\'s primary training loss. Instead of just comparing atom positions globally, FAPE aligns each residue\'s predicted local frame to the true frame, then measures distances. This captures both position AND orientation accuracy.',
    details: [
      'For each pair of residues (i, j): transform j\'s atoms into i\'s local frame, measure distance to ground truth',
      'FAPE = mean over all (i,j) pairs of ||T_i⁻¹ · x_j^pred − T_i⁻¹ · x_j^true||',
      'Why not just RMSD? RMSD requires a global alignment; FAPE captures local accuracy without it',
      'A residue can have low FAPE even if the global position is off — as long as local geometry is correct',
      'This is crucial for multi-domain proteins where domains may be correct individually but misoriented',
      'Example: two domains connected by a flexible linker — FAPE correctly credits each domain\'s internal accuracy even if the relative orientation is uncertain',
      'Also uses auxiliary losses: pLDDT head, distogram loss, masked MSA loss',
    ],
    formula: 'FAPE = (1/N²) Σ_i,j ||T_i⁻¹ · x_j^pred − T_i⁻¹ · x_j^true||',
  },
]

// ── Colors ─────────────────────────────────────────────

const CAT_COLORS = {
  ipa: { bg: '#e8eaf6', border: '#3f51b5', text: '#1a237e', glow: '#5c6bc0' },
  backbone: { bg: '#e0f2f1', border: '#00897b', text: '#004d40', glow: '#26a69a' },
  sidechain: { bg: '#fff3e0', border: '#ef6c00', text: '#e65100', glow: '#ff9800' },
  loss: { bg: '#fce4ec', border: '#c62828', text: '#b71c1c', glow: '#ef5350' },
}

// ── IPA Visualization ──────────────────────────────────

function IPAVisualization({ step: _step }: { step: number }) {
  // Show residues as frames in 2D, with attention between them
  const residues = [
    { id: 0, x: 80, y: 200, angle: -30, label: 'Ala₁₂', color: '#3f51b5' },
    { id: 1, x: 200, y: 120, angle: 15, label: 'Gly₁₃', color: '#3f51b5' },
    { id: 2, x: 340, y: 160, angle: -10, label: 'Val₁₄', color: '#3f51b5' },
    { id: 3, x: 460, y: 100, angle: 25, label: 'Leu₁₅', color: '#3f51b5' },
    { id: 4, x: 560, y: 210, angle: -20, label: 'Ser₁₆', color: '#3f51b5' },
  ]

  // Query/key point offsets (in local frame)
  const pointOffset = 20

  return (
    <svg viewBox="0 0 660 340" style={{ width: '100%', maxWidth: 660, height: 'auto' }}>
      <text x={330} y={20} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="#333" fontFamily="Inter, sans-serif">Invariant Point Attention — 3D-aware attention</text>

      {/* Attention lines (show which residues attend to which) */}
      {residues.map((r1, i) =>
        residues.map((r2, j) => {
          if (i >= j) return null
          const dist = Math.sqrt((r1.x - r2.x) ** 2 + (r1.y - r2.y) ** 2)
          const attnWeight = Math.max(0.05, Math.exp(-dist / 250))
          const isHighlight = (i === 0 && j === 3) || (i === 1 && j === 4)
          return (
            <g key={`a-${i}-${j}`}>
              <line x1={r1.x} y1={r1.y} x2={r2.x} y2={r2.y}
                stroke={isHighlight ? '#ff6f00' : '#90a4ae'}
                strokeWidth={isHighlight ? 2.5 : attnWeight * 3}
                opacity={isHighlight ? 0.7 : attnWeight * 0.5}
                strokeDasharray={isHighlight ? undefined : '4,3'}
              />
              {isHighlight && (
                <circle r={3.5} fill="#ff6f00" opacity={0.8}>
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
        const axisLen = 28
        // Local x and y axes of the frame
        const xEnd = { x: r.x + Math.cos(rad) * axisLen, y: r.y + Math.sin(rad) * axisLen }
        const yEnd = { x: r.x + Math.cos(rad + Math.PI / 2) * axisLen * 0.7, y: r.y + Math.sin(rad + Math.PI / 2) * axisLen * 0.7 }

        // Query point projected into space
        const qx = r.x + Math.cos(rad + 0.5) * pointOffset
        const qy = r.y + Math.sin(rad + 0.5) * pointOffset

        return (
          <g key={r.id}>
            {/* Frame glow */}
            <circle cx={r.x} cy={r.y} r={24} fill={r.color} opacity={0.08} />

            {/* Frame axes */}
            <line x1={r.x} y1={r.y} x2={xEnd.x} y2={xEnd.y}
              stroke="#c62828" strokeWidth={2} opacity={0.7} />
            <line x1={r.x} y1={r.y} x2={yEnd.x} y2={yEnd.y}
              stroke="#2e7d32" strokeWidth={2} opacity={0.7} />

            {/* Cα position */}
            <circle cx={r.x} cy={r.y} r={8} fill={r.color} opacity={0.85} />
            <text x={r.x} y={r.y + 3} textAnchor="middle" fontSize={7}
              fill="#fff" fontWeight={700} fontFamily="monospace">Cα</text>

            {/* Query/key point */}
            <circle cx={qx} cy={qy} r={4} fill="#ff6f00" opacity={0.6}>
              <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2s" repeatCount="indefinite" />
            </circle>
            <line x1={r.x} y1={r.y} x2={qx} y2={qy}
              stroke="#ff6f00" strokeWidth={1} strokeDasharray="2,2" opacity={0.4} />

            {/* Label */}
            <text x={r.x} y={r.y + 38} textAnchor="middle" fontSize={10}
              fill="#555" fontFamily="Inter, sans-serif">{r.label}</text>
          </g>
        )
      })}

      {/* Distance measurement lines between close residue pairs */}
      {[[0, 3], [1, 4]].map(([i, j]) => {
        const r1 = residues[i], r2 = residues[j]
        const mx = (r1.x + r2.x) / 2, my = (r1.y + r2.y) / 2
        return (
          <g key={`dist-${i}-${j}`}>
            {/* Distance indicator */}
            <text x={mx} y={my - 12} textAnchor="middle" fontSize={9}
              fill="#ff6f00" fontWeight={600} fontFamily="JetBrains Mono, monospace">
              {i === 0 ? '6.2Å' : '7.1Å'}
            </text>
            <text x={mx} y={my + 2} textAnchor="middle" fontSize={8}
              fill="#999" fontFamily="Inter, sans-serif">
              3D close!
            </text>
          </g>
        )
      })}

      {/* Sequence distance indicator */}
      <g transform="translate(20, 265)">
        <rect x={0} y={0} width={620} height={30} rx={6} fill="#fff3e0" opacity={0.5} />
        <text x={10} y={19} fontSize={10} fill="#e65100" fontWeight={600} fontFamily="Inter, sans-serif">
          Key insight:
        </text>
        <text x={90} y={19} fontSize={10} fill="#555" fontFamily="Inter, sans-serif">
          Residues 12↔15 are 3 apart in sequence but 6.2Å in 3D → IPA gives high attention
        </text>
      </g>

      {/* Legend */}
      <g transform="translate(20, 310)">
        <line x1={0} y1={5} x2={15} y2={5} stroke="#c62828" strokeWidth={2} />
        <text x={20} y={9} fontSize={10} fill="#666" fontFamily="Inter, sans-serif">x-axis (R)</text>
        <line x1={100} y1={5} x2={115} y2={5} stroke="#2e7d32" strokeWidth={2} />
        <text x={120} y={9} fontSize={10} fill="#666" fontFamily="Inter, sans-serif">y-axis (R)</text>
        <circle cx={210} cy={5} r={4} fill="#ff6f00" opacity={0.6} />
        <text x={220} y={9} fontSize={10} fill="#666" fontFamily="Inter, sans-serif">query/key point</text>
        <line x1={340} y1={5} x2={370} y2={5} stroke="#ff6f00" strokeWidth={2.5} />
        <text x={375} y={9} fontSize={10} fill="#666" fontFamily="Inter, sans-serif">high attention (3D-close)</text>
      </g>
    </svg>
  )
}

// ── Backbone refinement visualization ──────────────────

function BackboneRefinement({ block }: { block: number }) {
  // Show a small protein chain being refined over blocks
  // block 0 = flat line, block 7 = final folded structure
  const nRes = 8
  const t = block / 7 // 0 to 1

  // Interpolate from flat to folded
  const flatPositions = Array.from({ length: nRes }, (_, i) => ({
    x: 80 + i * 65, y: 160,
  }))

  const foldedPositions = [
    { x: 80, y: 180 }, { x: 140, y: 120 }, { x: 200, y: 160 },
    { x: 260, y: 100 }, { x: 320, y: 140 }, { x: 380, y: 80 },
    { x: 440, y: 130 }, { x: 500, y: 170 },
  ]

  const positions = flatPositions.map((flat, i) => ({
    x: flat.x + (foldedPositions[i].x - flat.x) * t,
    y: flat.y + (foldedPositions[i].y - flat.y) * t,
  }))

  const labels = ['Ala', 'Gly', 'Val', 'Leu', 'Ile', 'Pro', 'Phe', 'Trp']

  return (
    <svg viewBox="0 0 600 320" style={{ width: '100%', maxWidth: 600, height: 'auto' }}>
      <text x={300} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="#333" fontFamily="Inter, sans-serif">
        Backbone Refinement — Block {block + 1} of 8
      </text>

      {/* Block progress bar */}
      <g transform="translate(120, 35)">
        {Array.from({ length: 8 }).map((_, i) => (
          <g key={i}>
            <rect x={i * 45} y={0} width={38} height={16} rx={3}
              fill={i <= block ? '#00897b' : '#e0e0e0'}
              opacity={i === block ? 1 : (i < block ? 0.5 : 0.3)}
            />
            <text x={i * 45 + 19} y={12} textAnchor="middle" fontSize={9}
              fill={i <= block ? '#fff' : '#999'} fontFamily="Inter, sans-serif">
              {i + 1}
            </text>
          </g>
        ))}
      </g>

      {/* Backbone chain */}
      {positions.map((p, i) => {
        if (i === 0) return null
        const prev = positions[i - 1]
        return (
          <line key={`b-${i}`} x1={prev.x} y1={prev.y} x2={p.x} y2={p.y}
            stroke="#00897b" strokeWidth={3} opacity={0.6} />
        )
      })}

      {/* Residue nodes with frames */}
      {positions.map((p, i) => {
        const angle = i > 0
          ? Math.atan2(p.y - positions[i - 1].y, p.x - positions[i - 1].x) + t * (i * 0.3 - 0.8)
          : 0
        const axisLen = 14 * t + 6

        return (
          <g key={i}>
            {/* Frame axes (appear as refinement progresses) */}
            {t > 0.2 && (
              <>
                <line x1={p.x} y1={p.y}
                  x2={p.x + Math.cos(angle) * axisLen}
                  y2={p.y + Math.sin(angle) * axisLen}
                  stroke="#c62828" strokeWidth={1.5} opacity={t * 0.6} />
                <line x1={p.x} y1={p.y}
                  x2={p.x + Math.cos(angle + Math.PI / 2) * axisLen * 0.6}
                  y2={p.y + Math.sin(angle + Math.PI / 2) * axisLen * 0.6}
                  stroke="#2e7d32" strokeWidth={1.5} opacity={t * 0.6} />
              </>
            )}

            {/* N, Cα, C atoms */}
            {t > 0.4 && (
              <>
                <circle cx={p.x - 6 * Math.cos(angle)} cy={p.y - 6 * Math.sin(angle)}
                  r={3} fill="#1565c0" opacity={t * 0.8} />
                <circle cx={p.x + 6 * Math.cos(angle)} cy={p.y + 6 * Math.sin(angle)}
                  r={3} fill="#555" opacity={t * 0.8} />
              </>
            )}

            <circle cx={p.x} cy={p.y} r={10} fill="#00897b" opacity={0.85} />
            <text x={p.x} y={p.y + 3.5} textAnchor="middle" fontSize={7}
              fill="#fff" fontWeight={700} fontFamily="monospace">Cα</text>
            <text x={p.x} y={p.y + 24} textAnchor="middle" fontSize={9}
              fill="#666" fontFamily="Inter, sans-serif">{labels[i]}</text>
          </g>
        )
      })}

      {/* Stage description */}
      <text x={300} y={260} textAnchor="middle" fontSize={12} fill="#00897b" fontWeight={600}
        fontFamily="Inter, sans-serif">
        {block === 0 ? 'Starting: all frames at identity (flat)' :
         block <= 2 ? 'Early blocks: rough fold emerges' :
         block <= 5 ? 'Middle blocks: secondary structure forms, domains arrange' :
         'Final blocks: sub-angstrom refinement'}
      </text>

      {t > 0.4 && (
        <g transform="translate(140, 280)">
          <circle cx={0} cy={5} r={3} fill="#1565c0" />
          <text x={8} y={9} fontSize={9} fill="#666" fontFamily="Inter, sans-serif">N atom</text>
          <circle cx={80} cy={5} r={5} fill="#00897b" />
          <text x={90} y={9} fontSize={9} fill="#666" fontFamily="Inter, sans-serif">Cα atom</text>
          <circle cx={160} cy={5} r={3} fill="#555" />
          <text x={168} y={9} fontSize={9} fill="#666" fontFamily="Inter, sans-serif">C atom</text>
        </g>
      )}
    </svg>
  )
}

// ── Side-chain torsion visualization ───────────────────

function TorsionAngles() {
  const angles = [
    { name: 'χ₁', bond: 'Cα-Cβ', value: -60, color: '#ef6c00' },
    { name: 'χ₂', bond: 'Cβ-Cγ', value: 180, color: '#f57c00' },
    { name: 'χ₃', bond: 'Cγ-Cδ', value: 65, color: '#ff9800' },
    { name: 'χ₄', bond: 'Cδ-Cε', value: -170, color: '#ffa726' },
  ]

  return (
    <svg viewBox="0 0 600 340" style={{ width: '100%', maxWidth: 600, height: 'auto' }}>
      <text x={300} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="#333" fontFamily="Inter, sans-serif">
        Side-chain Torsion Angles — Example: Lysine (K)
      </text>

      {/* Backbone */}
      <g transform="translate(30, 100)">
        <line x1={0} y1={0} x2={50} y2={0} stroke="#999" strokeWidth={3} />
        <circle cx={0} cy={0} r={6} fill="#1565c0" />
        <text x={0} y={-12} textAnchor="middle" fontSize={9} fill="#1565c0" fontFamily="monospace">N</text>
        <circle cx={50} cy={0} r={8} fill="#00897b" />
        <text x={50} y={-12} textAnchor="middle" fontSize={9} fill="#00897b" fontFamily="monospace">Cα</text>
        <line x1={50} y1={0} x2={100} y2={0} stroke="#999" strokeWidth={3} />
        <circle cx={100} cy={0} r={6} fill="#555" />
        <text x={100} y={-12} textAnchor="middle" fontSize={9} fill="#555" fontFamily="monospace">C</text>

        {/* Side chain going down */}
        {angles.map((a, i) => {
          const baseY = 30 + i * 55
          const rad = (a.value * Math.PI) / 180
          const endX = 50 + Math.sin(rad) * 35
          const endY = baseY + Math.cos(rad) * 15

          return (
            <g key={a.name}>
              <line x1={50} y1={i === 0 ? 8 : baseY - 25} x2={50} y2={baseY}
                stroke={a.color} strokeWidth={2.5} />

              {/* Rotation arc */}
              <circle cx={50} cy={baseY} r={18} fill="none" stroke={a.color}
                strokeWidth={1.5} strokeDasharray="3,3" opacity={0.4} />

              {/* Torsion angle indicator */}
              <line x1={50} y1={baseY} x2={endX} y2={endY + baseY * 0.05}
                stroke={a.color} strokeWidth={2.5} />

              {/* Rotating indicator */}
              <circle cx={endX} cy={endY + baseY * 0.05} r={5} fill={a.color} opacity={0.8}>
                <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"
                  begin={`${i * 0.5}s`} />
              </circle>

              {/* Label */}
              <text x={100} y={baseY + 5} fontSize={12} fontWeight={700}
                fill={a.color} fontFamily="Inter, sans-serif">{a.name}</text>
              <text x={130} y={baseY + 5} fontSize={10}
                fill="#888" fontFamily="Inter, sans-serif">
                around {a.bond} = {a.value}°
              </text>
            </g>
          )
        })}

        {/* Atom labels along side chain */}
        {['Cβ', 'Cγ', 'Cδ', 'Cε', 'Nζ'].map((atom, i) => (
          <g key={atom}>
            <circle cx={50} cy={30 + i * 55} r={5} fill={i === 4 ? '#1565c0' : '#ef6c00'} opacity={0.7} />
            <text x={35} y={30 + i * 55 + 4} textAnchor="end" fontSize={8}
              fill="#555" fontFamily="monospace">{atom}</text>
          </g>
        ))}
      </g>

      {/* Rotamer explanation */}
      <g transform="translate(300, 80)">
        <text x={0} y={0} fontSize={12} fontWeight={600} fill="#333" fontFamily="Inter, sans-serif">
          Common rotamers for Leu:
        </text>
        {[
          { chi1: -60, chi2: -60, pop: '65%' },
          { chi1: -60, chi2: 180, pop: '20%' },
          { chi1: 180, chi2: 60, pop: '10%' },
        ].map((rot, i) => (
          <text key={i} x={0} y={22 + i * 18} fontSize={11} fill="#666" fontFamily="monospace">
            χ₁={rot.chi1}° χ₂={rot.chi2}° ({rot.pop})
          </text>
        ))}

        <text x={0} y={90} fontSize={11} fill="#999" fontStyle="italic" fontFamily="Inter, sans-serif">
          Wrong rotamer → side-chain tip
        </text>
        <text x={0} y={104} fontSize={11} fill="#999" fontStyle="italic" fontFamily="Inter, sans-serif">
          misplaced by 3–5Å
        </text>

        <text x={0} y={140} fontSize={12} fontWeight={600} fill="#333" fontFamily="Inter, sans-serif">
          Predicted as (sin χ, cos χ):
        </text>
        <text x={0} y={158} fontSize={11} fill="#666" fontFamily="monospace">
          → avoids discontinuity at ±180°
        </text>
        <text x={0} y={176} fontSize={11} fill="#666" fontFamily="monospace">
          → smooth loss landscape
        </text>
      </g>
    </svg>
  )
}

// ── FAPE Loss Visualization ────────────────────────────

function FAPEVisualization() {
  return (
    <svg viewBox="0 0 620 320" style={{ width: '100%', maxWidth: 620, height: 'auto' }}>
      <text x={310} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="#333" fontFamily="Inter, sans-serif">
        FAPE: Frame-Aligned Point Error
      </text>

      {/* Predicted structure */}
      <g transform="translate(40, 50)">
        <text x={100} y={0} textAnchor="middle" fontSize={12} fontWeight={600}
          fill="#3f51b5" fontFamily="Inter, sans-serif">Predicted</text>

        {/* Frames */}
        {[
          { x: 40, y: 60, angle: -20 },
          { x: 120, y: 40, angle: 10 },
          { x: 200, y: 70, angle: -5 },
        ].map((f, i) => {
          const rad = (f.angle * Math.PI) / 180
          return (
            <g key={i}>
              <line x1={f.x} y1={f.y} x2={f.x + Math.cos(rad) * 20} y2={f.y + Math.sin(rad) * 20}
                stroke="#c62828" strokeWidth={2} opacity={0.6} />
              <line x1={f.x} y1={f.y} x2={f.x + Math.cos(rad + Math.PI / 2) * 15}
                y2={f.y + Math.sin(rad + Math.PI / 2) * 15}
                stroke="#2e7d32" strokeWidth={2} opacity={0.6} />
              <circle cx={f.x} cy={f.y} r={8} fill="#3f51b5" opacity={0.8} />
              <text x={f.x} y={f.y + 3} textAnchor="middle" fontSize={8}
                fill="#fff" fontWeight={700} fontFamily="monospace">{i + 1}</text>
            </g>
          )
        })}

        {/* Backbone */}
        <line x1={40} y1={60} x2={120} y2={40} stroke="#3f51b5" strokeWidth={2} opacity={0.4} />
        <line x1={120} y1={40} x2={200} y2={70} stroke="#3f51b5" strokeWidth={2} opacity={0.4} />
      </g>

      {/* True structure */}
      <g transform="translate(340, 50)">
        <text x={100} y={0} textAnchor="middle" fontSize={12} fontWeight={600}
          fill="#2e7d32" fontFamily="Inter, sans-serif">Ground Truth</text>

        {[
          { x: 45, y: 55, angle: -25 },
          { x: 125, y: 38, angle: 15 },
          { x: 195, y: 65, angle: -8 },
        ].map((f, i) => {
          const rad = (f.angle * Math.PI) / 180
          return (
            <g key={i}>
              <line x1={f.x} y1={f.y} x2={f.x + Math.cos(rad) * 20} y2={f.y + Math.sin(rad) * 20}
                stroke="#c62828" strokeWidth={2} opacity={0.6} />
              <line x1={f.x} y1={f.y} x2={f.x + Math.cos(rad + Math.PI / 2) * 15}
                y2={f.y + Math.sin(rad + Math.PI / 2) * 15}
                stroke="#2e7d32" strokeWidth={2} opacity={0.6} />
              <circle cx={f.x} cy={f.y} r={8} fill="#2e7d32" opacity={0.8} />
              <text x={f.x} y={f.y + 3} textAnchor="middle" fontSize={8}
                fill="#fff" fontWeight={700} fontFamily="monospace">{i + 1}</text>
            </g>
          )
        })}

        <line x1={45} y1={55} x2={125} y2={38} stroke="#2e7d32" strokeWidth={2} opacity={0.4} />
        <line x1={125} y1={38} x2={195} y2={65} stroke="#2e7d32" strokeWidth={2} opacity={0.4} />
      </g>

      {/* FAPE concept */}
      <g transform="translate(40, 170)">
        <text x={270} y={0} textAnchor="middle" fontSize={12} fontWeight={600}
          fill="#c62828" fontFamily="Inter, sans-serif">
          Key insight: align locally, then measure
        </text>

        <rect x={30} y={15} width={480} height={70} rx={8} fill="#fce4ec" stroke="#c62828" strokeWidth={1} opacity={0.5} />

        <text x={270} y={38} textAnchor="middle" fontSize={11} fill="#555" fontFamily="Inter, sans-serif">
          For each residue i: transform all other atoms into residue i's local coordinate frame
        </text>
        <text x={270} y={55} textAnchor="middle" fontSize={11} fill="#555" fontFamily="Inter, sans-serif">
          → Compare predicted local positions to true local positions
        </text>
        <text x={270} y={72} textAnchor="middle" fontSize={11} fill="#555" fontFamily="Inter, sans-serif">
          → This captures both position accuracy AND orientation accuracy
        </text>

        <text x={270} y={108} textAnchor="middle" fontSize={11} fill="#999" fontStyle="italic"
          fontFamily="Inter, sans-serif">
          Unlike RMSD, FAPE doesn't need a global alignment step —
        </text>
        <text x={270} y={124} textAnchor="middle" fontSize={11} fill="#999" fontStyle="italic"
          fontFamily="Inter, sans-serif">
          each residue's local geometry is evaluated independently
        </text>
      </g>
    </svg>
  )
}

// ── Computational heat values for Structure Module ops ──
const SM_HEAT: Record<number, number> = {
  0: 0.95,  // IPA — expensive 3D attention
  1: 0.7,   // Backbone update
  2: 0.4,   // Atom positions (from ideal geometry)
  3: 0.6,   // Torsion angle prediction
  4: 0.85,  // FAPE loss computation
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

// ── Step connector ─────────────────────────────────────

function StepConnector({ done, active }: { done: boolean; active: boolean }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'center', padding: '0 0 0 22px', height: 16,
    }}>
      <div style={{
        width: 2, height: '100%',
        background: done ? '#00897b' : active ? '#80cbc4' : '#e0e0e0',
        transition: 'background 0.3s',
        position: 'relative' as const,
      }}>
        {active && (
          <div style={{
            position: 'absolute', left: -3, top: 2,
            width: 8, height: 8, borderRadius: '50%',
            background: '#00897b',
            animation: 'smPulse 1.5s infinite',
          }} />
        )}
      </div>
    </div>
  )
}

// ── Block list sidebar ─────────────────────────────────

function BlockList({ ops, activeId, onSelect }: {
  ops: SMOp[]; activeId: number; onSelect: (id: number) => void
}) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', padding: '8px 0',
    }}>
      <style>{`
        @keyframes smPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(0.6); }
        }
        @keyframes smNextArrow {
          0%, 100% { transform: translateX(0); opacity: 0.7; }
          50% { transform: translateX(3px); opacity: 1; }
        }
      `}</style>

      <div style={{
        fontSize: 11, color: '#999', textTransform: 'uppercase', letterSpacing: 1,
        padding: '0 12px 8px', fontFamily: 'Inter, sans-serif',
      }}>
        Structure Module
      </div>

      {/* Input */}
      <div style={{
        padding: '4px 12px', fontSize: 10, color: '#78909c',
        display: 'flex', alignItems: 'center', gap: 6,
        fontFamily: 'Inter, sans-serif',
      }}>
        <span style={{ fontSize: 12 }}>📥</span> Single repr + Pair repr in
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
                padding: '6px 10px 6px 6px', borderRadius: 8, cursor: 'pointer',
                background: active
                  ? `linear-gradient(135deg, ${cat.bg}, ${cat.bg}ee)`
                  : done ? 'rgba(0, 137, 123, 0.04)' : 'transparent',
                border: active
                  ? `2px solid ${cat.border}`
                  : isNext ? '2px dashed #80cbc4' : '2px solid transparent',
                transition: 'all 0.25s',
                display: 'flex', alignItems: 'center', gap: 6,
                position: 'relative' as const, overflow: 'hidden',
                margin: '0 4px',
              }}>
              {/* Heat bar */}
              <div style={{
                position: 'absolute', left: 0, top: 0, bottom: 0,
                width: `${heat * 100}%`,
                background: active
                  ? `linear-gradient(90deg, ${cat.glow}12, transparent)`
                  : `linear-gradient(90deg, ${hc}06, transparent)`,
                transition: 'all 0.3s',
              }} />

              {/* Step number */}
              <div style={{
                width: 26, height: 26, borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 12, fontWeight: 700, fontFamily: 'Inter, sans-serif',
                flexShrink: 0, position: 'relative' as const, zIndex: 1,
                background: active ? cat.border : done ? '#00897b' : '#e0e0e0',
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
                  color: active ? cat.text : done ? '#004d40' : '#666',
                  fontFamily: 'Inter, sans-serif',
                  display: 'flex', alignItems: 'center', gap: 5,
                  whiteSpace: 'nowrap' as const,
                }}>
                  {op.short}
                  <span style={{
                    fontSize: 7, padding: '1px 4px', borderRadius: 6,
                    background: `${hc}20`, color: hc, fontWeight: 700, letterSpacing: 0.5,
                    flexShrink: 0,
                  }}>
                    {heat >= 0.8 ? 'HOT' : heat >= 0.5 ? 'MED' : 'LOW'}
                  </span>
                </div>
                <div style={{
                  fontSize: 9.5, color: active ? cat.border : done ? '#26a69a' : '#bbb',
                  fontFamily: 'Inter, sans-serif',
                }}>
                  {op.category === 'ipa' ? '3D Attention' : op.category === 'backbone' ? 'Backbone' :
                   op.category === 'sidechain' ? 'Side-chain' : 'Training Loss'}
                </div>
              </div>

              {isNext && (
                <div style={{
                  position: 'relative' as const, zIndex: 1,
                  fontSize: 10, color: '#00897b', fontWeight: 600,
                  fontFamily: 'Inter, sans-serif',
                  animation: 'smNextArrow 1.5s infinite', flexShrink: 0,
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

      {/* Output */}
      <StepConnector done={activeId >= ops.length - 1} active={false} />
      <div style={{
        padding: '4px 12px', fontSize: 10, color: '#78909c',
        display: 'flex', alignItems: 'center', gap: 6,
        fontFamily: 'Inter, sans-serif',
      }}>
        <span style={{ fontSize: 12 }}>📤</span> 3D coordinates + confidence
      </div>

      {/* Repeat */}
      <div style={{
        margin: '10px 8px 0', padding: '8px 10px',
        background: 'linear-gradient(135deg, #e0f2f1, #e8f5e9)',
        borderRadius: 8, fontSize: 10, color: '#00897b', textAlign: 'center',
        fontFamily: 'Inter, sans-serif', border: '1px solid #a5d6a7', fontWeight: 600,
      }}>
        ↻ Steps 1–3 repeat × 8 blocks
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

// ── Main Structure Module Detail ───────────────────────

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

  // Auto-advance backbone block
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
      width: '100%', height: '100%', background: '#fff',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      fontFamily: 'Inter, system-ui, sans-serif',
    }}>
      {/* Top bar */}
      <div style={{
        padding: '10px 24px', borderBottom: '1px solid #eee',
        display: 'flex', alignItems: 'center', gap: 16, flexShrink: 0,
      }}>
        <button onClick={onBack} style={{
          border: '1px solid #ccc', borderRadius: 6, background: '#fff',
          padding: '6px 14px', cursor: 'pointer', fontSize: 13, color: '#555',
          display: 'flex', alignItems: 'center', gap: 6,
        }}>← Back to Overview</button>
        <h1 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: '#1a237e' }}>
          Structure Module Deep Dive
        </h1>
        <span style={{ fontSize: 13, color: '#78909c' }}>
          8 blocks — from abstract features to 3D atomic coordinates
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button onClick={() => { setActiveOpId(id => (id - 1 + SM_OPS.length) % SM_OPS.length); setPlaying(false) }}
            style={smallBtn}>◀ Prev</button>
          <button onClick={() => setPlaying(p => !p)}
            style={{ ...smallBtn, width: 60 }}>{playing ? '⏸ Pause' : '▶ Play'}</button>
          <button onClick={() => { goNext(); setPlaying(false) }}
            style={smallBtn}>Next ▶</button>
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        {/* Left sidebar */}
        <div style={{ width: 180, borderRight: '1px solid #eee', overflowY: 'auto', flexShrink: 0 }}>
          <BlockList ops={SM_OPS} activeId={activeOpId} onSelect={handleSelect} />
        </div>

        {/* Center visualization */}
        <div style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: 20, overflow: 'auto', minWidth: 0,
        }}>
          {activeOpId === 0 && <IPAVisualization step={0} />}
          {activeOpId === 1 && <BackboneRefinement block={backboneBlock} />}
          {activeOpId === 2 && <BackboneRefinement block={7} />}
          {activeOpId === 3 && <TorsionAngles />}
          {activeOpId === 4 && <FAPEVisualization />}
        </div>

        {/* Right explanation */}
        <div style={{
          width: 340, borderLeft: '1px solid #eee', overflowY: 'auto',
          padding: 20, flexShrink: 0,
        }}>
          <div style={{
            display: 'inline-block', padding: '3px 10px', borderRadius: 12,
            background: cat.bg, border: `1px solid ${cat.border}`,
            fontSize: 11, color: cat.text, fontWeight: 600, marginBottom: 10,
          }}>
            {activeOp.category === 'ipa' ? '3D Attention' :
             activeOp.category === 'backbone' ? 'Backbone Geometry' :
             activeOp.category === 'sidechain' ? 'Side-chain' : 'Training Loss'}
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
              fontFamily: 'monospace', fontSize: 11.5, color: '#555',
              marginBottom: 14, borderLeft: `3px solid ${cat.border}`,
              lineHeight: 1.5,
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

          {/* Backbone block slider */}
          {(activeOpId === 1 || activeOpId === 2) && (
            <div style={{
              marginTop: 20, padding: 12, background: '#e0f2f1',
              borderRadius: 8, border: '1px solid #80cbc4',
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#004d40', marginBottom: 6 }}>
                Refinement block: {backboneBlock + 1} / 8
              </div>
              <input type="range" min={0} max={7} value={backboneBlock}
                onChange={e => { setBackboneBlock(Number(e.target.value)); setPlaying(false) }}
                style={{ width: '100%' }} />
              <div style={{ fontSize: 11, color: '#00897b', marginTop: 4 }}>
                Drag to see progressive refinement from flat → folded
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const smallBtn: React.CSSProperties = {
  border: '1px solid #ccc', borderRadius: 4, background: '#fff',
  padding: '4px 10px', cursor: 'pointer', fontSize: 12, color: '#555',
}
