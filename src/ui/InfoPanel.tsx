import { useEffect, useRef } from 'react'
import { useAnimationStore, AnimStep } from '../store/useAnimationStore'
import { COLORS, DATA } from '../config'
import katex from 'katex'
import 'katex/dist/katex.min.css'

const STEP_INFO: Record<AnimStep, { title: string; desc: string; formula?: string }> = {
  [AnimStep.Idle]: {
    title: 'Triangle Attention',
    desc: 'A key mechanism in AlphaFold2\'s Evoformer that updates pair representations using triangular relationships between residues.',
  },
  [AnimStep.Entrance]: {
    title: 'Pair Representation',
    desc: 'Each cell (i, j) in the pair matrix encodes the relationship between residue i and residue j. The matrix captures evolutionary and structural signals.',
    formula: 'z_{ij} \\in \\mathbb{R}^c',
  },
  [AnimStep.SelectTarget]: {
    title: 'Step 1: Select Target Pair',
    desc: `We want to update the representation for residues (${DATA.targetI}, ${DATA.targetJ}). Triangle attention gathers information from all other residues to refine this pair.`,
    formula: 'z_{ij} \\leftarrow \\text{TriAttn}(z_{ij}, \\{z_{ik}, z_{jk}\\}_{k})',
  },
  [AnimStep.IterateK]: {
    title: 'Step 2: Find Third Residue k',
    desc: 'For each third residue k, we look at the pairs (i,k) and (j,k). These three pairs form a "triangle" in the pair matrix.',
    formula: '\\text{Triangle: } (i,j), (i,k), (j,k)',
  },
  [AnimStep.TriangleConverge]: {
    title: 'Step 3: Information Flows',
    desc: 'Information from (i,k) and (j,k) flows to (i,j). This is the key insight: if residue k is close to both i and j, that information should influence the i-j relationship.',
    formula: 'z_{ij} \\mathrel{+}= \\sum_k \\text{softmax}(q_i \\cdot k_k) \\cdot v_{jk}',
  },
  [AnimStep.SweepAllK]: {
    title: 'Step 4: Sweep All k',
    desc: 'We aggregate information across ALL possible third residues k=1..N. Each triangle contributes a weighted update.',
    formula: 'z_{ij} \\mathrel{+}= \\sum_{k=1}^{N} a_{ik} \\cdot z_{jk}',
  },
  [AnimStep.MatrixRefresh]: {
    title: 'Step 5: Full Matrix Update',
    desc: 'This process is applied to every pair (i,j) in parallel. The entire pair matrix is updated, encoding a richer representation of residue relationships.',
    formula: 'Z^{(l+1)} = Z^{(l)} + \\text{TriangleAttention}(Z^{(l)})',
  },
}

export function InfoPanel() {
  const currentStep = useAnimationStore((s) => s.currentStep)
  const formulaRef = useRef<HTMLDivElement>(null)

  const info = STEP_INFO[currentStep]

  useEffect(() => {
    if (formulaRef.current && info.formula) {
      try {
        katex.render(info.formula, formulaRef.current, {
          throwOnError: false,
          displayMode: true,
        })
      } catch (e) {
        // fallback: show raw formula
        formulaRef.current.textContent = info.formula
      }
    }
  }, [info.formula, currentStep])

  return (
    <div
      style={{
        position: 'absolute',
        top: '60px',
        right: '20px',
        width: '320px',
        padding: '20px',
        background: COLORS.panelBg,
        border: `1px solid ${COLORS.panelBorder}`,
        borderRadius: '12px',
        backdropFilter: 'blur(12px)',
        zIndex: 10,
        transition: 'opacity 0.3s',
      }}
    >
      <h2
        style={{
          fontFamily: "'Inter', sans-serif",
          fontSize: '14px',
          fontWeight: 600,
          color: COLORS.matrixLow,
          marginBottom: '8px',
          letterSpacing: '0.3px',
        }}
      >
        {info.title}
      </h2>
      <p
        style={{
          fontFamily: "'Inter', sans-serif",
          fontSize: '13px',
          lineHeight: 1.6,
          color: '#b0b0b0',
          marginBottom: info.formula ? '14px' : 0,
        }}
      >
        {info.desc}
      </p>
      {info.formula && (
        <div
          ref={formulaRef}
          style={{
            background: 'rgba(255,255,255,0.03)',
            borderRadius: '8px',
            padding: '10px 14px',
            fontSize: '14px',
            overflowX: 'auto',
          }}
        />
      )}
    </div>
  )
}
