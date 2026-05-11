import { useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import { TriangleAttentionScene } from './TriangleAttentionScene'
import { AnimationDriver } from './AnimationDriver'
import { useAnimationStore, AnimStep } from '../store/useAnimationStore'
import { KaTeXFormula } from '../ui/KaTeXFormula'
import { CAMERA } from '../config'

// ── Per-step narration synced with the AnimStep enum ─────────────────────────

const STEP_NARRATION: Record<number, {
  title: string
  body: string
  formula?: string
  algRef?: string
}> = {
  [AnimStep.Idle]: {
    title: 'Press ▶ to start',
    body: 'A 20-residue protein chain (left) and its pair representation z_ij (a 20×20 matrix, floating in space) will appear. We update one cell at a time using triangle inequality constraints.',
  },
  [AnimStep.Entrance]: {
    title: 'Setting the stage',
    body: 'The residue chain fades in on the left. Each sphere is one amino acid (Cα atom). The square plane is the pair representation z_ij — every (i,j) cell holds a learned 128-dim feature about residue pair (i,j). Colour encodes a scalar projection (cyan = low, magenta = high).',
  },
  [AnimStep.SelectTarget]: {
    title: 'Pick the cell to update',
    body: 'We will refresh z_ij for one specific pair (i, j). The "Triangle Multiplicative Update (Outgoing)" rule says: aggregate information from every third residue k that shares edges with both i and j. The current target cell is highlighted in white-gold.',
    formula: '\\tilde{z}_{ij} = g_{ij} \\odot \\text{Linear}\\!\\left(\\text{LayerNorm}\\!\\left(\\sum_k a_{ik} \\odot b_{jk}\\right)\\right)',
    algRef: 'Algorithm 11 — Triangle multiplicative update (outgoing)',
  },
  [AnimStep.IterateK]: {
    title: 'Visit one triangle (k = 0)',
    body: 'Pick a third residue k. The two edges (i, k) and (j, k) — together with the target edge (i, j) — form a triangle. The orange highlights on the chain show i, j, k; the green cells (i,k) and (j,k) light up in the matrix.',
    formula: 'a_{ik} = \\sigma(W_a z_{ik})\\odot W\'_a z_{ik}, \\quad b_{jk} = \\sigma(W_b z_{jk})\\odot W\'_b z_{jk}',
  },
  [AnimStep.TriangleConverge]: {
    title: 'Pull information along the triangle',
    body: 'Particles flow from the two side-edge cells into the target cell. Geometrically this enforces the triangle inequality: if i↔k and j↔k are "close" (high values), then i↔j is biased to be close too. The target cell starts to glow.',
  },
  [AnimStep.SweepAllK]: {
    title: 'Sum over all k',
    body: 'Now repeat for every k = 0…N−1. Each k contributes one outer-product term a_ik ⊙ b_jk. The full sum Σ_k is the message into z_ij. Watch the chain highlight scan through all residues.',
    formula: '\\text{msg}_{ij} = \\sum_{k=1}^{N_{\\text{res}}} a_{ik} \\odot b_{jk}',
  },
  [AnimStep.MatrixRefresh]: {
    title: 'Apply the update to every (i, j)',
    body: 'In practice this happens for every pair (i, j) in parallel — the whole 20×20 matrix is updated at once. Watch the wave sweep across the plane as each cell receives its new value. After this comes the gated output: z_ij ← g_ij ⊙ Linear(LayerNorm(msg_ij)).',
    formula: 'z_{ij} \\leftarrow z_{ij} + g_{ij} \\odot \\text{Linear}\\!\\left(\\text{LayerNorm}\\!\\left(\\text{msg}_{ij}\\right)\\right)',
  },
}

// ── Page ─────────────────────────────────────────────────────────────────────

export function TriangleAttention3DPage({ onBack }: { onBack: () => void }) {
  const currentStep = useAnimationStore(s => s.currentStep)
  const isPlaying = useAnimationStore(s => s.isPlaying)
  const currentK = useAnimationStore(s => s.currentK)
  const targetI = useAnimationStore(s => s.targetI)
  const targetJ = useAnimationStore(s => s.targetJ)
  const speed = useAnimationStore(s => s.speed)
  const play = useAnimationStore(s => s.play)
  const pause = useAnimationStore(s => s.pause)
  const togglePlay = useAnimationStore(s => s.togglePlay)
  const nextStep = useAnimationStore(s => s.nextStep)
  const prevStep = useAnimationStore(s => s.prevStep)
  const setSpeed = useAnimationStore(s => s.setSpeed)
  const reset = useAnimationStore(s => s.reset)

  // Auto-start when the page mounts
  useEffect(() => {
    reset()
    const t = setTimeout(() => play(), 200)
    return () => clearTimeout(t)
  }, [reset, play])

  const narration = STEP_NARRATION[currentStep] ?? STEP_NARRATION[AnimStep.Idle]

  return (
    <div style={{
      width: '100%', height: '100%', position: 'relative', background: '#070712',
      fontFamily: 'Inter, system-ui, sans-serif', overflow: 'hidden',
    }}>
      {/* Top bar */}
      <div style={topBar}>
        <button onClick={onBack} style={backBtn}>← Back</button>
        <div>
          <h1 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: '#c0c8d0' }}>
            Triangle Attention (3D)
          </h1>
          <div style={{ fontSize: 11, color: '#667788', marginTop: 2 }}>
            Algorithms 11–14 · Pair stack · 20 residues, target (i={targetI}, j={targetJ})
          </div>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
          <button onClick={() => { prevStep(); pause() }} style={pillBtn}>◀ Step</button>
          <button onClick={togglePlay} style={{ ...pillBtn, minWidth: 70, background: isPlaying ? 'rgba(239,83,80,0.15)' : 'rgba(102,187,106,0.15)', borderColor: isPlaying ? '#ef5350' : '#66bb6a', color: isPlaying ? '#ef9a9a' : '#a5d6a7' }}>
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button onClick={() => { nextStep(); pause() }} style={pillBtn}>Step ▶</button>
          <button onClick={reset} style={{ ...pillBtn, fontSize: 11 }}>⟲ Restart</button>
          <span style={{ fontSize: 11, color: '#667788', marginLeft: 8 }}>Speed</span>
          <input type="range" min={0.25} max={3} step={0.25} value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            style={{ width: 90, accentColor: '#42a5f5' }} />
          <span style={{ fontSize: 11, color: '#c0c8d0', fontFamily: 'JetBrains Mono, monospace', minWidth: 36 }}>
            {speed.toFixed(2)}×
          </span>
        </div>
      </div>

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: CAMERA.position, fov: CAMERA.fov, near: CAMERA.near, far: CAMERA.far }}
        style={{ width: '100%', height: '100%' }}
        gl={{ antialias: true, alpha: false }}
      >
        <color attach="background" args={['#070712']} />
        <fog attach="fog" args={['#070712', 22, 60]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[8, 10, 6]} intensity={0.9} color="#ffffff" />
        <directionalLight position={[-6, -4, -3]} intensity={0.25} color="#88aaff" />

        {/* Soft floor grid — adds scale and depth */}
        <Grid
          position={[0, -6, 0]}
          args={[40, 40]}
          cellColor="#1a2030"
          sectionColor="#2a3550"
          sectionThickness={0.8}
          fadeDistance={28}
          fadeStrength={1.4}
          infiniteGrid
        />

        <TriangleAttentionScene />
        <AnimationDriver />

        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={0.6}
          minDistance={6}
          maxDistance={40}
          target={[-2, 0, 0]}
        />

        <EffectComposer>
          <Bloom intensity={0.6} luminanceThreshold={0.55} luminanceSmoothing={0.4} mipmapBlur />
          <Vignette eskil={false} offset={0.15} darkness={0.7} />
        </EffectComposer>
      </Canvas>

      {/* Left narration panel */}
      <div style={leftPanel}>
        <div style={badge}>
          <span style={{ opacity: 0.7, fontSize: 9, fontWeight: 700, letterSpacing: 1, marginRight: 6 }}>STEP</span>
          {currentStep + 1} / 7
        </div>
        <h2 style={panelH2}>{narration.title}</h2>
        <p style={panelP}>{narration.body}</p>

        {narration.formula && (
          <div style={{ margin: '10px 0', padding: 10, background: 'rgba(66,165,245,0.07)', borderRadius: 6, border: '1px solid rgba(66,165,245,0.2)' }}>
            <KaTeXFormula
              formula={narration.formula}
              style={{ fontSize: 12, padding: 0, background: 'transparent', borderLeft: 'none' }}
            />
          </div>
        )}

        {narration.algRef && (
          <div style={{ fontSize: 10, color: '#667788', fontFamily: 'JetBrains Mono, monospace', marginBottom: 10 }}>
            {narration.algRef}
          </div>
        )}

        {/* Step progress dots */}
        <div style={{ display: 'flex', gap: 4, marginTop: 12 }}>
          {[1, 2, 3, 4, 5, 6].map(i => (
            <div key={i} style={{
              flex: 1, height: 4, borderRadius: 2,
              background: currentStep >= i ? '#42a5f5' : 'rgba(255,255,255,0.08)',
              transition: 'background 0.3s',
            }} />
          ))}
        </div>

        {/* Live k indicator */}
        {(currentStep === AnimStep.IterateK ||
          currentStep === AnimStep.TriangleConverge ||
          currentStep === AnimStep.SweepAllK) && (
          <div style={{
            marginTop: 14, padding: 10,
            background: 'rgba(76,175,80,0.08)', borderRadius: 6,
            border: '1px solid rgba(76,175,80,0.22)',
          }}>
            <div style={{ fontSize: 10.5, color: '#66bb6a', fontWeight: 600, marginBottom: 4 }}>
              Triangle currently being processed
            </div>
            <div style={{ fontSize: 13, color: '#a5d6a7', fontFamily: 'JetBrains Mono, monospace' }}>
              i={targetI}, j={targetJ}, <b>k={currentK}</b>
            </div>
            <div style={{ fontSize: 10, color: '#667788', marginTop: 3 }}>
              Contributes term: a<sub>i,{currentK}</sub> ⊙ b<sub>j,{currentK}</sub>
            </div>
          </div>
        )}
      </div>

      {/* Right legend + camera tips */}
      <div style={rightPanel}>
        <h2 style={panelH2}>Legend</h2>
        <LegendRow color="#4fc3f7" label="Residue (Cα)" />
        <LegendRow color="#ffeb3b" label="Highlighted residue (i, j, k)" />
        <LegendRow color="#00bcd4" label="Pair value (low)" />
        <LegendRow color="#e040fb" label="Pair value (high)" />
        <LegendRow color="#fffde7" label="Target cell (i, j)" />
        <LegendRow color="#ff9100" label="Triangle edge cell" />
        <LegendRow color="#ffe082" label="Information particle" />

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '14px 0' }} />

        <h3 style={panelH3}>What this proves</h3>
        <p style={panelP}>
          Triangle attention is geometric: it embeds the <b>triangle inequality</b>
          into the architecture. Without it, AlphaFold2's accuracy collapses
          (CASP14 ablation, Supp. Fig 10). Every pair update needs to be
          self-consistent across all three triangle edges.
        </p>

        <h3 style={panelH3}>Camera tips</h3>
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 11, color: '#8899aa', lineHeight: 1.6 }}>
          <li>Drag to orbit around the matrix.</li>
          <li>Scroll to zoom — see individual cells up close.</li>
          <li>Right-drag to pan.</li>
          <li>Try viewing the pair matrix edge-on — it's a thin plane.</li>
        </ul>

        <div style={{ marginTop: 12, padding: 10, background: 'rgba(255,145,0,0.06)', borderRadius: 6, border: '1px solid rgba(255,145,0,0.18)' }}>
          <div style={{ fontSize: 10.5, color: '#ffab40', fontWeight: 600, marginBottom: 4 }}>
            Note on the data
          </div>
          <div style={{ fontSize: 10.5, color: '#8899aa', lineHeight: 1.55 }}>
            Pair values are synthetic (sine waves + noise) for clear visualization.
            The structure of the update — the triangle pattern, the flow direction,
            the sweep over k — is exactly as in the paper.
          </div>
        </div>
      </div>
    </div>
  )
}

// ── UI helpers ──────────────────────────────────────────────────────────────

function LegendRow({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
      <div style={{ width: 14, height: 14, borderRadius: 3, background: color, boxShadow: `0 0 8px ${color}aa` }} />
      <span style={{ fontSize: 10.5, color: '#8899aa' }}>{label}</span>
    </div>
  )
}

// ── Styles ──────────────────────────────────────────────────────────────────

const topBar: React.CSSProperties = {
  position: 'absolute', top: 0, left: 0, right: 0,
  padding: '10px 20px', zIndex: 10,
  background: 'linear-gradient(180deg, rgba(7,7,18,0.92) 0%, rgba(7,7,18,0) 100%)',
  display: 'flex', alignItems: 'center', gap: 14, pointerEvents: 'auto',
}

const backBtn: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.12)', borderRadius: 6, background: 'rgba(20,20,40,0.6)',
  padding: '5px 12px', cursor: 'pointer', fontSize: 12, color: '#c0c8d0',
}

const pillBtn: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.12)', borderRadius: 6,
  background: 'rgba(20,20,40,0.6)',
  padding: '5px 12px', cursor: 'pointer', fontSize: 12, color: '#c0c8d0',
  fontFamily: 'Inter, sans-serif',
}

const leftPanel: React.CSSProperties = {
  position: 'absolute', top: 80, left: 16, width: 330, maxHeight: 'calc(100% - 100px)',
  overflowY: 'auto', padding: 16,
  background: 'rgba(15,15,28,0.78)', backdropFilter: 'blur(8px)',
  border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10,
  zIndex: 5, color: '#c0c8d0',
}

const rightPanel: React.CSSProperties = {
  position: 'absolute', top: 80, right: 16, width: 280, maxHeight: 'calc(100% - 100px)',
  overflowY: 'auto', padding: 16,
  background: 'rgba(15,15,28,0.78)', backdropFilter: 'blur(8px)',
  border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10,
  zIndex: 5, color: '#c0c8d0',
}

const panelH2: React.CSSProperties = {
  margin: '6px 0 8px', fontSize: 14, color: '#c0c8d0', fontWeight: 700,
}

const panelH3: React.CSSProperties = {
  margin: '14px 0 6px', fontSize: 12, color: '#c0c8d0', fontWeight: 700,
}

const panelP: React.CSSProperties = {
  margin: '0 0 8px', fontSize: 11.5, color: '#8899aa', lineHeight: 1.6,
}

const badge: React.CSSProperties = {
  display: 'inline-block', padding: '3px 10px', borderRadius: 12,
  background: 'rgba(66,165,245,0.12)', border: '1px solid #1565c060',
  fontSize: 11, color: '#64b5f6', fontWeight: 600, marginBottom: 8,
}
