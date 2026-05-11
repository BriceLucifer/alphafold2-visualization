import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, CatmullRomLine, Html } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { KaTeXFormula } from '../ui/KaTeXFormula'

// ── Constants ────────────────────────────────────────────────────────────────

const N_RES = 18
// Algorithm 29: v_bins = [1, 3, 5, ..., 99]^T — 50 bin centres, 2-lDDT-Cα wide.
const N_BINS = 50
const BIN_CENTRES = Array.from({ length: N_BINS }, (_, i) => 1 + 2 * i)

// Canonical AlphaFold pLDDT colour bands.
const CONFIDENCE_BANDS = [
  { min: 90, label: 'Very high', color: '#0053d6', text: '> 90' },
  { min: 70, label: 'Confident', color: '#65cbf3', text: '70 – 90' },
  { min: 50, label: 'Low',       color: '#ffdb13', text: '50 – 70' },
  { min: 0,  label: 'Very low',  color: '#ff7d45', text: '< 50' },
]

function plddtColor(v: number): string {
  for (const b of CONFIDENCE_BANDS) if (v >= b.min) return b.color
  return CONFIDENCE_BANDS[CONFIDENCE_BANDS.length - 1].color
}

// ── Synthetic backbone (alpha-helix + a low-confidence loop region) ──────────

function buildBackbone(): THREE.Vector3[] {
  // Mostly an alpha-helix; we splay residues 11–14 into a flexible loop so the
  // visualised pLDDT has a realistic-looking confident-vs-disordered contrast.
  const out: THREE.Vector3[] = []
  const radius = 0.42
  const rise = 0.18
  const tpr = (2 * Math.PI) / 3.6
  for (let i = 0; i < N_RES; i++) {
    const helixT = i * tpr
    const inLoop = i >= 11 && i <= 14
    const loopBlend = inLoop ? Math.sin(((i - 11) / 3) * Math.PI) * 0.5 : 0
    const px = Math.cos(helixT) * (radius + loopBlend) + loopBlend * 0.4
    const py = i * rise - (N_RES - 1) * rise * 0.5
    const pz = Math.sin(helixT) * (radius + loopBlend) - loopBlend * 0.3
    out.push(new THREE.Vector3(px, py, pz))
  }
  return out
}

// ── Synthetic pLDDT distribution per residue ─────────────────────────────────
// Synthesise a "predicted" 50-bin distribution that peaks near a target value.

function gaussianBins(target: number, sigma: number): number[] {
  let total = 0
  const probs = BIN_CENTRES.map(c => {
    const d = c - target
    const p = Math.exp(-(d * d) / (2 * sigma * sigma))
    total += p
    return p
  })
  return probs.map(p => p / total)
}

function buildPerResidueDist(): { dist: number[]; expected: number }[] {
  const targets: number[] = []
  const sigmas: number[] = []
  for (let i = 0; i < N_RES; i++) {
    // High-confidence helix residues: target ~90, sigma small.
    // Low-confidence loop residues (11–14): target ~45–55, sigma larger.
    // Termini (0–1, N-2..N-1): mid confidence.
    const inLoop = i >= 11 && i <= 14
    const isTerm = i < 2 || i > N_RES - 3
    let t: number, s: number
    if (inLoop) {
      t = 50 + Math.sin(i * 1.7) * 6
      s = 9
    } else if (isTerm) {
      t = 78 + Math.sin(i * 0.9) * 4
      s = 7
    } else {
      t = 91 + Math.sin(i * 0.5) * 3
      s = 4
    }
    targets.push(THREE.MathUtils.clamp(t, 1, 99))
    sigmas.push(s)
  }
  return targets.map((t, i) => {
    const dist = gaussianBins(t, sigmas[i])
    // Algorithm 29 line 5: r_i^pLDDT = p_i · v_bins (expected value).
    const expected = dist.reduce((acc, p, k) => acc + p * BIN_CENTRES[k], 0)
    return { dist, expected }
  })
}

// ── 3D primitives ────────────────────────────────────────────────────────────

function ConfidenceRibbon({ points, colors }: { points: THREE.Vector3[]; colors: string[] }) {
  // Render N−1 short tube segments between consecutive Cα, each coloured by
  // the average of its two endpoints. This gives the canonical AlphaFold
  // gradient look without needing a vertex-coloured custom shader.
  const segments = useMemo(() => {
    const segs: { p1: THREE.Vector3; p2: THREE.Vector3; color: string }[] = []
    for (let i = 0; i < points.length - 1; i++) {
      const a = new THREE.Color(colors[i])
      const b = new THREE.Color(colors[i + 1])
      const mix = a.clone().lerp(b, 0.5).getHexString()
      segs.push({ p1: points[i], p2: points[i + 1], color: `#${mix}` })
    }
    return segs
  }, [points, colors])

  return (
    <>
      {segments.map((s, i) => {
        const dir = new THREE.Vector3().subVectors(s.p2, s.p1)
        const len = dir.length()
        const mid = new THREE.Vector3().addVectors(s.p1, s.p2).multiplyScalar(0.5)
        const quat = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          dir.clone().normalize(),
        )
        return (
          <mesh key={`seg-${i}`} position={mid} quaternion={quat}>
            <cylinderGeometry args={[0.07, 0.07, len, 12, 1]} />
            <meshStandardMaterial
              color={s.color}
              emissive={s.color}
              emissiveIntensity={0.35}
              roughness={0.35}
              metalness={0.05}
            />
          </mesh>
        )
      })}
    </>
  )
}

function ResidueBeads({
  points, scores, selectedI, setSelectedI,
}: {
  points: THREE.Vector3[]
  scores: number[]
  selectedI: number
  setSelectedI: (i: number) => void
}) {
  return (
    <>
      {points.map((p, i) => {
        const c = plddtColor(scores[i])
        const isSel = i === selectedI
        return (
          <mesh
            key={`bead-${i}`}
            position={p}
            onClick={(e) => { e.stopPropagation(); setSelectedI(i) }}
            onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = 'pointer' }}
            onPointerOut={(e) => { e.stopPropagation(); document.body.style.cursor = 'auto' }}
          >
            <sphereGeometry args={[isSel ? 0.13 : 0.09, 22, 22]} />
            <meshStandardMaterial
              color={c}
              emissive={c}
              emissiveIntensity={isSel ? 0.95 : 0.45}
              roughness={0.3}
              metalness={0.08}
            />
          </mesh>
        )
      })}
    </>
  )
}

function SelectedResidueRing({ point }: { point: THREE.Vector3 }) {
  const ref = useRef<THREE.Mesh>(null)
  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.getElapsedTime()
    const s = 1 + Math.sin(t * 3) * 0.08
    ref.current.scale.set(s, 1, s)
  })
  return (
    <mesh ref={ref} position={point} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.18, 0.22, 48]} />
      <meshBasicMaterial color="#ffe082" transparent opacity={0.85} side={THREE.DoubleSide} />
    </mesh>
  )
}

function CameraDrift() {
  const groupRef = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    if (!groupRef.current) return
    groupRef.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.06) * 0.06
  })
  return <group ref={groupRef} />
}

// ── HTML panels ──────────────────────────────────────────────────────────────

function PerResidueBars({
  scores, selectedI, onClickBar,
}: {
  scores: number[]
  selectedI: number
  onClickBar: (i: number) => void
}) {
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{
        display: 'flex', alignItems: 'flex-end', gap: 2, height: 110,
        padding: '4px 4px', background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.06)', borderRadius: 6,
      }}>
        {scores.map((v, i) => {
          const h = Math.max(2, (v / 100) * 100)
          const isSel = i === selectedI
          return (
            <div
              key={`bar-${i}`}
              onClick={() => onClickBar(i)}
              title={`Residue ${i}: pLDDT ${v.toFixed(1)}`}
              style={{
                flex: 1, height: `${h}%`, minWidth: 6,
                background: plddtColor(v),
                opacity: isSel ? 1 : 0.78,
                outline: isSel ? '2px solid #ffe082' : 'none',
                outlineOffset: '-1px',
                borderRadius: 1.5,
                cursor: 'pointer',
                transition: 'opacity 0.15s, outline 0.1s',
              }}
            />
          )
        })}
      </div>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        fontSize: 9.5, color: '#667788',
        fontFamily: 'JetBrains Mono, monospace', marginTop: 3,
      }}>
        <span>residue 0</span>
        <span>residue {scores.length - 1}</span>
      </div>
    </div>
  )
}

function DistributionPlot({
  dist, expected,
}: {
  dist: number[]
  expected: number
}) {
  const maxP = Math.max(...dist)
  return (
    <div>
      <div style={{
        display: 'flex', alignItems: 'flex-end', gap: 0,
        height: 88, padding: '4px 0',
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.06)', borderRadius: 6,
        position: 'relative',
      }}>
        {dist.map((p, i) => {
          const h = (p / maxP) * 100
          return (
            <div key={`db-${i}`} style={{
              flex: 1, height: `${h}%`,
              background: plddtColor(BIN_CENTRES[i]),
              minWidth: 1,
              borderRadius: 0.5,
            }} />
          )
        })}
        {/* Expected-value marker */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          left: `${(expected / 100) * 100}%`,
          width: 1.5, background: '#ffe082',
          boxShadow: '0 0 6px #ffe082',
        }} />
      </div>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        fontSize: 9.5, color: '#667788',
        fontFamily: 'JetBrains Mono, monospace', marginTop: 3,
      }}>
        <span>bin 1</span>
        <span>50 (≈99 LDDT)</span>
      </div>
      <div style={{
        marginTop: 4, fontSize: 11, color: '#ffe082',
        fontFamily: 'JetBrains Mono, monospace',
      }}>
        r<sub>i</sub><sup>pLDDT</sup> = p<sub>i</sub> · v<sub>bins</sub> = <b>{expected.toFixed(1)}</b>
      </div>
    </div>
  )
}

function LegendStrip() {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 4,
      marginTop: 8,
    }}>
      {CONFIDENCE_BANDS.map(b => (
        <div key={b.label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 14, height: 14, borderRadius: 3,
            background: b.color, boxShadow: `0 0 6px ${b.color}99`,
          }} />
          <span style={{ fontSize: 10.5, color: '#c0c8d0', fontWeight: 600 }}>{b.label}</span>
          <span style={{
            fontSize: 10, color: '#8899aa',
            fontFamily: 'JetBrains Mono, monospace', marginLeft: 'auto',
          }}>{b.text}</span>
        </div>
      ))}
    </div>
  )
}

// ── Scene root ───────────────────────────────────────────────────────────────

function Scene({
  points, scores, selectedI, setSelectedI, showRibbon,
}: {
  points: THREE.Vector3[]
  scores: number[]
  selectedI: number
  setSelectedI: (i: number) => void
  showRibbon: boolean
}) {
  const colors = useMemo(() => scores.map(plddtColor), [scores])
  const curvePoints = useMemo(
    () => points.map(p => p.toArray() as [number, number, number]),
    [points],
  )
  return (
    <>
      <CameraDrift />
      {showRibbon && (
        <CatmullRomLine
          points={curvePoints}
          color="#5a7090"
          lineWidth={1}
          transparent
          opacity={0.35}
          segments={64}
        />
      )}
      <ConfidenceRibbon points={points} colors={colors} />
      <ResidueBeads
        points={points}
        scores={scores}
        selectedI={selectedI}
        setSelectedI={setSelectedI}
      />
      <SelectedResidueRing point={points[selectedI]} />
    </>
  )
}

// ── Top-level page ───────────────────────────────────────────────────────────

export function PlddtConfidence3DScene({ onBack }: { onBack: () => void }) {
  const points = useMemo(buildBackbone, [])
  const distData = useMemo(buildPerResidueDist, [])
  const scores = useMemo(() => distData.map(d => d.expected), [distData])
  const [selectedI, setSelectedI] = useState(7)
  const [showSpine, setShowSpine] = useState(true)
  const [autoSweep, setAutoSweep] = useState(false)
  const [speed, setSpeed] = useState(1.0)

  // Auto-sweep through residues so the side panel comes alive without input.
  useEffect(() => {
    if (!autoSweep) return
    const period = 1100 / speed
    const id = setInterval(() => setSelectedI(i => (i + 1) % N_RES), period)
    return () => clearInterval(id)
  }, [autoSweep, speed])

  useEffect(() => () => { document.body.style.cursor = 'auto' }, [])

  const chainPlddt = useMemo(
    () => scores.reduce((a, b) => a + b, 0) / scores.length,
    [scores],
  )

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
            pLDDT Confidence (3D)
          </h1>
          <div style={{ fontSize: 11, color: '#667788', marginTop: 2 }}>
            Algorithm 29 · 50 bins of width 2 · v<sub>bins</sub> = [1, 3, 5, …, 99]
          </div>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
          <button
            onClick={() => setAutoSweep(s => !s)}
            style={{
              ...pillBtn, minWidth: 90,
              background: autoSweep ? 'rgba(239,83,80,0.15)' : 'rgba(102,187,106,0.15)',
              borderColor: autoSweep ? '#ef5350' : '#66bb6a',
              color: autoSweep ? '#ef9a9a' : '#a5d6a7',
            }}>
            {autoSweep ? '⏸ Pause' : '▶ Sweep'}
          </button>
          <button onClick={() => setSelectedI(i => (i - 1 + N_RES) % N_RES)} style={pillBtn}>◀</button>
          <button onClick={() => setSelectedI(i => (i + 1) % N_RES)} style={pillBtn}>▶</button>
          <span style={{ fontSize: 11, color: '#667788', marginLeft: 8 }}>Speed</span>
          <input
            type="range" min={0.25} max={3} step={0.25} value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            style={{ width: 90, accentColor: '#42a5f5' }}
          />
          <span style={{
            fontSize: 11, color: '#c0c8d0',
            fontFamily: 'JetBrains Mono, monospace', minWidth: 36,
          }}>
            {speed.toFixed(2)}×
          </span>
        </div>
      </div>

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [2.6, 1.4, 2.8], fov: 45, near: 0.05, far: 50 }}
        style={{ width: '100%', height: '100%' }}
        gl={{ antialias: true, alpha: false }}
      >
        <color attach="background" args={['#070712']} />
        <fog attach="fog" args={['#070712', 5, 11]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[3, 4, 2]} intensity={0.85} color="#ffffff" />
        <directionalLight position={[-3, -2, -1]} intensity={0.22} color="#88aaff" />

        <Scene
          points={points}
          scores={scores}
          selectedI={selectedI}
          setSelectedI={setSelectedI}
          showRibbon={showSpine}
        />

        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={0.6}
          minDistance={1.5}
          maxDistance={6}
          target={[0, 0, 0]}
        />

        <EffectComposer>
          <Bloom intensity={0.6} luminanceThreshold={0.55} luminanceSmoothing={0.4} mipmapBlur />
          <Vignette eskil={false} offset={0.15} darkness={0.7} />
        </EffectComposer>

        <Html position={[0, 1.6, 0]} center distanceFactor={5} style={{ pointerEvents: 'none' }}>
          <div style={{
            padding: '3px 9px',
            background: 'rgba(20,20,40,0.7)',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 10, fontSize: 10,
            color: '#a5d6a7',
            fontFamily: 'JetBrains Mono, monospace',
            whiteSpace: 'nowrap', fontWeight: 700,
          }}>
            chain pLDDT = {chainPlddt.toFixed(1)}
          </div>
        </Html>
      </Canvas>

      {/* Left explainer */}
      <div style={leftPanel}>
        <div style={badge('rgba(0,83,214,0.18)', '#0053d6', '#7faaff')}>
          Confidence head · Algorithm 29
        </div>

        <h2 style={panelH2}>What you're seeing</h2>
        <p style={panelP}>
          Each Cα carries a predicted <b style={{ color: '#65cbf3' }}>pLDDT</b> score —
          AlphaFold's intrinsic confidence at the per-residue level. Colours follow
          the canonical AF2 palette: dark blue ≥ 90 (very high), light blue ≥ 70,
          yellow ≥ 50, orange &lt; 50 (likely disordered).
        </p>

        <p style={panelP}>
          The network projects the final single representation s<sub>i</sub> from
          Algorithm 20 line 30 into <b>50 bins</b> of width 2 (v<sub>bins</sub> = [1, 3, …, 99]),
          then takes the <b>expected value</b> to produce r<sub>i</sub><sup>pLDDT</sup> ∈ [0, 100].
        </p>

        <div style={{
          margin: '10px 0', padding: 10,
          background: 'rgba(101,203,243,0.07)', borderRadius: 6,
          border: '1px solid rgba(101,203,243,0.2)',
        }}>
          <KaTeXFormula
            formula={'\\mathbf{a}_i = \\mathrm{relu}\\!\\bigl(\\mathrm{Linear}(\\mathrm{relu}(\\mathrm{Linear}(\\mathrm{LayerNorm}(\\mathbf{s}_i))))\\bigr)'}
            style={{ fontSize: 11, padding: 0, background: 'transparent', borderLeft: 'none' }}
          />
          <KaTeXFormula
            formula={'\\mathbf{p}_i^{\\mathrm{pLDDT}} = \\mathrm{softmax}(\\mathrm{Linear}(\\mathbf{a}_i))'}
            style={{ fontSize: 11, padding: 0, background: 'transparent', borderLeft: 'none' }}
          />
          <KaTeXFormula
            formula={'r_i^{\\mathrm{pLDDT}} = {\\mathbf{p}_i^{\\mathrm{pLDDT}}}^{\\!\\top}\\mathbf{v}_{\\mathrm{bins}}'}
            style={{ fontSize: 11, padding: 0, background: 'transparent', borderLeft: 'none' }}
          />
        </div>

        <h3 style={panelH3}>Training loss</h3>
        <p style={panelP}>
          The ground-truth lDDT-Cα score is discretized into 50 bins and used as the target
          for a cross-entropy loss averaged over residues. Examples with resolution outside
          0.1 Å – 3.0 Å are excluded; NMR structures are ignored to keep the targets reliable.
        </p>
        <div style={{
          padding: 10, background: 'rgba(101,203,243,0.07)', borderRadius: 6,
          border: '1px solid rgba(101,203,243,0.2)',
        }}>
          <KaTeXFormula
            formula={'\\mathcal{L}_{\\mathrm{conf}} = \\mathrm{mean}_i\\!\\left({\\mathbf{p}_i^{\\mathrm{true\\,LDDT}}}^{\\!\\top}\\log \\mathbf{p}_i^{\\mathrm{pLDDT}}\\right)'}
            style={{ fontSize: 11, padding: 0, background: 'transparent', borderLeft: 'none' }}
          />
        </div>

        <h3 style={panelH3}>Reading this prediction</h3>
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 11, color: '#8899aa', lineHeight: 1.6 }}>
          <li>Stable α-helix or β-sheet core → almost always dark blue.</li>
          <li>Long flexible loops, IDRs, termini → orange / yellow.</li>
          <li>Chain pLDDT is the mean of per-residue pLDDT — useful but lossy.</li>
          <li>pLDDT is <i>local</i>: it does not reflect whether two confident domains are correctly oriented relative to each other (use pTM / TM-score head for that).</li>
        </ul>
      </div>

      {/* Right controls + readout */}
      <div style={rightPanel}>
        <h2 style={panelH2}>Per-residue pLDDT</h2>
        <PerResidueBars scores={scores} selectedI={selectedI} onClickBar={setSelectedI} />

        <div style={{ marginTop: 14 }}>
          <div style={{ fontSize: 11, color: '#c0c8d0', fontWeight: 600, marginBottom: 4 }}>
            Selected residue  i = {selectedI}
          </div>
          <input
            type="range" min={0} max={N_RES - 1} step={1} value={selectedI}
            onChange={(e) => { setSelectedI(Number(e.target.value)); setAutoSweep(false) }}
            style={{ width: '100%', accentColor: '#ffe082' }}
          />
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8, marginTop: 6,
            padding: '6px 10px', background: 'rgba(255,224,130,0.08)',
            border: '1px solid rgba(255,224,130,0.25)', borderRadius: 6,
          }}>
            <div style={{
              width: 16, height: 16, borderRadius: 4,
              background: plddtColor(scores[selectedI]),
              boxShadow: `0 0 8px ${plddtColor(scores[selectedI])}aa`,
            }} />
            <span style={{
              fontSize: 12, color: '#c0c8d0',
              fontFamily: 'JetBrains Mono, monospace',
            }}>
              pLDDT<sub>i</sub> = <b style={{ color: '#ffe082' }}>{scores[selectedI].toFixed(1)}</b>
            </span>
          </div>
        </div>

        <h3 style={panelH3}>Predicted 50-bin distribution</h3>
        <p style={{ ...panelP, margin: '0 0 4px' }}>
          p<sub>i</sub><sup>pLDDT</sup> over bin centres v<sub>bins</sub>.
          The yellow marker is the expected value — i.e. the displayed pLDDT score.
        </p>
        <DistributionPlot dist={distData[selectedI].dist} expected={distData[selectedI].expected} />

        <h3 style={panelH3}>Colour legend</h3>
        <LegendStrip />

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '14px 0' }} />

        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: '#c0c8d0', cursor: 'pointer' }}>
          <input type="checkbox" checked={showSpine} onChange={(e) => setShowSpine(e.target.checked)}
            style={{ accentColor: '#42a5f5' }} />
          Show backbone spline
        </label>

        <div style={tipBox}>
          <b style={{ color: '#42a5f5' }}>Tip:</b> click any bar (or any Cα bead in the 3D view)
          to select that residue. Toggle Sweep to autoplay through the chain.
        </div>
      </div>
    </div>
  )
}

// ── Styles ───────────────────────────────────────────────────────────────────

const panelBase: React.CSSProperties = {
  position: 'absolute', top: 80, maxHeight: 'calc(100% - 100px)',
  overflowY: 'auto', padding: 16,
  background: 'rgba(15,15,28,0.78)', backdropFilter: 'blur(8px)',
  border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10,
  zIndex: 5, color: '#c0c8d0',
}

const topBar: React.CSSProperties = {
  position: 'absolute', top: 0, left: 0, right: 0,
  padding: '10px 20px', zIndex: 10,
  background: 'linear-gradient(180deg, rgba(7,7,18,0.92) 0%, rgba(7,7,18,0) 100%)',
  display: 'flex', alignItems: 'center', gap: 14, pointerEvents: 'auto',
}

const backBtn: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.12)', borderRadius: 6,
  background: 'rgba(20,20,40,0.6)',
  padding: '5px 12px', cursor: 'pointer', fontSize: 12, color: '#c0c8d0',
}

const pillBtn: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.12)', borderRadius: 6,
  background: 'rgba(20,20,40,0.6)',
  padding: '5px 12px', cursor: 'pointer', fontSize: 12, color: '#c0c8d0',
}

const leftPanel: React.CSSProperties = { ...panelBase, left: 16, width: 330 }
const rightPanel: React.CSSProperties = { ...panelBase, right: 16, width: 300 }

const panelH2: React.CSSProperties = {
  margin: '6px 0 8px', fontSize: 14, color: '#c0c8d0', fontWeight: 700,
}

const panelH3: React.CSSProperties = {
  margin: '14px 0 6px', fontSize: 12, color: '#c0c8d0', fontWeight: 700,
}

const panelP: React.CSSProperties = {
  margin: '0 0 8px', fontSize: 11.5, color: '#8899aa', lineHeight: 1.6,
}

const tipBox: React.CSSProperties = {
  marginTop: 12, padding: 10,
  background: 'rgba(66,165,245,0.06)', borderRadius: 6,
  border: '1px solid rgba(66,165,245,0.18)',
  fontSize: 10.5, color: '#8899aa', lineHeight: 1.55,
}

function badge(bg: string, border: string, color: string): React.CSSProperties {
  return {
    display: 'inline-block', padding: '3px 10px', borderRadius: 12,
    background: bg, border: `1px solid ${border}60`,
    fontSize: 10.5, color, fontWeight: 600, marginBottom: 8,
  }
}
