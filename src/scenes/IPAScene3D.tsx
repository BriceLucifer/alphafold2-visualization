import { useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html, Line, CatmullRomLine } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { KaTeXFormula } from '../ui/KaTeXFormula'

// ── Synthetic backbone: an alpha-helical curl bending into a loop ─────────────
// 14 residues. Coordinates in nanometres (PDF: "We represent all coordinates
// within IPA in nanometres").

const N_RES = 14

function buildBackbone(): { pos: THREE.Vector3; tangent: THREE.Vector3; normal: THREE.Vector3 }[] {
  const out: { pos: THREE.Vector3; tangent: THREE.Vector3; normal: THREE.Vector3 }[] = []
  // Alpha-helix: ~3.6 residues per turn, rise 0.15 nm/residue, radius 0.23 nm.
  // We add a gentle bend so it doesn't look like a perfect cylinder.
  const radius = 0.42
  const rise = 0.18
  const tpr = (2 * Math.PI) / 3.6
  for (let i = 0; i < N_RES; i++) {
    const t = i * tpr
    const bend = Math.sin(i / N_RES * Math.PI) * 0.15
    const px = Math.cos(t) * radius + bend
    const py = i * rise - (N_RES - 1) * rise * 0.5
    const pz = Math.sin(t) * radius
    out.push({
      pos: new THREE.Vector3(px, py, pz),
      tangent: new THREE.Vector3(),
      normal: new THREE.Vector3(),
    })
  }
  // Compute frames via Gram-Schmidt on neighbours (Algorithm 21).
  // x1 = N (prev Cα), x2 = Cα (centre), x3 = C (next Cα).
  for (let i = 0; i < N_RES; i++) {
    const x1 = out[Math.max(0, i - 1)].pos
    const x2 = out[i].pos
    const x3 = out[Math.min(N_RES - 1, i + 1)].pos
    const v1 = new THREE.Vector3().subVectors(x3, x2)
    const v2 = new THREE.Vector3().subVectors(x1, x2)
    const e1 = v1.clone().normalize()
    const u2 = v2.clone().sub(e1.clone().multiplyScalar(e1.dot(v2)))
    const e2 = u2.normalize()
    out[i].tangent = e1
    out[i].normal = e2
  }
  return out
}

// ── Compute a frame's rotation matrix (R) so we can apply T_i ∘ vec ──────────

function frameMatrix(e1: THREE.Vector3, e2: THREE.Vector3, t: THREE.Vector3) {
  const e3 = new THREE.Vector3().crossVectors(e1, e2).normalize()
  const m = new THREE.Matrix4().makeBasis(e1, e2, e3)
  m.setPosition(t)
  return m
}

// Apply T to a local-frame vector
function applyFrame(m: THREE.Matrix4, local: THREE.Vector3): THREE.Vector3 {
  return local.clone().applyMatrix4(m)
}

// ── Local-frame query points (4 unit-ish offsets, ~tetrahedral) ──────────────

const LOCAL_QUERY_POINTS: THREE.Vector3[] = [
  new THREE.Vector3(0.45, 0.0, 0.0),
  new THREE.Vector3(-0.22, 0.39, 0.0),
  new THREE.Vector3(-0.22, -0.39, 0.0),
  new THREE.Vector3(0.0, 0.0, 0.45),
]

// ── Components ───────────────────────────────────────────────────────────────

function BackbonePath({ frames }: { frames: ReturnType<typeof buildBackbone> }) {
  const points = useMemo(() => frames.map(f => f.pos.toArray() as [number, number, number]), [frames])
  return (
    <CatmullRomLine
      points={points}
      color="#5a7090"
      lineWidth={2.2}
      transparent
      opacity={0.55}
      segments={64}
    />
  )
}

function ResidueSpheres({
  frames, selectedI, hoverJ, setHoverJ, setSelectedI,
}: {
  frames: ReturnType<typeof buildBackbone>
  selectedI: number
  hoverJ: number | null
  setHoverJ: (i: number | null) => void
  setSelectedI: (i: number) => void
}) {
  return (
    <>
      {frames.map((f, i) => {
        const isSelected = i === selectedI
        const isHover = i === hoverJ
        const color = isSelected ? '#ff9100' : isHover ? '#ffd54f' : '#4fc3f7'
        return (
          <mesh
            key={`r-${i}`}
            position={f.pos}
            onPointerOver={(e) => { e.stopPropagation(); setHoverJ(i); document.body.style.cursor = 'pointer' }}
            onPointerOut={(e) => { e.stopPropagation(); setHoverJ(null); document.body.style.cursor = 'auto' }}
            onClick={(e) => { e.stopPropagation(); setSelectedI(i) }}
          >
            <sphereGeometry args={[isSelected ? 0.085 : 0.06, 24, 24]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={isSelected ? 0.7 : isHover ? 0.45 : 0.18}
              roughness={0.3}
              metalness={0.1}
            />
          </mesh>
        )
      })}
    </>
  )
}

function FrameTriad({
  origin, e1, e2, scale = 0.18, dim = false,
}: {
  origin: THREE.Vector3
  e1: THREE.Vector3
  e2: THREE.Vector3
  scale?: number
  dim?: boolean
}) {
  const e3 = useMemo(() => new THREE.Vector3().crossVectors(e1, e2).normalize(), [e1, e2])
  const o = origin.toArray() as [number, number, number]
  const a = origin.clone().add(e1.clone().multiplyScalar(scale)).toArray() as [number, number, number]
  const b = origin.clone().add(e2.clone().multiplyScalar(scale)).toArray() as [number, number, number]
  const c = origin.clone().add(e3.clone().multiplyScalar(scale)).toArray() as [number, number, number]
  const o2 = o
  const a2 = a
  const b2 = b
  const c2 = c
  const opacity = dim ? 0.35 : 1
  return (
    <>
      <Line points={[o2, a2]} color="#ef5350" lineWidth={dim ? 1 : 1.8} transparent opacity={opacity} />
      <Line points={[o2, b2]} color="#66bb6a" lineWidth={dim ? 1 : 1.8} transparent opacity={opacity} />
      <Line points={[o2, c2]} color="#42a5f5" lineWidth={dim ? 1 : 1.8} transparent opacity={opacity} />
    </>
  )
}

function QueryKeyPoints({
  frame, color, asQuery, nPoints,
}: {
  frame: { pos: THREE.Vector3; tangent: THREE.Vector3; normal: THREE.Vector3 }
  color: string
  asQuery: boolean // big spheres if query
  nPoints: number
}) {
  const m = useMemo(
    () => frameMatrix(frame.tangent, frame.normal, frame.pos),
    [frame]
  )
  const points = useMemo(
    () => LOCAL_QUERY_POINTS.slice(0, nPoints).map(p => applyFrame(m, p)),
    [m, nPoints]
  )
  return (
    <>
      {points.map((p, i) => (
        <mesh key={`qp-${i}`} position={p}>
          <sphereGeometry args={[asQuery ? 0.038 : 0.022, 14, 14]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={asQuery ? 0.9 : 0.4}
            transparent
            opacity={asQuery ? 1 : 0.85}
          />
        </mesh>
      ))}
    </>
  )
}

function AttentionLinks({
  frames, selectedI, attn, showDistanceLines, nPoints,
}: {
  frames: ReturnType<typeof buildBackbone>
  selectedI: number
  attn: number[]
  showDistanceLines: boolean
  nPoints: number
}) {
  const fi = frames[selectedI]
  const mi = useMemo(() => frameMatrix(fi.tangent, fi.normal, fi.pos), [fi])
  const qiPts = useMemo(
    () => LOCAL_QUERY_POINTS.slice(0, nPoints).map(p => applyFrame(mi, p)),
    [mi, nPoints]
  )

  return (
    <>
      {frames.map((fj, j) => {
        if (j === selectedI) return null
        const w = attn[j]
        if (w < 0.01) return null
        const mj = frameMatrix(fj.tangent, fj.normal, fj.pos)
        const kjPts = LOCAL_QUERY_POINTS.slice(0, nPoints).map(p => applyFrame(mj, p))

        const opacity = Math.min(1, w * 6 + 0.05)
        return (
          <group key={`link-${j}`}>
            {/* Coarse residue-to-residue link (Cα → Cα) */}
            <Line
              points={[fi.pos.toArray() as [number, number, number], fj.pos.toArray() as [number, number, number]]}
              color="#ffab00"
              lineWidth={Math.max(0.6, w * 5)}
              transparent
              opacity={opacity * 0.9}
            />
            {/* Per-point squared-distance lines (||T_i∘q − T_j∘k||) — the IPA point term */}
            {showDistanceLines && qiPts.map((qp, p) => (
              <Line
                key={`pl-${j}-${p}`}
                points={[qp.toArray() as [number, number, number], kjPts[p].toArray() as [number, number, number]]}
                color="#4caf50"
                lineWidth={1}
                dashed
                dashScale={20}
                transparent
                opacity={opacity * 0.6}
              />
            ))}
          </group>
        )
      })}
    </>
  )
}

function FlowParticles({
  frames, selectedI, attn,
}: {
  frames: ReturnType<typeof buildBackbone>
  selectedI: number
  attn: number[]
}) {
  const groupRef = useRef<THREE.Group>(null)
  const fi = frames[selectedI]

  // Pre-build particle paths: from j → i, modulated by attn
  const lanes = useMemo(() => {
    return frames
      .map((fj, j) => ({ j, fj, w: attn[j] }))
      .filter(d => d.j !== selectedI && d.w > 0.05)
      .slice(0, 8)
  }, [frames, selectedI, attn])

  useFrame(({ clock }) => {
    if (!groupRef.current) return
    const t = clock.getElapsedTime()
    groupRef.current.children.forEach((child, idx) => {
      const lane = lanes[idx % lanes.length]
      if (!lane) return
      const speed = 0.6
      const phase = ((t * speed + idx * 0.13) % 1)
      const a = lane.fj.pos
      const b = fi.pos
      child.position.lerpVectors(a, b, phase)
    })
  })

  return (
    <group ref={groupRef}>
      {lanes.map((lane, i) => (
        <mesh key={`fp-${i}`}>
          <sphereGeometry args={[0.022, 10, 10]} />
          <meshBasicMaterial color="#ffe082" transparent opacity={Math.min(1, lane.w * 8)} />
        </mesh>
      ))}
    </group>
  )
}

// ── IPA scoring: synthetic softmax driven by spatial proximity ────────────────
// Per Algorithm 22 line 7. We use a stripped-down score:
//   logit_j = w_L * scalar_dot + b_ij - (γ w_C / 2) Σ_p ||T_i∘q − T_j∘k||²
// Scalar dot and pair bias are random per-residue scalars (synthetic).

function useSyntheticAttention(frames: ReturnType<typeof buildBackbone>, selectedI: number, gamma: number, nPoints: number) {
  return useMemo(() => {
    const N = frames.length
    const fi = frames[selectedI]
    const mi = frameMatrix(fi.tangent, fi.normal, fi.pos)
    const qiPts = LOCAL_QUERY_POINTS.slice(0, nPoints).map(p => applyFrame(mi, p))

    const w_L = Math.sqrt(1 / 3)
    const w_C = Math.sqrt(2 / (9 * Math.max(1, nPoints)))

    const logits: number[] = []
    for (let j = 0; j < N; j++) {
      if (j === selectedI) {
        logits.push(-Infinity)
        continue
      }
      const fj = frames[j]
      const mj = frameMatrix(fj.tangent, fj.normal, fj.pos)
      const kjPts = LOCAL_QUERY_POINTS.slice(0, nPoints).map(p => applyFrame(mj, p))

      // Scalar (synthetic): inverse sequence distance
      const seqDist = Math.abs(j - selectedI)
      const scalarDot = 1 / (seqDist + 1)
      // Pair bias (synthetic): periodic
      const bias = 0.2 * Math.sin(selectedI * 0.7 + j * 0.5)
      // Point term
      let pointSq = 0
      for (let p = 0; p < qiPts.length; p++) {
        pointSq += qiPts[p].distanceToSquared(kjPts[p])
      }
      const logit = w_L * scalarDot + bias - (gamma * w_C / 2) * pointSq
      logits.push(logit)
    }
    // softmax
    const maxL = Math.max(...logits.filter(v => isFinite(v)))
    const exps = logits.map(v => isFinite(v) ? Math.exp(v - maxL) : 0)
    const sum = exps.reduce((a, b) => a + b, 0) || 1
    return exps.map(v => v / sum)
  }, [frames, selectedI, gamma, nPoints])
}

// ── Scene composition ─────────────────────────────────────────────────────────

function Scene({
  selectedI, setSelectedI, gamma, nPoints, showAllFrames, showQueryPoints, showDistanceLines,
}: {
  selectedI: number
  setSelectedI: (i: number) => void
  gamma: number
  nPoints: number
  showAllFrames: boolean
  showQueryPoints: boolean
  showDistanceLines: boolean
}) {
  const frames = useMemo(buildBackbone, [])
  const [hoverJ, setHoverJ] = useState<number | null>(null)
  const attn = useSyntheticAttention(frames, selectedI, gamma, nPoints)
  const groupRef = useRef<THREE.Group>(null)

  // Subtle automatic rotation
  useFrame(({ clock }) => {
    if (!groupRef.current) return
    groupRef.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.08) * 0.15
  })

  return (
    <group ref={groupRef}>
      <BackbonePath frames={frames} />
      <ResidueSpheres
        frames={frames}
        selectedI={selectedI}
        hoverJ={hoverJ}
        setHoverJ={setHoverJ}
        setSelectedI={setSelectedI}
      />

      {/* Frames */}
      {frames.map((f, i) => {
        if (!showAllFrames && i !== selectedI && i !== hoverJ) return null
        const dim = i !== selectedI && i !== hoverJ
        return (
          <FrameTriad
            key={`tr-${i}`}
            origin={f.pos}
            e1={f.tangent}
            e2={f.normal}
            scale={i === selectedI ? 0.24 : 0.16}
            dim={dim}
          />
        )
      })}

      {/* Query/key points */}
      {showQueryPoints && (
        <>
          <QueryKeyPoints
            frame={frames[selectedI]}
            color="#ff9100"
            asQuery
            nPoints={nPoints}
          />
          {frames.map((f, j) => {
            if (j === selectedI) return null
            if (attn[j] < 0.02 && j !== hoverJ) return null
            return (
              <QueryKeyPoints
                key={`kp-${j}`}
                frame={f}
                color="#4caf50"
                asQuery={false}
                nPoints={nPoints}
              />
            )
          })}
        </>
      )}

      <AttentionLinks
        frames={frames}
        selectedI={selectedI}
        attn={attn}
        showDistanceLines={showDistanceLines}
        nPoints={nPoints}
      />

      <FlowParticles frames={frames} selectedI={selectedI} attn={attn} />
    </group>
  )
}

// ── Top-level scene with overlay panels ──────────────────────────────────────

export function IPAScene3D({ onBack }: { onBack: () => void }) {
  const [selectedI, setSelectedI] = useState(6)
  const [gamma, setGamma] = useState(1.0)
  const [nPoints, setNPoints] = useState(4)
  const [showAllFrames, setShowAllFrames] = useState(true)
  const [showQueryPoints, setShowQueryPoints] = useState(true)
  const [showDistanceLines, setShowDistanceLines] = useState(true)

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
            Invariant Point Attention (3D)
          </h1>
          <div style={{ fontSize: 11, color: '#667788', marginTop: 2 }}>
            Algorithm 22 · Supplementary Figure 8 · units in nanometres
          </div>
        </div>
      </div>

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [2.4, 1.4, 2.6], fov: 45, near: 0.05, far: 50 }}
        style={{ width: '100%', height: '100%' }}
        gl={{ antialias: true, alpha: false }}
      >
        <color attach="background" args={['#070712']} />
        <fog attach="fog" args={['#070712', 4, 10]} />
        <ambientLight intensity={0.35} />
        <directionalLight position={[3, 4, 2]} intensity={0.8} color="#ffffff" />
        <directionalLight position={[-3, -2, -1]} intensity={0.25} color="#88aaff" />

        <Scene
          selectedI={selectedI}
          setSelectedI={setSelectedI}
          gamma={gamma}
          nPoints={nPoints}
          showAllFrames={showAllFrames}
          showQueryPoints={showQueryPoints}
          showDistanceLines={showDistanceLines}
        />

        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={0.6}
          minDistance={1.2}
          maxDistance={6}
          target={[0, 0, 0]}
        />

        <EffectComposer>
          <Bloom intensity={0.55} luminanceThreshold={0.5} luminanceSmoothing={0.4} mipmapBlur />
          <Vignette eskil={false} offset={0.15} darkness={0.7} />
        </EffectComposer>

        {/* Floating axis ref in corner */}
        <Html position={[0, 0, 0]} fullscreen pointerEvents="none">
          <div />
        </Html>
      </Canvas>

      {/* Left explainer panel */}
      <div style={leftPanel}>
        <div style={badge('rgba(102, 187, 106, 0.12)', '#66bb6a', '#a5d6a7')}>
          Structure Module · Algorithm 22
        </div>

        <h2 style={panelH2}>What you're seeing</h2>
        <p style={panelP}>
          Each Cα carries a <b style={{ color: '#42a5f5' }}>rigid frame</b> T<sub>i</sub> = (R<sub>i</sub>, t<sub>i</sub>) —
          a local coordinate system (red/green/blue axes). Construction follows
          Algorithm 21: Gram–Schmidt on N, Cα, C atom positions.
        </p>

        <p style={panelP}>
          <b style={{ color: '#ff9100' }}>Orange spheres</b> are the selected residue's <i>query points</i>
          {' '}q̃<sub>i</sub><sup>hp</sup> — projected into global space via T<sub>i</sub>.
          <b style={{ color: '#66bb6a' }}> Green spheres</b> are <i>key points</i> of other residues.
        </p>

        <p style={panelP}>
          The IPA affinity penalises the <b style={{ color: '#4caf50' }}>squared distance</b> between
          paired projected points — dashed green lines. Closer ⇒ higher attention.
          That term <i>plus</i> the standard dot-product (w<sub>L</sub>·q·k) plus the pair-bias
          b<sub>ij</sub> give the final logits.
        </p>

        <div style={{ margin: '10px 0', padding: 10, background: 'rgba(76,175,80,0.06)', borderRadius: 6, border: '1px solid rgba(76,175,80,0.18)' }}>
          <KaTeXFormula
            formula={'a_{ij}^h = \\mathrm{softmax}_j\\!\\left( w_L\\frac{\\mathbf{q}_i^{h\\top}\\mathbf{k}_j^h}{\\sqrt{c}} + b_{ij}^h - \\frac{\\gamma^h w_C}{2}\\sum_p \\|T_i\\circ\\vec{q}_i^{hp} - T_j\\circ\\vec{k}_j^{hp}\\|^2 \\right)'}
            style={{ fontSize: 11, padding: 0, background: 'transparent', borderLeft: 'none' }}
          />
        </div>

        <h3 style={panelH3}>Why "invariant"?</h3>
        <p style={panelP}>
          Apply any global rigid motion T<sub>global</sub> to every frame, and the squared-distance
          term is unchanged: ‖T<sub>global</sub>(T<sub>i</sub>q̃ − T<sub>j</sub>k̃)‖² = ‖T<sub>i</sub>q̃ − T<sub>j</sub>k̃‖².
          So the network sees the same attention pattern no matter how you rotate or translate the protein.
        </p>

        <div style={legendBox}>
          <LegendRow color="#ff9100" label="selected residue i (click any to switch)" />
          <LegendRow color="#4fc3f7" label="other residues j" />
          <LegendRow color="#ffab00" label="attention strength (Cα ↔ Cα)" />
          <LegendRow color="#4caf50" label="‖T_i∘q − T_j∘k‖ point pair" />
          <LegendRow color="#ef5350" label="frame axis e₁ (tangent)" />
          <LegendRow color="#66bb6a" label="frame axis e₂ (normal)" />
          <LegendRow color="#42a5f5" label="frame axis e₃ (e₁ × e₂)" />
        </div>
      </div>

      {/* Right controls panel */}
      <div style={rightPanel}>
        <h2 style={panelH2}>Controls</h2>

        <SliderRow
          label={`Selected residue  i = ${selectedI}`}
          value={selectedI}
          min={0}
          max={N_RES - 1}
          step={1}
          onChange={setSelectedI}
        />

        <SliderRow
          label={`N_query_points = ${nPoints}`}
          value={nPoints}
          min={1}
          max={4}
          step={1}
          onChange={setNPoints}
          hint="Algorithm 22 default: 4"
        />

        <SliderRow
          label={`γᵃ (point weight) = ${gamma.toFixed(2)}`}
          value={gamma}
          min={0}
          max={3}
          step={0.05}
          onChange={setGamma}
          hint="softplus of a learnable scalar"
        />

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '14px 0' }} />

        <ToggleRow label="Show all frames" value={showAllFrames} onChange={setShowAllFrames} />
        <ToggleRow label="Show query/key points" value={showQueryPoints} onChange={setShowQueryPoints} />
        <ToggleRow label="Show point-pair distance lines" value={showDistanceLines} onChange={setShowDistanceLines} />

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '14px 0' }} />

        <div style={tipBox}>
          <b style={{ color: '#42a5f5' }}>Tip:</b> drag to orbit · scroll to zoom ·
          click any blue residue to set it as the query (i).
        </div>

        <div style={{ marginTop: 10, padding: 10, background: 'rgba(255,145,0,0.05)', borderRadius: 6, border: '1px solid rgba(255,145,0,0.18)' }}>
          <div style={{ fontSize: 10.5, color: '#ffab40', fontWeight: 600, marginBottom: 6 }}>
            Reading the geometry
          </div>
          <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 10.5, color: '#8899aa', lineHeight: 1.55 }}>
            <li>Each frame is a local coord system carried by a Cα.</li>
            <li>Query/key points are <i>fixed</i> in local frames; they move with their residue.</li>
            <li>Attention concentrates where projected q–k pairs land close in 3D.</li>
            <li>Frames will update after BackboneUpdate (Alg 23) — 8 layers of refinement.</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

// ── UI bits ──────────────────────────────────────────────────────────────────

function LegendRow({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
      <div style={{ width: 14, height: 14, borderRadius: 3, background: color, boxShadow: `0 0 6px ${color}88` }} />
      <span style={{ fontSize: 10.5, color: '#8899aa' }}>{label}</span>
    </div>
  )
}

function SliderRow({
  label, value, min, max, step, onChange, hint,
}: {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void; hint?: string
}) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 11, color: '#c0c8d0', fontWeight: 600, marginBottom: 4 }}>{label}</div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: '#42a5f5' }}
      />
      {hint && <div style={{ fontSize: 9.5, color: '#556677', marginTop: 2 }}>{hint}</div>}
    </div>
  )
}

function ToggleRow({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: '#c0c8d0', marginBottom: 6, cursor: 'pointer' }}>
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: '#42a5f5' }} />
      {label}
    </label>
  )
}

// ── Styles ───────────────────────────────────────────────────────────────────

const topBar: React.CSSProperties = {
  position: 'absolute', top: 0, left: 0, right: 0,
  padding: '10px 20px', zIndex: 10,
  background: 'linear-gradient(180deg, rgba(7,7,18,0.9) 0%, rgba(7,7,18,0) 100%)',
  display: 'flex', alignItems: 'center', gap: 14, pointerEvents: 'auto',
}

const backBtn: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.12)', borderRadius: 6, background: 'rgba(20,20,40,0.6)',
  padding: '5px 12px', cursor: 'pointer', fontSize: 12, color: '#c0c8d0',
}

const leftPanel: React.CSSProperties = {
  position: 'absolute', top: 80, left: 16, width: 320, maxHeight: 'calc(100% - 100px)',
  overflowY: 'auto', padding: 16,
  background: 'rgba(15,15,28,0.78)', backdropFilter: 'blur(8px)',
  border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10,
  zIndex: 5, color: '#c0c8d0',
}

const rightPanel: React.CSSProperties = {
  position: 'absolute', top: 80, right: 16, width: 280,
  padding: 16,
  background: 'rgba(15,15,28,0.78)', backdropFilter: 'blur(8px)',
  border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10,
  zIndex: 5, color: '#c0c8d0',
}

const panelH2: React.CSSProperties = {
  margin: '4px 0 8px', fontSize: 13, color: '#c0c8d0', fontWeight: 700,
}

const panelH3: React.CSSProperties = {
  margin: '12px 0 6px', fontSize: 12, color: '#c0c8d0', fontWeight: 700,
}

const panelP: React.CSSProperties = {
  margin: '0 0 8px', fontSize: 11.5, color: '#8899aa', lineHeight: 1.6,
}

const legendBox: React.CSSProperties = {
  marginTop: 10, padding: 10, background: 'rgba(255,255,255,0.03)', borderRadius: 6,
  border: '1px solid rgba(255,255,255,0.06)',
}

const tipBox: React.CSSProperties = {
  padding: 10, background: 'rgba(66,165,245,0.06)', borderRadius: 6,
  border: '1px solid rgba(66,165,245,0.18)',
  fontSize: 10.5, color: '#8899aa', lineHeight: 1.55,
}

function badge(bg: string, border: string, color: string): React.CSSProperties {
  return {
    display: 'inline-block', padding: '2px 8px', borderRadius: 10,
    background: bg, border: `1px solid ${border}60`,
    fontSize: 10, color, fontWeight: 600, marginBottom: 8,
  }
}
