import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, CatmullRomLine, Line } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { KaTeXFormula } from '../ui/KaTeXFormula'

// ── Constants ────────────────────────────────────────────────────────────────

const N_RES = 14
const N_LAYERS = 8 // Algorithm 20 outer loop runs the shared-weight stack 8×.

// ── Target fold: short alpha-helix bending into a loop ───────────────────────
// Adapted from Recycling3DScene.tsx::buildTargetFold so the user sees the same
// "final pose" target as the recycling scene.

function buildTargetPositions(): THREE.Vector3[] {
  const out: THREE.Vector3[] = []
  const radius = 0.42
  const rise = 0.18
  const tpr = (2 * Math.PI) / 3.6
  for (let i = 0; i < N_RES; i++) {
    const t = i * tpr
    const bend = Math.sin((i / (N_RES - 1)) * Math.PI) * 0.18
    out.push(new THREE.Vector3(
      Math.cos(t) * radius + bend,
      i * rise - (N_RES - 1) * rise * 0.5,
      Math.sin(t) * radius,
    ))
  }
  // Center so the cluster at origin (layer 0) doesn't sit way off-axis.
  const com = out.reduce((a, b) => a.clone().add(b), new THREE.Vector3())
  com.multiplyScalar(1 / out.length)
  return out.map(p => p.clone().sub(com))
}

// Per-residue target orientation derived Alg-21-style (Gram-Schmidt on neighbours)
function buildTargetOrientations(positions: THREE.Vector3[]): THREE.Quaternion[] {
  const qs: THREE.Quaternion[] = []
  for (let i = 0; i < positions.length; i++) {
    const x1 = positions[Math.max(0, i - 1)]
    const x2 = positions[i]
    const x3 = positions[Math.min(positions.length - 1, i + 1)]
    const v1 = new THREE.Vector3().subVectors(x3, x2)
    const v2 = new THREE.Vector3().subVectors(x1, x2)
    const e1 = v1.clone().normalize()
    const u2 = v2.clone().sub(e1.clone().multiplyScalar(e1.dot(v2)))
    const e2 = u2.normalize()
    const e3 = new THREE.Vector3().crossVectors(e1, e2).normalize()
    const m = new THREE.Matrix4().makeBasis(e1, e2, e3)
    const q = new THREE.Quaternion().setFromRotationMatrix(m)
    qs.push(q)
  }
  return qs
}

// ── Per-layer frame state for every residue ─────────────────────────────────
// Cumulative T_i^{(l)} = (R_i^{(l)}, t_i^{(l)}). Layer 0 is the "black hole":
// all positions at origin, all rotations = identity.

type FrameState = { pos: THREE.Vector3; quat: THREE.Quaternion }

function easeOutCubic(x: number) {
  const k = 1 - x
  return 1 - k * k * k
}

// Build cumulative frame states across the 9 snapshots (layer 0..8).
// At layer l, position = lerp(0, target, easeOut(l/8)), and the rotation slerps
// from identity to the target orientation along the same eased curve.
function buildFrameStates(): FrameState[][] {
  const targetPos = buildTargetPositions()
  const targetQ = buildTargetOrientations(targetPos)
  const out: FrameState[][] = []
  for (let l = 0; l <= N_LAYERS; l++) {
    const alpha = easeOutCubic(l / N_LAYERS)
    const layer: FrameState[] = []
    for (let i = 0; i < N_RES; i++) {
      const pos = targetPos[i].clone().multiplyScalar(alpha)
      const q = new THREE.Quaternion().slerpQuaternions(new THREE.Quaternion(), targetQ[i], alpha)
      layer.push({ pos, quat: q })
    }
    out.push(layer)
  }
  return out
}

// Recover the "predicted" pre-normalisation quaternion (1, b, c, d) the network
// would have emitted to take T_i^{(l-1)} → T_i^{(l)}. We compute the unit delta
// quaternion q_delta = q_l * q_{l-1}^{-1}, then synthesise a non-unit (1,b,c,d)
// by dividing the unit by its w-component (so w == 1) plus a residue-keyed
// noise term to make the numbers look "predicted" rather than perfectly clean.
function predictedQuatRaw(
  prev: THREE.Quaternion,
  curr: THREE.Quaternion,
  noiseSeed: number,
): { b: number; c: number; d: number } {
  const inv = prev.clone().invert()
  const delta = curr.clone().multiply(inv)
  // Force positive w hemisphere so division by w is stable.
  if (delta.w < 0) {
    delta.x *= -1; delta.y *= -1; delta.z *= -1; delta.w *= -1
  }
  const wSafe = Math.max(1e-6, delta.w)
  let b = delta.x / wSafe
  let c = delta.y / wSafe
  let d = delta.z / wSafe
  // Tiny deterministic noise so consecutive layers don't read as identical.
  let s = (noiseSeed * 1664525 + 1013904223) >>> 0
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return (s / 0xffffffff - 0.5) * 2
  }
  b += rnd() * 0.03
  c += rnd() * 0.03
  d += rnd() * 0.03
  return { b, c, d }
}

// ── Interpolated state used for the live 3D render ──────────────────────────

function lerpFrameStates(
  a: FrameState[],
  b: FrameState[],
  t: number,
): FrameState[] {
  return a.map((fa, i) => ({
    pos: fa.pos.clone().lerp(b[i].pos, t),
    quat: fa.quat.clone().slerp(b[i].quat, t),
  }))
}

// ── 3D sub-components ───────────────────────────────────────────────────────

// Module-level scratch — FrameTriad recomputes per frame for every residue,
// and allocations here add up fast (84+ Vector3s/frame for N_RES=14).
const _triadE = new THREE.Vector3()

function FrameTriad({
  origin, quat, scale, dim,
}: { origin: THREE.Vector3; quat: THREE.Quaternion; scale: number; dim: boolean }) {
  const ox = origin.x, oy = origin.y, oz = origin.z
  const o: [number, number, number] = [ox, oy, oz]
  _triadE.set(1, 0, 0).applyQuaternion(quat).multiplyScalar(scale)
  const a: [number, number, number] = [ox + _triadE.x, oy + _triadE.y, oz + _triadE.z]
  _triadE.set(0, 1, 0).applyQuaternion(quat).multiplyScalar(scale)
  const b: [number, number, number] = [ox + _triadE.x, oy + _triadE.y, oz + _triadE.z]
  _triadE.set(0, 0, 1).applyQuaternion(quat).multiplyScalar(scale)
  const c: [number, number, number] = [ox + _triadE.x, oy + _triadE.y, oz + _triadE.z]
  const opacity = dim ? 0.35 : 1
  return (
    <>
      <Line points={[o, a]} color="#ef5350" lineWidth={dim ? 1 : 1.8} transparent opacity={opacity} />
      <Line points={[o, b]} color="#66bb6a" lineWidth={dim ? 1 : 1.8} transparent opacity={opacity} />
      <Line points={[o, c]} color="#42a5f5" lineWidth={dim ? 1 : 1.8} transparent opacity={opacity} />
    </>
  )
}

function BackboneSpheres({
  frames, selectedI, setSelectedI,
}: {
  frames: FrameState[]
  selectedI: number
  setSelectedI: (i: number) => void
}) {
  return (
    <>
      {frames.map((f, i) => {
        const isSel = i === selectedI
        const color = isSel ? '#ff9100' : '#4fc3f7'
        return (
          <mesh
            key={`ca-${i}`}
            position={f.pos}
            onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = 'pointer' }}
            onPointerOut={(e) => { e.stopPropagation(); document.body.style.cursor = 'auto' }}
            onClick={(e) => { e.stopPropagation(); setSelectedI(i) }}
          >
            <sphereGeometry args={[isSel ? 0.085 : 0.065, 18, 18]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={isSel ? 0.7 : 0.22}
              roughness={0.32}
              metalness={0.1}
            />
          </mesh>
        )
      })}
    </>
  )
}

function BackbonePath({ frames }: { frames: FrameState[] }) {
  // CatmullRomLine needs distinct points; at layer 0 every Cα is at origin so
  // we just omit the line at extreme collapse to avoid degenerate geometry.
  const allAtOrigin = frames.every(f => f.pos.lengthSq() < 1e-6)
  if (allAtOrigin) return null
  const pts = frames.map(f => f.pos.toArray() as [number, number, number])
  return (
    <CatmullRomLine
      points={pts}
      color="#7bd3ff"
      lineWidth={2.4}
      transparent
      opacity={0.7}
      segments={56}
    />
  )
}

function PositionTrails({
  allLayers, currentLayer, selectedI,
}: {
  allLayers: FrameState[][]
  currentLayer: number
  selectedI: number
}) {
  // For each residue, a thin line through its layer-0..currentLayer positions.
  // The selected residue gets a brighter trail.
  // currentLayer is fractional, so we draw segments up to floor(currentLayer)
  // and a partial segment to the live interpolated point.
  const upper = Math.min(N_LAYERS, Math.floor(currentLayer))
  return (
    <>
      {Array.from({ length: N_RES }).map((_, i) => {
        const pts: [number, number, number][] = []
        for (let l = 0; l <= upper; l++) {
          const p = allLayers[l][i].pos
          // De-dup the trail at the black-hole layer so we don't waste segments
          // sitting on top of each other at the origin.
          if (l === 0) {
            pts.push([0.0001 * i, 0, 0]) // tiny stagger keeps the line geometry well-defined
            continue
          }
          pts.push(p.toArray() as [number, number, number])
        }
        if (pts.length < 2) return null
        const isSel = i === selectedI
        return (
          <Line
            key={`trail-${i}`}
            points={pts}
            color={isSel ? '#ffd180' : '#445577'}
            lineWidth={isSel ? 1.6 : 0.8}
            transparent
            opacity={isSel ? 0.85 : 0.45}
          />
        )
      })}
    </>
  )
}

function BlackHoleRing() {
  // Faint ring + dim sphere at origin showing where layer-0 sat.
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.12, 0.16, 64]} />
        <meshBasicMaterial color="#ffd54f" transparent opacity={0.4} side={THREE.DoubleSide} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.07, 18, 18]} />
        <meshBasicMaterial color="#ffd54f" transparent opacity={0.18} />
      </mesh>
    </group>
  )
}

// ── Scene root ──────────────────────────────────────────────────────────────

function Scene({
  liveFrames, allLayers, currentLayer, selectedI, setSelectedI,
  showTriads, showTrails, showBlackHole,
}: {
  liveFrames: FrameState[]
  allLayers: FrameState[][]
  currentLayer: number
  selectedI: number
  setSelectedI: (i: number) => void
  showTriads: boolean
  showTrails: boolean
  showBlackHole: boolean
}) {
  const groupRef = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    if (!groupRef.current) return
    groupRef.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.07) * 0.15
  })
  return (
    <group ref={groupRef}>
      <BackbonePath frames={liveFrames} />
      <BackboneSpheres frames={liveFrames} selectedI={selectedI} setSelectedI={setSelectedI} />
      {showTriads && liveFrames.map((f, i) => (
        <FrameTriad
          key={`tr-${i}`}
          origin={f.pos}
          quat={f.quat}
          scale={i === selectedI ? 0.22 : 0.14}
          dim={i !== selectedI}
        />
      ))}
      {showTrails && (
        <PositionTrails
          allLayers={allLayers}
          currentLayer={currentLayer}
          selectedI={selectedI}
        />
      )}
      {showBlackHole && <BlackHoleRing />}
    </group>
  )
}

// ── Narration per layer ─────────────────────────────────────────────────────

const NARRATION: Record<number, string> = {
  0: '"Black-hole init": every frame is T_i = (I, 0). All 14 residues sit on top of each other at the origin. No geometry yet — just identity rotations and zero translations.',
  1: 'First update — frames push out of the black-hole origin. The predicted (b, c, d) are still tiny: rotations are near-identity, translations carry most of the early signal.',
  2: 'Second layer. Residues spread apart along a coarse trajectory. The shared-weight stack is iterating; gradients flow on translations but NOT on rotations (Alg 20 line 20).',
  3: 'Third layer. Secondary-structure spacing starts to appear. Each frame composes a small ΔT onto its existing T_i.',
  4: 'Halfway through. The chain is recognisably extended. Quaternions are still close to identity per-layer — the cumulative R_i has done most of the rotation work.',
  5: 'Fifth layer. Subtle reorientations align the helical turn. The triads rotate visibly between layers.',
  6: 'Sixth layer. The fold is converging. Per-layer updates are getting smaller — the network is "fine-tuning".',
  7: 'Seventh layer. Almost final. Residual moves are now sub-Ångström adjustments.',
  8: 'Final layer. All frames are placed. Downstream the torsion network predicts side-chain χ angles; FAPE then scores the whole structure against ground truth.',
}

// ── Top-level component ─────────────────────────────────────────────────────

export function BackboneRefinement3DScene({ onBack }: { onBack: () => void }) {
  const [layerIdx, setLayerIdx] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1.0)
  const [selectedI, setSelectedI] = useState(Math.floor(N_RES / 2))
  const [showTriads, setShowTriads] = useState(true)
  const [showTrails, setShowTrails] = useState(true)
  const [showBlackHole, setShowBlackHole] = useState(true)

  // Continuous "live" layer position (fractional) used for smooth interpolation.
  const [liveLayer, setLiveLayer] = useState(0)
  const targetLayerRef = useRef(0)

  const allLayers = useMemo(buildFrameStates, [])

  // Drive liveLayer toward layerIdx smoothly (lerp every animation frame).
  useEffect(() => { targetLayerRef.current = layerIdx }, [layerIdx])

  // RAF loop for interpolation.
  useEffect(() => {
    let raf = 0
    let last = performance.now()
    const loop = (now: number) => {
      const dt = (now - last) / 1000
      last = now
      setLiveLayer(prev => {
        const target = targetLayerRef.current
        // Lerp ~1.8 layers per second baseline.
        const k = Math.min(1, dt * 2.4)
        return prev + (target - prev) * k
      })
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  // Auto-step through layers when playing.
  useEffect(() => {
    if (!playing) return
    const period = 1700 / speed
    const id = setInterval(() => {
      setLayerIdx(l => (l + 1) % (N_LAYERS + 1))
    }, period)
    return () => clearInterval(id)
  }, [playing, speed])

  useEffect(() => () => { document.body.style.cursor = 'auto' }, [])

  // Compute the live (interpolated) frames for rendering.
  const liveFrames = useMemo(() => {
    const lo = Math.floor(liveLayer)
    const hi = Math.min(N_LAYERS, lo + 1)
    const t = liveLayer - lo
    if (lo === hi) return allLayers[lo]
    return lerpFrameStates(allLayers[lo], allLayers[hi], t)
  }, [liveLayer, allLayers])

  // Quaternion / matrix readout for the panel — uses the discrete layerIdx,
  // not the live interpolation, so numbers don't jitter.
  const readout = useMemo(() => {
    if (layerIdx === 0) {
      // No predicted delta at the black-hole init step.
      return null
    }
    const prev = allLayers[layerIdx - 1][selectedI].quat
    const curr = allLayers[layerIdx][selectedI].quat
    const raw = predictedQuatRaw(prev, curr, layerIdx * 1009 + selectedI * 37)
    // Build rotation matrix from the *normalised* (1,b,c,d) → equivalent to
    // the per-layer delta quaternion. We display R_i as that 3×3.
    const norm = Math.sqrt(1 + raw.b * raw.b + raw.c * raw.c + raw.d * raw.d)
    const w = 1 / norm, x = raw.b / norm, y = raw.c / norm, z = raw.d / norm
    const R = [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    // Delta translation: position(layerIdx) − position(layerIdx-1).
    const dPos = allLayers[layerIdx][selectedI].pos.clone()
      .sub(allLayers[layerIdx - 1][selectedI].pos)
    return { b: raw.b, c: raw.c, d: raw.d, R, dt: dPos }
  }, [layerIdx, selectedI, allLayers])

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
            Backbone Refinement (3D)
          </h1>
          <div style={{ fontSize: 11, color: '#667788', marginTop: 2 }}>
            Algorithm 20 (outer loop) · Algorithm 23 (BackboneUpdate) · 8 shared-weight layers
          </div>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
          <button onClick={() => { setLayerIdx(l => Math.max(0, l - 1)); setPlaying(false) }} style={pillBtn}>◀ Step</button>
          <button
            onClick={() => setPlaying(p => !p)}
            style={{
              ...pillBtn, minWidth: 70,
              background: playing ? 'rgba(239,83,80,0.15)' : 'rgba(102,187,106,0.15)',
              borderColor: playing ? '#ef5350' : '#66bb6a',
              color: playing ? '#ef9a9a' : '#a5d6a7',
            }}>
            {playing ? '⏸ Pause' : '▶ Play'}
          </button>
          <button onClick={() => { setLayerIdx(l => Math.min(N_LAYERS, l + 1)); setPlaying(false) }} style={pillBtn}>Step ▶</button>
          <button onClick={() => { setLayerIdx(0); setPlaying(true) }} style={{ ...pillBtn, fontSize: 11 }}>⟲ Restart</button>
          <span style={{ fontSize: 11, color: '#667788', marginLeft: 8 }}>Speed</span>
          <input
            type="range" min={0.25} max={3} step={0.25} value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            style={{ width: 90, accentColor: '#42a5f5' }}
          />
          <span style={{ fontSize: 11, color: '#c0c8d0', fontFamily: 'JetBrains Mono, monospace', minWidth: 36 }}>
            {speed.toFixed(2)}×
          </span>
        </div>
      </div>

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [2.2, 1.3, 2.6], fov: 48, near: 0.05, far: 60 }}
        style={{ width: '100%', height: '100%' }}
        gl={{ antialias: true, alpha: false }}
      >
        <color attach="background" args={['#070712']} />
        <fog attach="fog" args={['#070712', 6, 14]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[4, 5, 3]} intensity={0.85} color="#ffffff" />
        <directionalLight position={[-3, -1, -2]} intensity={0.25} color="#88aaff" />

        <Scene
          liveFrames={liveFrames}
          allLayers={allLayers}
          currentLayer={liveLayer}
          selectedI={selectedI}
          setSelectedI={setSelectedI}
          showTriads={showTriads}
          showTrails={showTrails}
          showBlackHole={showBlackHole}
        />

        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={0.6}
          minDistance={1.0}
          maxDistance={8}
          target={[0, 0, 0]}
        />

        <EffectComposer>
          <Bloom intensity={0.55} luminanceThreshold={0.55} luminanceSmoothing={0.4} mipmapBlur />
          <Vignette eskil={false} offset={0.15} darkness={0.7} />
        </EffectComposer>
      </Canvas>

      {/* Left explainer panel */}
      <div style={leftPanel}>
        <div style={badge('rgba(102,187,106,0.12)', '#66bb6a', '#a5d6a7')}>
          Layer l = {layerIdx} / {N_LAYERS}
        </div>

        <h2 style={panelH2}>What's happening</h2>
        <p style={panelP}>{NARRATION[layerIdx]}</p>

        <h3 style={panelH3}>Composition</h3>
        <p style={panelP}>
          Each layer composes the network's predicted local update onto the existing frame:
        </p>
        <div style={{ margin: '6px 0 10px', padding: 10, background: 'rgba(66,165,245,0.06)', borderRadius: 6, border: '1px solid rgba(66,165,245,0.22)' }}>
          <KaTeXFormula
            formula={'T_i^{(l+1)} = T_i^{(l)} \\circ (R_i,\\, \\vec{\\Delta t}_i)'}
            style={{ fontSize: 11, padding: 0, background: 'transparent', borderLeft: 'none' }}
          />
        </div>

        <h3 style={panelH3}>Quaternion → rotation</h3>
        <p style={panelP}>
          The network emits three scalars (b, c, d); the rotation comes from forcing the
          first quaternion component to 1 and re-normalising:
        </p>
        <div style={{ margin: '6px 0 10px', padding: 10, background: 'rgba(186,104,200,0.06)', borderRadius: 6, border: '1px solid rgba(186,104,200,0.22)' }}>
          <KaTeXFormula
            formula={'(a,b,c,d) = (1,b,c,d)/\\sqrt{1+b^2+c^2+d^2}'}
            style={{ fontSize: 11, padding: 0, background: 'transparent', borderLeft: 'none' }}
          />
        </div>
        <p style={{ ...panelP, fontSize: 10.5, color: '#667788' }}>
          Because the unnormalised w = 1 &gt; 0, the resulting rotation always lies on the same
          hemisphere — no double-cover ambiguity. This is the trick from Algorithm 23.
        </p>

        <div style={{ margin: '10px 0', padding: 10, background: 'rgba(239,83,80,0.06)', borderRadius: 6, border: '1px solid rgba(239,83,80,0.22)' }}>
          <div style={{ fontSize: 11, color: '#ef9a9a', fontWeight: 600, marginBottom: 4 }}>
            Stop-gradient between iterations (Alg 20 line 20)
          </div>
          <p style={{ ...panelP, margin: 0, fontSize: 10.5 }}>
            Between layers the frame is updated as
            T<sub>i</sub> ← (stopgrad(R<sub>i</sub>), Δt<sub>i</sub>).
            Gradients <b style={{ color: '#ef9a9a' }}>DO flow</b> on translations
            but <b style={{ color: '#ef9a9a' }}>NOT</b> on rotations. This was critical for
            training stability — full backprop through 8 cascaded rotations made the
            loss blow up.
          </p>
        </div>

        <div style={legendBox}>
          <LegendRow color="#4fc3f7" label="Cα backbone atom (click to select)" />
          <LegendRow color="#ff9100" label="selected residue i" />
          <LegendRow color="#ef5350" label="frame e₁ (rotated by R_i)" />
          <LegendRow color="#66bb6a" label="frame e₂" />
          <LegendRow color="#42a5f5" label="frame e₃ = e₁ × e₂" />
          <LegendRow color="#ffd180" label="selected residue's position trail" />
          <LegendRow color="#ffd54f" label="black-hole origin (layer 0)" />
        </div>
      </div>

      {/* Right controls panel */}
      <div style={rightPanel}>
        <h2 style={panelH2}>Controls</h2>

        <SliderRow
          label={`Layer l = ${layerIdx}`}
          value={layerIdx}
          min={0}
          max={N_LAYERS}
          step={1}
          onChange={(v) => { setLayerIdx(v); setPlaying(false) }}
        />

        <SliderRow
          label={`Selected residue i = ${selectedI}`}
          value={selectedI}
          min={0}
          max={N_RES - 1}
          step={1}
          onChange={setSelectedI}
        />

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '12px 0' }} />

        <ToggleRow label="Show frame triads"      value={showTriads}    onChange={setShowTriads} />
        <ToggleRow label="Show position trails"   value={showTrails}    onChange={setShowTrails} />
        <ToggleRow label="Show black-hole guide"  value={showBlackHole} onChange={setShowBlackHole} />

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '12px 0' }} />

        <h3 style={panelH3}>Predicted (1, b, c, d) — residue {selectedI}, layer {layerIdx}</h3>
        {readout ? (
          <>
            <QuatHistogram b={readout.b} c={readout.c} d={readout.d} />
            <div style={{ marginTop: 6, fontSize: 10, color: '#667788', fontFamily: 'JetBrains Mono, monospace' }}>
              raw scalars before normalisation
            </div>

            <h3 style={panelH3}>R_i (rotation matrix)</h3>
            <MatrixReadout rows={readout.R} />

            <h3 style={panelH3}>Δt_i (translation)</h3>
            <MatrixReadout rows={[[readout.dt.x], [readout.dt.y], [readout.dt.z]]} />
          </>
        ) : (
          <div style={{ fontSize: 11, color: '#667788', fontStyle: 'italic', padding: '8px 0' }}>
            Layer 0 is the black-hole init — there is no predicted update yet. Step forward to
            see (b, c, d) and R<sub>i</sub>.
          </div>
        )}

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '12px 0' }} />

        <div style={tipBox}>
          <b style={{ color: '#42a5f5' }}>Tip:</b> drag to orbit · scroll to zoom · click any
          Cα to inspect its quaternion update. The same 8-layer stack shares weights — only
          the frames change between layers.
        </div>
      </div>
    </div>
  )
}

// ── UI sub-bits ─────────────────────────────────────────────────────────────

function QuatHistogram({ b, c, d }: { b: number; c: number; d: number }) {
  const maxAbs = Math.max(0.5, Math.abs(b), Math.abs(c), Math.abs(d))
  const bars: { label: string; v: number; color: string }[] = [
    { label: '1', v: 1, color: '#9ca3af' },
    { label: 'b', v: b, color: '#ef5350' },
    { label: 'c', v: c, color: '#66bb6a' },
    { label: 'd', v: d, color: '#42a5f5' },
  ]
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 70, marginTop: 6 }}>
      {bars.map(b2 => {
        // Map onto a centered axis. "1" gets full height; b/c/d are signed.
        const isUnit = b2.label === '1'
        const max = isUnit ? 1 : maxAbs
        const pct = Math.abs(b2.v) / max
        const height = Math.max(2, pct * 50)
        const above = b2.v >= 0
        return (
          <div key={b2.label} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', width: '100%', alignItems: 'center', position: 'relative' }}>
              {above ? (
                <div style={{ width: '70%', height, background: b2.color, opacity: 0.9, borderRadius: 2, boxShadow: `0 0 6px ${b2.color}66` }} />
              ) : (
                <div style={{ width: '70%', height, background: b2.color, opacity: 0.45, borderRadius: 2, marginTop: 4 }} />
              )}
            </div>
            <div style={{ fontSize: 10, color: '#c0c8d0', fontFamily: 'JetBrains Mono, monospace', marginTop: 2 }}>
              {b2.label}
            </div>
            <div style={{ fontSize: 9, color: '#667788', fontFamily: 'JetBrains Mono, monospace' }}>
              {b2.v.toFixed(2)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function MatrixReadout({ rows }: { rows: number[][] }) {
  return (
    <div style={{
      display: 'inline-block',
      padding: 8,
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: 6,
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: 11,
      color: '#c0c8d0',
      lineHeight: 1.55,
    }}>
      <table style={{ borderCollapse: 'collapse' }}>
        <tbody>
          {rows.map((row, i) => (
            <tr key={`r-${i}`}>
              {row.map((v, j) => (
                <td key={`c-${j}`} style={{ padding: '2px 8px', textAlign: 'right', minWidth: 56 }}>
                  {v.toFixed(3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function LegendRow({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
      <div style={{ width: 14, height: 14, borderRadius: 3, background: color, boxShadow: `0 0 6px ${color}88` }} />
      <span style={{ fontSize: 10.5, color: '#8899aa' }}>{label}</span>
    </div>
  )
}

function SliderRow({
  label, value, min, max, step, onChange,
}: {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void
}) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 11, color: '#c0c8d0', fontWeight: 600, marginBottom: 4 }}>{label}</div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: '#42a5f5' }}
      />
    </div>
  )
}

function ToggleRow({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <label style={{
      display: 'flex', alignItems: 'center', gap: 8,
      fontSize: 11, color: '#c0c8d0', marginBottom: 6, cursor: 'pointer',
    }}>
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

const panelBase: React.CSSProperties = {
  position: 'absolute', top: 80, maxHeight: 'calc(100% - 100px)',
  overflowY: 'auto', padding: 16,
  background: 'rgba(15,15,28,0.78)', backdropFilter: 'blur(8px)',
  border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10,
  zIndex: 5, color: '#c0c8d0',
}
const leftPanel: React.CSSProperties = { ...panelBase, left: 16, width: 340 }
const rightPanel: React.CSSProperties = { ...panelBase, right: 16, width: 300 }

const panelH2: React.CSSProperties = {
  margin: '8px 0 6px', fontSize: 13, color: '#c0c8d0', fontWeight: 700,
}

const panelH3: React.CSSProperties = {
  margin: '10px 0 6px', fontSize: 12, color: '#c0c8d0', fontWeight: 700,
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
    fontFamily: 'JetBrains Mono, monospace',
  }
}
