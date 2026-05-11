import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, CatmullRomLine, Line, Html, Grid } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { KaTeXFormula } from '../ui/KaTeXFormula'

// ── Constants ────────────────────────────────────────────────────────────────

const N_CYCLES = 4
const N_RES = 12
// Algorithm 32: 15 distance bins between 3.25 Å and 20.75 Å
const DGRAM_MIN = 3.25
const DGRAM_MAX = 20.75
const DGRAM_BINS = 15

// Stations laid out on an arc (semicircle facing the camera).
const RING_RADIUS = 4.6
function stationCenter(l: number): THREE.Vector3 {
  // l = 0..3 → angles spread along an arc behind the camera focal point
  const t = l / (N_CYCLES - 1) // 0..1
  const theta = Math.PI * (0.15 + t * 0.7) // arc from ~27° to ~153°
  const x = Math.cos(theta) * RING_RADIUS
  const z = -Math.sin(theta) * RING_RADIUS + 1.5
  return new THREE.Vector3(x, 0, z)
}

// ── Synthetic chains: cycle 1 is crumpled, cycle 4 is a clean helix ──────────

function buildTargetFold(): THREE.Vector3[] {
  // Target: short alpha-helix bending into a loop.
  const out: THREE.Vector3[] = []
  const radius = 0.28
  const rise = 0.16
  const tpr = (2 * Math.PI) / 3.6 // 3.6 residues per turn
  for (let i = 0; i < N_RES; i++) {
    const t = i * tpr
    const bend = Math.sin((i / (N_RES - 1)) * Math.PI) * 0.18
    out.push(new THREE.Vector3(
      Math.cos(t) * radius + bend,
      i * rise - (N_RES - 1) * rise * 0.5,
      Math.sin(t) * radius,
    ))
  }
  return out
}

// Deterministic pseudo-random walk (seeded by `seed`) for the cycle-1 starting state.
function buildCrumpledChain(seed: number): THREE.Vector3[] {
  const out: THREE.Vector3[] = [new THREE.Vector3(0, -0.9, 0)]
  let prev = out[0]
  // Linear congruential generator for deterministic noise
  let s = seed
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 0xffffffff
  }
  for (let i = 1; i < N_RES; i++) {
    // Roughly 0.38 nm Cα–Cα step length with random direction
    const theta = rnd() * Math.PI * 2
    const phi = (rnd() - 0.5) * Math.PI
    const step = 0.38
    const dx = step * Math.cos(phi) * Math.cos(theta)
    const dy = step * Math.sin(phi)
    const dz = step * Math.cos(phi) * Math.sin(theta)
    const next = new THREE.Vector3(prev.x + dx, prev.y + dy, prev.z + dz)
    out.push(next)
    prev = next
  }
  // Center it
  const com = out.reduce((a, b) => a.clone().add(b), new THREE.Vector3())
  com.multiplyScalar(1 / out.length)
  return out.map(p => p.clone().sub(com))
}

// Interpolate crumpled → target with progress t∈[0,1]. We mix coordinates and
// add small residual jitter that decays as we approach the target.
function buildCycleChain(l: number): THREE.Vector3[] {
  const target = buildTargetFold()
  const crumpled = buildCrumpledChain(42 + l) // each cycle has slightly different starting noise
  // Cycle 1 (l=0): mostly crumpled. Cycle 4 (l=3): clean target.
  const t = l / (N_CYCLES - 1)
  // Sharper acceleration near the end so cycle 4 looks crisp.
  const eased = t * t * (3 - 2 * t)
  const jitter = (1 - eased) * 0.25
  let s = 7 + l * 31
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return (s / 0xffffffff - 0.5) * 2
  }
  return target.map((tp, i) => {
    const cp = crumpled[i]
    const mixed = new THREE.Vector3().lerpVectors(cp, tp, eased)
    mixed.x += rnd() * jitter
    mixed.y += rnd() * jitter
    mixed.z += rnd() * jitter
    return mixed
  })
}

// ── Distogram from a chain: 15 bins of pairwise Cα distance ─────────────────

function chainToDistogram(chain: THREE.Vector3[]): number[][] {
  const N = chain.length
  const out: number[][] = []
  const bw = (DGRAM_MAX - DGRAM_MIN) / DGRAM_BINS
  for (let i = 0; i < N; i++) {
    const row: number[] = []
    for (let j = 0; j < N; j++) {
      // chain coords are in nm; the AF paper bins distances in Å.
      const d = chain[i].distanceTo(chain[j]) * 10
      let bin: number
      if (d < DGRAM_MIN) bin = 0
      else if (d >= DGRAM_MAX) bin = DGRAM_BINS - 1
      else bin = Math.floor((d - DGRAM_MIN) / bw)
      row.push(bin)
    }
    out.push(row)
  }
  return out
}

// ── Color helpers ────────────────────────────────────────────────────────────

// Diverging cyan → magenta colormap (matches existing project palette).
function distBinColor(bin: number): THREE.Color {
  const t = bin / (DGRAM_BINS - 1)
  const c1 = new THREE.Color('#00bcd4')
  const c2 = new THREE.Color('#ffffff')
  const c3 = new THREE.Color('#e040fb')
  if (t < 0.5) return c1.clone().lerp(c2, t * 2)
  return c2.clone().lerp(c3, (t - 0.5) * 2)
}

const STREAM_COLORS = {
  s: '#ffd54f',         // single representation
  z: '#64b5f6',         // pair representation
  distogram: '#ba68c8', // distogram derived from positions
}

// ── 3D Subcomponents ─────────────────────────────────────────────────────────

function BackboneChain({
  chain, active, dim,
}: { chain: THREE.Vector3[]; active: boolean; dim: boolean }) {
  const pts = useMemo(() => chain.map(p => p.toArray() as [number, number, number]), [chain])
  const baseColor = active ? '#4fc3f7' : '#3a5a78'
  const opacity = dim ? 0.45 : 1
  return (
    <>
      <CatmullRomLine
        points={pts}
        color={active ? '#7bd3ff' : '#5e7d99'}
        lineWidth={active ? 3 : 1.6}
        transparent
        opacity={opacity}
        segments={48}
      />
      {chain.map((p, i) => (
        <mesh key={`ca-${i}`} position={p}>
          <sphereGeometry args={[active ? 0.075 : 0.055, 16, 16]} />
          <meshStandardMaterial
            color={baseColor}
            emissive={baseColor}
            emissiveIntensity={active ? 0.6 : 0.18}
            roughness={0.35}
            metalness={0.1}
            transparent
            opacity={opacity}
          />
        </mesh>
      ))}
    </>
  )
}

function PairRepTile({
  position, seed, active,
}: { position: [number, number, number]; seed: number; active: boolean }) {
  // Small floating tile representing z_ij. Synthesised values for clear viz.
  const tileN = 8
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const colors = useMemo(() => {
    const arr: THREE.Color[] = []
    let s = seed
    const rnd = () => {
      s = (s * 1664525 + 1013904223) >>> 0
      return s / 0xffffffff
    }
    for (let i = 0; i < tileN; i++) {
      for (let j = 0; j < tileN; j++) {
        // Smoother in later cycles
        const noise = rnd()
        const smooth = 0.5 + 0.5 * Math.cos((i - j) * 0.6)
        const bin = Math.floor(THREE.MathUtils.clamp(smooth * DGRAM_BINS - 4 + (noise - 0.5) * 4, 0, DGRAM_BINS - 1))
        arr.push(distBinColor(bin))
      }
    }
    return arr
  }, [seed])

  useEffect(() => {
    if (!meshRef.current) return
    const dummy = new THREE.Object3D()
    const cell = 0.075
    const gap = 0.005
    let k = 0
    for (let i = 0; i < tileN; i++) {
      for (let j = 0; j < tileN; j++) {
        dummy.position.set(
          (j - (tileN - 1) / 2) * (cell + gap),
          0,
          (i - (tileN - 1) / 2) * (cell + gap),
        )
        dummy.scale.set(cell, 0.02, cell)
        dummy.updateMatrix()
        meshRef.current!.setMatrixAt(k, dummy.matrix)
        meshRef.current!.setColorAt(k, colors[k])
        k++
      }
    }
    meshRef.current.instanceMatrix.needsUpdate = true
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true
  }, [colors])

  return (
    <group position={position}>
      <instancedMesh ref={meshRef} args={[undefined, undefined, tileN * tileN]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          emissive="#ffffff"
          emissiveIntensity={active ? 0.45 : 0.15}
          roughness={0.55}
          metalness={0.05}
          transparent
          opacity={active ? 1 : 0.55}
          vertexColors
        />
      </instancedMesh>
      <Html position={[0, 0.12, 0]} center distanceFactor={6} style={{ pointerEvents: 'none' }}>
        <div style={{
          fontSize: 8, color: active ? '#c0c8d0' : '#556677', fontFamily: 'JetBrains Mono, monospace',
          textShadow: '0 0 4px #000', whiteSpace: 'nowrap',
        }}>
          z<sub>ij</sub>
        </div>
      </Html>
    </group>
  )
}

function DistogramFeedbackPanel({
  chain, position, visible,
}: { chain: THREE.Vector3[]; position: [number, number, number]; visible: boolean }) {
  // Renders the binned distogram of `chain` as an instanced N×N tile grid.
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const dgram = useMemo(() => chainToDistogram(chain), [chain])
  const N = dgram.length

  useEffect(() => {
    if (!meshRef.current) return
    const dummy = new THREE.Object3D()
    const cell = 0.085
    const gap = 0.005
    let k = 0
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        dummy.position.set(
          (j - (N - 1) / 2) * (cell + gap),
          0,
          (i - (N - 1) / 2) * (cell + gap),
        )
        dummy.scale.set(cell, 0.025, cell)
        dummy.updateMatrix()
        meshRef.current!.setMatrixAt(k, dummy.matrix)
        meshRef.current!.setColorAt(k, distBinColor(dgram[i][j]))
        k++
      }
    }
    meshRef.current.instanceMatrix.needsUpdate = true
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true
  }, [dgram, N])

  if (!visible) return null
  return (
    <group position={position} rotation={[-0.3, 0, 0]}>
      <instancedMesh ref={meshRef} args={[undefined, undefined, N * N]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          emissive="#ffffff"
          emissiveIntensity={0.35}
          roughness={0.45}
          metalness={0.05}
          vertexColors
        />
      </instancedMesh>
      <Html position={[0, 0.25, 0]} center distanceFactor={5} style={{ pointerEvents: 'none' }}>
        <div style={{
          fontSize: 9, color: STREAM_COLORS.distogram, fontFamily: 'JetBrains Mono, monospace',
          textShadow: '0 0 4px #000', whiteSpace: 'nowrap', fontWeight: 700,
        }}>
          distogram d<sub>ij</sub> (15 bins, 3.25–20.75 Å)
        </div>
      </Html>
    </group>
  )
}

function StationRing({ center, active }: { center: THREE.Vector3; active: boolean }) {
  const ringRef = useRef<THREE.Mesh>(null)
  useFrame(({ clock }) => {
    if (!ringRef.current || !active) return
    const t = clock.getElapsedTime()
    const s = 1 + Math.sin(t * 2) * 0.06
    ringRef.current.scale.set(s, 1, s)
  })
  return (
    <mesh ref={ringRef} position={[center.x, -1.05, center.z]} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.9, 1.05, 64]} />
      <meshBasicMaterial
        color={active ? '#ffd54f' : '#445566'}
        transparent
        opacity={active ? 0.9 : 0.25}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

function FeedbackStreamPath({
  from, to, color, particleCount, offset, visible,
}: {
  from: THREE.Vector3
  to: THREE.Vector3
  color: string
  particleCount: number
  offset: number
  visible: boolean
}) {
  // Curved path between consecutive stations, particles flowing.
  const curve = useMemo(() => {
    const mid = new THREE.Vector3().lerpVectors(from, to, 0.5)
    mid.y += 1.4 + offset * 0.4
    return new THREE.QuadraticBezierCurve3(from.clone(), mid, to.clone())
  }, [from, to, offset])

  const pathPts = useMemo(
    () => curve.getPoints(40).map(p => p.toArray() as [number, number, number]),
    [curve],
  )

  const groupRef = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    if (!groupRef.current || !visible) return
    const t = clock.getElapsedTime()
    groupRef.current.children.forEach((child, idx) => {
      const phase = ((t * 0.4 + (idx / particleCount) + offset * 0.2) % 1)
      const pt = curve.getPoint(phase)
      child.position.copy(pt)
    })
  })

  if (!visible) return null

  return (
    <>
      <Line points={pathPts} color={color} lineWidth={1.4} transparent opacity={0.35} dashed dashScale={20} />
      <group ref={groupRef}>
        {Array.from({ length: particleCount }).map((_, i) => (
          <mesh key={`p-${i}`}>
            <sphereGeometry args={[0.04, 10, 10]} />
            <meshBasicMaterial color={color} transparent opacity={0.95} />
          </mesh>
        ))}
      </group>
    </>
  )
}

// ── Scene root ──────────────────────────────────────────────────────────────

function Scene({
  activeCycle, setActiveCycle, showS, showZ, showDist,
}: {
  activeCycle: number
  setActiveCycle: (l: number) => void
  showS: boolean
  showZ: boolean
  showDist: boolean
}) {
  // Pre-compute all four chains
  const chains = useMemo(() => Array.from({ length: N_CYCLES }, (_, l) => buildCycleChain(l)), [])
  const centers = useMemo(() => Array.from({ length: N_CYCLES }, (_, l) => stationCenter(l)), [])

  // Subtle camera drift — gives the still scene some life.
  const groupRef = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    if (!groupRef.current) return
    groupRef.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.05) * 0.04
  })

  return (
    <group ref={groupRef}>
      {/* Stations */}
      {chains.map((chain, l) => {
        const c = centers[l]
        const active = l === activeCycle
        return (
          <group
            key={`station-${l}`}
            position={c}
            onClick={(e) => { e.stopPropagation(); setActiveCycle(l) }}
            onPointerOver={() => { document.body.style.cursor = 'pointer' }}
            onPointerOut={() => { document.body.style.cursor = 'auto' }}
          >
            <BackboneChain chain={chain} active={active} dim={!active} />
            <PairRepTile position={[0, 1.5, 0]} seed={l * 137 + 11} active={active} />
            <StationRing center={new THREE.Vector3(0, 0, 0)} active={active} />
            <Html position={[0, -1.35, 0]} center distanceFactor={7} style={{ pointerEvents: 'none' }}>
              <div style={{
                padding: '3px 10px',
                background: active ? 'rgba(255,213,79,0.18)' : 'rgba(20,20,40,0.55)',
                border: `1px solid ${active ? '#ffd54f' : '#445566'}`,
                borderRadius: 10,
                fontSize: 10,
                color: active ? '#ffe082' : '#8899aa',
                fontFamily: 'JetBrains Mono, monospace',
                whiteSpace: 'nowrap',
                fontWeight: 700,
              }}>
                Cycle l = {l + 1}
              </div>
            </Html>
          </group>
        )
      })}

      {/* Feedback paths: from station l → l+1 */}
      {chains.slice(0, -1).map((_, l) => {
        const from = centers[l].clone()
        const to = centers[l + 1].clone()
        return (
          <group key={`fb-${l}`}>
            <FeedbackStreamPath from={from} to={to} color={STREAM_COLORS.s}         particleCount={5} offset={0}    visible={showS} />
            <FeedbackStreamPath from={from} to={to} color={STREAM_COLORS.z}         particleCount={5} offset={0.6}  visible={showZ} />
            <FeedbackStreamPath from={from} to={to} color={STREAM_COLORS.distogram} particleCount={5} offset={1.2}  visible={showDist} />
          </group>
        )
      })}

      {/* Distogram preview: shows the distogram derived from the PREVIOUS cycle.
          That distogram becomes the input to the active cycle's z_ij update. */}
      {activeCycle > 0 && (
        <DistogramFeedbackPanel
          chain={chains[activeCycle - 1]}
          position={[0, 2.6, 1.5]}
          visible={showDist}
        />
      )}

      {/* Floor */}
      <Grid
        position={[0, -1.1, 0]}
        args={[24, 24]}
        cellColor="#1a2030"
        sectionColor="#2a3550"
        sectionThickness={0.8}
        fadeDistance={20}
        fadeStrength={1.4}
        infiniteGrid
      />
    </group>
  )
}

// ── Top-level component ─────────────────────────────────────────────────────

const NARRATION: Record<number, { incoming: string; produced: string }> = {
  0: {
    incoming: 'Nothing — for l = 1 the recycled inputs ŝ, ẑ, x̂_Cα are initialised to zero (Alg 30 line 1).',
    produced: 'An initial, mostly-extended backbone guess. Pair features are still rough.',
  },
  1: {
    incoming: 'The previous cycle\'s single repr ŝ_i, pair repr ẑ_ij, and predicted Cα positions x̂_Cα (turned into a distogram).',
    produced: 'A more compact intermediate fold. Pair features sharpen as the distogram constrains them.',
  },
  2: {
    incoming: 'Cycle 2\'s outputs (stop-gradient): refined ŝ, ẑ, and the distogram of the previous Cα positions.',
    produced: 'Most secondary structure is now in place. The geometry is converging.',
  },
  3: {
    incoming: 'Cycle 3\'s converged tensors. By now ŝ and ẑ are mostly stable; small refinements remain.',
    produced: 'The final predicted structure with confidence-weighted Cα positions.',
  },
}

export function Recycling3DScene({ onBack }: { onBack: () => void }) {
  const [activeCycle, setActiveCycle] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1.0)
  const [showS, setShowS] = useState(true)
  const [showZ, setShowZ] = useState(true)
  const [showDist, setShowDist] = useState(true)

  // Auto-advance through cycles when playing.
  useEffect(() => {
    if (!playing) return
    const period = 2600 / speed
    const id = setInterval(() => {
      setActiveCycle(c => (c + 1) % N_CYCLES)
    }, period)
    return () => clearInterval(id)
  }, [playing, speed])

  const narration = NARRATION[activeCycle]

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
            Recycling Iterations (3D)
          </h1>
          <div style={{ fontSize: 11, color: '#667788', marginTop: 2 }}>
            Algorithm 30 · Algorithm 32 (Recycling Embedder) · §1.10 · N_cycle = {N_CYCLES}
          </div>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
          <button onClick={() => { setActiveCycle(c => (c - 1 + N_CYCLES) % N_CYCLES); setPlaying(false) }} style={pillBtn}>◀ Step</button>
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
          <button onClick={() => { setActiveCycle(c => (c + 1) % N_CYCLES); setPlaying(false) }} style={pillBtn}>Step ▶</button>
          <button onClick={() => { setActiveCycle(0); setPlaying(true) }} style={{ ...pillBtn, fontSize: 11 }}>⟲ Restart</button>
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
        camera={{ position: [0, 3.2, 8], fov: 48, near: 0.1, far: 60 }}
        style={{ width: '100%', height: '100%' }}
        gl={{ antialias: true, alpha: false }}
      >
        <color attach="background" args={['#070712']} />
        <fog attach="fog" args={['#070712', 14, 28]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[6, 8, 4]} intensity={0.9} color="#ffffff" />
        <directionalLight position={[-5, -2, -3]} intensity={0.25} color="#88aaff" />

        <Scene
          activeCycle={activeCycle}
          setActiveCycle={setActiveCycle}
          showS={showS}
          showZ={showZ}
          showDist={showDist}
        />

        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={0.6}
          minDistance={3.5}
          maxDistance={18}
          target={[0, 0.5, 0]}
        />

        <EffectComposer>
          <Bloom intensity={0.55} luminanceThreshold={0.55} luminanceSmoothing={0.4} mipmapBlur />
          <Vignette eskil={false} offset={0.15} darkness={0.7} />
        </EffectComposer>
      </Canvas>

      {/* Left explainer panel */}
      <div style={leftPanel}>
        <div style={badge('rgba(255,213,79,0.12)', '#ffd54f', '#ffe082')}>
          Cycle l = {activeCycle + 1} / {N_CYCLES}
        </div>

        <h2 style={panelH2}>What the network sees coming in</h2>
        <p style={panelP}>{narration.incoming}</p>

        <h2 style={panelH2}>What it produces</h2>
        <p style={panelP}>{narration.produced}</p>

        <h3 style={panelH3}>Recycling embedder update</h3>
        <p style={panelP}>
          The pair representation absorbs both the previous cycle's <i>ẑ</i> and a fresh
          <b style={{ color: STREAM_COLORS.distogram }}> distogram</b> built from the previous Cα coords:
        </p>
        <div style={{ margin: '8px 0', padding: 10, background: 'rgba(186,104,200,0.06)', borderRadius: 6, border: '1px solid rgba(186,104,200,0.22)' }}>
          <KaTeXFormula
            formula={'z_{ij} \\leftarrow z_{ij} + \\text{Linear}(\\text{LayerNorm}(\\hat{z}_{ij})) + \\text{Linear}(\\text{one\\_hot}(d_{ij}))'}
            style={{ fontSize: 11, padding: 0, background: 'transparent', borderLeft: 'none' }}
          />
        </div>
        <div style={{ margin: '8px 0', padding: 10, background: 'rgba(255,213,79,0.06)', borderRadius: 6, border: '1px solid rgba(255,213,79,0.22)' }}>
          <KaTeXFormula
            formula={'s_i \\leftarrow s_i + \\text{Linear}(\\text{LayerNorm}(\\hat{s}_i))'}
            style={{ fontSize: 11, padding: 0, background: 'transparent', borderLeft: 'none' }}
          />
        </div>
        <p style={{ ...panelP, fontSize: 10.5, color: '#667788' }}>
          where d<sub>ij</sub> = ‖x<sub>Cα,i</sub> − x<sub>Cα,j</sub>‖ binned into 15 bins
          between 3.25 Å and 20.75 Å (Alg 32 line 2).
        </p>

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '14px 0' }} />

        <h3 style={panelH3}>Why recycling works</h3>
        <ul style={{ margin: 0, padding: '0 0 0 18px', fontSize: 10.5, color: '#8899aa', lineHeight: 1.6 }}>
          <li><b style={{ color: '#c0c8d0' }}>Shared weights</b> across all N_cycle iterations — no extra parameters.</li>
          <li><b style={{ color: '#c0c8d0' }}>Training trick:</b> N′ ~ Uniform[1, N_cycle] cycles per example, only the last cycle backprops (stop-gradient on ŝ, ẑ, x̂_Cα). Memory cost stays at one forward pass.</li>
          <li><b style={{ color: '#c0c8d0' }}>Ablation (§1.13):</b> removing recycling significantly drops CASP14 accuracy — the iterative refinement is doing real work.</li>
        </ul>
      </div>

      {/* Right controls / legend panel */}
      <div style={rightPanel}>
        <h2 style={panelH2}>Feedback streams</h2>
        <ToggleRow color={STREAM_COLORS.s}         label="s_i (single repr)"             value={showS}    onChange={setShowS} />
        <ToggleRow color={STREAM_COLORS.z}         label="z_ij (pair repr)"              value={showZ}    onChange={setShowZ} />
        <ToggleRow color={STREAM_COLORS.distogram} label="distogram(x_Cα) → z_ij update" value={showDist} onChange={setShowDist} />

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '14px 0' }} />

        <h3 style={panelH3}>Quick jump</h3>
        <div style={{ display: 'flex', gap: 6 }}>
          {[0, 1, 2, 3].map(l => (
            <button
              key={`jmp-${l}`}
              onClick={() => { setActiveCycle(l); setPlaying(false) }}
              style={{
                flex: 1, padding: '8px 0', fontSize: 12,
                fontFamily: 'JetBrains Mono, monospace',
                background: l === activeCycle ? 'rgba(255,213,79,0.18)' : 'rgba(20,20,40,0.55)',
                border: `1px solid ${l === activeCycle ? '#ffd54f' : 'rgba(255,255,255,0.12)'}`,
                borderRadius: 6,
                color: l === activeCycle ? '#ffe082' : '#c0c8d0',
                cursor: 'pointer',
              }}>
              l={l + 1}
            </button>
          ))}
        </div>

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '14px 0' }} />

        <h3 style={panelH3}>Reading the scene</h3>
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 10.5, color: '#8899aa', lineHeight: 1.6 }}>
          <li>Four <b>stations</b> on an arc — each is one full Evoformer+Structure forward pass.</li>
          <li>The <b>Cα backbone</b> at each station shows that cycle's predicted fold (crumpled → compact).</li>
          <li>Floating tiles above each backbone are that cycle's <b>z_ij</b> (8×8 preview).</li>
          <li>Curved <b>particle streams</b> carry s, z, and the distogram-from-x_Cα forward.</li>
          <li>The <b>distogram panel</b> above the active cycle is the signal feeding Alg 32 line 2.</li>
        </ul>

        <div style={{ marginTop: 12, padding: 10, background: 'rgba(66,165,245,0.06)', borderRadius: 6, border: '1px solid rgba(66,165,245,0.22)', fontSize: 10.5, color: '#8899aa', lineHeight: 1.55 }}>
          <b style={{ color: '#42a5f5' }}>Tip:</b> click any station to jump there ·
          drag to orbit · scroll to zoom · right-drag to pan.
        </div>
      </div>
    </div>
  )
}

// ── UI helpers ───────────────────────────────────────────────────────────────

function ToggleRow({ color, label, value, onChange }: {
  color: string; label: string; value: boolean; onChange: (v: boolean) => void
}) {
  return (
    <label style={{
      display: 'flex', alignItems: 'center', gap: 8,
      fontSize: 11, color: '#c0c8d0', marginBottom: 6, cursor: 'pointer',
    }}>
      <input
        type="checkbox" checked={value}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: color }}
      />
      <div style={{ width: 12, height: 12, borderRadius: 3, background: color, boxShadow: `0 0 6px ${color}aa` }} />
      <span>{label}</span>
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
const rightPanel: React.CSSProperties = { ...panelBase, right: 16, width: 290 }

const panelH2: React.CSSProperties = {
  margin: '8px 0 6px', fontSize: 13, color: '#c0c8d0', fontWeight: 700,
}

const panelH3: React.CSSProperties = {
  margin: '10px 0 6px', fontSize: 12, color: '#c0c8d0', fontWeight: 700,
}

const panelP: React.CSSProperties = {
  margin: '0 0 8px', fontSize: 11.5, color: '#8899aa', lineHeight: 1.6,
}

function badge(bg: string, border: string, color: string): React.CSSProperties {
  return {
    display: 'inline-block', padding: '2px 8px', borderRadius: 10,
    background: bg, border: `1px solid ${border}60`,
    fontSize: 10, color, fontWeight: 600, marginBottom: 8,
    fontFamily: 'JetBrains Mono, monospace',
  }
}
