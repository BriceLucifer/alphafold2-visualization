import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { COLORS, LAYOUT, ANIM } from '../config'

interface AttentionFlowProps {
  /** Source cell (row, col) */
  from: [number, number]
  /** Target cell (row, col) */
  to: [number, number]
  n: number
  active?: boolean
  position?: [number, number, number]
}

function cellCenter(row: number, col: number, n: number): THREE.Vector3 {
  const cellSize = LAYOUT.matrixCellSize
  const gap = LAYOUT.matrixGap
  const total = (cellSize + gap) * n
  return new THREE.Vector3(
    col * (cellSize + gap) - total / 2,
    (n - 1 - row) * (cellSize + gap) - total / 2,
    cellSize * 0.25
  )
}

export function AttentionFlow({ from, to, n, active = true, position = [LAYOUT.matrixX, LAYOUT.matrixY, LAYOUT.matrixZ] }: AttentionFlowProps) {
  const particlesRef = useRef<THREE.Points>(null)

  const { positions, lifetimes } = useMemo(() => {
    const count = ANIM.particleCount
    const fromPos = cellCenter(from[0], from[1], n)
    const toPos = cellCenter(to[0], to[1], n)
    const dir = new THREE.Vector3().subVectors(toPos, fromPos)

    const pos = new Float32Array(count * 3)
    const life = new Float32Array(count)

    for (let i = 0; i < count; i++) {
      // Spread particles along the path
      const t = i / count
      pos[i * 3] = fromPos.x + dir.x * t + (Math.random() - 0.5) * 0.1
      pos[i * 3 + 1] = fromPos.y + dir.y * t + (Math.random() - 0.5) * 0.1
      pos[i * 3 + 2] = fromPos.z + dir.z * t + (Math.random() - 0.5) * 0.05
      life[i] = t
    }

    return { positions: pos, lifetimes: life }
  }, [from, to, n])

  useFrame((_, delta) => {
    if (!particlesRef.current || !active) return
    const geo = particlesRef.current.geometry
    const posAttr = geo.getAttribute('position') as THREE.BufferAttribute
    const fromPos = cellCenter(from[0], from[1], n)
    const toPos = cellCenter(to[0], to[1], n)
    const dir = new THREE.Vector3().subVectors(toPos, fromPos)
    for (let i = 0; i < posAttr.count; i++) {
      lifetimes[i] += delta * 0.5
      if (lifetimes[i] > 1) lifetimes[i] = 0

      const t = lifetimes[i]
      posAttr.setXYZ(
        i,
        fromPos.x + dir.x * t + (Math.random() - 0.5) * 0.05,
        fromPos.y + dir.y * t + (Math.random() - 0.5) * 0.05,
        fromPos.z + dir.z * t * 0.5 + Math.sin(t * Math.PI) * 0.3
      )
    }
    posAttr.needsUpdate = true
  })

  if (!active) return null

  return (
    <group position={position} rotation={[LAYOUT.matrixTiltX, LAYOUT.matrixTiltY, 0]}>
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
            count={ANIM.particleCount}
          />
        </bufferGeometry>
        <pointsMaterial
          color={COLORS.particleFlow}
          size={0.08}
          transparent
          opacity={0.8}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          sizeAttenuation
        />
      </points>
    </group>
  )
}
