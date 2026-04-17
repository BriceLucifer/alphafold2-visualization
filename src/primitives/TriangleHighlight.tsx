import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import { COLORS, LAYOUT } from '../config'

interface TriangleHighlightProps {
  /** Indices (i,j), (i,k), (j,k) in the pair matrix */
  i: number
  j: number
  k: number
  n: number       // matrix size
  visible?: boolean
  position?: [number, number, number]
}

function cellCenter(row: number, col: number, n: number): [number, number, number] {
  const cellSize = LAYOUT.matrixCellSize
  const gap = LAYOUT.matrixGap
  const total = (cellSize + gap) * n
  return [
    col * (cellSize + gap) - total / 2,
    (n - 1 - row) * (cellSize + gap) - total / 2,
    cellSize * 0.2,  // slightly in front
  ]
}

export function TriangleHighlight({ i, j, k, n, visible = true, position = [LAYOUT.matrixX, LAYOUT.matrixY, LAYOUT.matrixZ] }: TriangleHighlightProps) {
  const materialRef = useRef<THREE.LineBasicMaterial>(null)

  const points = useMemo(() => {
    const pIJ = cellCenter(i, j, n)
    const pIK = cellCenter(i, k, n)
    const pJK = cellCenter(j, k, n)
    return [pIJ, pIK, pJK, pIJ] // closed triangle
  }, [i, j, k, n])

  useFrame(({ clock }) => {
    if (materialRef.current) {
      const t = clock.getElapsedTime()
      materialRef.current.opacity = 0.5 + 0.3 * Math.sin(t * 4)
    }
  })

  if (!visible) return null

  return (
    <group position={position} rotation={[LAYOUT.matrixTiltX, LAYOUT.matrixTiltY, 0]}>
      <Line
        points={points}
        color={COLORS.triangleEdge}
        lineWidth={3}
        transparent
        opacity={0.8}
      />
      {/* Glow line (wider, dimmer) */}
      <Line
        points={points}
        color={COLORS.triangleEdge}
        lineWidth={8}
        transparent
        opacity={0.15}
      />
    </group>
  )
}
