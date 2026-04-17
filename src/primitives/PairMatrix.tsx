import { useRef, useMemo, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { COLORS, LAYOUT } from '../config'

interface PairMatrixProps {
  matrix: number[][]
  highlightCell?: [number, number] | null
  highlightedCells?: Set<string>  // "i,j" strings
  updatingCells?: Set<string>     // cells currently being updated (glow)
  opacity?: number
  position?: [number, number, number]
}

function lerpColor(a: THREE.Color, b: THREE.Color, t: number): THREE.Color {
  return new THREE.Color().copy(a).lerp(b, t)
}

/** Map value in [-1,1] to diverging colormap */
function valueToColor(v: number): THREE.Color {
  const low = new THREE.Color(COLORS.matrixLow)
  const mid = new THREE.Color(COLORS.matrixMid)
  const high = new THREE.Color(COLORS.matrixHigh)
  if (v <= 0) {
    return lerpColor(low, mid, v + 1)
  } else {
    return lerpColor(mid, high, v)
  }
}

const _tempObj = new THREE.Object3D()
const _tempColor = new THREE.Color()

export function PairMatrix({
  matrix,
  highlightCell = null,
  highlightedCells = new Set(),
  updatingCells = new Set(),
  opacity = 1,
  position = [LAYOUT.matrixX, LAYOUT.matrixY, LAYOUT.matrixZ],
}: PairMatrixProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const n = matrix.length
  const count = n * n
  const cellSize = LAYOUT.matrixCellSize
  const gap = LAYOUT.matrixGap
  const totalSize = (cellSize + gap) * n

  // Pre-compute base colors
  const baseColors = useMemo(() => {
    return matrix.flatMap((row) =>
      row.map((val) => valueToColor(val))
    )
  }, [matrix])

  // Set initial transforms
  useEffect(() => {
    if (!meshRef.current) return
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const idx = i * n + j
        _tempObj.position.set(
          j * (cellSize + gap) - totalSize / 2,
          (n - 1 - i) * (cellSize + gap) - totalSize / 2,
          0
        )
        _tempObj.scale.setScalar(1)
        _tempObj.updateMatrix()
        meshRef.current.setMatrixAt(idx, _tempObj.matrix)
        meshRef.current.setColorAt(idx, baseColors[idx])
      }
    }
    meshRef.current.instanceMatrix.needsUpdate = true
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true
  }, [matrix, n, cellSize, gap, totalSize, baseColors])

  // Animate highlights
  useFrame(({ clock }) => {
    if (!meshRef.current) return
    const t = clock.getElapsedTime()

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const idx = i * n + j
        const key = `${i},${j}`
        const isHighlight = highlightCell && highlightCell[0] === i && highlightCell[1] === j
        const isInSet = highlightedCells.has(key)
        const isUpdating = updatingCells.has(key)

        // Scale animation for highlighted cells
        meshRef.current.getMatrixAt(idx, _tempObj.matrix)
        _tempObj.matrix.decompose(_tempObj.position, _tempObj.quaternion, _tempObj.scale)

        let targetScale = 1
        if (isHighlight) {
          targetScale = 1.3 + 0.15 * Math.sin(t * 5)
        } else if (isInSet) {
          targetScale = 1.15
        }
        _tempObj.scale.setScalar(THREE.MathUtils.lerp(_tempObj.scale.x, targetScale, 0.1))
        _tempObj.updateMatrix()
        meshRef.current.setMatrixAt(idx, _tempObj.matrix)

        // Color
        if (isHighlight) {
          _tempColor.set(COLORS.highlight)
        } else if (isUpdating) {
          const pulse = 0.5 + 0.5 * Math.sin(t * 6)
          _tempColor.copy(baseColors[idx]).lerp(new THREE.Color(COLORS.highlight), pulse * 0.4)
        } else if (isInSet) {
          _tempColor.copy(baseColors[idx]).lerp(new THREE.Color(COLORS.triangleEdge), 0.5)
        } else {
          _tempColor.copy(baseColors[idx])
        }
        meshRef.current.setColorAt(idx, _tempColor)
      }
    }
    meshRef.current.instanceMatrix.needsUpdate = true
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true
  })

  return (
    <group
      position={position}
      rotation={[LAYOUT.matrixTiltX, LAYOUT.matrixTiltY, 0]}
    >
      <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
        <boxGeometry args={[cellSize, cellSize, cellSize * 0.3]} />
        <meshStandardMaterial
          transparent
          opacity={opacity}
          roughness={0.4}
          metalness={0.05}
          toneMapped={false}
        />
      </instancedMesh>
    </group>
  )
}
