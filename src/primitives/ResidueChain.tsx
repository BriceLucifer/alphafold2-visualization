import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import { COLORS, LAYOUT } from '../config'
import { Residue } from './Residue'

interface ResidueChainProps {
  sequence: string[]
  highlightedIndices?: Set<number>
  opacity?: number
  basePosition?: [number, number, number]
}

export function ResidueChain({
  sequence,
  highlightedIndices = new Set(),
  opacity = 1,
  basePosition = [LAYOUT.chainX, LAYOUT.chainY, LAYOUT.chainZ],
}: ResidueChainProps) {
  const positions = useMemo(() => {
    return sequence.map((_, i) => {
      const x = basePosition[0]
      const y = basePosition[1] + (sequence.length / 2 - i) * LAYOUT.residueSpacing
      const z = basePosition[2]
      return [x, y, z] as [number, number, number]
    })
  }, [sequence, basePosition])

  // Bond line points
  const linePoints = useMemo(() => {
    return positions.map(p => new Float32Array(p)).map(p => [p[0], p[1], p[2]] as [number, number, number])
  }, [positions])

  return (
    <group>
      {/* Backbone bonds */}
      {positions.length > 1 && (
        <Line
          points={linePoints}
          color={COLORS.chainBond}
          lineWidth={1.5}
          transparent
          opacity={opacity * 0.6}
        />
      )}

      {/* Residue spheres */}
      {sequence.map((letter, i) => (
        <Residue
          key={i}
          letter={letter}
          index={i}
          position={positions[i]}
          highlighted={highlightedIndices.has(i)}
          opacity={opacity}
        />
      ))}
    </group>
  )
}
