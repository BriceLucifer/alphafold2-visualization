import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { COLORS, LAYOUT } from '../config'

interface ResidueProps {
  letter: string
  index: number
  position: [number, number, number]
  highlighted?: boolean
  opacity?: number
}

export function Residue({ letter, index, position, highlighted = false, opacity = 1 }: ResidueProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)

  const baseColor = useMemo(
    () => new THREE.Color(highlighted ? COLORS.residueHighlight : COLORS.residueDefault),
    [highlighted]
  )

  useFrame(({ clock }) => {
    if (!meshRef.current) return
    const t = clock.getElapsedTime()

    if (highlighted) {
      const scale = 1 + 0.15 * Math.sin(t * 4)
      meshRef.current.scale.setScalar(scale)
      if (glowRef.current) {
        glowRef.current.scale.setScalar(scale * 1.8)
        ;(glowRef.current.material as THREE.MeshBasicMaterial).opacity =
          0.15 + 0.1 * Math.sin(t * 4)
      }
    } else {
      meshRef.current.scale.setScalar(1)
    }
  })

  return (
    <group position={position}>
      {/* Glow sphere (behind) */}
      {highlighted && (
        <mesh ref={glowRef}>
          <sphereGeometry args={[LAYOUT.residueRadius * 1.8, 16, 16]} />
          <meshBasicMaterial
            color={COLORS.residueHighlight}
            transparent
            opacity={0.2}
            depthWrite={false}
          />
        </mesh>
      )}

      {/* Main sphere */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[LAYOUT.residueRadius, 24, 24]} />
        <meshStandardMaterial
          color={baseColor}
          emissive={highlighted ? COLORS.residueHighlight : '#000000'}
          emissiveIntensity={highlighted ? 0.5 : 0}
          transparent
          opacity={opacity}
          roughness={0.3}
          metalness={0.1}
        />
      </mesh>

      {/* Label */}
      <Html
        center
        distanceFactor={10}
        style={{
          color: '#fff',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '11px',
          fontWeight: 500,
          userSelect: 'none',
          pointerEvents: 'none',
          opacity: opacity,
          textShadow: '0 0 4px rgba(0,0,0,0.8)',
        }}
        position={[0, LAYOUT.residueRadius + 0.25, 0]}
      >
        <span>{letter}<sub style={{ fontSize: '8px' }}>{index}</sub></span>
      </Html>
    </group>
  )
}
