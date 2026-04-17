import { DATA } from '../config'

const AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

/** Seeded-ish random for reproducibility within a session */
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

const rand = seededRandom(42)

/** Generate a random amino acid sequence of length n */
export function generateSequence(n: number): string[] {
  return Array.from({ length: n }, () =>
    AMINO_ACIDS[Math.floor(rand() * AMINO_ACIDS.length)]
  )
}

/**
 * Generate a synthetic pair representation matrix (N×N).
 * Uses sine waves + noise to look plausible.
 * Values in [-1, 1].
 */
export function generatePairMatrix(n: number): number[][] {
  const matrix: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < n; j++) {
      const dist = Math.abs(i - j)
      // Sequence-distance-based signal + periodic pattern + noise
      const signal =
        0.5 * Math.exp(-dist / 5) * Math.cos((dist * Math.PI) / 4) +
        0.3 * Math.sin((i * 0.7 + j * 0.5)) +
        0.2 * (rand() * 2 - 1)
      row.push(Math.max(-1, Math.min(1, signal)))
    }
    matrix.push(row)
  }
  // Make symmetric
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const avg = (matrix[i][j] + matrix[j][i]) / 2
      matrix[i][j] = avg
      matrix[j][i] = avg
    }
  }
  return matrix
}

/**
 * Simulate the "updated" pair matrix after triangle attention.
 * Slightly shifts values toward nearby cells (smoothing effect).
 */
export function generateUpdatedMatrix(original: number[][]): number[][] {
  const n = original.length
  const updated = original.map(row => [...row])
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0
      let count = 0
      for (let k = 0; k < n; k++) {
        sum += original[i][k] * 0.3 + original[k][j] * 0.3
        count++
      }
      updated[i][j] = Math.max(-1, Math.min(1,
        original[i][j] * 0.6 + (sum / count) * 0.4 + (rand() * 0.1 - 0.05)
      ))
    }
  }
  return updated
}

// ── Pre-generated data ──────────────────────────────────
export const sequence = generateSequence(DATA.numResidues)
export const pairMatrix = generatePairMatrix(DATA.numResidues)
export const updatedPairMatrix = generateUpdatedMatrix(pairMatrix)
