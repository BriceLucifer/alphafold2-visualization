import { useMemo } from 'react'
import { useAnimationStore, AnimStep } from '../store/useAnimationStore'
import { ResidueChain } from '../primitives/ResidueChain'
import { PairMatrix } from '../primitives/PairMatrix'
import { TriangleHighlight } from '../primitives/TriangleHighlight'
import { AttentionFlow } from '../primitives/AttentionFlow'
import { sequence, pairMatrix, updatedPairMatrix } from '../data/synthetic'
import { DATA } from '../config'

export function TriangleAttentionScene() {
  const { currentStep, currentK, targetI, targetJ, stepProgress, entranceProgress } =
    useAnimationStore()

  const n = DATA.numResidues

  // Which residues to highlight on the chain
  const highlightedResidues = useMemo(() => {
    const set = new Set<number>()
    if (currentStep >= AnimStep.SelectTarget) {
      set.add(targetI)
      set.add(targetJ)
    }
    if (
      currentStep === AnimStep.IterateK ||
      currentStep === AnimStep.TriangleConverge ||
      currentStep === AnimStep.SweepAllK
    ) {
      set.add(currentK)
    }
    return set
  }, [currentStep, targetI, targetJ, currentK])

  // Which matrix cells to highlight
  const highlightedCells = useMemo(() => {
    const set = new Set<string>()
    if (
      currentStep === AnimStep.IterateK ||
      currentStep === AnimStep.TriangleConverge ||
      currentStep === AnimStep.SweepAllK
    ) {
      set.add(`${targetI},${currentK}`)
      set.add(`${targetJ},${currentK}`)
    }
    return set
  }, [currentStep, targetI, targetJ, currentK])

  // Cells being updated (for glow effect)
  const updatingCells = useMemo(() => {
    const set = new Set<string>()
    if (currentStep === AnimStep.TriangleConverge || currentStep === AnimStep.SweepAllK) {
      set.add(`${targetI},${targetJ}`)
    }
    if (currentStep === AnimStep.MatrixRefresh) {
      // Progressive wave — highlight cells based on stepProgress
      const totalCells = n * n
      const numToUpdate = Math.floor(stepProgress * totalCells)
      for (let idx = 0; idx < numToUpdate; idx++) {
        const i = Math.floor(idx / n)
        const j = idx % n
        set.add(`${i},${j}`)
      }
    }
    return set
  }, [currentStep, targetI, targetJ, stepProgress, n])

  // Choose matrix data: original or updated
  const displayMatrix = currentStep >= AnimStep.MatrixRefresh && stepProgress > 0.5
    ? updatedPairMatrix
    : pairMatrix

  // Primary highlight cell
  const highlightCell: [number, number] | null =
    currentStep >= AnimStep.SelectTarget ? [targetI, targetJ] : null

  // Show entrance based on progress
  const opacity = currentStep === AnimStep.Idle
    ? 0
    : currentStep === AnimStep.Entrance
      ? entranceProgress
      : 1

  const showTriangle =
    currentStep === AnimStep.IterateK ||
    currentStep === AnimStep.TriangleConverge ||
    currentStep === AnimStep.SweepAllK

  const showFlow =
    currentStep === AnimStep.TriangleConverge ||
    currentStep === AnimStep.SweepAllK

  return (
    <group>
      <ResidueChain
        sequence={sequence}
        highlightedIndices={highlightedResidues}
        opacity={opacity}
      />

      <PairMatrix
        matrix={displayMatrix}
        highlightCell={highlightCell}
        highlightedCells={highlightedCells}
        updatingCells={updatingCells}
        opacity={opacity}
      />

      {showTriangle && (
        <TriangleHighlight
          i={targetI}
          j={targetJ}
          k={currentK}
          n={n}
        />
      )}

      {showFlow && (
        <>
          <AttentionFlow
            from={[targetI, currentK]}
            to={[targetI, targetJ]}
            n={n}
          />
          <AttentionFlow
            from={[targetJ, currentK]}
            to={[targetI, targetJ]}
            n={n}
          />
        </>
      )}
    </group>
  )
}
