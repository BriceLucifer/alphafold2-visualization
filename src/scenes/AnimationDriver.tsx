import { useFrame } from '@react-three/fiber'
import { useAnimationStore, AnimStep } from '../store/useAnimationStore'
import { ANIM, DATA } from '../config'

/**
 * Drives the animation forward each frame when isPlaying is true.
 * This is a non-rendering component that lives inside <Canvas>.
 */
export function AnimationDriver() {
  const store = useAnimationStore

  useFrame((_, delta) => {
    const s = store.getState()
    if (!s.isPlaying) return

    const speed = s.speed
    const dt = delta * speed

    switch (s.currentStep) {
      case AnimStep.Idle: {
        // Auto-advance to entrance
        store.setState({ currentStep: AnimStep.Entrance, entranceProgress: 0 })
        break
      }

      case AnimStep.Entrance: {
        const next = Math.min(s.entranceProgress + dt / (ANIM.entranceDurationMs / 1000), 1)
        store.setState({ entranceProgress: next })
        if (next >= 1) {
          store.setState({ currentStep: AnimStep.SelectTarget, stepProgress: 0 })
        }
        break
      }

      case AnimStep.SelectTarget: {
        const next = s.stepProgress + dt / (ANIM.stepDurationMs / 1000)
        store.setState({ stepProgress: next })
        if (next >= 1) {
          store.setState({ currentStep: AnimStep.IterateK, stepProgress: 0, currentK: 0 })
        }
        break
      }

      case AnimStep.IterateK: {
        // Show the first k, then advance to converge
        const next = s.stepProgress + dt / (ANIM.stepDurationMs / 1000)
        store.setState({ stepProgress: next })
        if (next >= 1) {
          store.setState({
            currentStep: AnimStep.TriangleConverge,
            stepProgress: 0,
          })
        }
        break
      }

      case AnimStep.TriangleConverge: {
        const next = s.stepProgress + dt / (ANIM.stepDurationMs / 1000)
        store.setState({ stepProgress: next })
        if (next >= 1) {
          store.setState({
            currentStep: AnimStep.SweepAllK,
            stepProgress: 0,
            currentK: 0,
          })
        }
        break
      }

      case AnimStep.SweepAllK: {
        // Iterate k from 0 to N-1
        const sweepSpeed = dt / (ANIM.sweepDelayMs / 1000)
        const nextProgress = s.stepProgress + sweepSpeed
        const nextK = Math.min(
          Math.floor(nextProgress * DATA.numResidues),
          DATA.numResidues - 1
        )
        store.setState({ stepProgress: Math.min(nextProgress, 1), currentK: nextK })
        if (nextProgress >= 1) {
          store.setState({
            currentStep: AnimStep.MatrixRefresh,
            stepProgress: 0,
          })
        }
        break
      }

      case AnimStep.MatrixRefresh: {
        const next = s.stepProgress + dt / (ANIM.stepDurationMs * 2 / 1000)
        store.setState({ stepProgress: Math.min(next, 1) })
        if (next >= 1) {
          // Animation complete — pause
          store.setState({ isPlaying: false })
        }
        break
      }
    }
  })

  return null
}
