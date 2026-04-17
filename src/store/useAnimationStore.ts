import { create } from 'zustand'
import { DATA } from '../config'

export const AnimStep = {
  Idle: 0,
  Entrance: 1,
  SelectTarget: 2,
  IterateK: 3,
  TriangleConverge: 4,
  SweepAllK: 5,
  MatrixRefresh: 6,
} as const

export type AnimStep = (typeof AnimStep)[keyof typeof AnimStep]

interface AnimationState {
  currentStep: AnimStep
  isPlaying: boolean
  speed: number          // multiplier, 1 = normal
  currentK: number       // which k residue we're highlighting
  targetI: number
  targetJ: number
  stepProgress: number   // 0..1 within the current step
  entranceProgress: number // 0..1 for entrance animation

  // Actions
  play: () => void
  pause: () => void
  togglePlay: () => void
  nextStep: () => void
  prevStep: () => void
  setStep: (step: AnimStep) => void
  setSpeed: (speed: number) => void
  setCurrentK: (k: number) => void
  setStepProgress: (p: number) => void
  setEntranceProgress: (p: number) => void
  reset: () => void
}

export const useAnimationStore = create<AnimationState>((set) => ({
  currentStep: AnimStep.Idle,
  isPlaying: false,
  speed: 1,
  currentK: 0,
  targetI: DATA.targetI,
  targetJ: DATA.targetJ,
  stepProgress: 0,
  entranceProgress: 0,

  play: () => set({ isPlaying: true }),
  pause: () => set({ isPlaying: false }),
  togglePlay: () => set((s) => ({ isPlaying: !s.isPlaying })),

  nextStep: () =>
    set((s) => {
      const next = Math.min(s.currentStep + 1, AnimStep.MatrixRefresh)
      return { currentStep: next as AnimStep, stepProgress: 0, currentK: 0 }
    }),

  prevStep: () =>
    set((s) => {
      const prev = Math.max(s.currentStep - 1, AnimStep.Idle)
      return { currentStep: prev as AnimStep, stepProgress: 0, currentK: 0 }
    }),

  setStep: (step) => set({ currentStep: step, stepProgress: 0, currentK: 0 }),
  setSpeed: (speed) => set({ speed }),
  setCurrentK: (k) => set({ currentK: k }),
  setStepProgress: (p) => set({ stepProgress: p }),
  setEntranceProgress: (p) => set({ entranceProgress: p }),

  reset: () =>
    set({
      currentStep: AnimStep.Idle,
      isPlaying: false,
      stepProgress: 0,
      currentK: 0,
      entranceProgress: 0,
    }),
}))
