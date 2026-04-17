import { useAnimationStore } from '../store/useAnimationStore'
import { COLORS } from '../config'

const STEP_NAMES = [
  'Idle',
  'Entrance',
  'Select Target',
  'Iterate k',
  'Converge',
  'Sweep All k',
  'Matrix Refresh',
]

const buttonStyle: React.CSSProperties = {
  background: 'rgba(255,255,255,0.06)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '6px',
  color: '#e0e0e0',
  padding: '6px 14px',
  fontSize: '13px',
  fontFamily: "'Inter', sans-serif",
  cursor: 'pointer',
  transition: 'background 0.15s',
}

export function PlaybackControls() {
  const { currentStep, isPlaying, speed, togglePlay, nextStep, prevStep, setSpeed, reset } =
    useAnimationStore()

  return (
    <div
      style={{
        position: 'absolute',
        bottom: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '10px 20px',
        background: COLORS.panelBg,
        border: `1px solid ${COLORS.panelBorder}`,
        borderRadius: '12px',
        backdropFilter: 'blur(12px)',
        zIndex: 10,
      }}
    >
      <button style={buttonStyle} onClick={reset} title="Reset">
        ⏮
      </button>
      <button style={buttonStyle} onClick={prevStep} title="Previous step">
        ◀
      </button>
      <button
        style={{
          ...buttonStyle,
          background: isPlaying ? 'rgba(0,188,212,0.15)' : 'rgba(255,255,255,0.06)',
          minWidth: '44px',
        }}
        onClick={togglePlay}
        title={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? '⏸' : '▶'}
      </button>
      <button style={buttonStyle} onClick={nextStep} title="Next step">
        ▶
      </button>

      {/* Step indicator */}
      <span
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '12px',
          color: '#888',
          minWidth: '100px',
          textAlign: 'center',
        }}
      >
        {STEP_NAMES[currentStep]} ({currentStep}/6)
      </span>

      {/* Speed control */}
      <span style={{ fontSize: '11px', color: '#666' }}>Speed:</span>
      {[0.5, 1, 2].map((s) => (
        <button
          key={s}
          style={{
            ...buttonStyle,
            padding: '4px 8px',
            fontSize: '11px',
            background: speed === s ? 'rgba(0,188,212,0.2)' : 'rgba(255,255,255,0.04)',
          }}
          onClick={() => setSpeed(s)}
        >
          {s}×
        </button>
      ))}
    </div>
  )
}
