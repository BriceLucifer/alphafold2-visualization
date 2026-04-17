import { COLORS } from '../config'

export function TitleBar() {
  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        padding: '16px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        pointerEvents: 'none',
        zIndex: 10,
      }}
    >
      <h1
        style={{
          fontFamily: "'Inter', sans-serif",
          fontWeight: 600,
          fontSize: '16px',
          color: '#e0e0e0',
          letterSpacing: '0.5px',
        }}
      >
        AlphaFold2 Visually{' '}
        <span style={{ color: COLORS.matrixLow, fontWeight: 400 }}>
          — Triangle Attention
        </span>
      </h1>
    </div>
  )
}
