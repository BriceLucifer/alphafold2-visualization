// ── Colors ──────────────────────────────────────────────
export const COLORS = {
  bg: '#0a0a15',
  // Diverging colormap for pair matrix: cyan ↔ white ↔ magenta
  matrixLow: '#00bcd4',   // cyan
  matrixMid: '#ffffff',    // white
  matrixHigh: '#e040fb',   // magenta
  highlight: '#fffde7',    // warm white glow
  residueDefault: '#4fc3f7',
  residueHighlight: '#ffeb3b',
  chainBond: '#555570',
  triangleEdge: '#ff9100',
  particleFlow: '#ffe082',
  panelBg: 'rgba(10, 10, 21, 0.85)',
  panelBorder: 'rgba(255, 255, 255, 0.08)',
}

// ── Layout ──────────────────────────────────────────────
export const LAYOUT = {
  chainX: -12,         // residue chain x position
  chainY: 0,
  chainZ: 0,
  residueSpacing: 0.9,
  residueRadius: 0.3,

  matrixX: 2,          // pair matrix center
  matrixY: 0,
  matrixZ: 0,
  matrixCellSize: 0.4,
  matrixGap: 0.05,
  matrixTiltX: -0.3,   // slight tilt toward camera
  matrixTiltY: 0.15,
}

// ── Camera ──────────────────────────────────────────────
export const CAMERA = {
  position: [5, 6, 18] as [number, number, number],
  fov: 50,
  near: 0.1,
  far: 200,
}

// ── Animation ───────────────────────────────────────────
export const ANIM = {
  stepDurationMs: 2000,
  particleSpeed: 0.02,
  particleCount: 30,
  pulseFrequency: 2.0,
  entranceDurationMs: 1500,
  sweepDelayMs: 50,     // delay between k iterations
  autoPlayInterval: 3000,
  bloomIntensity: 0.6,
  bloomThreshold: 0.8,
}

// ── Data ────────────────────────────────────────────────
export const DATA = {
  numResidues: 20,
  // Demo triangle attention indices
  targetI: 5,
  targetJ: 12,
}
