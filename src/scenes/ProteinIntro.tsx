import { useState } from 'react'

// ── Section data ───────────────────────────────────────

interface Section {
  id: number
  title: string
  content: JSX.Element
}

// ── Visual components ──────────────────────────────────

function AminoAcidChain() {
  const acids = [
    { letter: 'M', name: 'Met', color: '#f9a825' },
    { letter: 'V', name: 'Val', color: '#43a047' },
    { letter: 'L', name: 'Leu', color: '#43a047' },
    { letter: 'S', name: 'Ser', color: '#1e88e5' },
    { letter: 'P', name: 'Pro', color: '#8e24aa' },
    { letter: 'A', name: 'Ala', color: '#757575' },
    { letter: 'D', name: 'Asp', color: '#e53935' },
    { letter: 'K', name: 'Lys', color: '#1565c0' },
    { letter: 'T', name: 'Thr', color: '#1e88e5' },
    { letter: 'N', name: 'Asn', color: '#1e88e5' },
    { letter: 'V', name: 'Val', color: '#43a047' },
    { letter: 'K', name: 'Lys', color: '#1565c0' },
    { letter: 'A', name: 'Ala', color: '#757575' },
    { letter: '...', name: '', color: '#bbb' },
  ]

  return (
    <svg viewBox="0 0 700 80" style={{ width: '100%', maxWidth: 700, height: 'auto' }}>
      {acids.map((a, i) => (
        <g key={i}>
          {i > 0 && (
            <line x1={i * 50 - 10} y1={35} x2={i * 50 + 5} y2={35}
              stroke="#ccc" strokeWidth={2} />
          )}
          <circle cx={i * 50 + 20} cy={35} r={16} fill={a.color} opacity={0.85} />
          <text x={i * 50 + 20} y={40} textAnchor="middle" fontSize={14}
            fill="#fff" fontWeight={700} fontFamily="monospace">{a.letter}</text>
          {a.name && (
            <text x={i * 50 + 20} y={65} textAnchor="middle" fontSize={9}
              fill="#999" fontFamily="Inter, sans-serif">{a.name}</text>
          )}
        </g>
      ))}
    </svg>
  )
}

function FoldingLevels() {
  return (
    <svg viewBox="0 0 800 200" style={{ width: '100%', maxWidth: 800, height: 'auto' }}>
      {/* Primary */}
      <g>
        <text x={90} y={20} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#e65100" fontFamily="Inter, sans-serif">Primary</text>
        <text x={90} y={34} textAnchor="middle" fontSize={10}
          fill="#999" fontFamily="Inter, sans-serif">sequence</text>
        {Array.from({ length: 8 }).map((_, i) => (
          <g key={i}>
            {i > 0 && <line x1={i * 22 + 18} y1={55} x2={i * 22 + 28} y2={55} stroke="#ddd" strokeWidth={1.5} />}
            <circle cx={i * 22 + 32} cy={55} r={7} fill="#e65100" opacity={0.7} />
          </g>
        ))}
      </g>

      {/* Secondary */}
      <g transform="translate(210, 0)">
        <text x={80} y={20} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#1565c0" fontFamily="Inter, sans-serif">Secondary</text>
        <text x={80} y={34} textAnchor="middle" fontSize={10}
          fill="#999" fontFamily="Inter, sans-serif">α-helix, β-sheet</text>
        <path d="M20,50 C35,35 50,65 65,50 C80,35 95,65 110,50 C125,35 140,65 155,50"
          fill="none" stroke="#1565c0" strokeWidth={4} strokeLinecap="round" />
        <text x={80} y={80} textAnchor="middle" fontSize={9} fill="#666" fontFamily="Inter, sans-serif">α-helix</text>
        {/* Beta sheet */}
        <g transform="translate(0, 85)">
          <line x1={20} y1={10} x2={155} y2={10} stroke="#1565c0" strokeWidth={3} />
          <line x1={20} y1={25} x2={155} y2={25} stroke="#1565c0" strokeWidth={3} />
          {[40, 80, 120].map(x => (
            <line key={x} x1={x} y1={12} x2={x} y2={23} stroke="#1565c0" strokeWidth={1} strokeDasharray="2,2" />
          ))}
          <text x={80} y={42} textAnchor="middle" fontSize={9} fill="#666" fontFamily="Inter, sans-serif">β-sheet</text>
        </g>
      </g>

      {/* Tertiary */}
      <g transform="translate(420, 0)">
        <text x={80} y={20} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#2e7d32" fontFamily="Inter, sans-serif">Tertiary</text>
        <text x={80} y={34} textAnchor="middle" fontSize={10}
          fill="#999" fontFamily="Inter, sans-serif">3D fold</text>
        <path d="M30,55 C45,30 60,75 75,50 C90,25 80,80 100,55 C115,35 125,70 140,55"
          fill="none" stroke="#2e7d32" strokeWidth={4} strokeLinecap="round" />
        <path d="M70,50 Q60,90 100,80 Q130,75 120,55"
          fill="none" stroke="#81c784" strokeWidth={2} strokeDasharray="3,3" />
        <circle cx={70} cy={50} r={3} fill="#c62828" />
        <circle cx={120} cy={55} r={3} fill="#c62828" />
        <text x={95} y={100} textAnchor="middle" fontSize={8} fill="#999" fontFamily="Inter, sans-serif">
          disulfide bond
        </text>
      </g>

      {/* Quaternary */}
      <g transform="translate(620, 0)">
        <text x={80} y={20} textAnchor="middle" fontSize={12} fontWeight={700}
          fill="#6a1b9a" fontFamily="Inter, sans-serif">Quaternary</text>
        <text x={80} y={34} textAnchor="middle" fontSize={10}
          fill="#999" fontFamily="Inter, sans-serif">multi-chain</text>
        <ellipse cx={55} cy={70} rx={30} ry={22} fill="#ce93d8" opacity={0.4} stroke="#6a1b9a" strokeWidth={1.5} />
        <ellipse cx={95} cy={65} rx={28} ry={20} fill="#b39ddb" opacity={0.4} stroke="#6a1b9a" strokeWidth={1.5} />
        <ellipse cx={70} cy={90} rx={25} ry={18} fill="#e1bee7" opacity={0.4} stroke="#6a1b9a" strokeWidth={1.5} />
        <ellipse cx={100} cy={88} rx={27} ry={19} fill="#d1c4e9" opacity={0.4} stroke="#6a1b9a" strokeWidth={1.5} />
        <text x={75} y={120} textAnchor="middle" fontSize={8} fill="#999" fontFamily="Inter, sans-serif">
          e.g. hemoglobin (4 subunits)
        </text>
      </g>
    </svg>
  )
}

function LevinthalDiagram() {
  return (
    <svg viewBox="0 0 520 260" style={{ width: '100%', maxWidth: 520, height: 'auto' }}>
      {/* Funnel */}
      <path d="M60,20 L260,230 L460,20" fill="rgba(25,118,210,0.06)" stroke="#1976d2" strokeWidth={1.5} />

      {/* Random dots at the top (huge conformational space) */}
      {Array.from({ length: 40 }).map((_, i) => (
        <circle key={i}
          cx={80 + Math.sin(i * 2.7) * (180 - i * 2) + i * 7}
          cy={25 + Math.random() * 30 + (i % 3) * 10}
          r={2.5} fill="#1976d2" opacity={0.3}
        />
      ))}

      {/* Arrow down the funnel */}
      <path d="M260,50 L260,220" fill="none" stroke="#e65100" strokeWidth={2} strokeDasharray="6,4" />
      <polygon points="260,225 254,213 266,213" fill="#e65100" />

      {/* Native state at bottom */}
      <circle cx={260} cy={240} r={8} fill="#2e7d32" />
      <text x={260} y={244} textAnchor="middle" fontSize={7} fill="#fff" fontWeight={700}>N</text>

      {/* Labels */}
      <text x={260} y={15} textAnchor="middle" fontSize={12} fontWeight={600}
        fill="#1565c0" fontFamily="Inter, sans-serif">~10³⁰⁰ possible conformations</text>
      <text x={260} y={258} textAnchor="middle" fontSize={11} fill="#2e7d32"
        fontWeight={600} fontFamily="Inter, sans-serif">1 native structure</text>

      {/* Time comparison */}
      <g transform="translate(320, 80)">
        <text x={0} y={0} fontSize={11} fill="#333" fontWeight={600} fontFamily="Inter, sans-serif">
          Brute-force search:
        </text>
        <text x={0} y={18} fontSize={11} fill="#c62828" fontFamily="Inter, sans-serif">
          longer than the age of the universe
        </text>
        <text x={0} y={40} fontSize={11} fill="#333" fontWeight={600} fontFamily="Inter, sans-serif">
          Actual protein folding:
        </text>
        <text x={0} y={58} fontSize={11} fill="#2e7d32" fontFamily="Inter, sans-serif">
          microseconds to seconds
        </text>
        <text x={0} y={84} fontSize={10} fill="#999" fontStyle="italic" fontFamily="Inter, sans-serif">
          — Levinthal's Paradox (1969)
        </text>
      </g>
    </svg>
  )
}

function ProteinExamples() {
  const examples = [
    {
      name: 'Hemoglobin',
      residues: '~574 (4 chains)',
      function: 'Carries oxygen in blood',
      difficulty: 'Quaternary structure — 4 subunits must assemble correctly',
      color: '#c62828',
    },
    {
      name: 'Insulin',
      residues: '51 (2 chains)',
      function: 'Regulates blood sugar',
      difficulty: 'Disulfide bonds between chains — must predict cross-chain contacts',
      color: '#1565c0',
    },
    {
      name: 'SARS-CoV-2 Spike',
      residues: '~1273',
      function: 'Viral entry into cells',
      difficulty: 'Very large, flexible, heavily glycosylated — multiple conformational states',
      color: '#2e7d32',
    },
    {
      name: 'p53 (tumor suppressor)',
      residues: '393',
      function: 'Prevents cancer',
      difficulty: 'Intrinsically disordered regions — no single stable structure',
      color: '#6a1b9a',
    },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, maxWidth: 720 }}>
      {examples.map(ex => (
        <div key={ex.name} style={{
          padding: '14px 16px', borderRadius: 8,
          border: `1px solid ${ex.color}22`,
          background: `${ex.color}08`,
        }}>
          <div style={{ fontSize: 15, fontWeight: 700, color: ex.color, marginBottom: 4 }}>
            {ex.name}
          </div>
          <div style={{ fontSize: 12, color: '#666', marginBottom: 2 }}>
            {ex.residues} residues — {ex.function}
          </div>
          <div style={{ fontSize: 11, color: '#999', fontStyle: 'italic' }}>
            Challenge: {ex.difficulty}
          </div>
        </div>
      ))}
    </div>
  )
}

function TimelineBar() {
  const events = [
    { year: '1972', label: 'Anfinsen wins Nobel:\nsequence → structure', x: 40 },
    { year: '1994', label: 'CASP1 competition\nlaunches', x: 160 },
    { year: '2018', label: 'AlphaFold1\nwins CASP13', x: 380 },
    { year: '2020', label: 'AlphaFold2\nCASP14: GDT ~92', x: 520 },
    { year: '2022', label: '200M structures\npredicted', x: 660 },
  ]

  return (
    <svg viewBox="0 0 750 110" style={{ width: '100%', maxWidth: 750, height: 'auto' }}>
      {/* Timeline line */}
      <line x1={30} y1={40} x2={720} y2={40} stroke="#ccc" strokeWidth={2} />

      {events.map((ev, i) => (
        <g key={i}>
          <circle cx={ev.x} cy={40} r={5}
            fill={i === 3 ? '#e65100' : i === 4 ? '#2e7d32' : '#1976d2'} />
          <text x={ev.x} y={28} textAnchor="middle" fontSize={11}
            fontWeight={700} fill={i === 3 ? '#e65100' : '#333'} fontFamily="Inter, sans-serif">
            {ev.year}
          </text>
          {ev.label.split('\n').map((line, li) => (
            <text key={li} x={ev.x} y={60 + li * 14} textAnchor="middle"
              fontSize={10} fill="#666" fontFamily="Inter, sans-serif">{line}</text>
          ))}
        </g>
      ))}
    </svg>
  )
}

function CASPScoreChart() {
  const data = [
    { year: 'CASP10\n2012', score: 40, color: '#90a4ae' },
    { year: 'CASP11\n2014', score: 45, color: '#90a4ae' },
    { year: 'CASP12\n2016', score: 50, color: '#90a4ae' },
    { year: 'CASP13\n2018', score: 60, color: '#1976d2' },
    { year: 'CASP14\n2020', score: 92, color: '#e65100' },
  ]
  const barW = 60
  const maxH = 160

  return (
    <svg viewBox="0 0 420 240" style={{ width: '100%', maxWidth: 420, height: 'auto' }}>
      <text x={210} y={18} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="#333" fontFamily="Inter, sans-serif">CASP Free-Modeling GDT-TS Scores</text>

      {/* Experimental threshold */}
      <line x1={40} y1={30 + maxH * (1 - 90 / 100)} x2={380} y2={30 + maxH * (1 - 90 / 100)}
        stroke="#2e7d32" strokeWidth={1} strokeDasharray="4,3" />
      <text x={385} y={30 + maxH * (1 - 90 / 100) + 4} fontSize={9} fill="#2e7d32"
        fontFamily="Inter, sans-serif">~experimental accuracy</text>

      {data.map((d, i) => {
        const h = maxH * d.score / 100
        const x = 55 + i * 72
        return (
          <g key={i}>
            <rect x={x} y={30 + maxH - h} width={barW} height={h}
              fill={d.color} opacity={0.75} rx={4} />
            <text x={x + barW / 2} y={30 + maxH - h - 6} textAnchor="middle"
              fontSize={12} fontWeight={700} fill={d.color} fontFamily="Inter, sans-serif">
              {d.score}
            </text>
            {d.year.split('\n').map((line, li) => (
              <text key={li} x={x + barW / 2} y={30 + maxH + 16 + li * 13}
                textAnchor="middle" fontSize={10} fill="#666" fontFamily="Inter, sans-serif">
                {line}
              </text>
            ))}
          </g>
        )
      })}

      {/* AF2 callout */}
      <text x={55 + 4 * 72 + barW / 2} y={225} textAnchor="middle" fontSize={10}
        fontWeight={700} fill="#e65100" fontFamily="Inter, sans-serif">AlphaFold2</text>
    </svg>
  )
}

// ── Sections ───────────────────────────────────────────

function makeSections(): Section[] {
  return [
    {
      id: 0,
      title: 'What are Proteins?',
      content: (
        <div>
          <p style={pStyle}>
            Proteins are the molecular machines of life. They are chains of <strong>amino acids</strong> —
            20 different types, each with unique chemical properties — linked together like beads on a string.
            A typical protein is 100–1000 amino acids long.
          </p>
          <AminoAcidChain />
          <p style={{ ...pStyle, marginTop: 16, fontSize: 13, color: '#666' }}>
            ↑ The first 13 residues of <strong>human hemoglobin</strong> β-chain: M-V-L-S-P-A-D-K-T-N-V-K-A...
          </p>
          <p style={pStyle}>
            This linear chain folds into a specific <strong>3D shape</strong> — and that shape determines
            everything about what the protein does. A misfolded protein can cause diseases like Alzheimer's,
            Parkinson's, and cystic fibrosis.
          </p>
          <div style={{ marginTop: 16 }}>
            <h4 style={h4Style}>Four levels of protein structure</h4>
            <FoldingLevels />
          </div>
        </div>
      ),
    },
    {
      id: 1,
      title: 'Why Is Structure Prediction So Hard?',
      content: (
        <div>
          <p style={pStyle}>
            Each amino acid has several rotatable bonds. For a protein with just 100 residues, the number
            of possible 3D arrangements is roughly <strong>10<sup>300</sup></strong> — more than the number
            of atoms in the observable universe (10<sup>80</sup>).
          </p>
          <LevinthalDiagram />
          <p style={pStyle}>
            Yet real proteins fold in milliseconds to seconds. This <strong>Levinthal's Paradox</strong> tells
            us that proteins don't randomly search — they follow an energy landscape "funnel" guided by physics.
            But simulating this from first principles requires enormous computing power.
          </p>
          <h4 style={h4Style}>Why not just use experiments?</h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, maxWidth: 700 }}>
            {[
              { method: 'X-ray Crystallography', time: 'Months–years', cost: '$50K–200K', note: 'Needs crystal (many proteins won\'t crystallize)' },
              { method: 'Cryo-EM', time: 'Weeks–months', cost: '$10K–100K', note: 'Limited resolution for small proteins' },
              { method: 'NMR', time: 'Months', cost: '$50K+', note: 'Only works for small proteins (<40 kDa)' },
            ].map(m => (
              <div key={m.method} style={{
                padding: 12, borderRadius: 8, background: '#f5f5f5', border: '1px solid #e0e0e0',
              }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#333', marginBottom: 4 }}>{m.method}</div>
                <div style={{ fontSize: 11, color: '#666' }}>Time: {m.time}</div>
                <div style={{ fontSize: 11, color: '#666' }}>Cost: {m.cost}</div>
                <div style={{ fontSize: 11, color: '#999', fontStyle: 'italic', marginTop: 4 }}>{m.note}</div>
              </div>
            ))}
          </div>
          <p style={{ ...pStyle, marginTop: 12 }}>
            As of 2020, only ~170,000 structures were experimentally determined — out of billions of known
            protein sequences. We desperately needed a computational solution.
          </p>
        </div>
      ),
    },
    {
      id: 2,
      title: 'Real Examples: Why Each Protein is a Puzzle',
      content: (
        <div>
          <p style={pStyle}>
            Every protein presents unique challenges. Here are four examples that illustrate why a general
            solution was considered one of biology's grand challenges:
          </p>
          <ProteinExamples />
          <p style={{ ...pStyle, marginTop: 16 }}>
            These challenges — multi-chain assembly, disulfide bonds, conformational flexibility, intrinsic
            disorder — mean that no single rule or template can predict all protein structures. The solution
            needed to learn from <strong>evolution itself</strong>.
          </p>
        </div>
      ),
    },
    {
      id: 3,
      title: 'The AlphaFold2 Breakthrough',
      content: (
        <div>
          <p style={pStyle}>
            In November 2020, AlphaFold2 achieved a median GDT-TS score of <strong>92.4</strong> at CASP14 —
            reaching experimental-level accuracy for the first time. The protein structure prediction problem,
            a 50-year grand challenge, was effectively solved.
          </p>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <CASPScoreChart />
            <div style={{ flex: 1, minWidth: 250 }}>
              <h4 style={h4Style}>Key ideas that made it work</h4>
              <ol style={{ margin: 0, padding: '0 0 0 18px', fontSize: 13, color: '#555', lineHeight: 1.8 }}>
                <li><strong>Evolution as a teacher</strong> — co-evolving residues in MSAs reveal which positions are in spatial contact</li>
                <li><strong>Triangle geometry</strong> — if A is near B, and B is near C, then A-C distance is constrained (triangle inequality)</li>
                <li><strong>End-to-end learning</strong> — directly predict 3D coordinates, not intermediate distance maps</li>
                <li><strong>Iterative refinement</strong> — recycle predictions 3× to self-correct errors</li>
                <li><strong>Invariant Point Attention</strong> — attention in 3D space that respects rotational symmetry</li>
              </ol>
            </div>
          </div>
          <TimelineBar />
        </div>
      ),
    },
  ]
}

const pStyle: React.CSSProperties = {
  margin: '0 0 12px', fontSize: 14, color: '#333', lineHeight: 1.65,
  maxWidth: 750, fontFamily: 'Inter, system-ui, sans-serif',
}

const h4Style: React.CSSProperties = {
  margin: '16px 0 10px', fontSize: 14, fontWeight: 700, color: '#1a237e',
  fontFamily: 'Inter, system-ui, sans-serif',
}

// ── Main Component ─────────────────────────────────────

export function ProteinIntro({ onContinue }: { onContinue: () => void }) {
  const sections = makeSections()
  const [activeSection, setActiveSection] = useState(0)
  const section = sections[activeSection]
  const isLast = activeSection === sections.length - 1

  return (
    <div style={{
      width: '100%', height: '100%', background: '#fff',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      fontFamily: 'Inter, system-ui, sans-serif',
    }}>
      {/* Top bar */}
      <div style={{
        padding: '14px 32px 10px',
        borderBottom: '1px solid #eee',
        display: 'flex', alignItems: 'center', gap: 16,
        flexShrink: 0,
      }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#1a237e' }}>
          AlphaFold2 Visually
        </h1>
        <span style={{ fontSize: 14, color: '#78909c' }}>
          Understanding the protein folding revolution
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          {sections.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setActiveSection(i)}
              style={{
                border: 'none', borderRadius: 16,
                padding: '4px 14px', cursor: 'pointer', fontSize: 12,
                background: i === activeSection ? '#1a237e' : '#f0f0f0',
                color: i === activeSection ? '#fff' : '#666',
                fontWeight: i === activeSection ? 600 : 400,
                transition: 'all 0.2s',
              }}
            >
              {i + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Content area */}
      <div style={{ flex: 1, overflow: 'auto', padding: '28px 40px 40px' }}>
        <h2 style={{
          margin: '0 0 20px', fontSize: 26, fontWeight: 800,
          color: '#1a237e',
        }}>
          {section.title}
        </h2>
        {section.content}
      </div>

      {/* Bottom nav */}
      <div style={{
        padding: '12px 32px',
        borderTop: '1px solid #eee',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <button
          onClick={() => setActiveSection(s => Math.max(0, s - 1))}
          disabled={activeSection === 0}
          style={{
            ...navBtnStyle,
            opacity: activeSection === 0 ? 0.4 : 1,
          }}
        >
          ← Previous
        </button>

        <div style={{ display: 'flex', gap: 6 }}>
          {sections.map((_, i) => (
            <div key={i} style={{
              width: 8, height: 8, borderRadius: '50%',
              background: i === activeSection ? '#1a237e' : i < activeSection ? '#90a4ae' : '#ddd',
              transition: 'background 0.2s',
            }} />
          ))}
        </div>

        {isLast ? (
          <button onClick={onContinue} style={{
            ...navBtnStyle,
            background: '#1a237e', color: '#fff',
            fontWeight: 700, fontSize: 14, padding: '10px 24px',
          }}>
            Explore the Architecture →
          </button>
        ) : (
          <button
            onClick={() => setActiveSection(s => s + 1)}
            style={navBtnStyle}
          >
            Next →
          </button>
        )}
      </div>
    </div>
  )
}

const navBtnStyle: React.CSSProperties = {
  border: '1px solid #ccc', borderRadius: 8, background: '#fff',
  padding: '8px 18px', cursor: 'pointer', fontSize: 13, color: '#555',
  fontFamily: 'Inter, system-ui, sans-serif',
}
