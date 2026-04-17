import { useState } from 'react'
import { ProteinIntro } from './scenes/ProteinIntro'
import { ArchitectureOverview } from './scenes/ArchitectureOverview'
import { EvoformerDetail } from './scenes/EvoformerDetail'
import { StructureModuleDetail } from './scenes/StructureModuleDetail'

type View = 'intro' | 'overview' | 'evoformer' | 'structure'

export default function App() {
  const [view, setView] = useState<View>('intro')

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {view === 'intro' && (
        <ProteinIntro onContinue={() => setView('overview')} />
      )}
      {view === 'overview' && (
        <ArchitectureOverview
          onDrillIn={(target) => setView(target as View)}
          onBack={() => setView('intro')}
        />
      )}
      {view === 'evoformer' && (
        <EvoformerDetail onBack={() => setView('overview')} />
      )}
      {view === 'structure' && (
        <StructureModuleDetail onBack={() => setView('overview')} />
      )}
    </div>
  )
}
