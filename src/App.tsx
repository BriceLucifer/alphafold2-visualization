import { useState, useRef, useEffect, useCallback } from 'react'
import { ProteinIntro } from './scenes/ProteinIntro'
import { ArchitectureOverview } from './scenes/ArchitectureOverview'
import { EvoformerDetail } from './scenes/EvoformerDetail'
import { StructureModuleDetail } from './scenes/StructureModuleDetail'
import { InputEmbeddingDetail } from './scenes/InputEmbeddingDetail'

type View = 'intro' | 'overview' | 'evoformer' | 'structure' | 'embeddings'

export default function App() {
  const [view, setView] = useState<View>('intro')
  const [transitioning, setTransitioning] = useState(false)
  const [displayView, setDisplayView] = useState<View>('intro')
  const containerRef = useRef<HTMLDivElement>(null)

  const navigateTo = useCallback((target: View) => {
    if (target === view || transitioning) return
    setTransitioning(true)
    setTimeout(() => {
      setDisplayView(target)
      setView(target)
      setTimeout(() => setTransitioning(false), 60)
    }, 350)
  }, [view, transitioning])

  useEffect(() => {
    setTransitioning(true)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => setTransitioning(false))
    })
  }, [])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden', background: '#0a0a15' }}>
      <style>{`
        .view-container {
          width: 100%; height: 100%;
          transition: opacity 0.35s cubic-bezier(0.4, 0, 0.2, 1), transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .view-container.fade-out {
          opacity: 0;
          transform: scale(0.97) translateY(4px);
        }
        .view-container.fade-in {
          opacity: 1;
          transform: scale(1) translateY(0);
        }
      `}</style>
      <div
        ref={containerRef}
        className={`view-container ${transitioning ? 'fade-out' : 'fade-in'}`}
      >
        {displayView === 'intro' && (
          <ProteinIntro onContinue={() => navigateTo('overview')} />
        )}
        {displayView === 'overview' && (
          <ArchitectureOverview
            onDrillIn={(target) => navigateTo(target as View)}
            onBack={() => navigateTo('intro')}
          />
        )}
        {displayView === 'evoformer' && (
          <EvoformerDetail onBack={() => navigateTo('overview')} />
        )}
        {displayView === 'structure' && (
          <StructureModuleDetail onBack={() => navigateTo('overview')} />
        )}
        {displayView === 'embeddings' && (
          <InputEmbeddingDetail onBack={() => navigateTo('overview')} />
        )}
      </div>
    </div>
  )
}
