import { useState, useRef, useEffect, useCallback } from 'react'
import { ProteinIntro } from './scenes/ProteinIntro'
import { ArchitectureOverview } from './scenes/ArchitectureOverview'
import { EvoformerDetail } from './scenes/EvoformerDetail'
import { StructureModuleDetail } from './scenes/StructureModuleDetail'

type View = 'intro' | 'overview' | 'evoformer' | 'structure'

export default function App() {
  const [view, setView] = useState<View>('intro')
  const [transitioning, setTransitioning] = useState(false)
  const [displayView, setDisplayView] = useState<View>('intro')
  const containerRef = useRef<HTMLDivElement>(null)

  const navigateTo = useCallback((target: View) => {
    if (target === view || transitioning) return
    setTransitioning(true)
    // Fade out
    setTimeout(() => {
      setDisplayView(target)
      setView(target)
      // Fade in
      setTimeout(() => setTransitioning(false), 50)
    }, 300)
  }, [view, transitioning])

  // On first mount, trigger fade-in
  useEffect(() => {
    setTransitioning(true)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => setTransitioning(false))
    })
  }, [])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden' }}>
      <style>{`
        .view-container {
          width: 100%; height: 100%;
          transition: opacity 0.3s ease, transform 0.35s ease;
        }
        .view-container.fade-out {
          opacity: 0;
          transform: scale(0.98);
        }
        .view-container.fade-in {
          opacity: 1;
          transform: scale(1);
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
      </div>
    </div>
  )
}
