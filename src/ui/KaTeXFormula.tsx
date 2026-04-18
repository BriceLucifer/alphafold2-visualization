import { useEffect, useRef } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

export function KaTeXFormula({
  formula,
  displayMode = true,
  style,
}: {
  formula: string
  displayMode?: boolean
  style?: React.CSSProperties
}) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (ref.current) {
      try {
        katex.render(formula, ref.current, {
          throwOnError: false,
          displayMode,
        })
      } catch {
        ref.current.textContent = formula
      }
    }
  }, [formula, displayMode])

  return (
    <div
      ref={ref}
      style={{
        background: 'rgba(255,255,255,0.03)',
        borderRadius: 6,
        padding: displayMode ? '12px 14px' : '6px 10px',
        overflowX: 'auto',
        ...style,
      }}
    />
  )
}
