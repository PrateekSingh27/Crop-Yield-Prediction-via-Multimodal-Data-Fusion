import { useState } from 'react'
import { fmt, ndviColor } from './ui'

// Month ticks per crop calendar (days after sowing at which each month starts)
const MONTH_TICKS = {
  Wheat: [[0, 'Nov'], [21, 'Dec'], [52, 'Jan'], [83, 'Feb'], [111, 'Mar'], [142, 'Apr']],
  Rice: [[0, 'Jun'], [11, 'Jul'], [42, 'Aug'], [73, 'Sep'], [103, 'Oct']],
}

/**
 * Each row = one crop season (all districts averaged), each cell = one 16-day
 * NDVI composite, coloured from bare soil (brown) to dense canopy (deep green).
 * The bar on the right is that season's mean yield.
 */
export default function NdviStrips({ data, crop }) {
  const [hover, setHover] = useState(null)
  if (!data) return null
  const { das, years } = data
  const seasonDays = crop === 'Wheat' ? 150 : 125
  const maxYield = Math.max(...years.map((y) => y.mean_yield))
  const minYield = Math.min(...years.map((y) => y.mean_yield))

  return (
    <div className="relative">
      <div className="grid grid-cols-[2.4rem_1fr_2.6rem] items-center gap-x-2 gap-y-[3px] sm:grid-cols-[3.5rem_1fr_8rem] sm:gap-x-3">
        {years.map((row) => (
          <Row key={row.year} row={row} das={das} seasonDays={seasonDays}
            maxYield={maxYield} minYield={minYield} onHover={setHover} hover={hover} />
        ))}
        <span />
        <div className="relative h-6 text-[0.7rem] text-muted sm:text-[0.8rem]">
          {MONTH_TICKS[crop].map(([d, m]) => (
            <span key={m} className="absolute top-1" style={{ left: `${(d / seasonDays) * 100}%` }}>{m}</span>
          ))}
        </div>
        <span className="pt-1 text-[0.7rem] leading-tight text-muted sm:text-[0.8rem]"><span className="hidden sm:inline">mean yield, </span>t/ha</span>
      </div>
      <p className="mt-2 h-5 text-[0.9rem] text-muted" aria-live="polite">
        {hover
          ? `${crop} ${hover.year}, day ${hover.das} after sowing: NDVI ${fmt(hover.ndvi)}`
          : 'Hover a cell to read its NDVI value.'}
      </p>
    </div>
  )
}

function Row({ row, das, seasonDays, maxYield, minYield, onHover, hover }) {
  const width = 16 / seasonDays
  const yieldPct = 35 + 65 * ((row.mean_yield - minYield) / Math.max(maxYield - minYield, 1e-6))
  return (
    <>
      <span className={`num text-right text-[0.9rem] ${hover?.year === row.year ? 'font-bold text-ink' : 'text-muted'}`}>
        {row.year}
      </span>
      <div className="relative h-[22px] overflow-hidden rounded-[3px] bg-wash" onMouseLeave={() => onHover(null)}>
        {row.ndvi.map((v, i) => (
          <span key={i} className="absolute inset-y-0 cursor-crosshair"
            style={{ left: `${(das[i] / seasonDays) * 100}%`, width: `calc(${width * 100}% + 1px)`, background: ndviColor(v) }}
            onMouseEnter={() => onHover({ year: row.year, das: das[i], ndvi: v })} />
        ))}
      </div>
      <div className="flex items-center gap-2">
        <span className="hidden h-[10px] rounded-[2px] bg-ink/70 sm:block" style={{ width: `${yieldPct * 0.55}%` }} />
        <span className="num text-[0.8rem] sm:text-[0.88rem]">{fmt(row.mean_yield)}</span>
      </div>
    </>
  )
}

export function NdviLegend() {
  const stops = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
  return (
    <div className="flex items-center gap-2 text-[0.82rem] text-muted">
      <span>bare soil</span>
      <span className="flex h-2.5 w-36 overflow-hidden rounded-sm">
        {stops.map((s) => <span key={s} className="flex-1" style={{ background: ndviColor(s) }} />)}
      </span>
      <span>dense canopy</span>
    </div>
  )
}
