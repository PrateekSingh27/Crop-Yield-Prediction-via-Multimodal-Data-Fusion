import { useEffect, useState } from 'react'

export const MODALITY_COLORS = {
  Satellite: '#2E7D4A',
  Weather: '#2F6FA6',
  Soil: '#8C6D3F',
  'Cross-modal': '#7A5AA6',
  Context: '#8A938C',
}
export const CROP_COLORS = { Wheat: '#C8902A', Rice: '#2E7D4A' }

// NDVI colour ramp: bare soil -> senescing -> green-up -> dense canopy
const NDVI_STOPS = [
  [0.1, [163, 132, 87]],
  [0.28, [216, 191, 106]],
  [0.45, [169, 194, 90]],
  [0.62, [78, 154, 76]],
  [0.8, [31, 94, 53]],
]
export function ndviColor(v) {
  if (v == null || Number.isNaN(v)) return '#ddd'
  if (v <= NDVI_STOPS[0][0]) return `rgb(${NDVI_STOPS[0][1]})`
  for (let i = 1; i < NDVI_STOPS.length; i++) {
    const [x1, c1] = NDVI_STOPS[i]
    const [x0, c0] = NDVI_STOPS[i - 1]
    if (v <= x1) {
      const t = (v - x0) / (x1 - x0)
      return `rgb(${c0.map((c, j) => Math.round(c + t * (c1[j] - c))).join(',')})`
    }
  }
  return `rgb(${NDVI_STOPS.at(-1)[1]})`
}

export const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d))

export function useApi(loader, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  useEffect(() => {
    let alive = true
    setState((s) => ({ ...s, loading: true }))
    loader()
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((error) => alive && setState({ data: null, error, loading: false }))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return state
}

export function Panel({ title, note, children, className = '', aside }) {
  return (
    <section className={`rounded-xl border border-rule bg-surface p-5 md:p-6 ${className}`}>
      {(title || aside) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div className="max-w-2xl">
            {title && <h3 className="text-[1.15rem] font-semibold leading-snug">{title}</h3>}
            {note && <p className="mt-1 text-[0.94rem] text-muted">{note}</p>}
          </div>
          {aside}
        </header>
      )}
      {children}
    </section>
  )
}

export function PageIntro({ title, children }) {
  return (
    <div className="mb-8 max-w-3xl">
      <h2 className="text-[2rem] font-bold leading-tight md:text-[2.4rem]">{title}</h2>
      {children && <p className="mt-3 text-[1.05rem] text-muted">{children}</p>}
    </div>
  )
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-rule bg-wash p-0.5">
      {options.map((o) => {
        const v = typeof o === 'string' ? o : o.value
        const text = typeof o === 'string' ? o : o.label
        const active = v === value
        return (
          <button key={v} role="radio" aria-checked={active} onClick={() => onChange(v)}
            className={`rounded-md px-3 py-1 text-sm font-semibold transition-colors ${active ? 'bg-surface text-ink shadow-sm' : 'text-muted hover:text-ink'}`}>
            {text}
          </button>
        )
      })}
    </div>
  )
}

export function Select({ label, value, onChange, children, className = '' }) {
  return (
    <label className={`flex flex-col gap-1 text-sm ${className}`}>
      <span className="font-semibold text-muted">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="rounded-lg border border-rule bg-surface px-3 py-2 text-[0.95rem] text-ink">
        {children}
      </select>
    </label>
  )
}

export function Loading({ what = 'data' }) {
  return <p className="py-10 text-muted">Loading {what}…</p>
}

export function ErrorNote({ error }) {
  return (
    <div className="rounded-lg border border-alert/40 bg-alert/5 p-4 text-[0.95rem]">
      <strong className="text-alert">Couldn't reach the backend.</strong> {error?.message}. Start it with
      <code className="mx-1 rounded bg-wash px-1.5 py-0.5">.\run.ps1</code> and reload this page.
    </div>
  )
}

export function Swatch({ color, children }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-sm text-muted">
      <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: color }} />
      {children}
    </span>
  )
}

export const chartTooltip = {
  contentStyle: { borderRadius: 8, border: '1px solid #D5DBD2', fontSize: 13, boxShadow: 'none' },
  labelStyle: { fontWeight: 600, color: '#1C2620' },
}
export const axisProps = { tick: { fontSize: 12, fill: '#5B675F' }, stroke: '#C6CEC3' }
