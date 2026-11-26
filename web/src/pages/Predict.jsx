import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { ErrorNote, Loading, MODALITY_COLORS, PageIntro, Segmented, Select, fmt, useApi } from '../components/ui'

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

// Scenario presets applied on top of the district's typical season
const SCENARIOS = [
  { id: 'typical', label: 'Typical season', apply: (x) => x },
  { id: 'heat', label: 'Heatwave at grain filling', apply: (x) => ({ ...x, heat_stress_days: clamp(x.heat_stress_days + 15, 0, 40), tmax_mean_c: x.tmax_mean_c + 2, ndvi_late: clamp(x.ndvi_late * 0.75, 0.05, 0.9) }) },
  { id: 'drought', label: 'Rainfall failure', apply: (x) => ({ ...x, rainfall_mm: x.rainfall_mm * 0.5, flowering_rainfall_mm: x.flowering_rainfall_mm * 0.4, dry_spell_days: clamp(x.dry_spell_days + 20, 0, 150), ndvi_mid: clamp(x.ndvi_mid * 0.9, 0.05, 0.95), ndvi_peak: clamp(x.ndvi_peak * 0.85, 0.1, 0.95) }) },
  { id: 'lowN', label: 'Nitrogen-poor soil', apply: (x) => ({ ...x, soil_n_kg_ha: 100, soil_oc_pct: clamp(x.soil_oc_pct * 0.6, 0.1, 2) }) },
  { id: 'good', label: 'Well-managed field', apply: (x) => ({ ...x, soil_n_kg_ha: 280, soil_oc_pct: 0.8, ndvi_peak: clamp(x.ndvi_peak + 0.06, 0.1, 0.92), ndvi_mid: clamp(x.ndvi_mid + 0.05, 0.05, 0.92) }) },
]

export default function Predict() {
  const meta = useApi(api.meta)
  const [district, setDistrict] = useState('Karnal')
  const [crop, setCrop] = useState('Wheat')
  const [scenario, setScenario] = useState('typical')
  const [inputs, setInputs] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const timer = useRef()

  const districts = meta.data?.districts ?? []
  const current = districts.find((d) => d.district === district)
  const preset = meta.data?.presets?.[`${district}|${crop}`]

  useEffect(() => {
    if (current && !current.crops.includes(crop)) setCrop(current.crops[0])
  }, [current, crop])

  // reset inputs when district / crop / scenario changes
  useEffect(() => {
    if (!preset) return
    const s = SCENARIOS.find((x) => x.id === scenario)
    setInputs(s.apply({ ...preset.inputs }))
  }, [preset, scenario])

  // debounced live prediction
  useEffect(() => {
    if (!inputs || !preset) return
    clearTimeout(timer.current)
    timer.current = setTimeout(() => {
      setBusy(true)
      const silt = clamp(100 - inputs.sand_pct - inputs.clay_pct, 0, 100)
      api.predict({ crop, district, inputs: { ...inputs, silt_pct: silt } })
        .then((r) => { setResult(r); setError(null) })
        .catch(setError)
        .finally(() => setBusy(false))
    }, 220)
    return () => clearTimeout(timer.current)
  }, [inputs, crop, district, preset])

  const groups = useMemo(() => {
    const out = {}
    for (const f of meta.data?.inputs ?? []) {
      if (f.name === 'silt_pct') continue
      ;(out[f.modality] ??= []).push(f)
    }
    return out
  }, [meta.data])

  if (meta.error) return <ErrorNote error={meta.error} />
  if (!meta.data || !inputs) return <Loading what="model" />

  const set = (name, value) => setInputs((x) => ({ ...x, [name]: value }))

  return (
    <div>
      <PageIntro title="Predict a season's yield">
        Start from a district's typical season, then change what the satellite, weather and soil would report.
        The forecast updates as you move a slider, with a breakdown of what pushed it up or down.
      </PageIntro>

      <div className="mb-6 flex flex-wrap items-end gap-4">
        <Select label="District" value={district} onChange={(v) => { setDistrict(v); setScenario('typical') }} className="min-w-56">
          {districts.map((d) => <option key={d.district} value={d.district}>{d.district}, {d.state}</option>)}
        </Select>
        {current && <Segmented label="Crop" options={current.crops} value={crop} onChange={(v) => { setCrop(v); setScenario('typical') }} />}
      </div>

      <fieldset className="mb-8">
        <legend className="mb-2 text-sm font-semibold text-muted">Try a scenario</legend>
        <div className="flex flex-wrap gap-2">
          {SCENARIOS.map((s) => (
            <button key={s.id} onClick={() => { setScenario(s.id); setInputs(s.apply({ ...preset.inputs })) }}
              aria-pressed={scenario === s.id}
              className={`rounded-full border px-3.5 py-1.5 text-[0.92rem] font-semibold transition-colors ${
                scenario === s.id ? 'border-ink bg-ink text-white' : 'border-rule bg-surface text-ink hover:border-ink'}`}>
              {s.label}
            </button>
          ))}
        </div>
      </fieldset>

      <div className="grid gap-8 lg:grid-cols-[1fr_420px]">
        <div className="space-y-6">
          {['Satellite', 'Weather', 'Soil'].map((modality) => (
            <section key={modality} className="rounded-xl border border-rule bg-surface p-5">
              <h3 className="mb-1 flex items-center gap-2 text-[1.1rem] font-semibold">
                <span className="h-3 w-3 rounded-sm" style={{ background: MODALITY_COLORS[modality] }} />
                {modality === 'Satellite' ? 'Satellite (NDVI by crop stage)' : modality === 'Weather' ? 'Weather over the season' : 'Soil test'}
              </h3>
              <div className="mt-3 grid gap-x-8 gap-y-4 sm:grid-cols-2">
                {groups[modality].map((f) => (
                  <Slider key={f.name} field={f} value={inputs[f.name]} onChange={(v) => set(f.name, v)}
                    typical={preset.inputs[f.name]} />
                ))}
              </div>
              {modality === 'Soil' && (
                <p className="mt-3 text-[0.9rem] text-muted">
                  Silt is filled in automatically: {fmt(clamp(100 - inputs.sand_pct - inputs.clay_pct, 0, 100), 0)}% so the texture adds up to 100%.
                </p>
              )}
            </section>
          ))}
        </div>

        <aside className="lg:sticky lg:top-24 lg:self-start">
          <ResultCard result={result} busy={busy} error={error} crop={crop} district={district} />
        </aside>
      </div>
    </div>
  )
}

function stepFor(field) {
  if (field.unit === 'index') return 0.01
  const span = field.p99 - field.p01
  return span > 500 ? 10 : span > 50 ? 1 : span > 5 ? 0.1 : 0.01
}

function Slider({ field, value, onChange, typical }) {
  const step = stepFor(field)
  const pad = (field.p99 - field.p01) * 0.25
  const lo = Math.max(field.min, Math.floor((field.p01 - pad) / step) * step)
  const hi = Math.min(field.max, Math.ceil((field.p99 + pad) / step) * step)
  const decimals = step < 0.1 ? 2 : step < 1 ? 1 : 0
  const id = `in-${field.name}`
  const changed = Math.abs(value - typical) > step / 2
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-[0.94rem]">{field.label}</label>
        <span className={`num text-[0.94rem] font-semibold ${changed ? 'text-monsoon' : ''}`}>
          {fmt(value, decimals)} <span className="font-normal text-muted">{field.unit !== 'index' ? field.unit : ''}</span>
        </span>
      </div>
      <input id={id} type="range" min={lo} max={hi} step={step} value={clamp(value, lo, hi)}
        onChange={(e) => onChange(Number(e.target.value))} />
      <div className="flex justify-between text-[0.78rem] text-muted">
        <span className="num">{fmt(lo, decimals)}</span>
        {changed && <button className="text-monsoon hover:underline" onClick={() => onChange(typical)}>reset to typical {fmt(typical, decimals)}</button>}
        <span className="num">{fmt(hi, decimals)}</span>
      </div>
    </div>
  )
}

function ResultCard({ result, busy, error, crop, district }) {
  if (error) return <div className="rounded-xl border border-alert/40 bg-surface p-5 text-alert">{error.message}</div>
  if (!result) return <div className="rounded-xl border border-rule bg-surface p-6"><Loading what="forecast" /></div>

  const [lo, hi] = result.interval_80
  const scaleMax = 7
  const top = result.contributions.filter((c) => c.modality !== 'Context').slice(0, 9)
  const maxAbs = Math.max(...top.map((c) => Math.abs(c.contribution)), 0.05)
  const delta = result.vs_district_mean_pct

  return (
    <div className={`rounded-xl border border-rule bg-surface p-6 transition-opacity ${busy ? 'opacity-70' : ''}`} aria-live="polite">
      <p className="text-[0.95rem] text-muted">Forecast yield, {crop.toLowerCase()} in {district}</p>
      <p className="num font-display text-[3.4rem] font-bold leading-none">
        {fmt(result.predicted_yield_t_ha)}<span className="ml-2 text-[1.3rem] font-semibold text-muted">t/ha</span>
      </p>
      {delta != null && (
        <p className={`mt-2 text-[0.98rem] font-semibold ${delta >= 0 ? 'text-canopy' : 'text-alert'}`}>
          {delta >= 0 ? '+' : ''}{fmt(delta, 1)}% against this district's average of {fmt(result.district_mean_yield)} t/ha
        </p>
      )}

      <div className="mt-5">
        <div className="relative h-3 rounded-full bg-wash">
          <div className="absolute inset-y-0 rounded-full bg-canopy/30" style={{ left: `${(lo / scaleMax) * 100}%`, width: `${((hi - lo) / scaleMax) * 100}%` }} />
          <div className="absolute -top-1 h-5 w-1 rounded bg-ink" style={{ left: `calc(${(result.predicted_yield_t_ha / scaleMax) * 100}% - 2px)` }} />
        </div>
        <p className="mt-2 text-[0.88rem] text-muted">
          80% of past forecasts landed between <span className="num font-semibold text-ink">{fmt(lo)}</span> and <span className="num font-semibold text-ink">{fmt(hi)}</span> t/ha of a value like this.
        </p>
      </div>

      <h4 className="mt-6 text-[1rem] font-semibold">Why this number</h4>
      <p className="text-[0.88rem] text-muted">
        Starting from the average season ({fmt(result.base_value)} t/ha), each input pushes the forecast up or down.
      </p>
      <ul className="mt-3 space-y-1.5">
        {top.map((c) => (
          <li key={c.feature} className="grid grid-cols-[1fr_120px_52px] items-center gap-2 text-[0.88rem]">
            <span className="truncate" title={c.label}>{c.label}</span>
            <span className="relative h-3">
              <span className="absolute inset-y-0 left-1/2 w-px bg-rule" />
              <span className="absolute inset-y-0 rounded-sm" style={{
                background: MODALITY_COLORS[c.modality],
                width: `${(Math.abs(c.contribution) / maxAbs) * 50}%`,
                left: c.contribution >= 0 ? '50%' : `${50 - (Math.abs(c.contribution) / maxAbs) * 50}%`,
              }} />
            </span>
            <span className={`num text-right font-semibold ${c.contribution >= 0 ? 'text-canopy' : 'text-alert'}`}>
              {c.contribution >= 0 ? '+' : ''}{fmt(c.contribution)}
            </span>
          </li>
        ))}
      </ul>

      <div className="mt-5 grid grid-cols-3 gap-2 border-t border-rule pt-4">
        {['Satellite', 'Weather', 'Soil'].map((m) => {
          const v = result.modality_contributions[m] ?? 0
          return (
            <div key={m}>
              <p className="text-[0.82rem] text-muted">{m}</p>
              <p className="num font-semibold" style={{ color: MODALITY_COLORS[m] }}>{v >= 0 ? '+' : ''}{fmt(v)} t/ha</p>
            </div>
          )
        })}
      </div>

      {result.warnings.length > 0 && (
        <ul className="mt-4 space-y-1 rounded-lg bg-ripe/10 p-3 text-[0.85rem] text-ink">
          {result.warnings.map((w) => <li key={w}>{w}</li>)}
        </ul>
      )}
      <p className="mt-4 text-[0.8rem] text-muted">Model: {result.model}. Trained on simulated district data; not an official forecast.</p>
    </div>
  )
}
