import { useEffect, useMemo, useState } from 'react'
import { Area, CartesianGrid, ComposedChart, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import { ErrorNote, Loading, PageIntro, Panel, Segmented, Select, axisProps, chartTooltip, fmt, useApi } from '../components/ui'

const TABLE_COLUMNS = [
  ['district', 'District'], ['crop', 'Crop'], ['year', 'Season'], ['ndvi_peak', 'Peak NDVI'],
  ['rainfall_mm', 'Rain (mm)'], ['heat_stress_days', 'Heat days'], ['tmax_mean_c', 'Tmax (°C)'],
  ['soil_ph', 'pH'], ['soil_n_kg_ha', 'N (kg/ha)'], ['soil_oc_pct', 'OC (%)'], ['yield_tonnes_ha', 'Yield (t/ha)'],
]
const PAGE_SIZE = 20

export default function Explore() {
  const meta = useApi(api.meta)
  const [district, setDistrict] = useState('Ludhiana')
  const [crop, setCrop] = useState('Wheat')

  const districts = meta.data?.districts ?? []
  const current = districts.find((d) => d.district === district)
  useEffect(() => {
    if (current && !current.crops.includes(crop)) setCrop(current.crops[0])
  }, [current, crop])

  const detail = useApi(() => api.district(district, crop), [district, crop])

  if (meta.error) return <ErrorNote error={meta.error} />
  return (
    <div className="space-y-8">
      <PageIntro title="Explore the fused dataset">
        Every row is one district, one crop and one season, with satellite, weather and soil features lined up
        side by side. Pick a district to see how its canopy and yield moved year to year.
      </PageIntro>

      <div className="flex flex-wrap items-end gap-4">
        <Select label="District" value={district} onChange={setDistrict} className="min-w-56">
          {Object.entries(groupBy(districts, 'state')).map(([state, list]) => (
            <optgroup key={state} label={state}>
              {list.map((d) => <option key={d.district} value={d.district}>{d.district}</option>)}
            </optgroup>
          ))}
        </Select>
        {current && <Segmented label="Crop" options={current.crops} value={crop} onChange={setCrop} />}
        {detail.data && (
          <p className="pb-1 text-[0.95rem] text-muted">
            {detail.data.state}, mean {crop.toLowerCase()} yield <strong className="text-ink">{fmt(detail.data.mean_yield)} t/ha</strong>
          </p>
        )}
      </div>

      {detail.error ? <ErrorNote error={detail.error} /> : detail.loading && !detail.data ? <Loading /> : detail.data && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Yield and peak NDVI by season"
            note="When the canopy is greener the yield usually follows, but not always: heat after flowering cuts yield without changing peak NDVI much.">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={detail.data.history} margin={{ right: 10 }}>
                <CartesianGrid vertical={false} stroke="#E6EBE3" />
                <XAxis dataKey="year" {...axisProps} />
                <YAxis yAxisId="y" {...axisProps} domain={['auto', 'auto']} />
                <YAxis yAxisId="n" orientation="right" {...axisProps} domain={[0, 1]} />
                <Tooltip {...chartTooltip} formatter={(v, n) => (n.includes('NDVI') ? fmt(v) : `${fmt(v)} t/ha`)} />
                <Legend wrapperStyle={{ fontSize: 13 }} />
                <Line yAxisId="y" type="linear" dataKey="yield_tonnes_ha" name="Yield (t/ha)" stroke="#1C2620" strokeWidth={2.2} dot={{ r: 3 }} />
                <Line yAxisId="n" type="linear" dataKey="ndvi_peak" name="Peak NDVI" stroke="#2E7D4A" strokeWidth={1.8} strokeDasharray="5 3" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Panel>

          <Panel title="Season NDVI curve"
            note={`Shaded band = range over all seasons. Best season ${detail.data.best_year}, worst ${detail.data.worst_year}.`}>
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={detail.data.ndvi_curve.map((p) => ({ ...p, band: [p.min, p.max] }))} margin={{ right: 10 }}>
                <CartesianGrid vertical={false} stroke="#E6EBE3" />
                <XAxis dataKey="das" {...axisProps} label={{ value: 'days after sowing', position: 'insideBottom', offset: -4, fontSize: 12, fill: '#5B675F' }} height={42} />
                <YAxis {...axisProps} domain={[0, 1]} />
                <Tooltip {...chartTooltip} formatter={(v) => (Array.isArray(v) ? `${fmt(v[0])}–${fmt(v[1])}` : fmt(v))} />
                <Legend wrapperStyle={{ fontSize: 13, paddingTop: 10 }} />
                <Area dataKey="band" name="Range" fill="#2E7D4A" fillOpacity={0.12} stroke="none" />
                <Line dataKey="best" name={`Best (${detail.data.best_year})`} stroke="#1F5E35" strokeWidth={2} dot={false} />
                <Line dataKey="worst" name={`Worst (${detail.data.worst_year})`} stroke="#B4472F" strokeWidth={2} dot={false} />
                <Line dataKey="mean" name="Mean" stroke="#5B675F" strokeDasharray="4 3" dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </Panel>
        </div>
      )}

      <DatasetTable states={[...new Set(districts.map((d) => d.state))]} />
    </div>
  )
}

function DatasetTable({ states }) {
  const [crop, setCrop] = useState('')
  const [state, setState] = useState('')
  const [page, setPage] = useState(0)
  const [sort, setSort] = useState({ key: 'year', dir: 1 })
  const all = useApi(() => api.dataset({ limit: 5000 }), [])

  const rows = useMemo(() => {
    const list = (all.data?.rows ?? []).filter((r) => (!crop || r.crop === crop) && (!state || r.state === state))
    return list.sort((a, b) => (a[sort.key] > b[sort.key] ? 1 : a[sort.key] < b[sort.key] ? -1 : 0) * sort.dir)
  }, [all.data, crop, state, sort])
  useEffect(() => setPage(0), [crop, state])

  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const visible = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const download = () => {
    const header = Object.keys(all.data.rows[0])
    const csv = [header.join(','), ...rows.map((r) => header.map((h) => r[h]).join(','))].join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    Object.assign(document.createElement('a'), { href: url, download: 'crop_yield_multimodal_filtered.csv' }).click()
    URL.revokeObjectURL(url)
  }

  return (
    <Panel title="Dataset" note={`${rows.length} of ${all.data?.total ?? '…'} rows. Click a column heading to sort.`}
      aside={
        <div className="flex flex-wrap items-end gap-3">
          <Select label="Crop" value={crop} onChange={setCrop}>
            <option value="">All crops</option><option>Wheat</option><option>Rice</option>
          </Select>
          <Select label="State" value={state} onChange={setState}>
            <option value="">All states</option>
            {states.map((s) => <option key={s}>{s}</option>)}
          </Select>
          <button onClick={download} disabled={!rows.length}
            className="rounded-lg border border-canopy px-3 py-2 text-sm font-semibold text-canopy hover:bg-canopy hover:text-white">
            Download CSV
          </button>
        </div>
      }>
      {all.loading ? <Loading /> : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[820px] text-[0.92rem]">
              <thead>
                <tr className="border-b border-rule text-left text-muted">
                  {TABLE_COLUMNS.map(([key, label]) => (
                    <th key={key} className="px-2 py-2 font-semibold">
                      <button onClick={() => setSort((s) => ({ key, dir: s.key === key ? -s.dir : 1 }))}
                        className="hover:text-ink">
                        {label}{sort.key === key ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="num">
                {visible.map((r) => (
                  <tr key={`${r.district}-${r.crop}-${r.year}`} className="border-b border-wash hover:bg-wash/60">
                    {TABLE_COLUMNS.map(([key]) => (
                      <td key={key} className={`px-2 py-1.5 ${key === 'yield_tonnes_ha' ? 'font-semibold' : ''}`}>
                        {typeof r[key] === 'number' && !['year'].includes(key) ? fmt(r[key], key.includes('ndvi') || key === 'soil_oc_pct' ? 2 : 1) : r[key]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 flex items-center justify-between text-sm text-muted">
            <span>Page {page + 1} of {pages}</span>
            <div className="flex gap-2">
              <button className="rounded-md border border-rule px-3 py-1 disabled:opacity-40" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
              <button className="rounded-md border border-rule px-3 py-1 disabled:opacity-40" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next</button>
            </div>
          </div>
        </>
      )}
    </Panel>
  )
}

function groupBy(list, key) {
  return list.reduce((acc, item) => ((acc[item[key]] ??= []).push(item), acc), {})
}
