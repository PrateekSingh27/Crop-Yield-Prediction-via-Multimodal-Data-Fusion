import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import NdviStrips, { NdviLegend } from '../components/NdviStrips'
import { CROP_COLORS, ErrorNote, Loading, MODALITY_COLORS, Panel, Segmented, axisProps, chartTooltip, fmt, useApi } from '../components/ui'

const PIPELINE = [
  ['Collect', 'Satellite NDVI composites, daily weather and soil samples for 32 districts, 2014–2024.'],
  ['Align', 'Aggregate everything to one row per district, crop and season using crop-calendar windows.'],
  ['Engineer', 'Phenology (green-up, peak, senescence), heat-stress days, GDD, and a rainfall × soil-N interaction.'],
  ['Fuse', 'Early fusion joins all 26 features in one table; late fusion stacks one model per modality.'],
  ['Validate', 'Leave-one-year-out: every season is forecast by a model that never saw it.'],
  ['Explain', 'SHAP values split each prediction into satellite, weather and soil contributions.'],
]

export default function Overview() {
  const [crop, setCrop] = useState('Wheat')
  const strips = useApi(api.strips)
  const overview = useApi(api.overview)
  const metrics = useApi(api.metrics)

  if (overview.error) return <ErrorNote error={overview.error} />

  const k = overview.data?.kpis
  const s = metrics.data?.summary
  return (
    <div className="space-y-14">
      {/* ---------------- Hero ---------------- */}
      <section>
        <h1 className="max-w-4xl text-[2.6rem] font-bold leading-[1.05] md:text-[3.6rem]">
          Forecasting crop yield by reading the field from three directions at once.
        </h1>
        <p className="mt-5 max-w-2xl text-[1.12rem] text-muted">
          Satellites see the canopy, weather records see the stress, soil tests see what lies underneath.
          This project fuses all three to predict wheat and rice yield for Indian districts before harvest.
        </p>

        <div className="mt-9 rounded-xl border border-rule bg-surface p-5 md:p-7">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
            <div className="max-w-xl">
              <h3 className="text-[1.15rem] font-semibold">Eleven seasons of canopy, as the satellite saw it</h3>
              <p className="mt-1 text-[0.94rem] text-muted">
                {crop === 'Wheat'
                  ? 'Each row is one wheat season averaged over all districts. In 2022 the crop browns early: that is the March heatwave, and the yield bar shortens with it.'
                  : 'Each row is one rice season. Monsoon clouds make kharif composites noisier, which is exactly why weather and soil data are needed alongside NDVI.'}
              </p>
            </div>
            <div className="flex flex-col items-end gap-2">
              <Segmented label="Crop" options={['Wheat', 'Rice']} value={crop} onChange={setCrop} />
              <NdviLegend />
            </div>
          </div>
          {strips.loading ? <Loading what="NDVI series" /> : (
            <div className="reveal"><NdviStrips data={strips.data?.crops?.[crop]} crop={crop} /></div>
          )}
        </div>

        {k && s && (
          <dl className="mt-6 grid grid-cols-2 gap-y-4 md:grid-cols-5 md:divide-x md:divide-rule">
            <Fact term="District seasons" value={k.rows} />
            <Fact term="Districts / states" value={`${k.districts} / ${k.states}`} />
            <Fact term="Features fused" value={s.n_features} />
            <Fact term="Forecast error (RMSE)" value={`${fmt(s.deployed_metrics.RMSE)} t/ha`} />
            <Fact term="Variance explained (R²)" value={fmt(s.deployed_metrics.R2)} />
          </dl>
        )}
      </section>

      {/* ---------------- Pipeline ---------------- */}
      <section>
        <h2 className="text-[1.75rem] font-bold">How a forecast is made</h2>
        <ol className="mt-6 grid gap-x-8 gap-y-6 md:grid-cols-3">
          {PIPELINE.map(([title, text], i) => (
            <li key={title} className="grid grid-cols-[2rem_1fr] gap-x-3 border-t border-rule pt-4">
              <span className="font-display text-[1.4rem] font-bold leading-none text-canopy">{i + 1}</span>
              <div>
                <h3 className="text-[1.05rem] font-semibold">{title}</h3>
                <p className="mt-1 text-[0.95rem] text-muted">{text}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      {/* ---------------- Key results ---------------- */}
      <section className="grid gap-6 lg:grid-cols-2">
        <Panel title="Fusion beats every single source"
          note="Leave-one-year-out R² of the deployed model trained on each combination of modalities.">
          {metrics.data ? <AblationChart rows={metrics.data.modality_ablation} /> : <Loading />}
        </Panel>
        <Panel title="Average yield by state" note="Mean over 2014–2024 seasons, tonnes per hectare.">
          {overview.data ? <StateChart rows={overview.data.by_state} /> : <Loading />}
        </Panel>
      </section>
    </div>
  )
}

function Fact({ term, value }) {
  return (
    <div className="md:px-5 md:first:pl-0">
      <dt className="text-[0.88rem] text-muted">{term}</dt>
      <dd className="num font-display text-[1.6rem] font-semibold leading-tight">{value}</dd>
    </div>
  )
}

function AblationChart({ rows }) {
  const data = [...rows].sort((a, b) => a.R2 - b.R2).map((r) => ({ ...r, label: r.Modalities.replace(' (early fusion)', '') }))
  const colorFor = (r) => (r.n_modalities === 3 ? '#1F5E35' : r.n_modalities === 2 ? '#7FAF85' : '#C3CEC4')
  return (
    <ResponsiveContainer width="100%" height={290}>
      <BarChart data={data} layout="vertical" margin={{ left: 10, right: 40 }}>
        <CartesianGrid horizontal={false} stroke="#E6EBE3" />
        <XAxis type="number" domain={[0, 1]} {...axisProps} />
        <YAxis type="category" dataKey="label" width={150} {...axisProps} />
        <Tooltip {...chartTooltip} formatter={(v) => fmt(v, 3)} />
        <Bar dataKey="R2" name="R²" radius={[0, 3, 3, 0]} label={{ position: 'right', fontSize: 12, formatter: (v) => fmt(v) }}>
          {data.map((r) => <Cell key={r.label} fill={colorFor(r)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function StateChart({ rows }) {
  const states = [...new Set(rows.map((r) => r.state))]
  const data = states.map((state) => {
    const entry = { state: state.replace('Pradesh', 'Pr.').replace('Chhattisgarh', 'Chhattisg.') }
    rows.filter((r) => r.state === state).forEach((r) => { entry[r.crop] = r.yield_tonnes_ha })
    return entry
  }).sort((a, b) => (b.Wheat ?? b.Rice) - (a.Wheat ?? a.Rice))
  return (
    <ResponsiveContainer width="100%" height={290}>
      <BarChart data={data} margin={{ bottom: 30 }}>
        <CartesianGrid vertical={false} stroke="#E6EBE3" />
        <XAxis dataKey="state" angle={-35} textAnchor="end" interval={0} height={60} {...axisProps} />
        <YAxis {...axisProps} />
        <Tooltip {...chartTooltip} formatter={(v) => `${fmt(v)} t/ha`} />
        <Legend wrapperStyle={{ fontSize: 13 }} verticalAlign="top" align="right" />
        <Bar dataKey="Wheat" fill={CROP_COLORS.Wheat} radius={[3, 3, 0, 0]} />
        <Bar dataKey="Rice" fill={CROP_COLORS.Rice} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export { MODALITY_COLORS }
