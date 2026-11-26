import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { api } from '../api'
import { CROP_COLORS, ErrorNote, Loading, MODALITY_COLORS, PageIntro, Panel, Swatch, axisProps, chartTooltip, fmt, useApi } from '../components/ui'

export default function Results() {
  const metrics = useApi(api.metrics)
  const preds = useApi(api.predictions)
  if (metrics.error) return <ErrorNote error={metrics.error} />
  if (!metrics.data) return <Loading what="results" />

  const m = metrics.data
  const s = m.summary
  const deployed = s.deployed_model
  const ablation = m.modality_ablation
  const fused = ablation.find((r) => r.n_modalities === 3)
  const bestSingle = ablation.filter((r) => r.n_modalities === 1).sort((a, b) => b.R2 - a.R2)[0]
  const ridge = m.model_comparison.find((r) => r.Model === 'Ridge Regression')
  const late = m.model_comparison.find((r) => r.Fusion === 'Late')

  return (
    <div className="space-y-8">
      <PageIntro title="Model results">
        Every number on this page is out-of-sample. Each season was predicted by a model trained only on the
        other ten seasons (leave-one-year-out), so the scores show how the system would do on a year it has never seen.
      </PageIntro>

      <div className="grid gap-4 md:grid-cols-3">
        <Finding value={`${fmt(s.deployed_metrics.RMSE)} t/ha`} title={`${deployed} error`}>
          RMSE across 440 held-out district seasons, or {fmt(s.deployed_metrics.MAPE_pct, 1)}% average error. R² {fmt(s.deployed_metrics.R2)}.
        </Finding>
        <Finding value={`+${fmt(fused.R2 - bestSingle.R2)} R²`} title="Gain from fusing">
          All three modalities together reach R² {fmt(fused.R2)}; the best single source ({bestSingle.Modalities.toLowerCase()}) reaches {fmt(bestSingle.R2)}.
        </Finding>
        <Finding value={`${fmt(late.RMSE)} vs ${fmt(fused.RMSE)}`} title="Early beats late fusion">
          Stacking separate per-modality models loses the cross-modal interactions (e.g. heat × canopy) that one joint model can learn.
        </Finding>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.15fr_1fr]">
        <Panel title="Model comparison" note="Sorted by RMSE. The deployed model is the most accurate one that can also be explained exactly.">
          <ComparisonTable rows={m.model_comparison} deployed={deployed} />
          <p className="mt-4 text-[0.92rem] text-muted">
            Gradient boosting beats ridge regression (RMSE {fmt(ridge.RMSE)}) because yield responds non-linearly:
            heat damage switches on above a threshold, and nitrogen response saturates. The neural network trails the trees
            on 440 rows, the data-hungry behaviour the deep-fusion option in the proposal warns about.
          </p>
        </Panel>
        <Panel title="Predicted vs actual" note={`${deployed}, each point forecast by a model that never saw its season.`}
          aside={<div className="flex gap-3"><Swatch color={CROP_COLORS.Wheat}>Wheat</Swatch><Swatch color={CROP_COLORS.Rice}>Rice</Swatch></div>}>
          {preds.data ? <PredVsActual rows={preds.data.rows} /> : <Loading />}
        </Panel>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="What drives the forecast" note="Mean absolute SHAP value: how far, on average, each feature moves a prediction (t/ha)."
          aside={<div className="flex flex-wrap gap-3">{['Satellite', 'Weather', 'Soil', 'Cross-modal'].map((k) => <Swatch key={k} color={MODALITY_COLORS[k]}>{k}</Swatch>)}</div>}>
          <ImportanceChart rows={m.feature_importance_shap.slice(0, 12)} />
        </Panel>
        <div className="space-y-6">
          <Panel title="Share of the explanation by modality" note="Sum of SHAP importance per data source for the deployed model.">
            <ModalityShare rows={m.modality_contribution.filter((r) => r.share_pct > 0.5)} />
            <p className="mt-4 text-[0.92rem] text-muted">
              Late-fusion meta-learner weights: {Object.entries(s.late_fusion_weights).map(([k, v]) => `${k} ${fmt(v)}`).join(', ')}.
              Both views agree that satellite carries the most signal, with weather and soil adding what the canopy cannot show.
            </p>
          </Panel>
          <Panel title="Generalising to unseen districts" note="Spatial check: whole districts are held out (8 folds) instead of whole years.">
            <SimpleTable rows={m.spatial_validation} cols={[['Model', 'Model'], ['RMSE', 'RMSE'], ['R2', 'R²'], ['MAPE_pct', 'MAPE %']]} />
            <p className="mt-3 text-[0.92rem] text-muted">
              Accuracy drops for districts the model has never seen, and the boosted trees lose their edge over ridge
              regression: part of what trees learn is each district's fixed soil signature. Forecasting new regions
              would need more districts in training.
            </p>
          </Panel>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Error by held-out season" note="RMSE when each year is the test year. 2022 wheat suffered a March heatwave.">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={m.per_year_metrics}>
              <CartesianGrid vertical={false} stroke="#E6EBE3" />
              <XAxis dataKey="year" {...axisProps} />
              <YAxis {...axisProps} />
              <Tooltip {...chartTooltip} formatter={(v) => `${fmt(v, 3)} t/ha`} />
              <Bar dataKey="RMSE" radius={[3, 3, 0, 0]}>
                {m.per_year_metrics.map((r) => <Cell key={r.year} fill={r.year === 2022 ? '#C8902A' : '#7FAF85'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Error by crop and state" note="Out-of-sample error for the deployed model.">
          <SimpleTable rows={m.per_crop_metrics} cols={[['crop', 'Crop'], ['n', 'Seasons'], ['RMSE', 'RMSE'], ['MAE', 'MAE'], ['R2', 'R²'], ['MAPE_pct', 'MAPE %']]} />
          <div className="mt-5 max-h-56 overflow-y-auto">
            <SimpleTable rows={m.per_state_metrics} cols={[['state', 'State'], ['n', 'Seasons'], ['RMSE', 'RMSE'], ['MAPE_pct', 'MAPE %']]} />
          </div>
        </Panel>
      </div>
    </div>
  )
}

function Finding({ value, title, children }) {
  return (
    <div className="border-l-[3px] border-canopy bg-surface py-4 pl-5 pr-4">
      <p className="text-[0.92rem] font-semibold text-muted">{title}</p>
      <p className="num font-display text-[1.9rem] font-bold leading-tight">{value}</p>
      <p className="mt-1 text-[0.92rem] text-muted">{children}</p>
    </div>
  )
}

function ComparisonTable({ rows, deployed }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[0.94rem]">
        <thead>
          <tr className="border-b border-rule text-left text-muted">
            <th className="py-2 pr-2 font-semibold">Model</th><th className="px-2 font-semibold">Fusion</th>
            <th className="px-2 text-right font-semibold">RMSE</th><th className="px-2 text-right font-semibold">MAE</th>
            <th className="px-2 text-right font-semibold">R²</th><th className="pl-2 text-right font-semibold">MAPE %</th>
          </tr>
        </thead>
        <tbody className="num">
          {rows.map((r) => {
            const isDeployed = r.Model === deployed
            return (
              <tr key={r.Model} className={`border-b border-wash ${isDeployed ? 'bg-canopy/8 font-semibold' : ''}`}>
                <td className="py-2 pr-2">{r.Model}{isDeployed && <span className="ml-2 rounded bg-canopy px-1.5 py-0.5 text-[0.75rem] font-semibold text-white">deployed</span>}</td>
                <td className="px-2">{r.Fusion}</td>
                <td className="px-2 text-right">{fmt(r.RMSE, 3)}</td><td className="px-2 text-right">{fmt(r.MAE, 3)}</td>
                <td className="px-2 text-right">{fmt(r.R2, 3)}</td><td className="pl-2 text-right">{fmt(r.MAPE_pct, 1)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function PredVsActual({ rows }) {
  const lo = Math.floor(Math.min(...rows.map((r) => Math.min(r.actual, r.predicted))) * 2) / 2
  const hi = Math.ceil(Math.max(...rows.map((r) => Math.max(r.actual, r.predicted))) * 2) / 2
  return (
    <ResponsiveContainer width="100%" height={330}>
      <ScatterChart margin={{ right: 10, bottom: 10 }}>
        <CartesianGrid stroke="#E6EBE3" />
        <XAxis type="number" dataKey="actual" name="Actual" domain={[lo, hi]} {...axisProps}
          label={{ value: 'actual t/ha', position: 'insideBottom', offset: -5, fontSize: 12, fill: '#5B675F' }} />
        <YAxis type="number" dataKey="predicted" name="Predicted" domain={[lo, hi]} {...axisProps}
          label={{ value: 'predicted t/ha', angle: -90, position: 'insideLeft', fontSize: 12, fill: '#5B675F' }} />
        <ZAxis range={[22, 22]} />
        <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#5B675F" strokeDasharray="4 3" />
        <Tooltip {...chartTooltip} cursor={{ strokeDasharray: '3 3' }}
          content={({ payload }) => payload?.[0] ? (
            <div className="rounded-lg border border-rule bg-surface px-3 py-2 text-[0.85rem]">
              <strong>{payload[0].payload.district}</strong> {payload[0].payload.crop} {payload[0].payload.year}<br />
              actual {fmt(payload[0].payload.actual)}, predicted {fmt(payload[0].payload.predicted)}
            </div>) : null} />
        {['Wheat', 'Rice'].map((crop) => (
          <Scatter key={crop} name={crop} data={rows.filter((r) => r.crop === crop)} fill={CROP_COLORS[crop]} fillOpacity={0.7} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  )
}

function ImportanceChart({ rows }) {
  return (
    <ResponsiveContainer width="100%" height={380}>
      <BarChart data={rows} layout="vertical" margin={{ left: 10, right: 30 }}>
        <CartesianGrid horizontal={false} stroke="#E6EBE3" />
        <XAxis type="number" {...axisProps} tickFormatter={(v) => fmt(v)} />
        <YAxis type="category" dataKey="label" width={190} {...axisProps} />
        <Tooltip {...chartTooltip} formatter={(v) => `${fmt(v, 3)} t/ha`} />
        <Bar dataKey="mean_abs_shap" name="mean |SHAP|" radius={[0, 3, 3, 0]}>
          {rows.map((r) => <Cell key={r.feature} fill={MODALITY_COLORS[r.modality]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function ModalityShare({ rows }) {
  return (
    <div>
      <div className="flex h-9 overflow-hidden rounded-md">
        {rows.map((r) => (
          <div key={r.modality} className="flex items-center justify-center text-[0.85rem] font-semibold text-white"
            style={{ width: `${r.share_pct}%`, background: MODALITY_COLORS[r.modality] }} title={`${r.modality} ${fmt(r.share_pct, 1)}%`}>
            {r.share_pct > 8 ? `${Math.round(r.share_pct)}%` : ''}
          </div>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-4">
        {rows.map((r) => <Swatch key={r.modality} color={MODALITY_COLORS[r.modality]}>{r.modality} {fmt(r.share_pct, 1)}%</Swatch>)}
      </div>
    </div>
  )
}

function SimpleTable({ rows, cols }) {
  return (
    <table className="w-full text-[0.92rem]">
      <thead>
        <tr className="border-b border-rule text-left text-muted">
          {cols.map(([k, label], i) => <th key={k} className={`py-1.5 font-semibold ${i ? 'text-right' : ''}`}>{label}</th>)}
        </tr>
      </thead>
      <tbody className="num">
        {rows.map((r, idx) => (
          <tr key={idx} className="border-b border-wash">
            {cols.map(([k], i) => (
              <td key={k} className={`py-1.5 ${i ? 'text-right' : ''}`}>
                {typeof r[k] === 'number' && k !== 'n' ? fmt(r[k], k === 'MAPE_pct' ? 1 : 3) : r[k]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
