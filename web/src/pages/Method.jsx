import { api } from '../api'
import { PageIntro, useApi } from '../components/ui'

const SOURCES = [
  ['Satellite', 'Sentinel-2 (10 m) and MODIS NDVI via Google Earth Engine', 'Simulated 16-day NDVI composites from a double-logistic canopy curve; cloud contamination in the monsoon.'],
  ['Weather', 'NASA POWER daily API (rain, Tmax/Tmin, solar, humidity)', 'Real NASA POWER data with one command, or a synthetic daily generator offline.'],
  ['Soil', 'SoilGrids (ISRIC) and India Soil Health Card', 'District soil profile plus yearly sampling noise, like repeated Soil Health Card tests.'],
  ['Yield', 'data.gov.in district crop statistics / ICRISAT', 'Crop-response model: water, heat, nitrogen, pH, organic carbon, texture, radiation.'],
]

const TEAM = [
  ['Snehaal Mukherjee', '1/23/SET/BCS/350'],
  ['Krishna Bhatia', '1/23/SET/BCS/358'],
  ['Prateek Singh', '1/23/SET/BCS/362'],
  ['Aryaman', '1/23/SET/BCS/364'],
]

export default function Method() {
  const meta = useApi(api.meta)
  const weather = meta.data?.summary?.data_source?.weather

  return (
    <div className="max-w-3xl space-y-12">
      <PageIntro title="Method and data">
        How the dataset is built, how the models are tested, and what the results can and cannot claim.
      </PageIntro>

      <section>
        <h3 className="text-[1.35rem] font-bold">Where each modality comes from</h3>
        <p className="mt-2 text-muted">
          Real district yields, satellite rasters and soil cards need accounts and days of GIS processing, so this build
          simulates them with crop-science response curves. Weather can be swapped for real NASA POWER records.
          The current model was trained with <strong className="text-ink">{weather ?? '…'}</strong> weather.
        </p>
        <div className="mt-5 overflow-x-auto">
          <table className="w-full text-[0.94rem]">
            <thead>
              <tr className="border-b border-rule text-left text-muted">
                <th className="py-2 pr-4 font-semibold">Modality</th>
                <th className="py-2 pr-4 font-semibold">Production source</th>
                <th className="py-2 font-semibold">In this build</th>
              </tr>
            </thead>
            <tbody>
              {SOURCES.map(([m, real, here]) => (
                <tr key={m} className="border-b border-wash align-top">
                  <td className="py-2.5 pr-4 font-semibold">{m}</td>
                  <td className="py-2.5 pr-4">{real}</td>
                  <td className="py-2.5 text-muted">{here}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h3 className="text-[1.35rem] font-bold">Why the simulation is not a shortcut</h3>
        <p className="mt-2 text-muted">
          Yield is generated with non-linear agronomy (a heat-damage threshold at grain filling, saturating nitrogen response,
          a pH optimum near 6.8, irrigation buffering rainfall deficits) and an unobserved management effect the model never sees.
          Only the satellite partly observes that effect through canopy vigour. That is why no single source is enough,
          why trees beat a linear model, and why the scores are realistic rather than perfect. Monsoon departures follow approximate
          IMD all-India figures, and the 2022 wheat season includes the March heatwave.
        </p>
      </section>

      <section>
        <h3 className="text-[1.35rem] font-bold">How the models are tested</h3>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-muted marker:text-canopy">
          <li><strong className="text-ink">Leave-one-year-out.</strong> Train on ten seasons, forecast the eleventh, repeat for all eleven. Random splits would let the model see the same season in training and test and inflate the score.</li>
          <li><strong className="text-ink">Leave-districts-out.</strong> A second check that holds out whole districts, to test places the model has never seen.</li>
          <li><strong className="text-ink">Metrics.</strong> RMSE and MAE in t/ha, R², and MAPE in percent.</li>
          <li><strong className="text-ink">Ablation.</strong> The same model retrained on every combination of satellite, weather and soil.</li>
          <li><strong className="text-ink">Explainability.</strong> Exact TreeSHAP contributions; each forecast equals the base value plus the sum of its contributions.</li>
          <li><strong className="text-ink">Uncertainty.</strong> 80% intervals from the spread of out-of-sample errors, per crop.</li>
        </ul>
      </section>

      <section>
        <h3 className="text-[1.35rem] font-bold">Limitations and next steps</h3>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-muted marker:text-canopy">
          <li>Scores describe the simulated world. Replace the yield labels with official district statistics before making agronomic claims.</li>
          <li>District averages hide field-level variation; Sentinel-2 at 10 m would allow field-scale models.</li>
          <li>Deep intermediate fusion (CNN for imagery, LSTM for daily weather) needs far more labelled seasons than 440.</li>
          <li>SHAP explains what the model learned, which is association, not cause.</li>
        </ul>
      </section>

      <section>
        <h3 className="text-[1.35rem] font-bold">Team</h3>
        <ul className="mt-3 grid gap-2 sm:grid-cols-2">
          {TEAM.map(([name, roll]) => (
            <li key={roll} className="flex justify-between border-b border-rule py-2">
              <span className="font-semibold">{name}</span><span className="num text-muted">{roll}</span>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-[0.92rem] text-muted">B.Tech CSE (AI &amp; ML), Manav Rachna International Institute of Research and Studies, Faridabad.</p>
      </section>
    </div>
  )
}
