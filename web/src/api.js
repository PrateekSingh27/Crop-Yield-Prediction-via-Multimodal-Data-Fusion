// Thin wrapper around the FastAPI backend. Same origin in production.
async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string' ? body.detail : body.detail?.[0]?.msg || detail
    } catch { /* keep default */ }
    throw new Error(detail)
  }
  return response.json()
}

export const api = {
  meta: () => request('/api/meta'),
  overview: () => request('/api/overview'),
  strips: () => request('/api/ndvi-strips'),
  metrics: () => request('/api/metrics'),
  predictions: () => request('/api/predictions'),
  district: (name, crop) => request(`/api/district/${encodeURIComponent(name)}?crop=${crop}`),
  dataset: (params = {}) => request(`/api/dataset?${new URLSearchParams(params)}`),
  predict: (body) => request('/api/predict', { method: 'POST', body: JSON.stringify(body) }),
}
