import { useEffect, useState } from 'react'
import Overview from './pages/Overview'
import Explore from './pages/Explore'
import Results from './pages/Results'
import Predict from './pages/Predict'
import Method from './pages/Method'

const PAGES = [
  { id: 'overview', label: 'Overview', Component: Overview },
  { id: 'explore', label: 'Explore data', Component: Explore },
  { id: 'results', label: 'Model results', Component: Results },
  { id: 'predict', label: 'Predict yield', Component: Predict },
  { id: 'method', label: 'Method & data', Component: Method },
]

function currentPage() {
  const id = window.location.hash.replace('#', '')
  return PAGES.some((p) => p.id === id) ? id : 'overview'
}

export default function App() {
  const [page, setPage] = useState(currentPage)
  useEffect(() => {
    const onHash = () => { setPage(currentPage()); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const { Component } = PAGES.find((p) => p.id === page)

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-rule bg-page/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1180px] flex-wrap items-center gap-x-8 gap-y-2 px-5 py-3">
          <a href="#overview" className="flex items-center gap-2.5">
            <Logo />
            <span className="font-display text-[1.05rem] font-bold leading-tight">
              Crop Yield Fusion
            </span>
          </a>
          <nav aria-label="Sections" className="-mb-3 flex gap-1 overflow-x-auto">
            {PAGES.map((p) => (
              <a key={p.id} href={`#${p.id}`} aria-current={p.id === page ? 'page' : undefined}
                className={`whitespace-nowrap border-b-2 px-3 pb-3 pt-1 text-[0.95rem] font-semibold transition-colors ${
                  p.id === page ? 'border-canopy text-ink' : 'border-transparent text-muted hover:text-ink'}`}>
                {p.label}
              </a>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-[1180px] px-5 pb-20 pt-10">
        <Component />
      </main>

      <footer className="border-t border-rule">
        <div className="mx-auto flex max-w-[1180px] flex-wrap justify-between gap-2 px-5 py-6 text-sm text-muted">
          <span>Crop Yield Prediction via Multimodal Data Fusion · B.Tech CSE (AI &amp; ML), MRIIRS</span>
          <a className="underline-offset-2 hover:underline" href="/docs">API documentation</a>
        </div>
      </footer>
    </div>
  )
}

function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="#2E7D4A" />
      <path d="M16 25V12M16 17c-4 0-6-3-6-6 4 0 6 2 6 6zm0-2c4 0 6-3 6-6-4 0-6 2-6 6z" stroke="white"
        strokeWidth="2" fill="none" strokeLinecap="round" />
    </svg>
  )
}
