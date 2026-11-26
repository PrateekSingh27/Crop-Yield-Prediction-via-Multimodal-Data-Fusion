import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// `npm run dev` -> http://localhost:5173 (proxies API calls to FastAPI on :8000)
// `npm run build` -> writes the production UI into ../api/static (served by FastAPI)
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: '../api/static', emptyOutDir: true, chunkSizeWarningLimit: 1200 },
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000', '/figures': 'http://127.0.0.1:8000' },
  },
})
