import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API and the board socket are served by the FastAPI app on :8000.
// Proxying both keeps the browser on one origin, so the CORS allowlist in
// api/config.py is not involved during local development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
