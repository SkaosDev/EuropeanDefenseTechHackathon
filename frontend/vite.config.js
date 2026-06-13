import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Le backend FastAPI tourne sur :8000 (CORS ouvert). On utilise des URLs absolues
// côté client (voir src/api.js), pas de proxy nécessaire.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true },
})
