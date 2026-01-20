// vite.config.js
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default ({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  return defineConfig({
    plugins: [react()],
    server: {
      host: '127.0.0.1',
      port: 5173,
      proxy: {
        // talk to Keycloak via same origin
        '/auth': {
          target: 'http://127.0.0.1:8088',
          changeOrigin: true,
          secure: false,
        },
        // keep your existing API proxies (if any)
        '/api/lic': {
          target: env.DEV_PROXY_LIC || 'http://127.0.0.1:1607',
          changeOrigin: true,
          secure: false,
          rewrite: p => p.replace(/^\/api\/lic/, '/license'),
        },
        '/api/predict': {
          target: env.DEV_PROXY_PREDICT || 'http://127.0.0.1:1605',
          changeOrigin: true,
          secure: false,
          // rewrite: p => p.replace(/^\/api\/predict/, '/predict_bots'),
        },
        '/api/appb': {
          target: env.DEV_PROXY_APPB || 'http://127.0.0.1:1611',
          changeOrigin: true,
          secure: false,
          rewrite: p => p.replace(/^\/api\/appb/, ''),
        },
      },
    },
  })
}
