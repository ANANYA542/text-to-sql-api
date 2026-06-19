import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    outDir: '../app/static',
    emptyOutDir: true,
  },
  base: '/static/',
  server: {
    proxy: {
      '/generate-sql': 'http://127.0.0.1:8001',
      '/retrieve': 'http://127.0.0.1:8001',
      '/health': 'http://127.0.0.1:8001',
      '/api': 'http://127.0.0.1:8001',
    }
  }
})
