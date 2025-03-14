import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@ai-tutor/shared': path.resolve(__dirname, '../shared/src'),
    },
  },
  optimizeDeps: {
    include: ['@ai-tutor/shared'],
  },
  build: {
    sourcemap: true,
    rollupOptions: {
      external: [],
    },
  },
});
