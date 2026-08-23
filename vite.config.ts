import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Library build (tsc) emits to dist/; the UI bundle goes to dist-ui/
// so the two outputs never collide.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist-ui',
  },
});
