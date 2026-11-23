import { defineConfig } from 'astro/config';

export default defineConfig({
  site: process.env.PAGES_ORIGIN || undefined,
  base: process.env.PAGES_BASE || '/',
  trailingSlash: 'always',
  output: 'static',
  devToolbar: { enabled: false },
});
