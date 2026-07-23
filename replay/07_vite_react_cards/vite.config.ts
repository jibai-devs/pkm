import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Minimal typing so we can read the shell env without pulling in @types/node.
declare const process: { env: Record<string, string | undefined> };

// The large replay.json / cards.json live in ./public (symlinked to ../).
// Expose shell env vars to the client (Vite only loads .env files into
// import.meta.env, not the process environment), so you can do:
//   VITE_REPLAY=/other.json bun run dev
// The Python play server (pkm/web/server.py) runs on :8000; proxy the game API
// to it so the browser can use same-origin relative /api paths in dev. Card art
// and cards.json are served by Vite from ./public (symlinks), so only /api is
// proxied. Override the target with PKM_PLAY_API if you run it elsewhere.
const PLAY_API = process.env.PKM_PLAY_API ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    open: false,
    proxy: { "/api": { target: PLAY_API, changeOrigin: true } },
  },
  define: {
    __REPLAY_URL__: JSON.stringify(process.env.VITE_REPLAY ?? ""),
    __CARDS_URL__: JSON.stringify(process.env.VITE_CARDS ?? ""),
  },
});
