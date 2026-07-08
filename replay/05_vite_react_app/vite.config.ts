import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Minimal typing so we can read the shell env without pulling in @types/node.
declare const process: { env: Record<string, string | undefined> };

// The large replay.json / cards.json live in ./public (symlinked to ../).
// Expose shell env vars to the client (Vite only loads .env files into
// import.meta.env, not the process environment), so you can do:
//   VITE_REPLAY=/other.json bun run dev
export default defineConfig({
  plugins: [react()],
  server: { port: 5175, open: false },
  define: {
    __REPLAY_URL__: JSON.stringify(process.env.VITE_REPLAY ?? ""),
    __CARDS_URL__: JSON.stringify(process.env.VITE_CARDS ?? ""),
  },
});
