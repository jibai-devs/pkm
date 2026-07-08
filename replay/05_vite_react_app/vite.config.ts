import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The large replay.json / cards.json live in ./public (symlinked to ../).
export default defineConfig({
  plugins: [react()],
  server: { port: 5175, open: false },
});
