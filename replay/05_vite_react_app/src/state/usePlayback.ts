import { useCallback, useEffect, useRef, useState } from "react";

export interface Playback {
  index: number;
  playing: boolean;
  speed: number; // steps per second
  count: number;
  setIndex: (i: number) => void;
  step: (delta: number) => void;
  toPct: (pct: number) => void;
  togglePlay: () => void;
  setSpeed: (s: number) => void;
}

// Owns playback position + auto-advance. Each step is a full snapshot, so
// jumping anywhere is O(1); play just advances the index on an interval.
export function usePlayback(count: number, initialIndex = 0): Playback {
  const [index, setIndexRaw] = useState(() =>
    Math.max(0, Math.min(count - 1, initialIndex)),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);

  const clamp = useCallback(
    (i: number) => Math.max(0, Math.min(count - 1, i)),
    [count],
  );
  const setIndex = useCallback((i: number) => setIndexRaw(clamp(i)), [clamp]);
  const step = useCallback((d: number) => setIndexRaw((i) => clamp(i + d)), [clamp]);
  const toPct = useCallback(
    (pct: number) => setIndexRaw(clamp(Math.round((pct / 100) * (count - 1)))),
    [clamp, count],
  );
  const togglePlay = useCallback(() => setPlaying((p) => !p), []);

  // Auto-advance loop.
  const idxRef = useRef(index);
  idxRef.current = index;
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      if (idxRef.current >= count - 1) {
        setPlaying(false);
        return;
      }
      setIndexRaw((i) => clamp(i + 1));
    }, 1000 / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, count, clamp]);

  return { index, playing, speed, count, setIndex, step, toPct, togglePlay, setSpeed };
}
