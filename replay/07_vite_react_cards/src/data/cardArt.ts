// Maps a card_id to its face-art image URL for either backend.
// local  -> /cards/<id>.png   (public/cards symlink -> pkm_data/replay/cards)
// cdn    -> https://ptcgvis.heroz.jp/img/<album>/<id>.png
export type CardBackend = "local" | "cdn";

const DEFAULT_ALBUM = "bqucewmzuceknw";

export function cardAlbum(): string {
  return (import.meta.env.VITE_CARD_ALBUM as string | undefined) || DEFAULT_ALBUM;
}

export function localCardUrl(id: number): string {
  return `/cards/${id}.png`;
}

export function cdnCardUrl(id: number, album: string = cardAlbum()): string {
  return `https://ptcgvis.heroz.jp/img/${album}/${id}.png`;
}

export function defaultBackend(): CardBackend {
  return (import.meta.env.VITE_CARD_BACKEND as string | undefined) === "cdn"
    ? "cdn"
    : "local";
}

export interface CardArtUrls {
  primary: string;
  fallback: string;
}

// The active backend's URL is `primary`; the other backend is `fallback`
// (used by <img> onError before finally giving up to the text card).
export function resolveCardArt(id: number, backend: CardBackend): CardArtUrls {
  const local = localCardUrl(id);
  const cdn = cdnCardUrl(id);
  return backend === "cdn"
    ? { primary: cdn, fallback: local }
    : { primary: local, fallback: cdn };
}
