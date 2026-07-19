import { describe, it, expect } from "vitest";
import { resolveCardArt, localCardUrl, cdnCardUrl } from "./cardArt";

describe("cardArt", () => {
  it("localCardUrl builds /cards/<id>.png", () => {
    expect(localCardUrl(121)).toBe("/cards/121.png");
  });

  it("cdnCardUrl uses the default album", () => {
    expect(cdnCardUrl(121)).toBe(
      "https://ptcgvis.heroz.jp/img/bqucewmzuceknw/121.png",
    );
  });

  it("local backend: primary is local, fallback is cdn", () => {
    expect(resolveCardArt(121, "local")).toEqual({
      primary: "/cards/121.png",
      fallback: "https://ptcgvis.heroz.jp/img/bqucewmzuceknw/121.png",
    });
  });

  it("cdn backend: primary is cdn, fallback is local", () => {
    expect(resolveCardArt(121, "cdn")).toEqual({
      primary: "https://ptcgvis.heroz.jp/img/bqucewmzuceknw/121.png",
      fallback: "/cards/121.png",
    });
  });
});
