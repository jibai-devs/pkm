import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { Card, CardBack } from "./Card";
import { CardDb } from "../data/cardDb";
import type { CardInstance } from "../data/types";

const db = new CardDb([
  { card_id: 121, name: "Dragapult ex", card_type: 0, energy_type: 0, hp: 320,
    basic: false, stage1: false, stage2: true, ex: true, evolves_from: null,
    weakness: null, resistance: null, retreat_cost: 2, attacks: [] },
]);
const inst: CardInstance = { id: 121, serial: 1, playerIndex: 0, hp: 200, maxHp: 320, energies: [2, 2, 5] };

afterEach(cleanup);

describe("Card art", () => {
  it("renders the local image by default", () => {
    render(<Card card={inst} db={db} variant="active" backend="local" />);
    const img = screen.getByRole("img") as HTMLImageElement;
    expect(img.getAttribute("src")).toBe("/cards/121.png");
  });

  it("renders the cdn image when backend is cdn", () => {
    render(<Card card={inst} db={db} variant="active" backend="cdn" />);
    const img = screen.getByRole("img") as HTMLImageElement;
    expect(img.getAttribute("src")).toBe(
      "https://ptcgvis.heroz.jp/img/bqucewmzuceknw/121.png",
    );
  });

  it("swaps to the fallback url on first error", () => {
    render(<Card card={inst} db={db} variant="active" backend="local" />);
    const img = screen.getByRole("img") as HTMLImageElement;
    fireEvent.error(img);
    expect(img.getAttribute("src")).toBe(
      "https://ptcgvis.heroz.jp/img/bqucewmzuceknw/121.png",
    );
  });

  it("keeps the card name in the DOM as text fallback", () => {
    render(<Card card={inst} db={db} variant="active" backend="local" />);
    expect(screen.getByText("Dragapult ex", { selector: ".card-name" })).toBeTruthy();
  });

  it("CardBack renders no image", () => {
    render(<CardBack />);
    expect(screen.queryByRole("img")).toBeNull();
  });
});
