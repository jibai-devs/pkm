// Energy-type codes, calibrated against the "Basic {X} Energy" cards in cards.json:
//   1={G} 2={R} 3={W} 4={L} 5={P} 6={F} 7={D} 8={M}
// 0 is Colorless {C}; 9/10/11 cover Dragon/Fairy variants seen in some sets.
export interface EnergyType {
  code: number;
  letter: string; // single-letter TCG symbol, e.g. "P"
  name: string;
  color: string; // background
  fg: string; // foreground text
}

export const ENERGY_TYPES: Record<number, EnergyType> = {
  0: { code: 0, letter: "C", name: "Colorless", color: "#d9d4c8", fg: "#2b2b2b" },
  1: { code: 1, letter: "G", name: "Grass", color: "#5fbb46", fg: "#0c2b09" },
  2: { code: 2, letter: "R", name: "Fire", color: "#e2493a", fg: "#2b0a06" },
  3: { code: 3, letter: "W", name: "Water", color: "#3d9ee0", fg: "#062033" },
  4: { code: 4, letter: "L", name: "Lightning", color: "#f2c53d", fg: "#332a05" },
  5: { code: 5, letter: "P", name: "Psychic", color: "#a25fd0", fg: "#22093a" },
  6: { code: 6, letter: "F", name: "Fighting", color: "#c8792e", fg: "#2e1605" },
  7: { code: 7, letter: "D", name: "Darkness", color: "#4a5568", fg: "#e5e9f0" },
  8: { code: 8, letter: "M", name: "Metal", color: "#8a97a5", fg: "#12181f" },
  9: { code: 9, letter: "N", name: "Dragon", color: "#b8912f", fg: "#241a04" },
  10: { code: 10, letter: "Y", name: "Fairy", color: "#e07ab5", fg: "#33061f" },
};

export function energyType(code: number): EnergyType {
  return (
    ENERGY_TYPES[code] ?? {
      code,
      letter: String(code),
      name: `Type ${code}`,
      color: "#9aa0a6",
      fg: "#111",
    }
  );
}
