export type Point3 = {
  x: number;
  y: number;
  z: number;
};

export type Landmark = {
  x: number;
  y: number;
  z: number;
};

export type HandPose =
  | "none"
  | "draw"
  | "pan2d"
  | "orbit3d"
  | "fist"
  | "openPalm"
  | "other";

export type GestureResult = {
  pose: HandPose;
  snap: boolean;
  fingerCount: number;
  palm: Point3;
  handScale: number;
  indexTip: Point3;
};

export const AIRLOOM_COLORS = [
  { name: "Ink", value: "#111111" },
  { name: "Graphite", value: "#454545" },
  { name: "Stone", value: "#858585" },
  { name: "Mist", value: "#c8c8c8" },
  { name: "Paper", value: "#f6f3ed" },
  { name: "Crimson", value: "#d7263d" },
  { name: "Coral", value: "#ff5a5f" },
  { name: "Peach", value: "#ff9f80" },
  { name: "Rose", value: "#e45c96" },
  { name: "Berry", value: "#9d174d" },
  { name: "Tangerine", value: "#f97316" },
  { name: "Marigold", value: "#f6b91a" },
  { name: "Lemon", value: "#f4df4e" },
  { name: "Meadow", value: "#68a357" },
  { name: "Forest", value: "#1e6f50" },
  { name: "Aqua", value: "#42c9b8" },
  { name: "Sky", value: "#52a7e0" },
  { name: "Cobalt", value: "#2864dc" },
  { name: "Navy", value: "#16325c" },
  { name: "Lagoon", value: "#087e8b" },
  { name: "Lavender", value: "#aa8dd8" },
  { name: "Violet", value: "#7a4bc2" },
  { name: "Plum", value: "#572f72" },
  { name: "Bubblegum", value: "#ef7fc5" },
  { name: "Clay", value: "#a2674a" },
] as const;

export const AIRLOOM_MIN_RADIUS = 0.012;
export const AIRLOOM_MAX_RADIUS = 0.14;

export function radiusFromThickness(thickness: number) {
  const clamped = Math.max(0, Math.min(1, thickness));
  return (
    AIRLOOM_MIN_RADIUS +
    (AIRLOOM_MAX_RADIUS - AIRLOOM_MIN_RADIUS) * clamped ** 1.45
  );
}

export function eraserRadiusFromThickness(thickness: number) {
  const clamped = Math.max(0, Math.min(1, thickness));
  return 0.09 + clamped ** 1.35 * 0.58;
}
