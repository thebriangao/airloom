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
  { name: "Volt", value: "#d7ff3f" },
  { name: "Solar", value: "#ffb000" },
  { name: "Ember", value: "#ff4f5e" },
  { name: "Pulse", value: "#ff58d6" },
  { name: "Iris", value: "#8c6bff" },
  { name: "Ion", value: "#54b9ff" },
  { name: "Glacier", value: "#73f7e8" },
  { name: "Paper", value: "#f4f0e8" },
] as const;

export const AIRLOOM_SIZES = [
  { name: "Hairline", value: 0.018 },
  { name: "Fine", value: 0.032 },
  { name: "Medium", value: 0.052 },
  { name: "Bold", value: 0.078 },
  { name: "Massive", value: 0.115 },
] as const;
