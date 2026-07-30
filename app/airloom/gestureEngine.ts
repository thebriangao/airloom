import type {
  GestureResult,
  HandPose,
  Landmark,
  Point3,
} from "./types";

const HOLD_MS = 115;
const DRAW_PINCH_START_RATIO = 0.46;
const DRAW_PINCH_RELEASE_RATIO = 0.68;
const DRAW_PINCH_RELEASE_HOLD_MS = 120;
const SNAP_COOLDOWN_MS = 900;
const SNAP_WINDOW_MS = 650;

function distance(a: Landmark, b: Landmark) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function midpoint(points: Landmark[]): Point3 {
  const total = points.reduce(
    (sum, point) => ({
      x: sum.x + point.x,
      y: sum.y + point.y,
      z: sum.z + point.z,
    }),
    { x: 0, y: 0, z: 0 },
  );

  return {
    x: total.x / points.length,
    y: total.y / points.length,
    z: total.z / points.length,
  };
}

function fingerExtended(
  landmarks: Landmark[],
  mcp: number,
  pip: number,
  tip: number,
) {
  const wrist = landmarks[0];
  const base = landmarks[mcp];
  const middle = landmarks[pip];
  const end = landmarks[tip];
  const a = {
    x: middle.x - base.x,
    y: middle.y - base.y,
    z: middle.z - base.z,
  };
  const b = {
    x: end.x - middle.x,
    y: end.y - middle.y,
    z: end.z - middle.z,
  };
  const dot = a.x * b.x + a.y * b.y + a.z * b.z;
  const lengthA = Math.hypot(a.x, a.y, a.z);
  const lengthB = Math.hypot(b.x, b.y, b.z);
  const straightness = dot / Math.max(0.0001, lengthA * lengthB);

  return (
    straightness > 0.45 &&
    distance(end, wrist) > distance(middle, wrist) * 1.12
  );
}

export function countExtendedFingers(landmarks: Landmark[]) {
  return [
    fingerExtended(landmarks, 5, 6, 8),
    fingerExtended(landmarks, 9, 10, 12),
    fingerExtended(landmarks, 13, 14, 16),
    fingerExtended(landmarks, 17, 18, 20),
  ].filter(Boolean).length;
}

function classifyPose(landmarks: Landmark[]) {
  const fingers = [
    fingerExtended(landmarks, 5, 6, 8),
    fingerExtended(landmarks, 9, 10, 12),
    fingerExtended(landmarks, 13, 14, 16),
    fingerExtended(landmarks, 17, 18, 20),
  ];

  const fingerCount = fingers.filter(Boolean).length;
  let pose: HandPose = "other";

  if (fingerCount === 0) pose = "fist";
  if (fingerCount === 4) pose = "openPalm";
  if (fingers[0] && fingers[1] && !fingers[2] && !fingers[3]) pose = "pan2d";
  if (fingers[0] && fingers[1] && fingers[2] && !fingers[3])
    pose = "orbit3d";

  return { pose, fingerCount };
}

export class GestureEngine {
  private candidate: HandPose = "none";
  private candidateSince = 0;
  private stable: HandPose = "none";
  private snapArmedAt = 0;
  private lastSnapAt = 0;
  private previousMiddleTip?: Landmark;
  private drawingPinch = false;
  private drawingPinchReleaseAt = 0;

  update(landmarks: Landmark[], timestamp: number): GestureResult {
    const { pose: rawPose, fingerCount } = classifyPose(landmarks);

    if (rawPose !== this.candidate) {
      this.candidate = rawPose;
      this.candidateSince = timestamp;
    } else if (timestamp - this.candidateSince >= HOLD_MS) {
      this.stable = rawPose;
    }

    const handScale = distance(landmarks[0], landmarks[9]);
    const palmSpan = Math.max(0.025, distance(landmarks[5], landmarks[17]));
    const thumbIndexRatio =
      distance(landmarks[4], landmarks[8]) / palmSpan;
    const thumbMiddleRatio =
      distance(landmarks[4], landmarks[12]) / palmSpan;
    const middleTip = landmarks[12];

    if (this.drawingPinch) {
      if (thumbIndexRatio > DRAW_PINCH_RELEASE_RATIO) {
        if (this.drawingPinchReleaseAt === 0) {
          this.drawingPinchReleaseAt = timestamp;
        } else if (
          timestamp - this.drawingPinchReleaseAt >=
          DRAW_PINCH_RELEASE_HOLD_MS
        ) {
          this.drawingPinch = false;
          this.drawingPinchReleaseAt = 0;
        }
      } else {
        this.drawingPinchReleaseAt = 0;
      }
    } else if (thumbIndexRatio < DRAW_PINCH_START_RATIO) {
      this.drawingPinch = true;
      this.drawingPinchReleaseAt = 0;
    }

    if (thumbMiddleRatio < 0.72) {
      this.snapArmedAt = timestamp;
    }

    const middleDrop = this.previousMiddleTip
      ? (middleTip.y - this.previousMiddleTip.y) / palmSpan
      : 0;
    const middlePalmRatio = distance(middleTip, landmarks[9]) / palmSpan;
    const recentlyArmed =
      this.snapArmedAt > 0 &&
      timestamp - this.snapArmedAt < SNAP_WINDOW_MS;
    const snap =
      recentlyArmed &&
      thumbMiddleRatio > 0.76 &&
      (middleDrop > 0.018 || middlePalmRatio < 1.08) &&
      timestamp - this.lastSnapAt > SNAP_COOLDOWN_MS;
    const snapPose =
      thumbMiddleRatio < 0.8 ||
      (recentlyArmed &&
        (thumbMiddleRatio < 1.4 || middlePalmRatio < 1.18));

    if (snap) {
      this.lastSnapAt = timestamp;
      this.snapArmedAt = 0;
    }

    this.previousMiddleTip = { ...middleTip };

    return {
      pose: this.stable,
      drawingPinch: this.drawingPinch,
      snap,
      snapPose,
      fingerCount,
      palm: midpoint([
        landmarks[0],
        landmarks[5],
        landmarks[9],
        landmarks[13],
        landmarks[17],
      ]),
      handScale,
      indexTip: { ...landmarks[8] },
    };
  }

  reset() {
    this.candidate = "none";
    this.stable = "none";
    this.candidateSince = 0;
    this.snapArmedAt = 0;
    this.previousMiddleTip = undefined;
    this.drawingPinch = false;
    this.drawingPinchReleaseAt = 0;
  }
}
