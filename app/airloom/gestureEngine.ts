import type {
  GestureResult,
  HandPose,
  Landmark,
  Point3,
} from "./types";

const HOLD_MS = 28;
const DRAW_PINCH_START_RATIO = 0.52;
const DRAW_PINCH_RELEASE_RATIO = 0.9;
const DRAW_PINCH_START_DISTANCE = 0.06;
const DRAW_PINCH_RELEASE_DISTANCE = 0.085;
const DRAW_PINCH_RELEASE_HOLD_MS = 28;
const OBJECT_GRAB_HOLD_MS = 28;
const OBJECT_GRAB_RELEASE_MS = 70;

function distance(a: Landmark, b: Landmark) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function distance2D(a: Landmark, b: Landmark) {
  return Math.hypot(a.x - b.x, a.y - b.y);
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

function classifyPose(landmarks: Landmark[]) {
  const fingers = [
    fingerExtended(landmarks, 5, 6, 8),
    fingerExtended(landmarks, 9, 10, 12),
    fingerExtended(landmarks, 13, 14, 16),
    fingerExtended(landmarks, 17, 18, 20),
  ];

  const fingerCount = fingers.filter(Boolean).length;
  const palmSpan = Math.max(
    0.025,
    distance2D(landmarks[5], landmarks[17]),
  );
  const palm = midpoint([
    landmarks[0],
    landmarks[5],
    landmarks[9],
    landmarks[13],
    landmarks[17],
  ]);
  const curledFingerCenter = midpoint([
    landmarks[8],
    landmarks[12],
    landmarks[16],
    landmarks[20],
  ]);
  const fist =
    fingerCount === 0 &&
    distance2D(curledFingerCenter, palm) < palmSpan * 0.92 &&
    distance2D(landmarks[4], palm) < palmSpan * 1.12;
  let pose: HandPose = "other";

  if (fist) pose = "fist";
  if (fingerCount === 4) pose = "openPalm";
  if (fingers[0] && fingers[1] && !fingers[2] && !fingers[3]) pose = "pan2d";
  if (fingers[0] && fingers[1] && fingers[2] && !fingers[3])
    pose = "orbit3d";

  const indexVectorX = landmarks[8].x - landmarks[6].x;
  const indexVectorY = landmarks[8].y - landmarks[6].y;
  const indexPointingUp =
    fingers[0] &&
    !fingers[1] &&
    !fingers[2] &&
    !fingers[3] &&
    indexVectorY < -0.025 &&
    Math.abs(indexVectorX) < Math.abs(indexVectorY) * 0.85;

  return { pose, fingerCount, fist, indexPointingUp, palm };
}

export class GestureEngine {
  private candidate: HandPose = "none";
  private candidateSince = 0;
  private stable: HandPose = "none";
  private drawingPinch = false;
  private drawingPinchReleaseAt = 0;
  private filteredThumbIndexRatio = 1;
  private objectGrab = false;
  private objectGrabCandidateAt = 0;
  private objectGrabReleaseAt = 0;

  update(landmarks: Landmark[], timestamp: number): GestureResult {
    const {
      pose: rawPose,
      fingerCount,
      fist,
      indexPointingUp,
      palm,
    } = classifyPose(landmarks);

    if (fist) {
      this.objectGrabReleaseAt = 0;
      if (this.objectGrabCandidateAt === 0) {
        this.objectGrabCandidateAt = timestamp;
      } else if (
        timestamp - this.objectGrabCandidateAt >= OBJECT_GRAB_HOLD_MS
      ) {
        this.objectGrab = true;
      }
    } else {
      this.objectGrabCandidateAt = 0;
      if (this.objectGrab) {
        if (this.objectGrabReleaseAt === 0) {
          this.objectGrabReleaseAt = timestamp;
        } else if (
          timestamp - this.objectGrabReleaseAt >= OBJECT_GRAB_RELEASE_MS
        ) {
          this.objectGrab = false;
          this.objectGrabReleaseAt = 0;
        }
      }
    }

    if (rawPose !== this.candidate) {
      this.candidate = rawPose;
      this.candidateSince = timestamp;
    } else if (timestamp - this.candidateSince >= HOLD_MS) {
      this.stable = rawPose;
    }

    const handScale = distance(landmarks[0], landmarks[9]);
    const palmSpan2D = Math.max(
      0.025,
      distance2D(landmarks[5], landmarks[17]),
    );
    const thumbIndexDistance = distance2D(landmarks[4], landmarks[8]);
    const thumbIndexRatio = thumbIndexDistance / palmSpan2D;
    this.filteredThumbIndexRatio =
      this.filteredThumbIndexRatio * 0.45 + thumbIndexRatio * 0.55;
    if (fist || this.objectGrab) {
      this.drawingPinch = false;
      this.drawingPinchReleaseAt = 0;
    } else if (this.drawingPinch) {
      const releaseSignal =
        thumbIndexRatio > DRAW_PINCH_RELEASE_RATIO &&
        thumbIndexDistance > DRAW_PINCH_RELEASE_DISTANCE;
      if (releaseSignal) {
        if (this.drawingPinchReleaseAt === 0) {
          this.drawingPinchReleaseAt = timestamp;
        } else if (
          timestamp - this.drawingPinchReleaseAt >=
            DRAW_PINCH_RELEASE_HOLD_MS &&
          this.filteredThumbIndexRatio >
            DRAW_PINCH_RELEASE_RATIO * 0.82
        ) {
          this.drawingPinch = false;
          this.drawingPinchReleaseAt = 0;
        }
      } else {
        this.drawingPinchReleaseAt = 0;
      }
    } else if (
      thumbIndexRatio < DRAW_PINCH_START_RATIO ||
      thumbIndexDistance < DRAW_PINCH_START_DISTANCE
    ) {
      this.drawingPinch = true;
      this.drawingPinchReleaseAt = 0;
      this.filteredThumbIndexRatio = thumbIndexRatio;
    }

    const sideViewControl =
      indexPointingUp &&
      !this.drawingPinch &&
      !this.objectGrab &&
      !fist;
    const brushHover =
      !this.drawingPinch &&
      !this.objectGrab &&
      !fist &&
      !sideViewControl &&
      thumbIndexRatio > 0.68 &&
      rawPose !== "pan2d" &&
      rawPose !== "orbit3d" &&
      rawPose !== "fist";

    return {
      pose: this.objectGrab ? "grab" : this.stable,
      drawingPinch: this.drawingPinch,
      objectGrab: this.objectGrab,
      objectGrabIntent: fist || this.objectGrab,
      sideViewControl,
      brushHover,
      fingerCount,
      palm,
      handScale,
      indexTip: { ...landmarks[8] },
      grabPoint: palm,
    };
  }

  reset() {
    this.candidate = "none";
    this.stable = "none";
    this.candidateSince = 0;
    this.drawingPinch = false;
    this.drawingPinchReleaseAt = 0;
    this.filteredThumbIndexRatio = 1;
    this.objectGrab = false;
    this.objectGrabCandidateAt = 0;
    this.objectGrabReleaseAt = 0;
  }
}
