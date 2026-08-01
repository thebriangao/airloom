import type {
  GestureResult,
  HandPose,
  Landmark,
  Point3,
} from "./types";

const HOLD_MS = 28;
// Require the detected thumb and index tips to pass calibrated 2D and 3D
// hand-relative contact limits at the same time.
const DRAW_TOUCH_START_RATIO_2D = 0.48;
const DRAW_TOUCH_START_RATIO_3D = 0.68;
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
    const palmSpan3D = Math.max(
      0.025,
      distance(landmarks[5], landmarks[17]),
    );
    const thumbIndexRatio2D =
      distance2D(landmarks[4], landmarks[8]) / palmSpan2D;
    const thumbIndexRatio3D =
      distance(landmarks[4], landmarks[8]) / palmSpan3D;
    const fingertipsTouching =
      thumbIndexRatio2D <= DRAW_TOUCH_START_RATIO_2D &&
      thumbIndexRatio3D <= DRAW_TOUCH_START_RATIO_3D;
    // Evaluate contact from the raw landmarks on every camera frame. There is
    // intentionally no release delay or latched pinch state: the first frame
    // that no longer shows fingertip contact ends the stroke.
    this.drawingPinch = !fist && !this.objectGrab && fingertipsTouching;

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
      !fingertipsTouching &&
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
    this.objectGrab = false;
    this.objectGrabCandidateAt = 0;
    this.objectGrabReleaseAt = 0;
  }
}
