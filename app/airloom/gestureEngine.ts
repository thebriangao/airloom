import type {
  GestureResult,
  HandPose,
  Landmark,
  Point3,
} from "./types";

const HOLD_MS = 115;
const DRAW_PINCH_START_RATIO = 0.52;
const DRAW_PINCH_RELEASE_RATIO = 0.9;
const DRAW_PINCH_START_DISTANCE = 0.06;
const DRAW_PINCH_RELEASE_DISTANCE = 0.085;
const DRAW_PINCH_RELEASE_HOLD_MS = 240;
const OBJECT_GRAB_HOLD_MS = 45;
const OBJECT_GRAB_RELEASE_MS = 150;
const SNAP_COOLDOWN_MS = 900;
const SNAP_WINDOW_MS = 650;

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

function fingertipsTogether(landmarks: Landmark[]) {
  const fingertips = [4, 8, 12, 16, 20].map((index) => landmarks[index]);
  const palmSpan = Math.max(
    0.025,
    distance2D(landmarks[5], landmarks[17]),
  );
  let maximumSpread = 0;
  for (let first = 0; first < fingertips.length; first += 1) {
    for (let second = first + 1; second < fingertips.length; second += 1) {
      maximumSpread = Math.max(
        maximumSpread,
        distance2D(fingertips[first], fingertips[second]),
      );
    }
  }
  const center = midpoint(fingertips);
  const reachesPastPalm =
    distance2D(center, landmarks[0]) > palmSpan * 1.02;
  const spreadRatio = maximumSpread / palmSpan;
  return {
    active: spreadRatio < 0.9 && reachesPastPalm,
    tightIntent: spreadRatio < 1.18 && reachesPastPalm,
    spreadRatio,
    reachesPastPalm,
    center,
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
  private filteredThumbIndexRatio = 1;
  private objectGrab = false;
  private objectGrabCandidateAt = 0;
  private objectGrabReleaseAt = 0;
  private lastGrabPoint?: Point3;
  private previousGrabSpread = Infinity;
  private grabIntentUntil = 0;

  update(landmarks: Landmark[], timestamp: number): GestureResult {
    const { pose: rawPose, fingerCount } = classifyPose(landmarks);
    const grab = fingertipsTogether(landmarks);
    const fingertipsConverging =
      grab.reachesPastPalm &&
      grab.spreadRatio < 2.1 &&
      this.previousGrabSpread - grab.spreadRatio > 0.055;
    if (grab.active || grab.tightIntent || fingertipsConverging) {
      this.grabIntentUntil = timestamp + 190;
    }
    const grabIntent = grab.active || timestamp < this.grabIntentUntil;
    this.previousGrabSpread = grab.spreadRatio;

    if (grab.active) {
      this.lastGrabPoint = { ...grab.center };
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
    const palmSpan = Math.max(0.025, distance(landmarks[5], landmarks[17]));
    const palmSpan2D = Math.max(
      0.025,
      distance2D(landmarks[5], landmarks[17]),
    );
    const thumbIndexDistance = distance2D(landmarks[4], landmarks[8]);
    const thumbIndexRatio = thumbIndexDistance / palmSpan2D;
    this.filteredThumbIndexRatio =
      this.filteredThumbIndexRatio * 0.78 + thumbIndexRatio * 0.22;
    const thumbMiddleRatio =
      distance(landmarks[4], landmarks[12]) / palmSpan;
    const middleTip = landmarks[12];

    if (grabIntent || this.objectGrab) {
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
      pose: this.objectGrab ? "grab" : this.stable,
      drawingPinch: this.drawingPinch,
      objectGrab: this.objectGrab,
      objectGrabIntent: grabIntent || this.objectGrab,
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
      grabPoint:
        this.objectGrab && !grab.active && this.lastGrabPoint
          ? { ...this.lastGrabPoint }
          : grab.center,
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
    this.filteredThumbIndexRatio = 1;
    this.objectGrab = false;
    this.objectGrabCandidateAt = 0;
    this.objectGrabReleaseAt = 0;
    this.lastGrabPoint = undefined;
    this.previousGrabSpread = Infinity;
    this.grabIntentUntil = 0;
  }
}
