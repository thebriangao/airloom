import * as THREE from "three";

const MINIMUM_POINT_COUNT = 8;
const MINIMUM_PATH_LENGTH = 0.42;
const ROUGHNESS_PERCENTILE = 0.68;
const ROUGHNESS_THRESHOLD = 0.055;
const MAXIMUM_BLEND = 0.34;

function pathLength(points: THREE.Vector3[]) {
  let length = 0;
  for (let index = 1; index < points.length; index += 1) {
    length += points[index - 1].distanceTo(points[index]);
  }
  return length;
}

function turnAngle(
  previous: THREE.Vector3,
  current: THREE.Vector3,
  next: THREE.Vector3,
) {
  const incoming = current.clone().sub(previous);
  const outgoing = next.clone().sub(current);
  if (incoming.lengthSq() < 0.0000001 || outgoing.lengthSq() < 0.0000001) {
    return 0;
  }
  return incoming.angleTo(outgoing);
}

function localRoughness(points: THREE.Vector3[], index: number) {
  const previous = points[index - 1];
  const current = points[index];
  const next = points[index + 1];
  const localSpan = previous.distanceTo(current) + current.distanceTo(next);
  if (localSpan < 0.000001) return 0;

  const localAverage = previous
    .clone()
    .addScaledVector(current, 2)
    .add(next)
    .multiplyScalar(0.25);
  return current.distanceTo(localAverage) / localSpan;
}

function isDeliberateCorner(points: THREE.Vector3[], index: number) {
  if (index < 2 || index > points.length - 3) return false;
  const cornerAngle = points[index]
    .clone()
    .sub(points[index - 2])
    .angleTo(points[index + 2].clone().sub(points[index]));
  if (cornerAngle < 0.62) return false;

  const incomingTurn = turnAngle(
    points[index - 2],
    points[index - 1],
    points[index],
  );
  const outgoingTurn = turnAngle(
    points[index],
    points[index + 1],
    points[index + 2],
  );
  return incomingTurn < 0.28 && outgoingTurn < 0.28;
}

export function smoothStroke(points: THREE.Vector3[]) {
  if (
    points.length < MINIMUM_POINT_COUNT ||
    pathLength(points) < MINIMUM_PATH_LENGTH
  ) {
    return points;
  }

  const roughness = points
    .slice(1, -1)
    .map((_, offset) => localRoughness(points, offset + 1))
    .sort((first, second) => first - second);
  const percentileIndex = Math.min(
    roughness.length - 1,
    Math.floor(roughness.length * ROUGHNESS_PERCENTILE),
  );
  if (roughness[percentileIndex] < ROUGHNESS_THRESHOLD) return points;

  let changed = false;
  const smoothed = points.map((point, index) => {
    if (
      index === 0 ||
      index === points.length - 1 ||
      isDeliberateCorner(points, index)
    ) {
      return point.clone();
    }

    const roughnessAtPoint = localRoughness(points, index);
    const response = THREE.MathUtils.clamp(
      (roughnessAtPoint - ROUGHNESS_THRESHOLD * 0.45) /
        (ROUGHNESS_THRESHOLD * 2.4),
      0,
      1,
    );
    const blend = response * MAXIMUM_BLEND;
    if (blend < 0.01) return point.clone();

    changed = true;
    const localAverage = points[index - 1]
      .clone()
      .addScaledVector(point, 2)
      .add(points[index + 1])
      .multiplyScalar(0.25);
    return point.clone().lerp(localAverage, blend);
  });

  return changed ? smoothed : points;
}
