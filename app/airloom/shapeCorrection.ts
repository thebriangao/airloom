import * as THREE from "three";

export type CorrectedShapeKind =
  | "line"
  | "circle"
  | "square"
  | "rectangle";

export type ShapeCorrection = {
  kind: CorrectedShapeKind;
  points: THREE.Vector3[];
};

type Point2 = { x: number; y: number };

const LINE_ERROR_LIMIT = 0.085;
const CIRCLE_ERROR_LIMIT = 0.09;
const RECTANGLE_ERROR_LIMIT = 0.075;

function pathLength(points: THREE.Vector3[]) {
  let length = 0;
  for (let index = 1; index < points.length; index += 1) {
    length += points[index - 1].distanceTo(points[index]);
  }
  return length;
}

function evenlySpacedLine(
  start: THREE.Vector3,
  end: THREE.Vector3,
  segments = 20,
) {
  return Array.from({ length: segments + 1 }, (_, index) =>
    start.clone().lerp(end, index / segments),
  );
}

function principalDirection(points: THREE.Vector3[], center: THREE.Vector3) {
  const covariance = new THREE.Matrix3().set(0, 0, 0, 0, 0, 0, 0, 0, 0);
  const elements = covariance.elements;
  for (const point of points) {
    const offset = point.clone().sub(center);
    elements[0] += offset.x * offset.x;
    elements[1] += offset.x * offset.y;
    elements[2] += offset.x * offset.z;
    elements[3] += offset.y * offset.x;
    elements[4] += offset.y * offset.y;
    elements[5] += offset.y * offset.z;
    elements[6] += offset.z * offset.x;
    elements[7] += offset.z * offset.y;
    elements[8] += offset.z * offset.z;
  }

  let direction = points.at(-1)!.clone().sub(points[0]);
  if (direction.lengthSq() < 0.000001) direction.set(1, 0, 0);
  direction.normalize();
  for (let iteration = 0; iteration < 10; iteration += 1) {
    direction.applyMatrix3(covariance);
    if (direction.lengthSq() < 0.000001) return undefined;
    direction.normalize();
  }
  return direction;
}

function fitLine(points: THREE.Vector3[], length: number) {
  const center = points
    .reduce((sum, point) => sum.add(point), new THREE.Vector3())
    .multiplyScalar(1 / points.length);
  const direction = principalDirection(points, center);
  if (!direction) return undefined;

  let minimum = Infinity;
  let maximum = -Infinity;
  let squaredError = 0;
  for (const point of points) {
    const offset = point.clone().sub(center);
    const projection = offset.dot(direction);
    minimum = Math.min(minimum, projection);
    maximum = Math.max(maximum, projection);
    squaredError += offset
      .addScaledVector(direction, -projection)
      .lengthSq();
  }

  const extent = maximum - minimum;
  if (extent < 0.28) return undefined;
  const directness = points[0].distanceTo(points.at(-1)!) / length;
  const normalizedError = Math.sqrt(squaredError / points.length) / extent;
  if (directness < 0.8 || normalizedError > LINE_ERROR_LIMIT) {
    return undefined;
  }

  const start = center.clone().addScaledVector(direction, minimum);
  const end = center.clone().addScaledVector(direction, maximum);
  if (start.distanceTo(points[0]) > end.distanceTo(points[0])) {
    return { kind: "line" as const, points: evenlySpacedLine(end, start) };
  }
  return { kind: "line" as const, points: evenlySpacedLine(start, end) };
}

function projectionPlane(points: THREE.Vector3[]) {
  const center = points
    .reduce((sum, point) => sum.add(point), new THREE.Vector3())
    .multiplyScalar(1 / points.length);
  const normal = new THREE.Vector3();

  for (let index = 0; index < points.length; index += 1) {
    const current = points[index].clone().sub(center);
    const next = points[(index + 1) % points.length].clone().sub(center);
    normal.add(current.cross(next));
  }
  if (normal.lengthSq() < 0.000001) return undefined;
  normal.normalize();

  let axisX = points[0].clone().sub(center);
  axisX.addScaledVector(normal, -axisX.dot(normal));
  if (axisX.lengthSq() < 0.000001) {
    const reference = Math.abs(normal.y) < 0.9
      ? new THREE.Vector3(0, 1, 0)
      : new THREE.Vector3(1, 0, 0);
    axisX = reference.cross(normal);
  }
  axisX.normalize();
  const axisY = normal.clone().cross(axisX).normalize();
  const projected = points.map((point) => {
    const offset = point.clone().sub(center);
    return { x: offset.dot(axisX), y: offset.dot(axisY) };
  });

  return { center, axisX, axisY, projected };
}

function solveThreeByThree(matrix: number[][], values: number[]) {
  const rows = matrix.map((row, index) => [...row, values[index]]);
  for (let column = 0; column < 3; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < 3; row += 1) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) {
        pivot = row;
      }
    }
    if (Math.abs(rows[pivot][column]) < 0.0000001) return undefined;
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const divisor = rows[column][column];
    for (let entry = column; entry < 4; entry += 1) {
      rows[column][entry] /= divisor;
    }
    for (let row = 0; row < 3; row += 1) {
      if (row === column) continue;
      const factor = rows[row][column];
      for (let entry = column; entry < 4; entry += 1) {
        rows[row][entry] -= factor * rows[column][entry];
      }
    }
  }
  return rows.map((row) => row[3]);
}

function fitCircle2D(points: Point2[], path: number) {
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumYY = 0;
  let sumXY = 0;
  let sumR2 = 0;
  let sumXR2 = 0;
  let sumYR2 = 0;
  for (const point of points) {
    const radiusSquared = point.x * point.x + point.y * point.y;
    sumX += point.x;
    sumY += point.y;
    sumXX += point.x * point.x;
    sumYY += point.y * point.y;
    sumXY += point.x * point.y;
    sumR2 += radiusSquared;
    sumXR2 += point.x * radiusSquared;
    sumYR2 += point.y * radiusSquared;
  }
  const count = points.length;
  const solution = solveThreeByThree(
    [
      [sumXX, sumXY, sumX],
      [sumXY, sumYY, sumY],
      [sumX, sumY, count],
    ],
    [-sumXR2, -sumYR2, -sumR2],
  );
  if (!solution) return undefined;
  const [a, b, c] = solution;
  const center = { x: -a / 2, y: -b / 2 };
  const radiusSquared = center.x ** 2 + center.y ** 2 - c;
  if (radiusSquared <= 0) return undefined;
  const radius = Math.sqrt(radiusSquared);
  if (radius < 0.14) return undefined;

  const error = Math.sqrt(
    points.reduce((sum, point) => {
      const distance = Math.hypot(point.x - center.x, point.y - center.y);
      return sum + (distance - radius) ** 2;
    }, 0) / count,
  ) / radius;
  const circumferenceCoverage = path / (Math.PI * 2 * radius);
  if (
    error > CIRCLE_ERROR_LIMIT ||
    circumferenceCoverage < 0.68 ||
    circumferenceCoverage > 1.38
  ) {
    return undefined;
  }
  return { center, radius, error };
}

function fitRectangle2D(points: Point2[], path: number) {
  let best:
    | {
        angle: number;
        minX: number;
        maxX: number;
        minY: number;
        maxY: number;
        error: number;
      }
    | undefined;

  for (let step = 0; step < 90; step += 1) {
    const angle = (step / 90) * (Math.PI / 2);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const rotated = points.map((point) => ({
      x: point.x * cosine + point.y * sine,
      y: -point.x * sine + point.y * cosine,
    }));
    const minX = Math.min(...rotated.map((point) => point.x));
    const maxX = Math.max(...rotated.map((point) => point.x));
    const minY = Math.min(...rotated.map((point) => point.y));
    const maxY = Math.max(...rotated.map((point) => point.y));
    const width = maxX - minX;
    const height = maxY - minY;
    const diagonal = Math.hypot(width, height);
    if (Math.min(width, height) < 0.16 || diagonal < 0.3) continue;

    const edgeCounts = [0, 0, 0, 0];
    let squaredError = 0;
    for (const point of rotated) {
      const distances = [
        Math.abs(point.x - minX),
        Math.abs(maxX - point.x),
        Math.abs(point.y - minY),
        Math.abs(maxY - point.y),
      ];
      const distance = Math.min(...distances);
      edgeCounts[distances.indexOf(distance)] += 1;
      squaredError += distance * distance;
    }
    if (edgeCounts.some((count) => count < points.length * 0.04)) continue;
    const error = Math.sqrt(squaredError / points.length) / diagonal;
    if (!best || error < best.error) {
      best = { angle, minX, maxX, minY, maxY, error };
    }
  }
  if (!best || best.error > RECTANGLE_ERROR_LIMIT) return undefined;

  let { minX, maxX, minY, maxY } = best;
  const originalWidth = maxX - minX;
  const originalHeight = maxY - minY;
  const ratio = originalWidth / originalHeight;
  const square = ratio >= 0.78 && ratio <= 1.28;
  if (square) {
    const size = (originalWidth + originalHeight) / 2;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    minX = centerX - size / 2;
    maxX = centerX + size / 2;
    minY = centerY - size / 2;
    maxY = centerY + size / 2;
  }

  const perimeter = 2 * ((maxX - minX) + (maxY - minY));
  const coverage = path / perimeter;
  if (coverage < 0.62 || coverage > 1.42) return undefined;
  return { ...best, minX, maxX, minY, maxY, square };
}

function pointFromPlane(
  point: Point2,
  plane: NonNullable<ReturnType<typeof projectionPlane>>,
) {
  return plane.center
    .clone()
    .addScaledVector(plane.axisX, point.x)
    .addScaledVector(plane.axisY, point.y);
}

function circlePoints(
  fit: NonNullable<ReturnType<typeof fitCircle2D>>,
  plane: NonNullable<ReturnType<typeof projectionPlane>>,
  sourceStart: Point2,
) {
  const startAngle = Math.atan2(
    sourceStart.y - fit.center.y,
    sourceStart.x - fit.center.x,
  );
  const segments = 64;
  return Array.from({ length: segments + 1 }, (_, index) => {
    const angle = startAngle + (index / segments) * Math.PI * 2;
    return pointFromPlane(
      {
        x: fit.center.x + Math.cos(angle) * fit.radius,
        y: fit.center.y + Math.sin(angle) * fit.radius,
      },
      plane,
    );
  });
}

function rectanglePoints(
  fit: NonNullable<ReturnType<typeof fitRectangle2D>>,
  plane: NonNullable<ReturnType<typeof projectionPlane>>,
) {
  const cosine = Math.cos(fit.angle);
  const sine = Math.sin(fit.angle);
  const unrotate = (point: Point2) => ({
    x: point.x * cosine - point.y * sine,
    y: point.x * sine + point.y * cosine,
  });
  const corners = [
    { x: fit.minX, y: fit.minY },
    { x: fit.maxX, y: fit.minY },
    { x: fit.maxX, y: fit.maxY },
    { x: fit.minX, y: fit.maxY },
  ].map(unrotate);

  let startIndex = 0;
  let bestDistance = Infinity;
  for (let index = 0; index < corners.length; index += 1) {
    const distance = Math.hypot(
      corners[index].x - plane.projected[0].x,
      corners[index].y - plane.projected[0].y,
    );
    if (distance < bestDistance) {
      bestDistance = distance;
      startIndex = index;
    }
  }
  const ordered = Array.from(
    { length: 5 },
    (_, index) => corners[(startIndex + index) % 4],
  );
  return ordered.flatMap((corner, index) => {
    if (index === ordered.length - 1) return [];
    const next = ordered[index + 1];
    const edge = Array.from({ length: 9 }, (_, step) =>
      pointFromPlane(
        {
          x: corner.x + (next.x - corner.x) * (step / 8),
          y: corner.y + (next.y - corner.y) * (step / 8),
        },
        plane,
      ),
    );
    return index === 0 ? edge : edge.slice(1);
  });
}

export function correctSimpleGeometry(
  sourcePoints: THREE.Vector3[],
): ShapeCorrection | undefined {
  if (sourcePoints.length < 5) return undefined;
  const points = sourcePoints.map((point) => point.clone());
  const length = pathLength(points);
  if (length < 0.32) return undefined;

  const line = fitLine(points, length);
  if (line) return line;

  const bounds = new THREE.Box3().setFromPoints(points);
  const scale = bounds.getSize(new THREE.Vector3()).length();
  if (
    scale < 0.28 ||
    points[0].distanceTo(points.at(-1)!) / scale > 0.32
  ) {
    return undefined;
  }
  const plane = projectionPlane(points);
  if (!plane) return undefined;

  const circle = fitCircle2D(plane.projected, length);
  const rectangle = fitRectangle2D(plane.projected, length);
  if (circle && (!rectangle || circle.error <= rectangle.error * 1.18)) {
    return {
      kind: "circle",
      points: circlePoints(circle, plane, plane.projected[0]),
    };
  }
  if (rectangle) {
    return {
      kind: rectangle.square ? "square" : "rectangle",
      points: rectanglePoints(rectangle, plane),
    };
  }
  return undefined;
}
