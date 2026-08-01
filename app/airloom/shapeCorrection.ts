import * as THREE from "three";

export type CorrectedShapeKind =
  | "line"
  | "arc"
  | "circle"
  | "ellipse"
  | "triangle"
  | "square"
  | "rectangle"
  | "pentagon"
  | "hexagon";

export type ShapeCorrection = {
  kind: CorrectedShapeKind;
  points: THREE.Vector3[];
};

type Point2 = { x: number; y: number };
type ProjectionPlane = NonNullable<ReturnType<typeof projectionPlane>>;

const LINE_ERROR_LIMIT = 0.085;
const CURVE_ERROR_LIMIT = 0.065;
const RECTANGLE_ERROR_LIMIT = 0.06;
const POLYGON_ERROR_LIMIT = 0.05;
const CRAFT_BLEND = 0.14;
const CRAFT_OFFSET_LIMIT = 0.014;

const distance2 = (first: Point2, second: Point2) =>
  Math.hypot(first.x - second.x, first.y - second.y);

function pathLength3(points: THREE.Vector3[]) {
  let length = 0;
  for (let index = 1; index < points.length; index += 1) {
    length += points[index - 1].distanceTo(points[index]);
  }
  return length;
}

function pathLength2(points: Point2[]) {
  let length = 0;
  for (let index = 1; index < points.length; index += 1) {
    length += distance2(points[index - 1], points[index]);
  }
  return length;
}

function signedArea(points: Point2[]) {
  let area = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    area += current.x * next.y - next.x * current.y;
  }
  return area / 2;
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

function resamplePath(points: THREE.Vector3[], count: number) {
  if (points.length === 1) {
    return Array.from({ length: count }, () => points[0].clone());
  }
  const cumulative = [0];
  for (let index = 1; index < points.length; index += 1) {
    cumulative.push(
      cumulative[index - 1] + points[index - 1].distanceTo(points[index]),
    );
  }
  const total = cumulative.at(-1)!;
  if (total < 0.000001) {
    return Array.from({ length: count }, () => points[0].clone());
  }

  let segment = 1;
  return Array.from({ length: count }, (_, index) => {
    const target = (index / Math.max(1, count - 1)) * total;
    while (segment < cumulative.length - 1 && cumulative[segment] < target) {
      segment += 1;
    }
    const startDistance = cumulative[segment - 1];
    const endDistance = cumulative[segment];
    const progress =
      endDistance === startDistance
        ? 0
        : (target - startDistance) / (endDistance - startDistance);
    return points[segment - 1].clone().lerp(points[segment], progress);
  });
}

function addCraftCharacter(
  idealPoints: THREE.Vector3[],
  sourcePoints: THREE.Vector3[],
  closed: boolean,
) {
  const sampledSource = resamplePath(sourcePoints, idealPoints.length);
  const bounds = new THREE.Box3().setFromPoints(idealPoints);
  const scale = Math.max(0.001, bounds.getSize(new THREE.Vector3()).length());
  const rawOffsets = idealPoints.map((point, index) =>
    sampledSource[index].clone().sub(point).multiplyScalar(CRAFT_BLEND),
  );
  const offsets = rawOffsets.map((offset, index) => {
    const previous = rawOffsets[Math.max(0, index - 1)];
    const next = rawOffsets[Math.min(rawOffsets.length - 1, index + 1)];
    const smoothed = previous
      .clone()
      .addScaledVector(offset, 2)
      .add(next)
      .multiplyScalar(0.25);
    return smoothed.clampLength(0, scale * CRAFT_OFFSET_LIMIT);
  });
  const crafted = idealPoints.map((point, index) =>
    point.clone().add(offsets[index]),
  );
  if (closed && crafted.length > 1) {
    const closure = crafted[0].clone().add(crafted.at(-1)!).multiplyScalar(0.5);
    crafted[0] = closure;
    crafted[crafted.length - 1] = closure.clone();
  } else if (crafted.length > 1) {
    crafted[0] = idealPoints[0].clone();
    crafted[crafted.length - 1] = idealPoints.at(-1)!.clone();
  }
  return crafted;
}

function matchBoundsToSource(
  shapedPoints: THREE.Vector3[],
  sourcePoints: THREE.Vector3[],
  plane: ProjectionPlane,
) {
  const source = plane.projected;
  const shaped = shapedPoints.map((point) => {
    const offset = point.clone().sub(plane.center);
    return {
      x: offset.dot(plane.axisX),
      y: offset.dot(plane.axisY),
      depth: offset.dot(plane.normal),
    };
  });
  const sourceMinX = Math.min(...source.map((point) => point.x));
  const sourceMaxX = Math.max(...source.map((point) => point.x));
  const sourceMinY = Math.min(...source.map((point) => point.y));
  const sourceMaxY = Math.max(...source.map((point) => point.y));
  const shapedMinX = Math.min(...shaped.map((point) => point.x));
  const shapedMaxX = Math.max(...shaped.map((point) => point.x));
  const shapedMinY = Math.min(...shaped.map((point) => point.y));
  const shapedMaxY = Math.max(...shaped.map((point) => point.y));
  const scaleX =
    (sourceMaxX - sourceMinX) / Math.max(0.0001, shapedMaxX - shapedMinX);
  const scaleY =
    (sourceMaxY - sourceMinY) / Math.max(0.0001, shapedMaxY - shapedMinY);
  const sourceCenterX = (sourceMinX + sourceMaxX) / 2;
  const sourceCenterY = (sourceMinY + sourceMaxY) / 2;
  const shapedCenterX = (shapedMinX + shapedMaxX) / 2;
  const shapedCenterY = (shapedMinY + shapedMaxY) / 2;

  return shaped.map((point) =>
    plane.center
      .clone()
      .addScaledVector(
        plane.axisX,
        sourceCenterX + (point.x - shapedCenterX) * scaleX,
      )
      .addScaledVector(
        plane.axisY,
        sourceCenterY + (point.y - shapedCenterY) * scaleY,
      )
      .addScaledVector(plane.normal, point.depth),
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
    squaredError += offset.addScaledVector(direction, -projection).lengthSq();
  }

  const extent = maximum - minimum;
  if (extent < 0.28) return undefined;
  const directness = points[0].distanceTo(points.at(-1)!) / length;
  const normalizedError = Math.sqrt(squaredError / points.length) / extent;
  if (directness < 0.8 || normalizedError > LINE_ERROR_LIMIT) return undefined;

  let start = center.clone().addScaledVector(direction, minimum);
  let end = center.clone().addScaledVector(direction, maximum);
  if (start.distanceTo(points[0]) > end.distanceTo(points[0])) {
    [start, end] = [end, start];
  }
  const ideal = evenlySpacedLine(start, end);
  return {
    kind: "line" as const,
    points: addCraftCharacter(ideal, points, false),
  };
}

function projectionPlane(points: THREE.Vector3[]) {
  const center = points
    .reduce((sum, point) => sum.add(point), new THREE.Vector3())
    .multiplyScalar(1 / points.length);
  const normal = new THREE.Vector3();

  for (let index = 0; index < points.length - 1; index += 1) {
    const current = points[index].clone().sub(center);
    const next = points[index + 1].clone().sub(center);
    normal.add(current.cross(next));
  }
  normal.add(
    points.at(-1)!.clone().sub(center).cross(points[0].clone().sub(center)),
  );
  if (normal.lengthSq() < 0.000001) return undefined;
  normal.normalize();

  let axisX = points[0].clone().sub(center);
  axisX.addScaledVector(normal, -axisX.dot(normal));
  if (axisX.lengthSq() < 0.000001) {
    const reference =
      Math.abs(normal.y) < 0.9
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
  return { center, normal, axisX, axisY, projected };
}

function pointFromPlane(point: Point2, plane: ProjectionPlane) {
  return plane.center
    .clone()
    .addScaledVector(plane.axisX, point.x)
    .addScaledVector(plane.axisY, point.y);
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

function fitCircle2D(points: Point2[]) {
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
  const error =
    Math.sqrt(
      points.reduce((sum, point) => {
        const distance = Math.hypot(point.x - center.x, point.y - center.y);
        return sum + (distance - radius) ** 2;
      }, 0) / count,
    ) / radius;
  return { center, radius, error };
}

function fitEllipse2D(points: Point2[]) {
  let best:
    | {
        angle: number;
        center: Point2;
        radiusX: number;
        radiusY: number;
        error: number;
      }
    | undefined;

  for (let step = 0; step < 90; step += 1) {
    const angle = (step / 90) * Math.PI;
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
    const radiusX = (maxX - minX) / 2;
    const radiusY = (maxY - minY) / 2;
    if (Math.min(radiusX, radiusY) < 0.12) continue;
    const centerRotated = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
    const error = Math.sqrt(
      rotated.reduce((sum, point) => {
        const radial = Math.hypot(
          (point.x - centerRotated.x) / radiusX,
          (point.y - centerRotated.y) / radiusY,
        );
        return sum + (radial - 1) ** 2;
      }, 0) / points.length,
    );
    if (!best || error < best.error) {
      best = {
        angle,
        center: {
          x: centerRotated.x * cosine - centerRotated.y * sine,
          y: centerRotated.x * sine + centerRotated.y * cosine,
        },
        radiusX,
        radiusY,
        error,
      };
    }
  }
  return best;
}

function ellipsePoint(
  fit: NonNullable<ReturnType<typeof fitEllipse2D>>,
  angle: number,
) {
  const cosine = Math.cos(fit.angle);
  const sine = Math.sin(fit.angle);
  const x = Math.cos(angle) * fit.radiusX;
  const y = Math.sin(angle) * fit.radiusY;
  return {
    x: fit.center.x + x * cosine - y * sine,
    y: fit.center.y + x * sine + y * cosine,
  };
}

function angleOnEllipse(
  point: Point2,
  fit: NonNullable<ReturnType<typeof fitEllipse2D>>,
) {
  const cosine = Math.cos(fit.angle);
  const sine = Math.sin(fit.angle);
  const offsetX = point.x - fit.center.x;
  const offsetY = point.y - fit.center.y;
  const x = offsetX * cosine + offsetY * sine;
  const y = -offsetX * sine + offsetY * cosine;
  return Math.atan2(y / fit.radiusY, x / fit.radiusX);
}

function closedCurvePoints(
  fit: NonNullable<ReturnType<typeof fitEllipse2D>>,
  plane: ProjectionPlane,
  sourcePoints: THREE.Vector3[],
) {
  const direction = signedArea(plane.projected) >= 0 ? 1 : -1;
  const startAngle = angleOnEllipse(plane.projected[0], fit);
  const segments = 64;
  const ideal = Array.from({ length: segments + 1 }, (_, index) =>
    pointFromPlane(
      ellipsePoint(fit, startAngle + direction * (index / segments) * Math.PI * 2),
      plane,
    ),
  );
  return matchBoundsToSource(
    addCraftCharacter(ideal, sourcePoints, true),
    sourcePoints,
    plane,
  );
}

function unwrapSweep(points: Point2[], center: Point2) {
  const angles = points.map((point) =>
    Math.atan2(point.y - center.y, point.x - center.x),
  );
  let sweep = 0;
  let absoluteSweep = 0;
  for (let index = 1; index < angles.length; index += 1) {
    let delta = angles[index] - angles[index - 1];
    while (delta > Math.PI) delta -= Math.PI * 2;
    while (delta < -Math.PI) delta += Math.PI * 2;
    sweep += delta;
    absoluteSweep += Math.abs(delta);
  }
  return { start: angles[0], sweep, absoluteSweep };
}

function fitArc(
  points: THREE.Vector3[],
  plane: ProjectionPlane,
  length: number,
) {
  const circle = fitCircle2D(plane.projected);
  if (!circle || circle.error > CURVE_ERROR_LIMIT) return undefined;
  const { start, sweep, absoluteSweep } = unwrapSweep(
    plane.projected,
    circle.center,
  );
  const coverage = Math.abs(sweep) / (Math.PI * 2);
  const pathCoverage = length / (Math.PI * 2 * circle.radius);
  const directionConsistency = Math.abs(sweep) / Math.max(0.0001, absoluteSweep);
  if (
    coverage < 0.12 ||
    coverage > 0.86 ||
    Math.abs(pathCoverage - coverage) > 0.18 ||
    directionConsistency < 0.95
  ) {
    return undefined;
  }
  const segments = Math.max(18, Math.round(64 * coverage));
  const ideal = Array.from({ length: segments + 1 }, (_, index) =>
    pointFromPlane(
      {
        x: circle.center.x + Math.cos(start + sweep * (index / segments)) * circle.radius,
        y: circle.center.y + Math.sin(start + sweep * (index / segments)) * circle.radius,
      },
      plane,
    ),
  );
  return {
    kind: "arc" as const,
    points: matchBoundsToSource(
      addCraftCharacter(ideal, points, false),
      points,
      plane,
    ),
  };
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
    const edgeSequence: number[] = [];
    let squaredError = 0;
    for (const point of rotated) {
      const distances = [
        Math.abs(point.x - minX),
        Math.abs(maxX - point.x),
        Math.abs(point.y - minY),
        Math.abs(maxY - point.y),
      ];
      const distance = Math.min(...distances);
      const edgeIndex = distances.indexOf(distance);
      edgeCounts[edgeIndex] += 1;
      const cycleIndex = [3, 1, 0, 2][edgeIndex];
      if (edgeSequence.at(-1) !== cycleIndex) edgeSequence.push(cycleIndex);
      squaredError += distance * distance;
    }
    if (edgeCounts.some((count) => count < points.length * 0.04)) continue;
    if (edgeSequence.length > 1 && edgeSequence[0] === edgeSequence.at(-1)) {
      edgeSequence.pop();
    }
    if (edgeSequence.length < 4 || edgeSequence.length > 8) continue;
    let forwardTurns = 0;
    let backwardTurns = 0;
    let invalidTurns = 0;
    for (let index = 0; index < edgeSequence.length; index += 1) {
      const current = edgeSequence[index];
      const next = edgeSequence[(index + 1) % edgeSequence.length];
      const delta = (next - current + 4) % 4;
      if (delta === 1) forwardTurns += 1;
      else if (delta === 3) backwardTurns += 1;
      else if (delta !== 0) invalidTurns += 1;
    }
    if (invalidTurns > 0 || Math.min(forwardTurns, backwardTurns) > 1) continue;
    const error = Math.sqrt(squaredError / points.length) / diagonal;
    if (!best || error < best.error) {
      best = { angle, minX, maxX, minY, maxY, error };
    }
  }
  if (!best || best.error > RECTANGLE_ERROR_LIMIT) return undefined;
  const width = best.maxX - best.minX;
  const height = best.maxY - best.minY;
  const coverage = path / (2 * (width + height));
  if (coverage < 0.62 || coverage > 1.42) return undefined;
  return { ...best, square: width / height >= 0.78 && width / height <= 1.28 };
}

function edgePoints(corners: Point2[], plane: ProjectionPlane) {
  return corners.flatMap((corner, index) => {
    const next = corners[(index + 1) % corners.length];
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

function rectanglePoints(
  fit: NonNullable<ReturnType<typeof fitRectangle2D>>,
  plane: ProjectionPlane,
  sourcePoints: THREE.Vector3[],
) {
  const cosine = Math.cos(fit.angle);
  const sine = Math.sin(fit.angle);
  const unrotate = (point: Point2) => ({
    x: point.x * cosine - point.y * sine,
    y: point.x * sine + point.y * cosine,
  });
  let corners = [
    { x: fit.minX, y: fit.minY },
    { x: fit.maxX, y: fit.minY },
    { x: fit.maxX, y: fit.maxY },
    { x: fit.minX, y: fit.maxY },
  ].map(unrotate);
  if (signedArea(plane.projected) * signedArea(corners) < 0) {
    corners = corners.reverse();
  }
  let startIndex = 0;
  let bestDistance = Infinity;
  for (let index = 0; index < corners.length; index += 1) {
    const distance = distance2(corners[index], plane.projected[0]);
    if (distance < bestDistance) {
      bestDistance = distance;
      startIndex = index;
    }
  }
  corners = Array.from(
    { length: corners.length },
    (_, index) => corners[(startIndex + index) % corners.length],
  );
  return matchBoundsToSource(
    addCraftCharacter(edgePoints(corners, plane), sourcePoints, true),
    sourcePoints,
    plane,
  );
}

function pointToSegmentDistance(point: Point2, start: Point2, end: Point2) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 0.0000001) return distance2(point, start);
  const t = THREE.MathUtils.clamp(
    ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared,
    0,
    1,
  );
  return Math.hypot(point.x - (start.x + dx * t), point.y - (start.y + dy * t));
}

function simplifyPath(points: Point2[], epsilon: number): Point2[] {
  if (points.length <= 2) return points;
  let furthestIndex = 0;
  let furthestDistance = 0;
  for (let index = 1; index < points.length - 1; index += 1) {
    const distance = pointToSegmentDistance(
      points[index],
      points[0],
      points.at(-1)!,
    );
    if (distance > furthestDistance) {
      furthestDistance = distance;
      furthestIndex = index;
    }
  }
  if (furthestDistance <= epsilon) return [points[0], points.at(-1)!];
  const first = simplifyPath(points.slice(0, furthestIndex + 1), epsilon);
  const second = simplifyPath(points.slice(furthestIndex), epsilon);
  return [...first.slice(0, -1), ...second];
}

function fitPolygon2D(points: Point2[], scale: number) {
  const source = distance2(points[0], points.at(-1)!) < scale * 0.08
    ? points.slice(0, -1)
    : points;
  const center = source.reduce(
    (sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }),
    { x: 0, y: 0 },
  );
  center.x /= source.length;
  center.y /= source.length;
  let anchor = 0;
  let anchorDistance = 0;
  for (let index = 0; index < source.length; index += 1) {
    const distance = distance2(source[index], center);
    if (distance > anchorDistance) {
      anchorDistance = distance;
      anchor = index;
    }
  }
  const rotated = [...source.slice(anchor), ...source.slice(0, anchor)];
  const simplified = simplifyPath(
    [...rotated, rotated[0]],
    scale * 0.045,
  );
  const corners = simplified.slice(0, -1);
  while (corners.length > 3) {
    let shortestIndex = 0;
    let shortestLength = Infinity;
    for (let index = 0; index < corners.length; index += 1) {
      const edgeLength = distance2(
        corners[index],
        corners[(index + 1) % corners.length],
      );
      if (edgeLength < shortestLength) {
        shortestLength = edgeLength;
        shortestIndex = index;
      }
    }
    if (shortestLength >= scale * 0.18) break;
    const nextIndex = (shortestIndex + 1) % corners.length;
    const midpoint = {
      x: (corners[shortestIndex].x + corners[nextIndex].x) / 2,
      y: (corners[shortestIndex].y + corners[nextIndex].y) / 2,
    };
    if (nextIndex === 0) {
      corners[0] = midpoint;
      corners.pop();
    } else {
      corners[shortestIndex] = midpoint;
      corners.splice(nextIndex, 1);
    }
  }
  if (corners.length < 3 || corners.length > 6) return undefined;

  const shortestEdge = Math.min(
    ...corners.map((corner, index) =>
      distance2(corner, corners[(index + 1) % corners.length]),
    ),
  );
  if (shortestEdge < scale * 0.12) return undefined;
  const squaredError = points.reduce((sum, point) => {
    const distance = Math.min(
      ...corners.map((corner, index) =>
        pointToSegmentDistance(
          point,
          corner,
          corners[(index + 1) % corners.length],
        ),
      ),
    );
    return sum + distance * distance;
  }, 0);
  const error = Math.sqrt(squaredError / points.length) / scale;
  if (error > POLYGON_ERROR_LIMIT) return undefined;

  const perimeter = corners.reduce(
    (sum, corner, index) =>
      sum + distance2(corner, corners[(index + 1) % corners.length]),
    0,
  );
  const coverage = pathLength2(points) / perimeter;
  if (coverage < 0.72 || coverage > 1.32) return undefined;

  let turnSign = 0;
  for (let index = 0; index < corners.length; index += 1) {
    const previous = corners[(index - 1 + corners.length) % corners.length];
    const current = corners[index];
    const next = corners[(index + 1) % corners.length];
    const cross =
      (current.x - previous.x) * (next.y - current.y) -
      (current.y - previous.y) * (next.x - current.x);
    if (Math.abs(cross) < scale * scale * 0.015) return undefined;
    const sign = Math.sign(cross);
    if (turnSign !== 0 && sign !== turnSign) return undefined;
    turnSign = sign;
  }
  return corners;
}

function polygonKind(count: number): CorrectedShapeKind | undefined {
  if (count === 3) return "triangle";
  if (count === 5) return "pentagon";
  if (count === 6) return "hexagon";
  return undefined;
}

export function correctSimpleGeometry(
  sourcePoints: THREE.Vector3[],
): ShapeCorrection | undefined {
  if (sourcePoints.length < 5) return undefined;
  const points = sourcePoints.map((point) => point.clone());
  const length = pathLength3(points);
  if (length < 0.32) return undefined;

  const line = fitLine(points, length);
  if (line) return line;

  const bounds = new THREE.Box3().setFromPoints(points);
  const scale = bounds.getSize(new THREE.Vector3()).length();
  if (scale < 0.28) return undefined;
  const plane = projectionPlane(points);
  if (!plane) return undefined;
  const closed = points[0].distanceTo(points.at(-1)!) / scale <= 0.28;

  if (!closed) return fitArc(points, plane, length);

  const polygon = fitPolygon2D(plane.projected, scale);
  const polygonShape = polygonKind(polygon?.length ?? 0);
  if (polygon && polygonShape) {
    return {
      kind: polygonShape,
      points: matchBoundsToSource(
        addCraftCharacter(edgePoints(polygon, plane), points, true),
        points,
        plane,
      ),
    };
  }

  const ellipse = fitEllipse2D(plane.projected);
  if (ellipse && ellipse.error <= CURVE_ERROR_LIMIT) {
    const ratio = Math.max(ellipse.radiusX, ellipse.radiusY) /
      Math.min(ellipse.radiusX, ellipse.radiusY);
    return {
      kind: ratio <= 1.2 ? "circle" : "ellipse",
      points: closedCurvePoints(ellipse, plane, points),
    };
  }

  const rectangle = fitRectangle2D(plane.projected, length);
  if (rectangle) {
    return {
      kind: rectangle.square ? "square" : "rectangle",
      points: rectanglePoints(rectangle, plane, points),
    };
  }

  return undefined;
}
