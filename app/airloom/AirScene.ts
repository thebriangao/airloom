import * as THREE from "three";
import type { Point3 } from "./types";

type Stroke = {
  points: THREE.Vector3[];
  color: string;
  radius: number;
  mesh: THREE.Mesh;
  startCap?: THREE.Mesh;
  endCap?: THREE.Mesh;
};

export type SnapKind = "none" | "vertex" | "edge";

type GrabState = {
  strokes: Stroke[];
  originalPoints: Map<Stroke, THREE.Vector3[]>;
  anchor: THREE.Vector3;
  worldDepth: number;
};

type SnapResult = {
  translation: THREE.Vector3;
  kind: SnapKind;
};

export function projectNormalizedPointToArtwork(
  camera: THREE.PerspectiveCamera,
  artwork: THREE.Object3D,
  raycaster: THREE.Raycaster,
  point: Point3,
  depth: number,
) {
  artwork.updateMatrixWorld(true);
  raycaster.setFromCamera(
    new THREE.Vector2(
      1 - THREE.MathUtils.clamp(point.x, 0.018, 0.982) * 2,
      1 - THREE.MathUtils.clamp(point.y, 0.018, 0.982) * 2,
    ),
    camera,
  );

  const targetWorldZ =
    artwork.getWorldPosition(new THREE.Vector3()).z +
    THREE.MathUtils.clamp(depth, -2.15, 2.15);
  if (Math.abs(raycaster.ray.direction.z) < 0.0001) return undefined;
  const distanceToDepth =
    (targetWorldZ - raycaster.ray.origin.z) / raycaster.ray.direction.z;
  if (distanceToDepth < 0) return undefined;

  const worldPoint = raycaster.ray.origin
    .clone()
    .add(raycaster.ray.direction.clone().multiplyScalar(distanceToDepth));
  const inverseArtworkMatrix = artwork.matrixWorld.clone().invert();
  return worldPoint.applyMatrix4(inverseArtworkMatrix);
}

export function splitStrokeOutsideEraser(
  points: THREE.Vector3[],
  eraserPoint: THREE.Vector3,
  radius: number,
  eraserStart = eraserPoint,
) {
  if (points.length === 0) return [];

  const sampleStep = THREE.MathUtils.clamp(radius * 0.42, 0.025, 0.1);
  const sampledPoints: THREE.Vector3[] = [points[0]];
  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1];
    const end = points[index];
    const steps = Math.min(
      18,
      Math.max(1, Math.ceil(start.distanceTo(end) / sampleStep)),
    );
    for (let step = 1; step <= steps; step += 1) {
      sampledPoints.push(start.clone().lerp(end, step / steps));
    }
  }

  const eraserPath = new THREE.Line3(eraserStart, eraserPoint);
  const closestPoint = new THREE.Vector3();
  const eraserMoved =
    eraserStart.distanceToSquared(eraserPoint) > Number.EPSILON;
  const outside = sampledPoints.map((point) => {
    if (!eraserMoved) {
      return point.distanceToSquared(eraserPoint) > radius * radius;
    }
    eraserPath.closestPointToPoint(point, true, closestPoint);
    return point.distanceToSquared(closestPoint) > radius * radius;
  });
  if (outside.every(Boolean)) return [points];

  const runs: THREE.Vector3[][] = [];
  let run: THREE.Vector3[] = [];
  for (let index = 0; index < sampledPoints.length; index += 1) {
    const point = sampledPoints[index];
    if (outside[index]) {
      run.push(point);
    } else if (run.length > 0) {
      runs.push(run);
      run = [];
    }
  }
  if (run.length > 0) runs.push(run);
  return runs;
}

function featureVertices(points: THREE.Vector3[]) {
  if (points.length <= 2) return points.map((point) => point.clone());
  const vertices = [points[0].clone()];
  let lastVertex = points[0];
  const windowSize = Math.min(4, Math.max(1, Math.floor(points.length / 8)));

  for (
    let index = windowSize;
    index < points.length - windowSize;
    index += 1
  ) {
    const incoming = points[index]
      .clone()
      .sub(points[index - windowSize]);
    const outgoing = points[index + windowSize]
      .clone()
      .sub(points[index]);
    if (
      incoming.lengthSq() > 0.0001 &&
      outgoing.lengthSq() > 0.0001 &&
      incoming.angleTo(outgoing) > 0.38 &&
      points[index].distanceTo(lastVertex) > 0.14
    ) {
      vertices.push(points[index].clone());
      lastVertex = points[index];
    }
  }

  const end = points.at(-1)!;
  if (end.distanceTo(lastVertex) > 0.03) vertices.push(end.clone());
  return vertices;
}

function segmentsForPoints(
  points: THREE.Vector3[],
  translation = new THREE.Vector3(),
) {
  const segments: THREE.Line3[] = [];
  for (let index = 1; index < points.length; index += 1) {
    segments.push(
      new THREE.Line3(
        points[index - 1].clone().add(translation),
        points[index].clone().add(translation),
      ),
    );
  }
  return segments;
}

export function snapObjectTranslation(
  movingStrokes: THREE.Vector3[][],
  targetStrokes: THREE.Vector3[][],
  translation: THREE.Vector3,
  threshold = 0.28,
): SnapResult {
  const unsnapped = { translation: translation.clone(), kind: "none" as const };
  if (movingStrokes.length === 0 || targetStrokes.length === 0) {
    return unsnapped;
  }

  const movingVertices = movingStrokes.flatMap((points) =>
    featureVertices(points).map((point) => point.add(translation)),
  );
  const targetVertices = targetStrokes.flatMap(featureVertices);
  const movingSegments = movingStrokes.flatMap((points) =>
    segmentsForPoints(points, translation),
  );
  const targetSegments = targetStrokes.flatMap((points) =>
    segmentsForPoints(points),
  );

  let bestVertexDistance = Infinity;
  let bestVertexCorrection: THREE.Vector3 | undefined;
  for (const moving of movingVertices) {
    for (const target of targetVertices) {
      const distance = moving.distanceTo(target);
      if (distance >= bestVertexDistance) continue;
      bestVertexDistance = distance;
      bestVertexCorrection = target.clone().sub(moving);
    }
  }
  if (bestVertexCorrection && bestVertexDistance <= threshold * 0.82) {
    return {
      translation: translation.clone().add(bestVertexCorrection),
      kind: "vertex",
    };
  }

  let bestEdgeDistance = Infinity;
  let bestEdgeCorrection: THREE.Vector3 | undefined;
  const closest = new THREE.Vector3();
  for (const moving of movingVertices) {
    for (const targetEdge of targetSegments) {
      targetEdge.closestPointToPoint(moving, true, closest);
      const distance = moving.distanceTo(closest);
      if (distance >= bestEdgeDistance) continue;
      bestEdgeDistance = distance;
      bestEdgeCorrection = closest.clone().sub(moving);
    }
  }
  for (const target of targetVertices) {
    for (const movingEdge of movingSegments) {
      movingEdge.closestPointToPoint(target, true, closest);
      const distance = target.distanceTo(closest);
      if (distance >= bestEdgeDistance) continue;
      bestEdgeDistance = distance;
      bestEdgeCorrection = target.clone().sub(closest);
    }
  }
  if (bestEdgeCorrection && bestEdgeDistance <= threshold) {
    return {
      translation: translation.clone().add(bestEdgeCorrection),
      kind: "edge",
    };
  }
  return unsnapped;
}

export class AirScene {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
  private sideCamera = new THREE.OrthographicCamera(-2, 2, 1.4, -1.4, 0.1, 100);
  private raycaster = new THREE.Raycaster();
  private renderer: THREE.WebGLRenderer;
  private sideRenderer: THREE.WebGLRenderer;
  private artwork = new THREE.Group();
  private targetPosition = new THREE.Vector3();
  private targetRotation = new THREE.Vector2();
  private strokes: Stroke[] = [];
  private activeStroke?: Stroke;
  private previousEraserPoint?: THREE.Vector3;
  private grabState?: GrabState;
  private lastArtworkPresence = false;
  private onArtworkPresenceChange?: (hasArtwork: boolean) => void;
  private frame = 0;
  private width = 1;
  private height = 1;

  constructor(
    canvas: HTMLCanvasElement,
    sideCanvas: HTMLCanvasElement,
    onArtworkPresenceChange?: (hasArtwork: boolean) => void,
  ) {
    this.onArtworkPresenceChange = onArtworkPresenceChange;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.sideRenderer = new THREE.WebGLRenderer({
      canvas: sideCanvas,
      alpha: true,
      antialias: true,
    });
    this.sideRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.sideRenderer.outputColorSpace = THREE.SRGBColorSpace;
    this.sideRenderer.setClearColor(0xffffff, 0);
    this.camera.position.set(0, 0, 7);
    this.sideCamera.up.set(0, 1, 0);

    const ambient = new THREE.AmbientLight("#ffffff", 2.2);
    const key = new THREE.DirectionalLight("#ffffff", 3.8);
    key.position.set(4, 5, 7);
    const rim = new THREE.PointLight("#d9d9d9", 18, 18);
    rim.position.set(-4, -2, 4);
    this.scene.add(ambient, key, rim, this.artwork);

    const loop = () => {
      this.frame = window.requestAnimationFrame(loop);
      this.artwork.position.lerp(this.targetPosition, 0.14);
      this.artwork.rotation.x = THREE.MathUtils.lerp(
        this.artwork.rotation.x,
        this.targetRotation.x,
        0.12,
      );
      this.artwork.rotation.y = THREE.MathUtils.lerp(
        this.artwork.rotation.y,
        this.targetRotation.y,
        0.12,
      );
      this.renderer.render(this.scene, this.camera);
      if (this.strokes.length > 0) {
        this.updateSideCamera();
        this.sideRenderer.render(this.scene, this.sideCamera);
      }
    };
    loop();
  }

  resize(width: number, height: number) {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height, false);
    const sideCanvas = this.sideRenderer.domElement;
    this.sideRenderer.setSize(
      Math.max(1, sideCanvas.clientWidth || 260),
      Math.max(1, sideCanvas.clientHeight || 176),
      false,
    );
  }

  normalizedToArtwork(point: Point3, depth: number) {
    this.scene.updateMatrixWorld(true);
    return projectNormalizedPointToArtwork(
      this.camera,
      this.artwork,
      this.raycaster,
      point,
      depth,
    );
  }

  private updateSideCamera() {
    this.scene.updateMatrixWorld(true);
    const bounds = new THREE.Box3();
    for (const stroke of this.strokes) {
      bounds.expandByObject(stroke.mesh);
      if (stroke.startCap) bounds.expandByObject(stroke.startCap);
      if (stroke.endCap) bounds.expandByObject(stroke.endCap);
    }
    if (bounds.isEmpty()) return;

    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const canvas = this.sideRenderer.domElement;
    const aspect =
      Math.max(1, canvas.clientWidth) / Math.max(1, canvas.clientHeight);
    const halfHeight = Math.max(
      0.8,
      size.y * 0.64,
      (size.z / Math.max(0.5, aspect)) * 0.64,
    );
    const halfWidth = halfHeight * aspect;

    this.sideCamera.left = -halfWidth;
    this.sideCamera.right = halfWidth;
    this.sideCamera.top = halfHeight;
    this.sideCamera.bottom = -halfHeight;
    this.sideCamera.position.set(center.x + 12, center.y, center.z);
    this.sideCamera.lookAt(center);
    this.sideCamera.updateProjectionMatrix();
  }

  private notifyArtworkPresence() {
    const hasArtwork = this.strokes.length > 0;
    if (hasArtwork === this.lastArtworkPresence) return;
    this.lastArtworkPresence = hasArtwork;
    this.onArtworkPresenceChange?.(hasArtwork);
  }

  private geometryForPoints(points: THREE.Vector3[], radius: number) {
    if (points.length === 1) {
      return new THREE.SphereGeometry(radius, 10, 10);
    }
    const curvePoints =
      points.length === 2
        ? [points[0], points[0].clone().lerp(points[1], 0.5), points[1]]
        : points;
    const curve = new THREE.CatmullRomCurve3(curvePoints, false, "centripetal");
    return new THREE.TubeGeometry(
      curve,
      Math.min(420, Math.max(10, points.length * 3)),
      radius,
      8,
      false,
    );
  }

  private createStroke(points: THREE.Vector3[], color: string, radius: number) {
    const material = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.52,
      roughness: 0.32,
      metalness: 0.06,
    });
    const mesh = new THREE.Mesh(
      this.geometryForPoints(points, radius),
      material,
    );
    if (points.length === 1) mesh.position.copy(points[0]);
    this.artwork.add(mesh);
    const stroke: Stroke = {
      points: points.map((point) => point.clone()),
      color,
      radius,
      mesh,
    };
    this.updateStrokeCaps(stroke);
    return stroke;
  }

  private updateStrokeCaps(stroke: Stroke) {
    if (stroke.points.length < 2) {
      for (const cap of [stroke.startCap, stroke.endCap]) {
        if (!cap) continue;
        cap.geometry.dispose();
        this.artwork.remove(cap);
      }
      stroke.startCap = undefined;
      stroke.endCap = undefined;
      return;
    }

    const material = stroke.mesh.material as THREE.Material;
    if (!stroke.startCap) {
      stroke.startCap = new THREE.Mesh(
        new THREE.SphereGeometry(stroke.radius, 10, 10),
        material,
      );
      this.artwork.add(stroke.startCap);
    }
    if (!stroke.endCap) {
      stroke.endCap = new THREE.Mesh(
        new THREE.SphereGeometry(stroke.radius, 10, 10),
        material,
      );
      this.artwork.add(stroke.endCap);
    }
    stroke.startCap.position.copy(stroke.points[0]);
    stroke.endCap.position.copy(stroke.points.at(-1)!);
  }

  private rebuildStroke(stroke: Stroke) {
    stroke.mesh.geometry.dispose();
    stroke.mesh.geometry = this.geometryForPoints(stroke.points, stroke.radius);
    if (stroke.points.length === 1) {
      stroke.mesh.position.copy(stroke.points[0]);
    } else {
      stroke.mesh.position.set(0, 0, 0);
    }
    this.updateStrokeCaps(stroke);
  }

  private removeStroke(stroke: Stroke) {
    stroke.mesh.geometry.dispose();
    for (const cap of [stroke.startCap, stroke.endCap]) {
      if (!cap) continue;
      cap.geometry.dispose();
      this.artwork.remove(cap);
    }
    (stroke.mesh.material as THREE.Material).dispose();
    this.artwork.remove(stroke.mesh);
  }

  private strokesTouch(first: Stroke, second: Stroke) {
    const threshold = Math.max(
      0.16,
      first.radius + second.radius + 0.08,
    );
    const thresholdSquared = threshold * threshold;
    for (const firstPoint of first.points) {
      for (const secondPoint of second.points) {
        if (firstPoint.distanceToSquared(secondPoint) <= thresholdSquared) {
          return true;
        }
      }
    }
    return false;
  }

  private connectedObject(seed: Stroke) {
    const connected = new Set<Stroke>([seed]);
    const queue = [seed];
    while (queue.length > 0) {
      const current = queue.shift()!;
      for (const candidate of this.strokes) {
        if (
          connected.has(candidate) ||
          !this.strokesTouch(current, candidate)
        ) {
          continue;
        }
        connected.add(candidate);
        queue.push(candidate);
      }
    }
    return [...connected];
  }

  private setStrokeSelected(stroke: Stroke, selected: boolean) {
    const material = stroke.mesh.material as THREE.MeshStandardMaterial;
    material.emissiveIntensity = selected ? 1.45 : 0.52;
    material.roughness = selected ? 0.18 : 0.32;
  }

  private strokeUnderPoint(point: Point3) {
    if (this.strokes.length === 0) return undefined;
    const ndc = new THREE.Vector2(
      1 - THREE.MathUtils.clamp(point.x, 0.018, 0.982) * 2,
      1 - THREE.MathUtils.clamp(point.y, 0.018, 0.982) * 2,
    );
    this.scene.updateMatrixWorld(true);
    this.raycaster.setFromCamera(ndc, this.camera);
    const meshToStroke = new Map(
      this.strokes.map((stroke) => [stroke.mesh, stroke]),
    );
    const exactHit = this.raycaster.intersectObjects(
      [...meshToStroke.keys()],
      false,
    )[0];
    if (exactHit) {
      return {
        stroke: meshToStroke.get(exactHit.object as THREE.Mesh)!,
        worldPoint: exactHit.point,
      };
    }

    let nearest:
      | { stroke: Stroke; worldPoint: THREE.Vector3; distance: number }
      | undefined;
    const aspect = this.width / Math.max(1, this.height);
    for (const stroke of this.strokes) {
      for (const localPoint of stroke.points) {
        const worldPoint = localPoint
          .clone()
          .applyMatrix4(this.artwork.matrixWorld);
        const projected = worldPoint.clone().project(this.camera);
        const screenDistance = Math.hypot(
          (projected.x - ndc.x) * aspect,
          projected.y - ndc.y,
        );
        if (screenDistance > 0.075 || screenDistance >= (nearest?.distance ?? Infinity)) {
          continue;
        }
        nearest = { stroke, worldPoint, distance: screenDistance };
      }
    }
    return nearest;
  }

  beginObjectGrab(point: Point3) {
    this.endStroke();
    this.endObjectGrab();
    const hit = this.strokeUnderPoint(point);
    if (!hit) return false;

    const strokes = this.connectedObject(hit.stroke);
    const inverseArtwork = this.artwork.matrixWorld.clone().invert();
    const anchor = hit.worldPoint.clone().applyMatrix4(inverseArtwork);
    const artworkOrigin = this.artwork.getWorldPosition(new THREE.Vector3());
    this.grabState = {
      strokes,
      originalPoints: new Map(
        strokes.map((stroke) => [
          stroke,
          stroke.points.map((strokePoint) => strokePoint.clone()),
        ]),
      ),
      anchor,
      worldDepth: hit.worldPoint.z - artworkOrigin.z,
    };
    for (const stroke of strokes) this.setStrokeSelected(stroke, true);
    return true;
  }

  moveObjectGrab(point: Point3, depthDelta: number): SnapKind {
    const grab = this.grabState;
    if (!grab) return "none";
    const cursorPoint = this.normalizedToArtwork(
      point,
      grab.worldDepth + THREE.MathUtils.clamp(depthDelta, -2.15, 2.15),
    );
    if (!cursorPoint) return "none";

    const rawTranslation = cursorPoint.clone().sub(grab.anchor);
    const movingPoints = grab.strokes.map(
      (stroke) => grab.originalPoints.get(stroke)!,
    );
    const selected = new Set(grab.strokes);
    const targetPoints = this.strokes
      .filter((stroke) => !selected.has(stroke))
      .map((stroke) => stroke.points);
    const snap = snapObjectTranslation(
      movingPoints,
      targetPoints,
      rawTranslation,
    );

    for (const stroke of grab.strokes) {
      stroke.points = grab.originalPoints
        .get(stroke)!
        .map((strokePoint) => strokePoint.clone().add(snap.translation));
      this.rebuildStroke(stroke);
    }
    return snap.kind;
  }

  endObjectGrab() {
    if (!this.grabState) return;
    for (const stroke of this.grabState.strokes) {
      this.setStrokeSelected(stroke, false);
    }
    this.grabState = undefined;
  }

  addPoint(point: THREE.Vector3, color: string, radius: number) {
    this.previousEraserPoint = undefined;
    if (!this.activeStroke) {
      this.activeStroke = this.createStroke([point], color, radius);
      this.strokes.push(this.activeStroke);
      this.notifyArtworkPresence();
      return;
    }

    const previous = this.activeStroke.points.at(-1)!;
    const distance = point.distanceTo(previous);
    if (!Number.isFinite(distance)) return;

    const target =
      distance > 0.32
        ? previous
            .clone()
            .add(point.clone().sub(previous).setLength(0.32))
        : point;
    const smoothed = previous.clone().lerp(target, 0.46);
    if (smoothed.distanceTo(previous) < 0.014) return;

    this.activeStroke.points.push(smoothed);
    this.rebuildStroke(this.activeStroke);
  }

  eraseAt(point: THREE.Vector3, radius: number) {
    this.activeStroke = undefined;
    const previous = this.previousEraserPoint;
    const eraserStart =
      previous && previous.distanceTo(point) <= radius * 4.5
        ? previous
        : point;
    let erased = false;

    for (const stroke of [...this.strokes]) {
      const eraserRadius = radius + stroke.radius * 1.15;
      const runs = splitStrokeOutsideEraser(
        stroke.points,
        point,
        eraserRadius,
        eraserStart,
      );
      if (runs.length === 1 && runs[0] === stroke.points) continue;

      erased = true;
      this.strokes = this.strokes.filter((candidate) => candidate !== stroke);
      this.removeStroke(stroke);

      for (const survivingPoints of runs) {
        const survivingStroke = this.createStroke(
          survivingPoints,
          stroke.color,
          stroke.radius,
        );
        this.strokes.push(survivingStroke);
      }
    }

    this.previousEraserPoint = point.clone();
    if (erased) this.notifyArtworkPresence();
    return erased;
  }

  endStroke() {
    this.activeStroke = undefined;
    this.previousEraserPoint = undefined;
  }

  pan(deltaX: number, deltaY: number) {
    this.targetPosition.x += deltaX * 8;
    this.targetPosition.y -= deltaY * 8;
  }

  orbit(deltaX: number, deltaY: number, deltaDepth: number) {
    this.targetRotation.y -= deltaX * 5.8;
    this.targetRotation.x = THREE.MathUtils.clamp(
      this.targetRotation.x - deltaY * 5.8,
      -1.45,
      1.45,
    );
    this.targetPosition.z = THREE.MathUtils.clamp(
      this.targetPosition.z + deltaDepth * 34,
      -4.6,
      4.6,
    );
  }

  undo() {
    this.endObjectGrab();
    this.endStroke();
    const stroke = this.strokes.pop();
    if (!stroke) return;
    this.removeStroke(stroke);
    this.notifyArtworkPresence();
  }

  clear() {
    this.endObjectGrab();
    this.endStroke();
    for (const stroke of this.strokes) {
      this.removeStroke(stroke);
    }
    this.strokes = [];
    this.notifyArtworkPresence();
    this.resetView();
  }

  resetView() {
    this.endObjectGrab();
    this.targetPosition.set(0, 0, 0);
    this.targetRotation.set(0, 0);
    this.artwork.position.set(0, 0, 0);
    this.artwork.rotation.set(0, 0, 0);
  }

  getCanvas() {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement;
  }

  dispose() {
    window.cancelAnimationFrame(this.frame);
    this.onArtworkPresenceChange = undefined;
    this.clear();
    this.renderer.dispose();
    this.sideRenderer.dispose();
  }
}
