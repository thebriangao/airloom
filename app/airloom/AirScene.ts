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

export class AirScene {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
  private raycaster = new THREE.Raycaster();
  private renderer: THREE.WebGLRenderer;
  private artwork = new THREE.Group();
  private targetPosition = new THREE.Vector3();
  private targetRotation = new THREE.Vector2();
  private cursorCoreMaterial = new THREE.MeshStandardMaterial({
    color: "#111111",
    emissive: "#111111",
    emissiveIntensity: 0.4,
    roughness: 0.3,
    metalness: 0.04,
    transparent: true,
  });
  private cursorDepthMaterial = new THREE.MeshBasicMaterial({
    color: "#111111",
    wireframe: true,
    transparent: true,
    opacity: 0.9,
    depthTest: false,
  });
  private cursorAnchorMaterial = new THREE.MeshBasicMaterial({
    color: "#111111",
    wireframe: true,
    transparent: true,
    opacity: 0.28,
    depthTest: false,
  });
  private cursorLineMaterial = new THREE.LineDashedMaterial({
    color: "#111111",
    transparent: true,
    opacity: 0.7,
    dashSize: 0.08,
    gapSize: 0.05,
    depthTest: false,
  });
  private cursorGroup = new THREE.Group();
  private cursorCore = new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 14),
    this.cursorCoreMaterial,
  );
  private cursorDepthHalo = new THREE.Mesh(
    new THREE.SphereGeometry(1, 14, 12),
    this.cursorDepthMaterial,
  );
  private cursorAnchor = new THREE.Mesh(
    new THREE.SphereGeometry(1, 12, 10),
    this.cursorAnchorMaterial,
  );
  private cursorLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3(),
    ]),
    this.cursorLineMaterial,
  );
  private strokes: Stroke[] = [];
  private activeStroke?: Stroke;
  private previousEraserPoint?: THREE.Vector3;
  private frame = 0;
  private width = 1;
  private height = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.camera.position.set(0, 0, 7);

    const ambient = new THREE.AmbientLight("#ffffff", 2.2);
    const key = new THREE.DirectionalLight("#ffffff", 3.8);
    key.position.set(4, 5, 7);
    const rim = new THREE.PointLight("#d9d9d9", 18, 18);
    rim.position.set(-4, -2, 4);
    this.cursorGroup.add(this.cursorCore, this.cursorDepthHalo);
    this.cursorGroup.visible = false;
    this.cursorAnchor.visible = false;
    this.cursorLine.visible = false;
    this.cursorDepthHalo.renderOrder = 20;
    this.cursorAnchor.renderOrder = 19;
    this.cursorLine.renderOrder = 18;
    this.artwork.add(
      this.cursorGroup,
      this.cursorAnchor,
      this.cursorLine,
    );
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
    };
    loop();
  }

  resize(width: number, height: number) {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height, false);
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

  setBrushCursor(
    point: THREE.Vector3,
    radius: number,
    color: string,
    eraser: boolean,
    active: boolean,
    depth: number,
  ) {
    if (!point.toArray().every(Number.isFinite)) {
      this.hideBrushCursor();
      return;
    }

    const safeRadius = THREE.MathUtils.clamp(radius, 0.001, 0.72);
    const depthColor =
      depth > 0.065
        ? "#ff5a5f"
        : depth < -0.065
          ? "#2864dc"
          : "#111111";

    this.cursorGroup.visible = true;
    this.cursorGroup.position.copy(point);
    this.cursorCore.scale.setScalar(safeRadius);
    this.cursorDepthHalo.scale.setScalar(
      safeRadius * (active ? 1.42 : 1.24),
    );
    this.cursorCoreMaterial.color.set(eraser ? "#fbfbf8" : color);
    this.cursorCoreMaterial.emissive.set(eraser ? "#777777" : color);
    this.cursorCoreMaterial.opacity = eraser ? 0.3 : 0.88;
    this.cursorCoreMaterial.wireframe = eraser;
    this.cursorDepthMaterial.color.set(depthColor);
    this.cursorDepthMaterial.opacity = active ? 1 : 0.76;

    this.artwork.updateMatrixWorld(true);
    const worldPoint = point.clone().applyMatrix4(this.artwork.matrixWorld);
    const artworkOrigin = this.artwork.getWorldPosition(new THREE.Vector3());
    const worldAnchor = new THREE.Vector3(
      worldPoint.x,
      worldPoint.y,
      artworkOrigin.z,
    );
    const localAnchor = worldAnchor.applyMatrix4(
      this.artwork.matrixWorld.clone().invert(),
    );
    const showDepthGuide = Math.abs(depth) > 0.045;

    this.cursorAnchor.visible = showDepthGuide;
    this.cursorLine.visible = showDepthGuide;
    if (!showDepthGuide) return;

    this.cursorAnchor.position.copy(localAnchor);
    this.cursorAnchor.scale.setScalar(safeRadius * 0.82);
    this.cursorAnchorMaterial.color.set(depthColor);
    this.cursorLineMaterial.color.set(depthColor);
    const positions = this.cursorLine.geometry.getAttribute(
      "position",
    ) as THREE.BufferAttribute;
    positions.setXYZ(0, point.x, point.y, point.z);
    positions.setXYZ(1, localAnchor.x, localAnchor.y, localAnchor.z);
    positions.needsUpdate = true;
    this.cursorLine.geometry.computeBoundingSphere();
    this.cursorLine.computeLineDistances();
  }

  hideBrushCursor() {
    this.cursorGroup.visible = false;
    this.cursorAnchor.visible = false;
    this.cursorLine.visible = false;
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
    stroke.mesh.position.set(0, 0, 0);
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

  addPoint(point: THREE.Vector3, color: string, radius: number) {
    this.previousEraserPoint = undefined;
    if (!this.activeStroke) {
      this.activeStroke = this.createStroke([point], color, radius);
      this.strokes.push(this.activeStroke);
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
    this.endStroke();
    const stroke = this.strokes.pop();
    if (!stroke) return;
    this.removeStroke(stroke);
  }

  clear() {
    this.endStroke();
    for (const stroke of this.strokes) {
      this.removeStroke(stroke);
    }
    this.strokes = [];
    this.resetView();
  }

  resetView() {
    this.targetPosition.set(0, 0, 0);
    this.targetRotation.set(0, 0);
    this.artwork.position.set(0, 0, 0);
    this.artwork.rotation.set(0, 0, 0);
  }

  getCanvas() {
    const cursorVisible = this.cursorGroup.visible;
    const anchorVisible = this.cursorAnchor.visible;
    const lineVisible = this.cursorLine.visible;
    this.hideBrushCursor();
    this.renderer.render(this.scene, this.camera);
    this.cursorGroup.visible = cursorVisible;
    this.cursorAnchor.visible = anchorVisible;
    this.cursorLine.visible = lineVisible;
    return this.renderer.domElement;
  }

  dispose() {
    window.cancelAnimationFrame(this.frame);
    this.clear();
    this.cursorCore.geometry.dispose();
    this.cursorDepthHalo.geometry.dispose();
    this.cursorAnchor.geometry.dispose();
    this.cursorLine.geometry.dispose();
    this.cursorCoreMaterial.dispose();
    this.cursorDepthMaterial.dispose();
    this.cursorAnchorMaterial.dispose();
    this.cursorLineMaterial.dispose();
    this.renderer.dispose();
  }
}
