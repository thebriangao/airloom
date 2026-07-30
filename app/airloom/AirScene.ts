import * as THREE from "three";
import type { Point3 } from "./types";

type Stroke = {
  points: THREE.Vector3[];
  color: string;
  radius: number;
  mesh: THREE.Mesh;
};

export class AirScene {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
  private renderer: THREE.WebGLRenderer;
  private artwork = new THREE.Group();
  private strokes: Stroke[] = [];
  private activeStroke?: Stroke;
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
    this.scene.add(ambient, key, rim, this.artwork);

    const loop = () => {
      this.frame = window.requestAnimationFrame(loop);
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

  normalizedToWorld(point: Point3, depth: number) {
    const ndc = new THREE.Vector3(1 - point.x * 2, 1 - point.y * 2, 0.1);
    ndc.unproject(this.camera);
    const direction = ndc.sub(this.camera.position).normalize();
    const targetZ = THREE.MathUtils.clamp(depth, -2.15, 2.15);
    const distanceToPlane =
      (targetZ - this.camera.position.z) / direction.z;
    return this.camera.position
      .clone()
      .add(direction.multiplyScalar(distanceToPlane));
  }

  addPoint(point: THREE.Vector3, color: string, radius: number) {
    if (!this.activeStroke) {
      const geometry = new THREE.SphereGeometry(radius, 10, 10);
      const material = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.52,
        roughness: 0.32,
        metalness: 0.06,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(point);
      this.artwork.add(mesh);
      this.activeStroke = {
        points: [point.clone()],
        color,
        radius,
        mesh,
      };
      this.strokes.push(this.activeStroke);
      return;
    }

    const previous = this.activeStroke.points.at(-1)!;
    const smoothed = previous.clone().lerp(point, 0.38);
    if (smoothed.distanceTo(previous) < 0.018) return;

    this.activeStroke.points.push(smoothed);
    const points = this.activeStroke.points;
    const curvePoints =
      points.length === 2
        ? [points[0], points[0].clone().lerp(points[1], 0.5), points[1]]
        : points;
    const curve = new THREE.CatmullRomCurve3(curvePoints, false, "centripetal");
    const geometry = new THREE.TubeGeometry(
      curve,
      Math.min(420, Math.max(10, points.length * 3)),
      this.activeStroke.radius,
      8,
      false,
    );
    this.activeStroke.mesh.geometry.dispose();
    this.activeStroke.mesh.geometry = geometry;
    this.activeStroke.mesh.position.set(0, 0, 0);
  }

  endStroke() {
    this.activeStroke = undefined;
  }

  pan(deltaX: number, deltaY: number) {
    this.artwork.position.x += deltaX * 8;
    this.artwork.position.y -= deltaY * 8;
  }

  orbit(deltaX: number, deltaY: number, deltaDepth: number) {
    this.artwork.rotation.y -= deltaX * 5.5;
    this.artwork.rotation.x -= deltaY * 5.5;
    this.artwork.position.z = THREE.MathUtils.clamp(
      this.artwork.position.z + deltaDepth * 12,
      -3.2,
      3.2,
    );
  }

  undo() {
    this.endStroke();
    const stroke = this.strokes.pop();
    if (!stroke) return;
    stroke.mesh.geometry.dispose();
    (stroke.mesh.material as THREE.Material).dispose();
    this.artwork.remove(stroke.mesh);
  }

  clear() {
    this.endStroke();
    for (const stroke of this.strokes) {
      stroke.mesh.geometry.dispose();
      (stroke.mesh.material as THREE.Material).dispose();
      this.artwork.remove(stroke.mesh);
    }
    this.strokes = [];
    this.artwork.position.set(0, 0, 0);
    this.artwork.rotation.set(0, 0, 0);
  }

  resetView() {
    this.artwork.position.set(0, 0, 0);
    this.artwork.rotation.set(0, 0, 0);
  }

  getCanvas() {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement;
  }

  dispose() {
    window.cancelAnimationFrame(this.frame);
    this.clear();
    this.renderer.dispose();
  }
}
