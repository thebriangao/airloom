import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";
import * as THREE from "three";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html", host: "airloom.test" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Airloom product shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Airloom \| Draw in the space between<\/title>/i);
  assert.match(html, /AIRLOOM/);
  assert.match(html, /SPACE IS THE CANVAS/);
  assert.match(html, /Enable camera/);
  assert.match(html, /Draw with your mouse now/);
  assert.match(html, /og:image/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("camera access is requested before loading hand tracking", async () => {
  const source = await readFile(
    new URL("../app/airloom/AirloomStudio.tsx", import.meta.url),
    "utf8",
  );
  const cameraRequest = source.indexOf(
    "await navigator.mediaDevices.getUserMedia",
  );
  const trackingImport = source.indexOf(
    'await import(\n        "@mediapipe/tasks-vision"',
  );

  assert.notEqual(cameraRequest, -1);
  assert.notEqual(trackingImport, -1);
  assert.ok(cameraRequest < trackingImport);
  assert.match(source, /if \(!context\) return;/);
  assert.match(source, /if \(event\.target !== event\.currentTarget\) return;/);
  assert.match(
    source,
    /className="camera-bubble"\s+onPointerDown=\{\(event\) => event\.stopPropagation\(\)\}/,
  );
});

test("projects cursor rays into transformed artwork space and splits erased strokes", async () => {
  const { projectNormalizedPointToArtwork, splitStrokeOutsideEraser } =
    await import("../app/airloom/AirScene.ts");
  const camera = new THREE.PerspectiveCamera(48, 1.4, 0.1, 100);
  camera.position.set(0, 0, 7);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);

  const artwork = new THREE.Group();
  artwork.position.set(0.8, -0.35, 0.4);
  artwork.rotation.set(0.28, -0.46, 0.08);
  artwork.updateMatrixWorld(true);

  const normalized = { x: 0.31, y: 0.42, z: 0 };
  const localPoint = projectNormalizedPointToArtwork(
    camera,
    artwork,
    new THREE.Raycaster(),
    normalized,
    0,
  );
  assert.ok(localPoint);

  const projected = localPoint
    .clone()
    .applyMatrix4(artwork.matrixWorld)
    .project(camera);
  assert.ok(Math.abs(projected.x - (1 - normalized.x * 2)) < 0.00001);
  assert.ok(Math.abs(projected.y - (1 - normalized.y * 2)) < 0.00001);

  const points = [0, 1, 2, 3, 4].map(
    (x) => new THREE.Vector3(x, 0, 0),
  );
  const runs = splitStrokeOutsideEraser(
    points,
    new THREE.Vector3(2, 0, 0),
    0.4,
  );
  assert.deepEqual(
    runs.map((run) => run.map((point) => point.x)),
    [
      [0, 1],
      [3, 4],
    ],
  );
});

test("keeps tracking, gestures, and rendering in separate modules", async () => {
  const [studio, gestures, scene, architecture, packageJson, styles] =
    await Promise.all([
      readFile(
        new URL("../app/airloom/AirloomStudio.tsx", import.meta.url),
        "utf8",
      ),
      readFile(
        new URL("../app/airloom/gestureEngine.ts", import.meta.url),
        "utf8",
      ),
      readFile(new URL("../app/airloom/AirScene.ts", import.meta.url), "utf8"),
      readFile(new URL("../ARCHITECTURE.md", import.meta.url), "utf8"),
      readFile(new URL("../package.json", import.meta.url), "utf8"),
      readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    ]);

  assert.match(studio, /HandLandmarker/);
  assert.match(studio, /result\.pose === "draw"/);
  assert.match(studio, /result\.pose === "pan2d"/);
  assert.match(studio, /result\.pose === "orbit3d"/);
  assert.match(studio, /menuOpenRef\.current/);
  assert.match(studio, /selectThickness/);
  assert.match(studio, /selectEraserThickness/);
  assert.match(studio, /AudioContext/);
  assert.match(studio, /confirmSnap/);
  assert.match(studio, /confirmClap/);
  assert.match(studio, /getFloatTimeDomainData/);
  assert.match(studio, /numHands: 2/);
  assert.match(studio, /audio: \{/);
  assert.match(studio, /eraserEnabled/);
  assert.match(studio, /brush-cartridge/);
  assert.match(studio, /cartridge-color-grid/);
  assert.match(studio, /type="range"/);
  assert.doesNotMatch(studio, /brush-dial|thickness-arc/);
  assert.match(gestures, /SNAP_COOLDOWN_MS/);
  assert.match(gestures, /fingerExtended/);
  assert.match(scene, /TubeGeometry/);
  assert.match(scene, /CatmullRomCurve3/);
  assert.match(scene, /normalizedToArtwork/);
  assert.match(scene, /inverseArtworkMatrix/);
  assert.match(scene, /eraseAt/);
  assert.match(architecture, /Gesture precedence/);
  assert.match(packageJson, /"@mediapipe\/tasks-vision"/);
  assert.match(packageJson, /"three"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.match(styles, /\.brush-cartridge \{[\s\S]*?translate3d/);
  assert.match(styles, /\.brush-cartridge \{[\s\S]*?will-change: transform/);
  assert.match(styles, /\.brush-cartridge \{[\s\S]*?border-radius: 18px;/);
  assert.match(styles, /\.cartridge-tab \{[\s\S]*?right: 6px;/);
  assert.match(styles, /\.brush-cartridge-shell\.is-eraser/);
  assert.doesNotMatch(styles, /0 34px 54px/);
  await access(new URL("../public/og-v2.png", import.meta.url));
});
