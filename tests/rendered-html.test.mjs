import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

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
});

test("keeps tracking, gestures, and rendering in separate modules", async () => {
  const [studio, gestures, scene, architecture, packageJson] =
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
    ]);

  assert.match(studio, /HandLandmarker/);
  assert.match(studio, /result\.pose === "draw"/);
  assert.match(studio, /result\.pose === "pan2d"/);
  assert.match(studio, /result\.pose === "orbit3d"/);
  assert.match(studio, /menuOpenRef\.current/);
  assert.match(studio, /selectThickness/);
  assert.match(studio, /AudioContext/);
  assert.match(studio, /dial-color-grid/);
  assert.match(gestures, /SNAP_COOLDOWN_MS/);
  assert.match(gestures, /fingerExtended/);
  assert.match(scene, /TubeGeometry/);
  assert.match(scene, /CatmullRomCurve3/);
  assert.match(architecture, /Gesture precedence/);
  assert.match(packageJson, /"@mediapipe\/tasks-vision"/);
  assert.match(packageJson, /"three"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await access(new URL("../public/og-v2.png", import.meta.url));
});
