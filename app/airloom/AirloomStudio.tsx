"use client";

import type { HandLandmarker } from "@mediapipe/tasks-vision";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { AirScene } from "./AirScene";
import { GestureEngine } from "./gestureEngine";
import {
  AIRLOOM_COLORS,
  AIRLOOM_SIZES,
  type HandPose,
  type Landmark,
} from "./types";

type CameraState = "idle" | "loading" | "active" | "error";

const POSE_LABELS: Record<HandPose, string> = {
  none: "Show your hand",
  draw: "Drawing in 3D",
  pan2d: "Panning canvas",
  orbit3d: "Orbiting canvas",
  fist: "Fist detected",
  openPalm: "Open palm",
  other: "Gesture not assigned",
};

const clampIndex = (value: number, length: number) =>
  Math.max(0, Math.min(length - 1, Math.floor(value * length)));

export function AirloomStudio() {
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<AirScene | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const gestureEngineRef = useRef(new GestureEngine());
  const gestureRef = useRef<HandPose>("none");
  const previousControlRef = useRef<{
    pose: HandPose;
    x: number;
    y: number;
    scale: number;
  } | null>(null);
  const depthBaselineRef = useRef<number | null>(null);
  const menuOpenRef = useRef(false);
  const colorIndexRef = useRef(0);
  const sizeIndexRef = useRef(2);
  const pointerDrawingRef = useRef(false);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState("");
  const [gesture, setGesture] = useState<HandPose>("none");
  const [menuOpen, setMenuOpen] = useState(false);
  const [colorIndex, setColorIndex] = useState(0);
  const [sizeIndex, setSizeIndex] = useState(2);
  const [demoMode, setDemoMode] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  const selectColor = useCallback((index: number) => {
    colorIndexRef.current = index;
    setColorIndex(index);
  }, []);

  const selectSize = useCallback((index: number) => {
    sizeIndexRef.current = index;
    setSizeIndex(index);
  }, []);

  const toggleMenu = useCallback(() => {
    const next = !menuOpenRef.current;
    menuOpenRef.current = next;
    setMenuOpen(next);
    sceneRef.current?.endStroke();
    previousControlRef.current = null;
  }, []);

  const updateGesture = useCallback((next: HandPose) => {
    if (gestureRef.current === next) return;
    gestureRef.current = next;
    setGesture(next);
  }, []);

  useEffect(() => {
    if (!canvasRef.current || !stageRef.current) return;

    const scene = new AirScene(canvasRef.current);
    sceneRef.current = scene;
    const resize = () => {
      if (!stageRef.current) return;
      scene.resize(
        stageRef.current.clientWidth,
        stageRef.current.clientHeight,
      );
    };
    const observer = new ResizeObserver(resize);
    observer.observe(stageRef.current);
    resize();

    return () => {
      observer.disconnect();
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === "m") toggleMenu();
      if (event.key.toLowerCase() === "z") sceneRef.current?.undo();
      if (event.key === "Escape" && menuOpenRef.current) toggleMenu();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [toggleMenu]);

  const processHand = useCallback(
    (landmarks: Landmark[], timestamp: number) => {
      const result = gestureEngineRef.current.update(landmarks, timestamp);

      if (cursorRef.current) {
        cursorRef.current.style.opacity = "1";
        cursorRef.current.style.left = `${(1 - result.indexTip.x) * 100}%`;
        cursorRef.current.style.top = `${result.indexTip.y * 100}%`;
      }

      if (result.snap) {
        toggleMenu();
        return;
      }

      updateGesture(result.pose);

      if (menuOpenRef.current) {
        sceneRef.current?.endStroke();
        previousControlRef.current = null;
        if (result.pose === "fist") {
          selectColor(clampIndex(1 - result.palm.x, AIRLOOM_COLORS.length));
        } else if (result.pose === "openPalm") {
          selectSize(clampIndex(1 - result.palm.x, AIRLOOM_SIZES.length));
        }
        return;
      }

      const previous = previousControlRef.current;
      if (result.pose === "draw") {
        if (previous?.pose !== "draw") {
          depthBaselineRef.current = result.handScale;
          sceneRef.current?.endStroke();
        }
        const baseline = depthBaselineRef.current ?? result.handScale;
        const depth = (result.handScale - baseline) * 19;
        const point = sceneRef.current?.normalizedToWorld(result.indexTip, depth);
        if (point) {
          sceneRef.current?.addPoint(
            point,
            AIRLOOM_COLORS[colorIndexRef.current].value,
            AIRLOOM_SIZES[sizeIndexRef.current].value,
          );
        }
      } else {
        sceneRef.current?.endStroke();
      }

      if (previous?.pose === result.pose && result.pose === "pan2d") {
        sceneRef.current?.pan(
          -(result.palm.x - previous.x),
          result.palm.y - previous.y,
        );
      }

      if (previous?.pose === result.pose && result.pose === "orbit3d") {
        sceneRef.current?.orbit(
          result.palm.x - previous.x,
          result.palm.y - previous.y,
          result.handScale - previous.scale,
        );
      }

      previousControlRef.current = {
        pose: result.pose,
        x: result.palm.x,
        y: result.palm.y,
        scale: result.handScale,
      };
    },
    [selectColor, selectSize, toggleMenu, updateGesture],
  );

  const stopCamera = useCallback(() => {
    window.cancelAnimationFrame(animationRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    handLandmarkerRef.current?.close();
    handLandmarkerRef.current = null;
    gestureEngineRef.current.reset();
    previousControlRef.current = null;
    if (cursorRef.current) cursorRef.current.style.opacity = "0";
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const startTrackingLoop = useCallback(() => {
    const loop = () => {
      const video = videoRef.current;
      const landmarker = handLandmarkerRef.current;

      if (
        video &&
        landmarker &&
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        video.currentTime !== lastVideoTimeRef.current
      ) {
        lastVideoTimeRef.current = video.currentTime;
        const result = landmarker.detectForVideo(video, performance.now());
        const landmarks = result.landmarks[0] as Landmark[] | undefined;

        if (landmarks) {
          processHand(landmarks, performance.now());
        } else {
          gestureEngineRef.current.reset();
          sceneRef.current?.endStroke();
          previousControlRef.current = null;
          updateGesture("none");
          if (cursorRef.current) cursorRef.current.style.opacity = "0";
        }
      }

      animationRef.current = window.requestAnimationFrame(loop);
    };
    animationRef.current = window.requestAnimationFrame(loop);
  }, [processHand, updateGesture]);

  const startCamera = async () => {
    setCameraState("loading");
    setCameraError("");
    setDemoMode(false);

    try {
      const { FilesetResolver, HandLandmarker } = await import(
        "@mediapipe/tasks-vision"
      );
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;

      if (!videoRef.current) throw new Error("Camera surface unavailable.");
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/wasm",
      );

      try {
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.58,
            minHandPresenceConfidence: 0.55,
            minTrackingConfidence: 0.55,
          },
        );
      } catch {
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
              delegate: "CPU",
            },
            runningMode: "VIDEO",
            numHands: 1,
          },
        );
      }

      lastVideoTimeRef.current = -1;
      setCameraState("active");
      startTrackingLoop();
    } catch (error) {
      stopCamera();
      setCameraState("error");
      setCameraError(
        error instanceof Error
          ? error.message
          : "Airloom could not start the camera.",
      );
    }
  };

  const startDemo = () => {
    stopCamera();
    setDemoMode(true);
    setCameraState("idle");
    setCameraError("");
    updateGesture("draw");
  };

  const pointerPosition = (event: ReactPointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / rect.width,
      y: (event.clientY - rect.top) / rect.height,
      z: 0,
    };
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!demoMode || menuOpenRef.current) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    pointerDrawingRef.current = true;
    const pointer = pointerPosition(event);
    const point = sceneRef.current?.normalizedToWorld(
      { ...pointer, x: 1 - pointer.x },
      0,
    );
    if (point) {
      sceneRef.current?.addPoint(
        point,
        AIRLOOM_COLORS[colorIndexRef.current].value,
        AIRLOOM_SIZES[sizeIndexRef.current].value,
      );
    }
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!pointerDrawingRef.current || !demoMode) return;
    const pointer = pointerPosition(event);
    const point = sceneRef.current?.normalizedToWorld(
      { ...pointer, x: 1 - pointer.x },
      0,
    );
    if (point) {
      sceneRef.current?.addPoint(
        point,
        AIRLOOM_COLORS[colorIndexRef.current].value,
        AIRLOOM_SIZES[sizeIndexRef.current].value,
      );
    }
  };

  const handlePointerUp = () => {
    pointerDrawingRef.current = false;
    sceneRef.current?.endStroke();
  };

  const exportArtwork = async () => {
    const stage = stageRef.current;
    const sceneCanvas = sceneRef.current?.getCanvas();
    if (!stage || !sceneCanvas) return;

    const width = Math.max(1280, stage.clientWidth);
    const height = Math.round(width * (stage.clientHeight / stage.clientWidth));
    const output = document.createElement("canvas");
    output.width = width;
    output.height = height;
    const context = output.getContext("2d");
    if (!context) return;

    const background = context.createLinearGradient(0, 0, width, height);
    background.addColorStop(0, "#101719");
    background.addColorStop(1, "#050606");
    context.fillStyle = background;
    context.fillRect(0, 0, width, height);

    if (videoRef.current && cameraState === "active") {
      context.save();
      context.translate(width, 0);
      context.scale(-1, 1);
      context.globalAlpha = 0.6;
      context.drawImage(videoRef.current, 0, 0, width, height);
      context.restore();
    }

    context.drawImage(sceneCanvas, 0, 0, width, height);
    const blob = await new Promise<Blob | null>((resolve) =>
      output.toBlob(resolve, "image/png"),
    );
    if (!blob) return;

    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `airloom-${new Date().toISOString().slice(0, 10)}.png`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const selectedColor = AIRLOOM_COLORS[colorIndex];
  const selectedSize = AIRLOOM_SIZES[sizeIndex];
  const isStarted = cameraState === "active" || demoMode;

  return (
    <main className="studio-shell">
      <header className="studio-header">
        <a className="wordmark" href="#" aria-label="Airloom home">
          <span className="wordmark-mark">A</span>
          <span>
            AIRLOOM
            <small>DRAW IN THE SPACE BETWEEN</small>
          </span>
        </a>

        <div className="header-status" aria-live="polite">
          <span
            className={`status-light ${cameraState === "active" ? "is-live" : ""}`}
          />
          {demoMode
            ? "Mouse demo"
            : cameraState === "active"
              ? "Hand tracking live"
              : "Camera offline"}
        </div>

        <button
          className="quiet-button"
          onClick={() => setHelpOpen((value) => !value)}
          aria-expanded={helpOpen}
        >
          How to move
        </button>
      </header>

      <section
        ref={stageRef}
        className={`studio-stage ${demoMode ? "is-demo" : ""}`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      >
        <video
          ref={videoRef}
          className={`camera-feed ${cameraState === "active" ? "is-visible" : ""}`}
          muted
          playsInline
        />
        <div className="camera-wash" />
        <canvas ref={canvasRef} className="air-canvas" />
        <div
          ref={cursorRef}
          className="finger-cursor"
          style={{
            color: selectedColor.value,
            width: `${18 + sizeIndex * 5}px`,
            height: `${18 + sizeIndex * 5}px`,
          }}
        />

        <div className="corner-frame corner-frame-a" />
        <div className="corner-frame corner-frame-b" />

        {isStarted && (
          <>
            <div className="gesture-readout">
              <span>{menuOpen ? "MENU MODE" : demoMode ? "MOUSE INPUT" : "GESTURE"}</span>
              <strong>{menuOpen ? "Choose your brush" : POSE_LABELS[gesture]}</strong>
            </div>

            <div className="brush-readout">
              <span
                className="brush-color"
                style={{ background: selectedColor.value }}
              />
              <div>
                <strong>{selectedColor.name}</strong>
                <span>{selectedSize.name} stroke</span>
              </div>
            </div>
          </>
        )}

        {!isStarted && (
          <div className="welcome-panel">
            <p className="eyebrow">YOUR HAND IS THE BRUSH</p>
            <h1>
              Paint beyond
              <br />
              the flat canvas.
            </h1>
            <p className="welcome-copy">
              Draw luminous strokes in three dimensions using one finger. Pan
              with two. Orbit with three. Snap to change your brush.
            </p>
            <div className="welcome-actions">
              <button
                className="primary-button"
                onClick={startCamera}
                disabled={cameraState === "loading"}
              >
                {cameraState === "loading"
                  ? "Preparing hand tracking…"
                  : "Enable camera"}
              </button>
              <button className="secondary-button" onClick={startDemo}>
                Try with a mouse
              </button>
            </div>
            <p className="privacy-note">
              Camera frames are processed on this device and are never uploaded.
            </p>
            {cameraError && <p className="error-note">{cameraError}</p>}
          </div>
        )}

        {helpOpen && (
          <aside className="help-card">
            <button
              className="help-close"
              onClick={() => setHelpOpen(false)}
              aria-label="Close gesture guide"
            >
              Close
            </button>
            <p className="eyebrow">GESTURE MAP</p>
            <h2>Move with intention.</h2>
            <ol>
              <li>
                <strong>1 finger</strong>
                <span>Draw a stroke</span>
              </li>
              <li>
                <strong>2 fingers</strong>
                <span>Pan in 2D</span>
              </li>
              <li>
                <strong>3 fingers</strong>
                <span>Orbit and zoom</span>
              </li>
              <li>
                <strong>Snap</strong>
                <span>Toggle brush menu</span>
              </li>
            </ol>
          </aside>
        )}

        {menuOpen && (
          <section
            className="brush-menu"
            role="dialog"
            aria-modal="true"
            aria-label="Brush menu"
          >
            <div className="menu-heading">
              <div>
                <p className="eyebrow">SNAP MENU</p>
                <h2>Choose your matter.</h2>
              </div>
              <button className="menu-close" onClick={toggleMenu}>
                Done
              </button>
            </div>

            <div className="menu-section">
              <div className="menu-instruction">
                <span>01</span>
                <p>
                  <strong>Make a fist</strong>
                  Move left or right to select color
                </p>
              </div>
              <div className="color-options">
                {AIRLOOM_COLORS.map((color, index) => (
                  <button
                    key={color.name}
                    className={`color-option ${index === colorIndex ? "is-selected" : ""}`}
                    onClick={() => selectColor(index)}
                    aria-label={`Select ${color.name}`}
                    aria-pressed={index === colorIndex}
                  >
                    <span style={{ background: color.value }} />
                    <small>{color.name}</small>
                  </button>
                ))}
              </div>
            </div>

            <div className="menu-section">
              <div className="menu-instruction">
                <span>02</span>
                <p>
                  <strong>Open your hand</strong>
                  Move left or right to select stroke size
                </p>
              </div>
              <div className="size-options">
                {AIRLOOM_SIZES.map((size, index) => (
                  <button
                    key={size.name}
                    className={`size-option ${index === sizeIndex ? "is-selected" : ""}`}
                    onClick={() => selectSize(index)}
                    aria-label={`Select ${size.name} stroke`}
                    aria-pressed={index === sizeIndex}
                  >
                    <span
                      style={{
                        width: `${7 + index * 5}px`,
                        height: `${7 + index * 5}px`,
                        background: selectedColor.value,
                      }}
                    />
                    <small>{size.name}</small>
                  </button>
                ))}
              </div>
            </div>
            <p className="menu-tip">
              Snap again, press M, or choose Done to return to painting.
            </p>
          </section>
        )}

        {isStarted && (
          <nav className="tool-rail" aria-label="Artwork controls">
            <button onClick={toggleMenu}>
              <span>Brush</span>
              <small>M</small>
            </button>
            <button onClick={() => sceneRef.current?.undo()}>
              <span>Undo</span>
              <small>Z</small>
            </button>
            <button onClick={() => sceneRef.current?.resetView()}>
              <span>Reset view</span>
            </button>
            <button onClick={() => sceneRef.current?.clear()}>
              <span>Clear</span>
            </button>
            <button className="export-button" onClick={exportArtwork}>
              <span>Export PNG</span>
            </button>
          </nav>
        )}

        {isStarted && !menuOpen && (
          <div className="gesture-strip" aria-hidden="true">
            <span className={gesture === "draw" ? "is-active" : ""}>
              <b>01</b> Draw
            </span>
            <span className={gesture === "pan2d" ? "is-active" : ""}>
              <b>02</b> Pan
            </span>
            <span className={gesture === "orbit3d" ? "is-active" : ""}>
              <b>03</b> Orbit
            </span>
            <span>
              <b>✦</b> Snap for brush
            </span>
          </div>
        )}
      </section>

      <footer className="studio-footer">
        <span>Experimental browser instrument</span>
        <span>MediaPipe + Three.js</span>
        <span>Camera data stays local</span>
      </footer>
    </main>
  );
}
