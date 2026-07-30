"use client";

import type { HandLandmarker } from "@mediapipe/tasks-vision";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { AirScene } from "./AirScene";
import { GestureEngine } from "./gestureEngine";
import {
  AIRLOOM_COLORS,
  radiusFromThickness,
  type HandPose,
  type Landmark,
} from "./types";

type CameraState =
  | "idle"
  | "requesting"
  | "calibrating"
  | "active"
  | "error";
type SoundKind = "open" | "close" | "color" | "size";

const POSE_LABELS: Record<HandPose, string> = {
  none: "Show your hand",
  draw: "Drawing",
  pan2d: "Panning",
  orbit3d: "Orbiting",
  fist: "Choosing color",
  openPalm: "Adjusting thickness",
  other: "Reading gesture",
};

const clamp = (value: number, minimum = 0, maximum = 1) =>
  Math.max(minimum, Math.min(maximum, value));

const gridPosition = (value: number) => clamp((value - 0.12) / 0.76);

export function AirloomStudio() {
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<AirScene | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
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
  const thicknessRef = useRef(0.32);
  const lastSizeTickRef = useRef(4);
  const pointerDrawingRef = useRef(false);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState("");
  const [gesture, setGesture] = useState<HandPose>("none");
  const [menuOpen, setMenuOpen] = useState(false);
  const [colorIndex, setColorIndex] = useState(0);
  const [thickness, setThickness] = useState(0.32);
  const [demoMode, setDemoMode] = useState(true);
  const [helpOpen, setHelpOpen] = useState(false);

  const selectedColor = AIRLOOM_COLORS[colorIndex];

  const primeAudio = useCallback((): AudioContext | null => {
    try {
      const AudioContextConstructor =
        window.AudioContext ??
        (
          window as Window & {
            webkitAudioContext?: typeof AudioContext;
          }
        ).webkitAudioContext;
      if (!AudioContextConstructor) return null;
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContextConstructor();
      }
      if (audioContextRef.current.state === "suspended") {
        void audioContextRef.current.resume().catch(() => undefined);
      }
      return audioContextRef.current;
    } catch {
      return null;
    }
  }, []);

  const playSound = useCallback(
    (kind: SoundKind) => {
      const context = primeAudio();
      if (!context) return;
      const now = context.currentTime;
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const filter = context.createBiquadFilter();
      const frequencies: Record<SoundKind, [number, number]> = {
        open: [220, 520],
        close: [420, 180],
        color: [520, 610],
        size: [170, 220],
      };
      const [start, end] = frequencies[kind];

      oscillator.type = kind === "size" ? "triangle" : "sine";
      oscillator.frequency.setValueAtTime(start, now);
      oscillator.frequency.exponentialRampToValueAtTime(end, now + 0.075);
      filter.type = "lowpass";
      filter.frequency.setValueAtTime(kind === "open" ? 1800 : 1100, now);
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(
        kind === "open" ? 0.075 : 0.038,
        now + 0.012,
      );
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.11);
      oscillator.connect(filter);
      filter.connect(gain);
      gain.connect(context.destination);
      oscillator.start(now);
      oscillator.stop(now + 0.12);
    },
    [primeAudio],
  );

  const selectColor = useCallback(
    (index: number, audible = true) => {
      const next = clamp(index, 0, AIRLOOM_COLORS.length - 1);
      if (colorIndexRef.current === next) return;
      colorIndexRef.current = next;
      setColorIndex(next);
      if (audible) playSound("color");
    },
    [playSound],
  );

  const selectThickness = useCallback(
    (value: number, audible = true) => {
      const next = clamp(value);
      if (Math.abs(thicknessRef.current - next) < 0.003) return;
      thicknessRef.current = next;
      setThickness(next);
      const tick = Math.floor(next * 14);
      if (audible && tick !== lastSizeTickRef.current) {
        lastSizeTickRef.current = tick;
        playSound("size");
      }
    },
    [playSound],
  );

  const toggleMenu = useCallback(() => {
    const next = !menuOpenRef.current;
    menuOpenRef.current = next;
    setMenuOpen(next);
    sceneRef.current?.endStroke();
    previousControlRef.current = null;
    playSound(next ? "open" : "close");
  }, [playSound]);

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
          const column = Math.min(
            4,
            Math.floor(gridPosition(1 - result.palm.x) * 5),
          );
          const row = Math.min(
            4,
            Math.floor(gridPosition(result.palm.y) * 5),
          );
          selectColor(row * 5 + column);
        } else if (result.pose === "openPalm") {
          selectThickness(gridPosition(1 - result.palm.x));
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
            radiusFromThickness(thicknessRef.current),
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
    [selectColor, selectThickness, toggleMenu, updateGesture],
  );

  const stopCamera = useCallback(() => {
    window.cancelAnimationFrame(animationRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
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
    setCameraState("requesting");
    setCameraError("");
    setDemoMode(false);
    try {
      primeAudio();
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          "Camera access is unavailable here. Open Airloom in a current browser over HTTPS.",
        );
      }
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
      setCameraState("calibrating");

      const { FilesetResolver, HandLandmarker } = await import(
        "@mediapipe/tasks-vision"
      );

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
      setDemoMode(true);
      setCameraState("error");
      const errorName =
        error instanceof DOMException || error instanceof Error
          ? error.name
          : "";
      setCameraError(
        errorName === "NotAllowedError"
          ? "Camera access was blocked. Allow it in your browser's site settings, then try again."
          : errorName === "NotFoundError"
            ? "No camera was found on this device."
            : error instanceof Error
              ? error.message
              : "Airloom could not start the camera.",
      );
    }
  };

  const useMouse = () => {
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

  const handleCanvasPointerDown = (
    event: ReactPointerEvent<HTMLDivElement>,
  ) => {
    if (!demoMode || menuOpenRef.current) return;
    // The stage owns mouse drawing, but its floating controls are children.
    // Never capture a press that began on a button or another overlay.
    if (event.target !== event.currentTarget) return;
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
        radiusFromThickness(thicknessRef.current),
      );
    }
  };

  const handleCanvasPointerMove = (
    event: ReactPointerEvent<HTMLDivElement>,
  ) => {
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
        radiusFromThickness(thicknessRef.current),
      );
    }
  };

  const handleCanvasPointerUp = () => {
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
    context.fillStyle = "#fbfbf8";
    context.fillRect(0, 0, width, height);
    context.fillStyle = "#d8d8d4";
    const gap = 26 * (width / Math.max(1, stage.clientWidth));
    for (let x = gap; x < width; x += gap) {
      for (let y = gap; y < height; y += gap) {
        context.beginPath();
        context.arc(x, y, 1.1, 0, Math.PI * 2);
        context.fill();
      }
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

  const cameraHasVideo =
    cameraState === "calibrating" || cameraState === "active";
  const cartridgeStyle = {
    "--cartridge-color": selectedColor.value,
    "--thickness-position": `${thickness * 100}%`,
    "--marker-size": `${13 + thickness * 9}px`,
    "--stroke-height": `${2 + thickness * 20}px`,
  } as CSSProperties;

  return (
    <main className="airloom-app">
      <section
        ref={stageRef}
        className={`white-workspace ${demoMode ? "mouse-mode" : ""}`}
        onPointerDown={handleCanvasPointerDown}
        onPointerMove={handleCanvasPointerMove}
        onPointerUp={handleCanvasPointerUp}
        onPointerCancel={handleCanvasPointerUp}
      >
        <canvas ref={canvasRef} className="air-canvas" />
        <div
          ref={cursorRef}
          className="finger-cursor"
          style={{
            color: selectedColor.value,
            width: `${16 + thickness * 24}px`,
            height: `${16 + thickness * 24}px`,
          }}
        />

        <header className="floating-brand">
          <span className="brand-dot" />
          <div>
            <strong>AIRLOOM</strong>
            <small>SPACE IS THE CANVAS</small>
          </div>
        </header>

        <div className="gesture-pill" aria-live="polite">
          <span className={cameraState === "active" ? "live-dot" : "idle-dot"} />
          {cameraState === "requesting"
            ? "Waiting for camera permission"
            : cameraState === "calibrating"
              ? "Loading hand tracking"
              : demoMode
                ? "Mouse drawing"
                : POSE_LABELS[gesture]}
        </div>

        <aside
          className="camera-bubble"
          onPointerDown={(event) => event.stopPropagation()}
        >
          <div className="camera-bubble-depth" />
          <div className="camera-window">
            <video
              ref={videoRef}
              className={`camera-feed ${cameraHasVideo ? "is-visible" : ""}`}
              muted
              playsInline
            />
            {!cameraHasVideo && (
              <div className="camera-placeholder">
                <span>CAMERA 01</span>
                <strong>
                  {cameraState === "requesting"
                    ? "Allow camera access…"
                    : cameraState === "error"
                      ? "Camera needs attention"
                      : "Ready when you are"}
                </strong>
                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    void startCamera();
                  }}
                  disabled={cameraState === "requesting"}
                >
                  {cameraState === "error" ? "Try again" : "Enable camera"}
                </button>
                {!demoMode && (
                  <button className="camera-text-button" onClick={useMouse}>
                    Use mouse
                  </button>
                )}
                {cameraError && <small>{cameraError}</small>}
              </div>
            )}
            {cameraHasVideo && (
              <>
                <div className="camera-label">
                  <span />
                  {cameraState === "active" ? "LIVE HAND" : "LOADING TRACKING"}
                </div>
                <button
                  className="camera-exit"
                  onClick={(event) => {
                    event.stopPropagation();
                    useMouse();
                  }}
                >
                  Exit
                </button>
              </>
            )}
          </div>
        </aside>

        <aside
          className={`brush-cartridge-shell ${menuOpen ? "is-open" : ""}`}
          style={cartridgeStyle}
          onPointerDown={(event) => event.stopPropagation()}
        >
          <button
            className="cartridge-tab"
            onClick={() => {
              primeAudio();
              toggleMenu();
            }}
            aria-expanded={menuOpen}
            aria-controls="brush-cartridge"
            aria-label={
              menuOpen ? "Close brush cartridge" : "Open brush cartridge"
            }
          >
            <span className="cartridge-tab-light" />
            <span className="cartridge-tab-grip">
              <i />
              <i />
              <i />
            </span>
            <small>BRUSH</small>
          </button>

          <div
            id="brush-cartridge"
            className="brush-cartridge"
            aria-hidden={!menuOpen}
          >
            <div className="cartridge-backplate" />
            <header className="cartridge-header">
              <div>
                <span>AIRLOOM TOOL</span>
                <strong>BRUSH CARTRIDGE</strong>
              </div>
              <div className="cartridge-status">
                <i />
                {menuOpen ? "EJECTED" : "LOCKED"}
              </div>
            </header>

            <div className="cartridge-body">
              <section className="cartridge-section color-section">
                <div className="cartridge-section-label">
                  <span>01</span>
                  <div>
                    <strong>COLOR</strong>
                    <small>FIST + MOVE</small>
                  </div>
                </div>

                <div className="cartridge-color-grid">
                {AIRLOOM_COLORS.map((color, index) => (
                  <button
                    key={color.name}
                    className={index === colorIndex ? "is-selected" : ""}
                    onClick={() => {
                      primeAudio();
                      selectColor(index);
                    }}
                    aria-label={`Select ${color.name}`}
                    aria-pressed={index === colorIndex}
                    tabIndex={menuOpen ? 0 : -1}
                  >
                    <span
                      className="color-swatch-core"
                      style={{ background: color.value }}
                    />
                  </button>
                ))}
              </div>
              </section>

              <section className="cartridge-section thickness-section">
                <div className="cartridge-section-label">
                  <span>02</span>
                  <div>
                    <strong>THICKNESS</strong>
                    <small>OPEN PALM + MOVE</small>
                  </div>
                  <b>{Math.round(1 + thickness * 99)}</b>
                </div>

                <div className="thickness-control">
                  <div className="thickness-profile">
                    <span className="thickness-marker" />
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.005"
                      value={thickness}
                      onPointerDown={() => primeAudio()}
                      onChange={(event) =>
                        selectThickness(Number(event.currentTarget.value))
                      }
                      aria-label="Continuous stroke thickness"
                      tabIndex={menuOpen ? 0 : -1}
                    />
                  </div>
                  <div className="thickness-scale">
                    <span>HAIRLINE</span>
                    <span>HEAVY</span>
                  </div>
                </div>
              </section>
            </div>

            <footer className="cartridge-footer">
              <div className="selected-brush">
                <span
                  className="selected-color-chip"
                  style={{ background: selectedColor.value }}
                />
                <div>
                  <small>ACTIVE BRUSH</small>
                  <strong>{selectedColor.name}</strong>
                </div>
              </div>
              <span className="selected-stroke-preview" />
              <small className="cartridge-gesture-hint">
                {gesture === "fist"
                  ? "MOVE TO PICK COLOR"
                  : gesture === "openPalm"
                    ? "MOVE TO SIZE"
                    : "SNAP TO HOLSTER"}
              </small>
            </footer>
          </div>
        </aside>

        <nav className="minimal-tools" aria-label="Artwork controls">
          <button onClick={() => sceneRef.current?.undo()}>Undo</button>
          <button onClick={() => sceneRef.current?.resetView()}>Reset view</button>
          <button onClick={() => sceneRef.current?.clear()}>Clear</button>
          <button onClick={exportArtwork}>Export</button>
          <button
            className="help-button"
            onClick={() => setHelpOpen((value) => !value)}
            aria-expanded={helpOpen}
          >
            ?
          </button>
        </nav>

        {helpOpen && (
          <aside className="gesture-guide">
            <div>
              <span>1</span>
              <strong>Draw</strong>
            </div>
            <div>
              <span>2</span>
              <strong>Pan</strong>
            </div>
            <div>
              <span>3</span>
              <strong>Orbit + zoom</strong>
            </div>
            <div>
              <span>✦</span>
              <strong>Snap cartridge</strong>
            </div>
          </aside>
        )}

        <p className="canvas-hint">
          {cameraState === "active"
            ? "One finger draws · Two pan · Three orbit · Snap ejects the cartridge"
            : cameraState === "calibrating"
              ? "Camera ready · Loading hand tracking…"
            : "Draw with your mouse now, or enable the camera above"}
        </p>
      </section>
    </main>
  );
}
