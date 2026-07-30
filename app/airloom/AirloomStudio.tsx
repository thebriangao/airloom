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
import { SuddenSoundDetector } from "./audioGesture";
import {
  countExtendedFingers,
  GestureEngine,
} from "./gestureEngine";
import {
  AIRLOOM_COLORS,
  eraserRadiusFromThickness,
  radiusFromThickness,
  type GestureResult,
  type HandPose,
  type Landmark,
  type Point3,
} from "./types";

type CameraState =
  | "idle"
  | "requesting"
  | "calibrating"
  | "active"
  | "error";
type SoundKind =
  | "open"
  | "close"
  | "color"
  | "size"
  | "eraserOn"
  | "eraserOff";

const POSE_LABELS: Record<HandPose, string> = {
  none: "Show your hand",
  draw: "Drawing",
  pan2d: "Panning",
  orbit3d: "Orbiting",
  fist: "Choosing color",
  openPalm: "Adjusting thickness",
  other: "Pinch to draw",
};

const clamp = (value: number, minimum = 0, maximum = 1) =>
  Math.max(minimum, Math.min(maximum, value));

const gridPosition = (value: number) => clamp((value - 0.12) / 0.76);
const stableDelta = (value: number, deadZone: number) =>
  Math.abs(value) < deadZone ? 0 : value;

type DepthCalibration = {
  handScale: number;
  fingerOffset: number;
  filteredDepth: number;
};

const fingerDepthOffset = (result: GestureResult) =>
  (result.indexTip.z - result.palm.z) / Math.max(0.025, result.handScale);

const brushDepth = (
  result: GestureResult,
  calibration: DepthCalibration,
) => {
  const palmDepth =
    Math.log(
      Math.max(0.025, result.handScale) /
        Math.max(0.025, calibration.handScale),
    ) * 4.6;
  const fingertipDepth =
    (calibration.fingerOffset - fingerDepthOffset(result)) * 1.65;
  return clamp(palmDepth + fingertipDepth, -2.15, 2.15);
};

const palmCenter = (landmarks: Landmark[]) => {
  const indices = [0, 5, 9, 13, 17];
  return indices.reduce(
    (center, index) => ({
      x: center.x + landmarks[index].x / indices.length,
      y: center.y + landmarks[index].y / indices.length,
    }),
    { x: 0, y: 0 },
  );
};

export function AirloomStudio() {
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<AirScene | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const microphoneSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const microphoneAnalyserRef = useRef<AnalyserNode | null>(null);
  const microphoneSamplesRef = useRef(new Float32Array(1024));
  const suddenSoundDetectorRef = useRef(new SuddenSoundDetector());
  const lastAudioTransientRef = useRef(-Infinity);
  const lastVisualSnapRef = useRef(-Infinity);
  const lastConfirmedSnapRef = useRef(-Infinity);
  const lastVisualClapRef = useRef(-Infinity);
  const lastConfirmedClapRef = useRef(-Infinity);
  const clapContactRef = useRef(false);
  const clapMotionRef = useRef<{
    separation: number;
    timestamp: number;
    approachAt: number;
  } | null>(null);
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
  const filteredTipRef = useRef<Point3 | null>(null);
  const depthCalibrationRef = useRef<DepthCalibration | null>(null);
  const menuOpenRef = useRef(false);
  const colorIndexRef = useRef(0);
  const thicknessRef = useRef(0.32);
  const eraserThicknessRef = useRef(0.42);
  const eraserEnabledRef = useRef(false);
  const lastSizeTickRef = useRef(4);
  const pointerDrawingRef = useRef(false);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState("");
  const [gesture, setGesture] = useState<HandPose>("none");
  const [menuOpen, setMenuOpen] = useState(false);
  const [colorIndex, setColorIndex] = useState(0);
  const [thickness, setThickness] = useState(0.32);
  const [eraserThickness, setEraserThickness] = useState(0.42);
  const [eraserEnabled, setEraserEnabled] = useState(false);
  const [demoMode, setDemoMode] = useState(true);
  const [helpOpen, setHelpOpen] = useState(false);

  const selectedColor = AIRLOOM_COLORS[colorIndex];
  const activeThickness = eraserEnabled ? eraserThickness : thickness;

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
        eraserOn: [310, 155],
        eraserOff: [180, 360],
      };
      const [start, end] = frequencies[kind];

      oscillator.type =
        kind === "size" || kind.startsWith("eraser") ? "triangle" : "sine";
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

  const selectEraserThickness = useCallback(
    (value: number, audible = true) => {
      const next = clamp(value);
      if (Math.abs(eraserThicknessRef.current - next) < 0.003) return;
      eraserThicknessRef.current = next;
      setEraserThickness(next);
      const tick = Math.floor(next * 14);
      if (audible && tick !== lastSizeTickRef.current) {
        lastSizeTickRef.current = tick;
        playSound("size");
      }
    },
    [playSound],
  );

  const toggleEraser = useCallback(() => {
    const next = !eraserEnabledRef.current;
    eraserEnabledRef.current = next;
    setEraserEnabled(next);
    sceneRef.current?.endStroke();
    previousControlRef.current = null;
    playSound(next ? "eraserOn" : "eraserOff");
  }, [playSound]);

  const toggleMenu = useCallback(() => {
    const next = !menuOpenRef.current;
    menuOpenRef.current = next;
    setMenuOpen(next);
    sceneRef.current?.endStroke();
    previousControlRef.current = null;
    playSound(next ? "open" : "close");
  }, [playSound]);

  const confirmSnap = useCallback(
    (timestamp: number) => {
      const paired =
        Math.abs(timestamp - lastAudioTransientRef.current) <= 560 &&
        Math.abs(timestamp - lastVisualSnapRef.current) <= 560;
      const cooledDown = timestamp - lastConfirmedSnapRef.current > 950;
      const clapIsNotWinning =
        Math.abs(timestamp - lastVisualClapRef.current) > 650 &&
        timestamp - lastConfirmedClapRef.current > 650;
      if (!paired || !cooledDown || !clapIsNotWinning) return false;
      lastConfirmedSnapRef.current = timestamp;
      lastAudioTransientRef.current = -Infinity;
      lastVisualSnapRef.current = -Infinity;
      toggleMenu();
      return true;
    },
    [toggleMenu],
  );

  const confirmClap = useCallback(
    (timestamp: number) => {
      const paired =
        Math.abs(timestamp - lastAudioTransientRef.current) <= 650 &&
        Math.abs(timestamp - lastVisualClapRef.current) <= 650;
      const cooledDown = timestamp - lastConfirmedClapRef.current > 1150;
      if (!paired || !cooledDown) return false;
      lastConfirmedClapRef.current = timestamp;
      lastAudioTransientRef.current = -Infinity;
      lastVisualClapRef.current = -Infinity;
      lastVisualSnapRef.current = -Infinity;
      toggleEraser();
      return true;
    },
    [toggleEraser],
  );

  const sampleMicrophone = useCallback(
    (timestamp: number) => {
      const analyser = microphoneAnalyserRef.current;
      if (!analyser) return;
      const samples = microphoneSamplesRef.current;
      analyser.getFloatTimeDomainData(samples);
      const { transient } = suddenSoundDetectorRef.current.update(
        samples,
        timestamp,
      );
      if (!transient) return;

      lastAudioTransientRef.current = timestamp;
      if (!confirmClap(timestamp)) confirmSnap(timestamp);
    },
    [confirmClap, confirmSnap],
  );

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

  const applyToolAtPoint = useCallback((point: Parameters<AirScene["addPoint"]>[0]) => {
    if (eraserEnabledRef.current) {
      sceneRef.current?.eraseAt(
        point,
        eraserRadiusFromThickness(eraserThicknessRef.current),
      );
      return;
    }
    sceneRef.current?.addPoint(
      point,
      AIRLOOM_COLORS[colorIndexRef.current].value,
      radiusFromThickness(thicknessRef.current),
    );
  }, []);

  const detectClap = useCallback(
    (hands: Landmark[][], timestamp: number) => {
      if (hands.length < 2) {
        const recentApproach =
          clapMotionRef.current &&
          timestamp - clapMotionRef.current.approachAt < 430;
        if (!recentApproach || clapContactRef.current) return false;
        clapContactRef.current = true;
        lastVisualClapRef.current = timestamp;
        return confirmClap(timestamp);
      }

      const firstPalm = palmCenter(hands[0]);
      const secondPalm = palmCenter(hands[1]);
      const separation = Math.hypot(
        firstPalm.x - secondPalm.x,
        firstPalm.y - secondPalm.y,
      );
      const firstSpan = Math.hypot(
        hands[0][5].x - hands[0][17].x,
        hands[0][5].y - hands[0][17].y,
      );
      const secondSpan = Math.hypot(
        hands[1][5].x - hands[1][17].x,
        hands[1][5].y - hands[1][17].y,
      );
      const averageSpan = Math.max(0.04, (firstSpan + secondSpan) / 2);
      const contactDistance = clamp(averageSpan * 1.55, 0.15, 0.25);
      const handsLookOpen =
        countExtendedFingers(hands[0]) >= 2 &&
        countExtendedFingers(hands[1]) >= 2;
      const previousMotion = clapMotionRef.current;
      const elapsed = previousMotion
        ? Math.max(16, timestamp - previousMotion.timestamp)
        : 16;
      const closingRate = previousMotion
        ? ((previousMotion.separation - separation) / elapsed) * 1000
        : 0;
      const approaching =
        handsLookOpen && separation < 0.46 && closingRate > 0.2;
      const approachAt = approaching
        ? timestamp
        : previousMotion?.approachAt ?? -Infinity;

      clapMotionRef.current = {
        separation,
        timestamp,
        approachAt,
      };

      if (separation > Math.max(0.34, contactDistance * 1.65)) {
        clapContactRef.current = false;
      }
      const recentApproach = timestamp - approachAt < 520;
      if (
        !handsLookOpen ||
        !recentApproach ||
        separation > contactDistance ||
        clapContactRef.current
      ) {
        return false;
      }

      clapContactRef.current = true;
      lastVisualClapRef.current = timestamp;
      return confirmClap(timestamp);
    },
    [confirmClap],
  );

  const processHands = useCallback(
    (hands: Landmark[][], timestamp: number) => {
      if (detectClap(hands, timestamp)) return;
      const landmarks = hands[0];
      if (!landmarks) return;
      const result = gestureEngineRef.current.update(landmarks, timestamp);
      if (!filteredTipRef.current) {
        filteredTipRef.current = { ...result.indexTip };
      } else {
        filteredTipRef.current.x +=
          (result.indexTip.x - filteredTipRef.current.x) * 0.34;
        filteredTipRef.current.y +=
          (result.indexTip.y - filteredTipRef.current.y) * 0.34;
        filteredTipRef.current.z +=
          (result.indexTip.z - filteredTipRef.current.z) * 0.28;
      }
      const filteredTip = filteredTipRef.current;

      if (cursorRef.current) {
        cursorRef.current.style.opacity = "1";
        cursorRef.current.style.left = `${(1 - filteredTip.x) * 100}%`;
        cursorRef.current.style.top = `${filteredTip.y * 100}%`;
      }

      if (
        hands.length === 1 &&
        (result.snapPose || result.snap)
      ) {
        lastVisualSnapRef.current = timestamp;
        if (confirmSnap(timestamp)) return;
      }

      const controlPose = result.drawingPinch ? "draw" : result.pose;
      updateGesture(controlPose);

      if (menuOpenRef.current) {
        sceneRef.current?.endStroke();
        previousControlRef.current = null;

        if (result.pose === "fist" && !eraserEnabledRef.current) {
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
          const nextThickness = gridPosition(1 - result.palm.x);
          if (eraserEnabledRef.current) {
            selectEraserThickness(nextThickness);
          } else {
            selectThickness(nextThickness);
          }
        }
        return;
      }

      const previous = previousControlRef.current;
      const controlSample =
        previous?.pose === controlPose
          ? {
              pose: controlPose,
              x: previous.x + (result.palm.x - previous.x) * 0.22,
              y: previous.y + (result.palm.y - previous.y) * 0.22,
              scale:
                previous.scale +
                (result.handScale - previous.scale) * 0.18,
            }
          : {
              pose: controlPose,
              x: result.palm.x,
              y: result.palm.y,
              scale: result.handScale,
            };
      if (controlPose === "draw") {
        if (previous?.pose !== "draw") {
          sceneRef.current?.endStroke();
        }
        if (!depthCalibrationRef.current) {
          depthCalibrationRef.current = {
            handScale: result.handScale,
            fingerOffset: fingerDepthOffset(result),
            filteredDepth: 0,
          };
        }
        const calibration = depthCalibrationRef.current;
        const trackedDepth = brushDepth(result, calibration);
        calibration.filteredDepth +=
          (trackedDepth - calibration.filteredDepth) * 0.22;
        const point = sceneRef.current?.normalizedToArtwork(
          filteredTip,
          calibration.filteredDepth,
        );
        if (point) {
          applyToolAtPoint(point);
        }
      } else {
        sceneRef.current?.endStroke();
      }

      if (previous?.pose === controlPose && controlPose === "pan2d") {
        sceneRef.current?.pan(
          -stableDelta(controlSample.x - previous.x, 0.00045),
          stableDelta(controlSample.y - previous.y, 0.00045),
        );
      }
      if (previous?.pose === controlPose && controlPose === "orbit3d") {
        sceneRef.current?.orbit(
          stableDelta(controlSample.x - previous.x, 0.0004),
          stableDelta(controlSample.y - previous.y, 0.0004),
          stableDelta(controlSample.scale - previous.scale, 0.00022),
        );
      }

      previousControlRef.current = controlSample;
    },
    [
      applyToolAtPoint,
      confirmSnap,
      detectClap,
      selectColor,
      selectEraserThickness,
      selectThickness,
      updateGesture,
    ],
  );

  const setupMicrophone = useCallback(
    (stream: MediaStream) => {
      const context = primeAudio();
      if (!context || stream.getAudioTracks().length === 0) {
        throw new Error(
          "Microphone access is required to confirm snaps and claps.",
        );
      }
      microphoneSourceRef.current?.disconnect();
      microphoneAnalyserRef.current?.disconnect();
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.1;
      source.connect(analyser);
      microphoneSourceRef.current = source;
      microphoneAnalyserRef.current = analyser;
      suddenSoundDetectorRef.current.reset();
      lastAudioTransientRef.current = -Infinity;
    },
    [primeAudio],
  );

  const stopCamera = useCallback(() => {
    window.cancelAnimationFrame(animationRef.current);
    microphoneSourceRef.current?.disconnect();
    microphoneAnalyserRef.current?.disconnect();
    microphoneSourceRef.current = null;
    microphoneAnalyserRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    handLandmarkerRef.current?.close();
    handLandmarkerRef.current = null;
    gestureEngineRef.current.reset();
    clapContactRef.current = false;
    clapMotionRef.current = null;
    suddenSoundDetectorRef.current.reset();
    previousControlRef.current = null;
    filteredTipRef.current = null;
    depthCalibrationRef.current = null;
    if (cursorRef.current) cursorRef.current.style.opacity = "0";
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const startTrackingLoop = useCallback(() => {
    const loop = () => {
      const timestamp = performance.now();
      const video = videoRef.current;
      const landmarker = handLandmarkerRef.current;
      if (
        video &&
        landmarker &&
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        video.currentTime !== lastVideoTimeRef.current
      ) {
        lastVideoTimeRef.current = video.currentTime;
        const result = landmarker.detectForVideo(video, timestamp);
        const hands = result.landmarks as Landmark[][];
        if (hands.length > 0) {
          processHands(hands, timestamp);
        } else {
          gestureEngineRef.current.reset();
          clapContactRef.current = false;
          sceneRef.current?.endStroke();
          previousControlRef.current = null;
          filteredTipRef.current = null;
          updateGesture("none");
          if (cursorRef.current) cursorRef.current.style.opacity = "0";
        }
      }
      sampleMicrophone(timestamp);
      animationRef.current = window.requestAnimationFrame(loop);
    };
    animationRef.current = window.requestAnimationFrame(loop);
  }, [processHands, sampleMicrophone, updateGesture]);

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
        audio: {
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: false,
        },
      });
      streamRef.current = stream;
      if (!videoRef.current) throw new Error("Camera surface unavailable.");
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setupMicrophone(stream);
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
            numHands: 2,
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
            numHands: 2,
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
          ? "Camera or microphone access was blocked. Allow both in your browser's site settings, then try again."
          : errorName === "NotFoundError"
            ? "No camera or microphone was found on this device."
            : error instanceof Error
              ? error.message
              : "Airloom could not start the camera and microphone.",
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
    const point = sceneRef.current?.normalizedToArtwork(
      { ...pointer, x: 1 - pointer.x },
      0,
    );
    if (point) applyToolAtPoint(point);
  };

  const handleCanvasPointerMove = (
    event: ReactPointerEvent<HTMLDivElement>,
  ) => {
    if (!pointerDrawingRef.current || !demoMode) return;
    const pointer = pointerPosition(event);
    const point = sceneRef.current?.normalizedToArtwork(
      { ...pointer, x: 1 - pointer.x },
      0,
    );
    if (point) applyToolAtPoint(point);
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
    "--cartridge-color": eraserEnabled ? "#6f6f6a" : selectedColor.value,
    "--thickness-position": `${activeThickness * 100}%`,
    "--marker-size": `${13 + activeThickness * 9}px`,
    "--stroke-height": `${2 + activeThickness * 20}px`,
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
          className={`finger-cursor ${eraserEnabled ? "is-eraser" : ""} ${gesture === "draw" ? "is-drawing" : "is-hovering"}`}
          style={{
            color: eraserEnabled ? "#111111" : selectedColor.value,
            width: eraserEnabled
              ? `${28 + eraserThickness * 54}px`
              : `${16 + thickness * 24}px`,
            height: eraserEnabled
              ? `${28 + eraserThickness * 54}px`
              : `${16 + thickness * 24}px`,
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
                ? eraserEnabled
                  ? "Mouse eraser"
                  : "Mouse drawing"
                : eraserEnabled
                  ? `Eraser · ${POSE_LABELS[gesture]}`
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
                  {cameraState === "error"
                    ? "Try again"
                    : "Enable camera + mic"}
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
                  {cameraState === "active"
                    ? "LIVE HAND + MIC"
                    : "LOADING TRACKING"}
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
          className={`brush-cartridge-shell ${menuOpen ? "is-open" : ""} ${eraserEnabled ? "is-eraser" : ""}`}
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
              menuOpen
                ? `Close ${eraserEnabled ? "eraser" : "brush"} cartridge`
                : `Open ${eraserEnabled ? "eraser" : "brush"} cartridge`
            }
          >
            <span className="cartridge-tab-light" />
            <span className="cartridge-tab-grip">
              <i />
              <i />
              <i />
            </span>
            <small>{eraserEnabled ? "ERASE" : "BRUSH"}</small>
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
                <strong>
                  {eraserEnabled ? "ERASER CARTRIDGE" : "BRUSH CARTRIDGE"}
                </strong>
              </div>
              <div className="cartridge-status">
                <i />
                {menuOpen ? "EJECTED" : "LOCKED"}
              </div>
            </header>

            <div className="cartridge-body">
              {!eraserEnabled && (
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
              )}

              <section className="cartridge-section thickness-section">
                <div className="cartridge-section-label">
                  <span>{eraserEnabled ? "01" : "02"}</span>
                  <div>
                    <strong>
                      {eraserEnabled ? "ERASER SIZE" : "THICKNESS"}
                    </strong>
                    <small>OPEN PALM + MOVE</small>
                  </div>
                  <b>{Math.round(1 + activeThickness * 99)}</b>
                </div>

                <div className="thickness-control">
                  <div className="thickness-profile">
                    <span className="thickness-marker" />
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.005"
                      value={activeThickness}
                      onPointerDown={() => primeAudio()}
                      onChange={(event) => {
                        const value = Number(event.currentTarget.value);
                        if (eraserEnabled) {
                          selectEraserThickness(value);
                        } else {
                          selectThickness(value);
                        }
                      }}
                      aria-label={
                        eraserEnabled
                          ? "Continuous eraser size"
                          : "Continuous stroke thickness"
                      }
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
                  className={`selected-color-chip ${eraserEnabled ? "eraser-chip" : ""}`}
                  style={{
                    background: eraserEnabled ? "#deded8" : selectedColor.value,
                  }}
                />
                <div>
                  <small>ACTIVE TOOL</small>
                  <strong>
                    {eraserEnabled ? "ERASER" : selectedColor.name}
                  </strong>
                </div>
              </div>
              <span className="selected-stroke-preview" />
              <small className="cartridge-gesture-hint">
                {gesture === "fist"
                  ? eraserEnabled
                    ? "OPEN PALM TO SIZE"
                    : "MOVE TO PICK COLOR"
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
          <button
            onClick={() => {
              sceneRef.current?.clear();
              depthCalibrationRef.current = null;
            }}
          >
            Clear
          </button>
          <button
            className={eraserEnabled ? "is-active" : ""}
            onClick={toggleEraser}
          >
            Eraser
          </button>
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
              <strong>Pinch draw</strong>
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
              <strong>Snap + sound</strong>
            </div>
            <div>
              <span>◉</span>
              <strong>Clap eraser</strong>
            </div>
          </aside>
        )}

        <p className="canvas-hint">
          {cameraState === "active"
            ? "Pinch to draw · Release to stop · Two pan · Three orbit · Move hand in or out to zoom"
            : cameraState === "calibrating"
              ? "Camera ready · Loading hand tracking…"
            : "Draw with your mouse now, or enable the camera and microphone above"}
        </p>
      </section>
    </main>
  );
}
