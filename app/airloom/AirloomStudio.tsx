"use client";

import type {
  FaceLandmarker,
  HandLandmarker,
} from "@mediapipe/tasks-vision";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { AirScene, type SnapKind } from "./AirScene";
import { GestureEngine } from "./gestureEngine";
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
type VisionFileset = Parameters<
  typeof HandLandmarker.createFromOptions
>[0];
type SoundKind =
  | "open"
  | "close"
  | "color"
  | "size"
  | "eraserOn"
  | "eraserOff"
  | "undo";
const POSE_LABELS: Record<HandPose, string> = {
  none: "Show your hand",
  draw: "Drawing",
  grab: "Close fist over an object",
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
const HAND_TRACKING_GRACE_MS = 350;
const HEAD_HAND_FREE_MS = 260;
const FACE_SAMPLE_INTERVAL_MS = 55;
const HEAD_TURN_THRESHOLD = 0.115;
const HEAD_TURN_VELOCITY = 0.34;
const HEAD_NEUTRAL_THRESHOLD = 0.045;
const HEAD_REARM_MS = 160;
const HEAD_TURN_COOLDOWN_MS = 700;
const WRIST_ROLL_START_ORIENTATION = 0.48;
const WRIST_ROLL_EDGE_ORIENTATION = 0.2;
const WRIST_ROLL_END_ORIENTATION = 0.42;
const WRIST_ROLL_MAX_MS = 520;
const WRIST_ROLL_COOLDOWN_MS = 900;
const CANVAS_INPUT_BLOCKER = "[data-block-canvas-input]";

const pointerIsOverCanvasUi = (clientX: number, clientY: number) =>
  Boolean(
    document
      .elementFromPoint(clientX, clientY)
      ?.closest(CANVAS_INPUT_BLOCKER),
  );

const responsiveBlend = (
  distance: number,
  elapsedSeconds: number,
  restingBlend: number,
  fullResponseSpeed: number,
) =>
  clamp(
    restingBlend +
      (distance / Math.max(1 / 120, elapsedSeconds) / fullResponseSpeed) *
        (1 - restingBlend),
    restingBlend,
    1,
  );

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

const palmFacingScore = (landmarks: Landmark[]) => {
  const wrist = landmarks[0];
  const indexBase = landmarks[5];
  const pinkyBase = landmarks[17];
  if (!wrist || !indexBase || !pinkyBase) return 0;
  const indexVector = {
    x: indexBase.x - wrist.x,
    y: indexBase.y - wrist.y,
    z: indexBase.z - wrist.z,
  };
  const pinkyVector = {
    x: pinkyBase.x - wrist.x,
    y: pinkyBase.y - wrist.y,
    z: pinkyBase.z - wrist.z,
  };
  const normal = {
    x: indexVector.y * pinkyVector.z - indexVector.z * pinkyVector.y,
    y: indexVector.z * pinkyVector.x - indexVector.x * pinkyVector.z,
    z: indexVector.x * pinkyVector.y - indexVector.y * pinkyVector.x,
  };
  const magnitude = Math.hypot(normal.x, normal.y, normal.z);
  return magnitude > 0.00001 ? normal.z / magnitude : 0;
};

const mirroredFaceYaw = (landmarks: Landmark[]) => {
  const nose = landmarks[1];
  const firstCheek = landmarks[234];
  const secondCheek = landmarks[454];
  if (!nose || !firstCheek || !secondCheek) return null;
  const mirroredNoseX = 1 - nose.x;
  const mirroredFirstX = 1 - firstCheek.x;
  const mirroredSecondX = 1 - secondCheek.x;
  const faceWidth = Math.abs(mirroredFirstX - mirroredSecondX);
  if (faceWidth < 0.04) return null;
  const faceCenter = (mirroredFirstX + mirroredSecondX) / 2;
  return (mirroredNoseX - faceCenter) / faceWidth;
};

type WristRollState = {
  startOrientation: number;
  startAt: number;
  previousOrientation: number;
  previousAt: number;
  startPalm: { x: number; y: number };
  active: boolean;
  crossedEdge: boolean;
};

type HeadTurnState = {
  neutralYaw: number;
  previousYaw: number;
  previousAt: number;
  neutralSince: number;
  armed: boolean;
};

export function AirloomStudio() {
  const stageRef = useRef<HTMLElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sideCanvasRef = useRef<HTMLCanvasElement>(null);
  const hoverCursorRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<AirScene | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const handLandmarkerPromiseRef = useRef<Promise<HandLandmarker> | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const faceLandmarkerPromiseRef = useRef<Promise<FaceLandmarker> | null>(null);
  const visionFilesetPromiseRef = useRef<Promise<VisionFileset> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const trackingFrameRef = useRef(0);
  const trackingActiveRef = useRef(false);
  const lastVideoTimeRef = useRef(-1);
  const lastHandsSeenAtRef = useRef(-Infinity);
  const noHandsSinceRef = useRef(-Infinity);
  const lastFaceSampleAtRef = useRef(-Infinity);
  const headTurnRef = useRef<HeadTurnState | null>(null);
  const lastHeadTurnAtRef = useRef(-Infinity);
  const wristRollRef = useRef<WristRollState | null>(null);
  const lastWristRollAtRef = useRef(-Infinity);
  const lastHandSampleAtRef = useRef(-Infinity);
  const gestureEngineRef = useRef(new GestureEngine());
  const gestureRef = useRef<HandPose>("none");
  const previousControlRef = useRef<{
    pose: HandPose;
    x: number;
    y: number;
    scale: number;
  } | null>(null);
  const filteredTipRef = useRef<Point3 | null>(null);
  const filteredGrabRef = useRef<Point3 | null>(null);
  const depthCalibrationRef = useRef<DepthCalibration | null>(null);
  const grabDepthCalibrationRef = useRef<DepthCalibration | null>(null);
  const menuOpenRef = useRef(false);
  const colorIndexRef = useRef(0);
  const thicknessRef = useRef(0.32);
  const eraserThicknessRef = useRef(0.42);
  const eraserEnabledRef = useRef(false);
  const objectGrabActiveRef = useRef(false);
  const snapKindRef = useRef<SnapKind>("none");
  const lastSizeTickRef = useRef(4);
  const pointerDrawingRef = useRef(false);
  const pointerHoverActiveRef = useRef(false);
  const pointerObjectGrabRef = useRef(false);
  const pointerGrabDepthRef = useRef(0);
  const pointerNavigationRef = useRef<{
    mode: "pan" | "orbit";
    x: number;
    y: number;
  } | null>(null);
  const lastPointerPositionRef = useRef<Point3 | null>(null);
  const pointerPressRef = useRef<{
    x: number;
    y: number;
    moved: boolean;
  } | null>(null);
  const pointerClickStrokeCountRef = useRef(0);
  const lastPointerClickAtRef = useRef(-Infinity);
  const lastPointerClickPositionRef = useRef<Point3 | null>(null);
  const sideViewControlRef = useRef<{ x: number; y: number } | null>(null);
  const sideViewOrbitActiveRef = useRef(false);
  const sideViewPointerRef = useRef<{
    x: number;
    pointerId: number;
  } | null>(null);
  const undoSwipeRef = useRef<{
    x: number;
    startedAt: number;
    travel: number;
    timestamp: number;
  } | null>(null);
  const lastUndoAtRef = useRef(-Infinity);

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
  const [hasArtwork, setHasArtwork] = useState(false);
  const [objectGrabSelected, setObjectGrabSelected] = useState(false);
  const [snapKind, setSnapKind] = useState<SnapKind>("none");
  const [sideViewOrbiting, setSideViewOrbiting] = useState(false);
  const [sideViewReturning, setSideViewReturning] = useState(false);

  const selectedColor = AIRLOOM_COLORS[colorIndex];
  const activeThickness = eraserEnabled ? eraserThickness : thickness;

  const hideHoverCursor = useCallback(() => {
    if (hoverCursorRef.current) {
      hoverCursorRef.current.style.opacity = "0";
    }
  }, []);

  const showHoverCursor = useCallback((tip: Point3) => {
    const cursor = hoverCursorRef.current;
    const stage = stageRef.current;
    if (!cursor || !stage) return;
    const radius = eraserEnabledRef.current
      ? eraserRadiusFromThickness(eraserThicknessRef.current)
      : radiusFromThickness(thicknessRef.current);
    const visibleWorldHeight =
      2 * 7 * Math.tan((48 * Math.PI) / 360);
    const pixelsPerWorld = stage.clientHeight / visibleWorldHeight;
    const diameter = clamp(radius * 2 * pixelsPerWorld, 4, 170);
    cursor.style.left = `${(1 - tip.x) * 100}%`;
    cursor.style.top = `${tip.y * 100}%`;
    cursor.style.width = `${diameter}px`;
    cursor.style.height = `${diameter}px`;
    cursor.style.opacity = "1";
  }, []);

  const updateSnapKind = useCallback((next: SnapKind) => {
    if (snapKindRef.current === next) return;
    snapKindRef.current = next;
    setSnapKind(next);
  }, []);

  const releaseObjectGrab = useCallback(() => {
    sceneRef.current?.endObjectGrab();
    objectGrabActiveRef.current = false;
    filteredGrabRef.current = null;
    grabDepthCalibrationRef.current = null;
    setObjectGrabSelected(false);
    updateSnapKind("none");
  }, [updateSnapKind]);

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
        undo: [430, 245],
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

  const undoArtwork = useCallback(() => {
    releaseObjectGrab();
    sceneRef.current?.undo();
    playSound("undo");
  }, [playSound, releaseObjectGrab]);

  const releaseSideViewOrbit = useCallback(() => {
    if (!sideViewOrbitActiveRef.current) return;
    sideViewOrbitActiveRef.current = false;
    sideViewControlRef.current = null;
    sideViewPointerRef.current = null;
    sceneRef.current?.endSideViewOrbit();
    setSideViewOrbiting(false);
    setSideViewReturning(true);
  }, []);

  const resetSideViewControl = useCallback(() => {
    sideViewOrbitActiveRef.current = false;
    sideViewControlRef.current = null;
    sideViewPointerRef.current = null;
    sceneRef.current?.resetSideView();
    setSideViewOrbiting(false);
    setSideViewReturning(false);
  }, []);

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
    releaseObjectGrab();
    hideHoverCursor();
    previousControlRef.current = null;
    playSound(next ? "eraserOn" : "eraserOff");
  }, [hideHoverCursor, playSound, releaseObjectGrab]);

  const toggleMenu = useCallback(() => {
    const next = !menuOpenRef.current;
    menuOpenRef.current = next;
    setMenuOpen(next);
    sceneRef.current?.endStroke();
    releaseObjectGrab();
    hideHoverCursor();
    previousControlRef.current = null;
    playSound(next ? "open" : "close");
  }, [hideHoverCursor, playSound, releaseObjectGrab]);

  const detectWristRoll = useCallback(
    (
      landmarks: Landmark[],
      result: GestureResult,
      timestamp: number,
    ): "idle" | "tracking" | "triggered" => {
      const orientation = palmFacingScore(landmarks);
      const center = palmCenter(landmarks);
      let state = wristRollRef.current;

      if (result.drawingPinch || result.sideViewControl) {
        wristRollRef.current = null;
        return "idle";
      }

      if (!state) {
        if (
          result.fingerCount >= 2 &&
          Math.abs(orientation) >= WRIST_ROLL_START_ORIENTATION
        ) {
          wristRollRef.current = {
            startOrientation: orientation,
            startAt: timestamp,
            previousOrientation: orientation,
            previousAt: timestamp,
            startPalm: center,
            active: false,
            crossedEdge: false,
          };
        }
        return "idle";
      }

      const frameElapsed = Math.max(8, timestamp - state.previousAt);
      const turningTowardEdge =
        Math.sign(orientation) === Math.sign(state.previousOrientation) &&
        Math.abs(state.previousOrientation) - Math.abs(orientation) > 0.065 &&
        frameElapsed < 120;
      const skippedAcrossEdge =
        Math.sign(orientation) === -Math.sign(state.previousOrientation) &&
        Math.abs(state.previousOrientation) >= WRIST_ROLL_START_ORIENTATION &&
        Math.abs(orientation) >= WRIST_ROLL_END_ORIENTATION &&
        frameElapsed < 120;

      if (!state.active) {
        if (turningTowardEdge || skippedAcrossEdge) {
          state.active = true;
          state.startOrientation = state.previousOrientation;
          state.startAt = state.previousAt;
          state.startPalm = center;
          state.crossedEdge = skippedAcrossEdge;
        } else if (
          result.fingerCount >= 2 &&
          Math.abs(orientation) >= WRIST_ROLL_START_ORIENTATION
        ) {
          state.startOrientation = orientation;
          state.startAt = timestamp;
          state.startPalm = center;
        } else if (timestamp - state.startAt > 180) {
          wristRollRef.current = null;
          return "idle";
        }
        state.previousOrientation = orientation;
        state.previousAt = timestamp;
        return state.active ? "tracking" : "idle";
      }

      const elapsed = timestamp - state.startAt;
      const palmTravel = Math.hypot(
        center.x - state.startPalm.x,
        center.y - state.startPalm.y,
      );
      if (
        Math.abs(orientation) <= WRIST_ROLL_EDGE_ORIENTATION ||
        Math.sign(orientation) === -Math.sign(state.previousOrientation)
      ) {
        state.crossedEdge = true;
      }

      const finishedOpposite =
        state.crossedEdge &&
        Math.sign(orientation) === -Math.sign(state.startOrientation) &&
        Math.abs(orientation) >= WRIST_ROLL_END_ORIENTATION;
      const cooledDown =
        timestamp - lastWristRollAtRef.current > WRIST_ROLL_COOLDOWN_MS;
      if (
        finishedOpposite &&
        elapsed >= 45 &&
        elapsed <= WRIST_ROLL_MAX_MS &&
        palmTravel < 0.26 &&
        cooledDown
      ) {
        lastWristRollAtRef.current = timestamp;
        wristRollRef.current = null;
        toggleEraser();
        return "triggered";
      }

      if (elapsed > WRIST_ROLL_MAX_MS || palmTravel >= 0.26) {
        wristRollRef.current = null;
        return "idle";
      }

      state.previousOrientation = orientation;
      state.previousAt = timestamp;
      return "tracking";
    },
    [toggleEraser],
  );

  const processHeadTurn = useCallback(
    (video: HTMLVideoElement, timestamp: number) => {
      const landmarker = faceLandmarkerRef.current;
      if (
        !landmarker ||
        timestamp - lastFaceSampleAtRef.current < FACE_SAMPLE_INTERVAL_MS
      ) {
        return;
      }
      lastFaceSampleAtRef.current = timestamp;
      const faceResult = landmarker.detectForVideo(video, timestamp);
      const landmarks = faceResult.faceLandmarks[0] as Landmark[] | undefined;
      if (!landmarks) {
        headTurnRef.current = null;
        return;
      }
      const yaw = mirroredFaceYaw(landmarks);
      if (yaw === null) {
        headTurnRef.current = null;
        return;
      }

      let state = headTurnRef.current;
      if (!state) {
        headTurnRef.current = {
          neutralYaw: yaw,
          previousYaw: yaw,
          previousAt: timestamp,
          neutralSince: timestamp,
          armed: false,
        };
        return;
      }

      const elapsedSeconds = Math.max(
        0.008,
        (timestamp - state.previousAt) / 1000,
      );
      const velocity = (yaw - state.previousYaw) / elapsedSeconds;
      const delta = yaw - state.neutralYaw;
      const nearNeutral = Math.abs(delta) <= HEAD_NEUTRAL_THRESHOLD;
      const cooledDown =
        timestamp - lastHeadTurnAtRef.current > HEAD_TURN_COOLDOWN_MS;

      if (!state.armed) {
        if (nearNeutral) {
          if (state.neutralSince === 0) state.neutralSince = timestamp;
          state.neutralYaw += (yaw - state.neutralYaw) * 0.08;
          if (
            cooledDown &&
            timestamp - state.neutralSince >= HEAD_REARM_MS
          ) {
            state.armed = true;
          }
        } else {
          state.neutralSince = 0;
        }
      } else if (nearNeutral && Math.abs(velocity) < 0.2) {
        state.neutralYaw += (yaw - state.neutralYaw) * 0.04;
      }

      const openJerk =
        !menuOpenRef.current &&
        state.armed &&
        delta <= -HEAD_TURN_THRESHOLD &&
        velocity <= -HEAD_TURN_VELOCITY;
      const closeJerk =
        menuOpenRef.current &&
        state.armed &&
        delta >= HEAD_TURN_THRESHOLD &&
        velocity >= HEAD_TURN_VELOCITY;
      if ((openJerk || closeJerk) && cooledDown) {
        lastHeadTurnAtRef.current = timestamp;
        state.armed = false;
        state.neutralSince = 0;
        toggleMenu();
      }

      state.previousYaw = yaw;
      state.previousAt = timestamp;
    },
    [toggleMenu],
  );

  const updateGesture = useCallback((next: HandPose) => {
    if (gestureRef.current === next) return;
    gestureRef.current = next;
    setGesture(next);
  }, []);

  useEffect(() => {
    if (
      !canvasRef.current ||
      !sideCanvasRef.current ||
      !stageRef.current
    ) {
      return;
    }
    const scene = new AirScene(
      canvasRef.current,
      sideCanvasRef.current,
      setHasArtwork,
      () => setSideViewReturning(false),
    );
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
      if (!demoMode) return;
      const target = event.target as HTMLElement | null;
      if (
        target?.isContentEditable ||
        ["INPUT", "SELECT", "TEXTAREA"].includes(target?.tagName ?? "")
      ) {
        return;
      }
      const key = event.key.toLowerCase();
      if ((event.metaKey || event.ctrlKey) && !event.shiftKey && key === "z") {
        event.preventDefault();
        undoArtwork();
      } else if (event.key === "Escape") {
        releaseObjectGrab();
        if (menuOpenRef.current) toggleMenu();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [
    demoMode,
    releaseObjectGrab,
    toggleMenu,
    undoArtwork,
  ]);

  useEffect(() => {
    const finishSideViewPointer = () => {
      if (sideViewPointerRef.current) releaseSideViewOrbit();
    };
    window.addEventListener("pointerup", finishSideViewPointer);
    window.addEventListener("pointercancel", finishSideViewPointer);
    window.addEventListener("blur", finishSideViewPointer);
    return () => {
      window.removeEventListener("pointerup", finishSideViewPointer);
      window.removeEventListener("pointercancel", finishSideViewPointer);
      window.removeEventListener("blur", finishSideViewPointer);
    };
  }, [releaseSideViewOrbit]);

  useEffect(() => {
    if (demoMode) return;
    pointerDrawingRef.current = false;
    pointerHoverActiveRef.current = false;
    pointerObjectGrabRef.current = false;
    pointerNavigationRef.current = null;
    pointerPressRef.current = null;
    sceneRef.current?.endStroke();
    resetSideViewControl();
    releaseObjectGrab();
    hideHoverCursor();
  }, [
    demoMode,
    hideHoverCursor,
    releaseObjectGrab,
    resetSideViewControl,
  ]);

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

  const processHands = useCallback(
    (hands: Landmark[][], timestamp: number) => {
      if (
        pointerDrawingRef.current ||
        pointerObjectGrabRef.current ||
        pointerNavigationRef.current
      ) {
        return;
      }
      if (hands.length !== 1) {
        sceneRef.current?.endStroke();
        releaseSideViewOrbit();
        releaseObjectGrab();
        hideHoverCursor();
        previousControlRef.current = null;
        undoSwipeRef.current = null;
        return;
      }
      const landmarks = hands[0];
      if (!landmarks) return;
      const result = gestureEngineRef.current.update(landmarks, timestamp);
      const wristRoll = detectWristRoll(landmarks, result, timestamp);
      if (wristRoll !== "idle") {
        gestureEngineRef.current.reset();
        sceneRef.current?.endStroke();
        releaseSideViewOrbit();
        releaseObjectGrab();
        hideHoverCursor();
        previousControlRef.current = null;
        undoSwipeRef.current = null;
        updateGesture("other");
        return;
      }
      const elapsedSeconds = Number.isFinite(lastHandSampleAtRef.current)
        ? clamp(
            (timestamp - lastHandSampleAtRef.current) / 1000,
            1 / 120,
            0.1,
          )
        : 1 / 60;
      lastHandSampleAtRef.current = timestamp;
      if (!filteredTipRef.current) {
        filteredTipRef.current = { ...result.indexTip };
      } else {
        const tipBlend = responsiveBlend(
          Math.hypot(
            result.indexTip.x - filteredTipRef.current.x,
            result.indexTip.y - filteredTipRef.current.y,
          ),
          elapsedSeconds,
          result.drawingPinch ? 0.62 : 0.5,
          1.2,
        );
        const tipDepthBlend = responsiveBlend(
          Math.abs(result.indexTip.z - filteredTipRef.current.z),
          elapsedSeconds,
          0.42,
          0.65,
        );
        filteredTipRef.current.x +=
          (result.indexTip.x - filteredTipRef.current.x) * tipBlend;
        filteredTipRef.current.y +=
          (result.indexTip.y - filteredTipRef.current.y) * tipBlend;
        filteredTipRef.current.z +=
          (result.indexTip.z - filteredTipRef.current.z) * tipDepthBlend;
      }
      const filteredTip = filteredTipRef.current;
      if (!pointerHoverActiveRef.current) {
        if (result.brushHover && !menuOpenRef.current) {
          showHoverCursor(filteredTip);
        } else {
          hideHoverCursor();
        }
      }
      if (!filteredGrabRef.current) {
        filteredGrabRef.current = { ...result.grabPoint };
      } else {
        const grabBlend = responsiveBlend(
          Math.hypot(
            result.grabPoint.x - filteredGrabRef.current.x,
            result.grabPoint.y - filteredGrabRef.current.y,
          ),
          elapsedSeconds,
          0.5,
          1,
        );
        const grabDepthBlend = responsiveBlend(
          Math.abs(result.grabPoint.z - filteredGrabRef.current.z),
          elapsedSeconds,
          0.38,
          0.55,
        );
        filteredGrabRef.current.x +=
          (result.grabPoint.x - filteredGrabRef.current.x) * grabBlend;
        filteredGrabRef.current.y +=
          (result.grabPoint.y - filteredGrabRef.current.y) * grabBlend;
        filteredGrabRef.current.z +=
          (result.grabPoint.z - filteredGrabRef.current.z) * grabDepthBlend;
      }

      if (result.sideViewControl && !menuOpenRef.current) {
        sceneRef.current?.endStroke();
        releaseObjectGrab();
        hideHoverCursor();
        depthCalibrationRef.current = null;
        previousControlRef.current = null;
        undoSwipeRef.current = null;

        const screenX = 1 - filteredTip.x;
        const previousSideView = sideViewControlRef.current;
        if (!sideViewOrbitActiveRef.current) {
          sceneRef.current?.beginSideViewOrbit();
          sideViewOrbitActiveRef.current = true;
          setSideViewOrbiting(true);
          setSideViewReturning(false);
        } else if (previousSideView) {
          sceneRef.current?.rotateSideView(
            (screenX - previousSideView.x) * 15,
          );
        }
        sideViewControlRef.current = {
          x: screenX,
          y: filteredTip.y,
        };
        updateGesture("other");
        return;
      }
      releaseSideViewOrbit();

      if (result.pose === "openPalm" && !menuOpenRef.current) {
        hideHoverCursor();
        const screenX = 1 - result.palm.x;
        const previousSwipe = undoSwipeRef.current;
        if (
          !previousSwipe ||
          timestamp - previousSwipe.timestamp > 120
        ) {
          undoSwipeRef.current = {
            x: screenX,
            startedAt: timestamp,
            travel: 0,
            timestamp,
          };
        } else {
          const deltaX = screenX - previousSwipe.x;
          const travel =
            deltaX < 0
              ? previousSwipe.travel - deltaX
              : Math.max(0, previousSwipe.travel - deltaX * 0.5);
          const elapsed = timestamp - previousSwipe.startedAt;
          undoSwipeRef.current = {
            x: screenX,
            startedAt: previousSwipe.startedAt,
            travel,
            timestamp,
          };
          if (
            travel > 0.16 &&
            elapsed < 440 &&
            timestamp - lastUndoAtRef.current > 850
          ) {
            lastUndoAtRef.current = timestamp;
            undoSwipeRef.current = null;
            sceneRef.current?.endStroke();
            previousControlRef.current = null;
            undoArtwork();
            updateGesture("openPalm");
            return;
          }
        }
      } else {
        undoSwipeRef.current = null;
      }

      const menuFist =
        menuOpenRef.current &&
        result.fingerCount === 0 &&
        result.objectGrabIntent;
      const controlPose = menuFist
        ? "fist"
        : result.objectGrab
          ? "grab"
          : result.drawingPinch
            ? "draw"
            : result.objectGrabIntent
              ? "other"
              : result.pose;
      updateGesture(controlPose);

      if (menuOpenRef.current) {
        sceneRef.current?.endStroke();
        releaseObjectGrab();
        previousControlRef.current = null;

        if (menuFist && !eraserEnabledRef.current) {
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
      const controlBlend = previous
        ? responsiveBlend(
            Math.hypot(
              result.palm.x - previous.x,
              result.palm.y - previous.y,
            ),
            elapsedSeconds,
            0.5,
            0.85,
          )
        : 1;
      const scaleBlend = previous
        ? responsiveBlend(
            Math.abs(result.handScale - previous.scale),
            elapsedSeconds,
            0.44,
            0.28,
          )
        : 1;
      const controlSample =
        previous?.pose === controlPose
          ? {
              pose: controlPose,
              x: previous.x + (result.palm.x - previous.x) * controlBlend,
              y: previous.y + (result.palm.y - previous.y) * controlBlend,
              scale:
                previous.scale +
                (result.handScale - previous.scale) * scaleBlend,
            }
          : {
              pose: controlPose,
              x: result.palm.x,
              y: result.palm.y,
              scale: result.handScale,
            };

      if (controlPose === "grab") {
        sceneRef.current?.endStroke();
        depthCalibrationRef.current = null;
        if (!objectGrabActiveRef.current) {
          const selected = sceneRef.current?.beginObjectGrab(
            filteredGrabRef.current,
          );
          if (selected) {
            objectGrabActiveRef.current = true;
            setObjectGrabSelected(true);
            grabDepthCalibrationRef.current = {
              handScale: result.handScale,
              fingerOffset: 0,
              filteredDepth: 0,
            };
          }
        }
        const grabCalibration = grabDepthCalibrationRef.current;
        if (objectGrabActiveRef.current && grabCalibration) {
          const trackedDepth =
            Math.log(
              Math.max(0.025, result.handScale) /
                Math.max(0.025, grabCalibration.handScale),
            ) * 4.6;
          const grabDepthTarget = clamp(trackedDepth, -2.15, 2.15);
          const grabDepthBlend = responsiveBlend(
            Math.abs(
              grabDepthTarget - grabCalibration.filteredDepth,
            ),
            elapsedSeconds,
            0.42,
            2,
          );
          grabCalibration.filteredDepth +=
            (grabDepthTarget - grabCalibration.filteredDepth) *
            grabDepthBlend;
          updateSnapKind(
            sceneRef.current?.moveObjectGrab(
              filteredGrabRef.current,
              grabCalibration.filteredDepth,
            ) ?? "none",
          );
        }
        previousControlRef.current = controlSample;
        return;
      }

      if (objectGrabActiveRef.current) {
        releaseObjectGrab();
      }

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
        const depthBlend = responsiveBlend(
          Math.abs(trackedDepth - calibration.filteredDepth),
          elapsedSeconds,
          0.46,
          2,
        );
        calibration.filteredDepth +=
          (trackedDepth - calibration.filteredDepth) * depthBlend;
        const point = sceneRef.current?.normalizedToArtwork(
          filteredTip,
          calibration.filteredDepth,
        );
        if (point) applyToolAtPoint(point);
      } else {
        sceneRef.current?.endStroke();
      }

      if (previous?.pose === controlPose && controlPose === "pan2d") {
        depthCalibrationRef.current = null;
        sceneRef.current?.pan(
          -stableDelta(controlSample.x - previous.x, 0.00045),
          stableDelta(controlSample.y - previous.y, 0.00045),
        );
      }
      if (previous?.pose === controlPose && controlPose === "orbit3d") {
        depthCalibrationRef.current = null;
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
      detectWristRoll,
      hideHoverCursor,
      releaseSideViewOrbit,
      releaseObjectGrab,
      selectColor,
      selectEraserThickness,
      selectThickness,
      showHoverCursor,
      updateGesture,
      updateSnapKind,
      undoArtwork,
    ],
  );

  const prepareVisionFileset = useCallback(() => {
    if (visionFilesetPromiseRef.current) {
      return visionFilesetPromiseRef.current;
    }
    const promise = import("@mediapipe/tasks-vision")
      .then(({ FilesetResolver }) =>
        FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/wasm",
        ),
      )
      .catch((error) => {
        visionFilesetPromiseRef.current = null;
        throw error;
      });
    visionFilesetPromiseRef.current = promise;
    return promise;
  }, []);

  const prepareHandLandmarker = useCallback(() => {
    if (handLandmarkerRef.current) {
      return Promise.resolve(handLandmarkerRef.current);
    }
    if (handLandmarkerPromiseRef.current) {
      return handLandmarkerPromiseRef.current;
    }

    const promise = (async () => {
      const [{ HandLandmarker }, vision] = await Promise.all([
        import("@mediapipe/tasks-vision"),
        prepareVisionFileset(),
      ]);
      try {
        return await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.52,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      } catch {
        return HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
      }
    })().then(
      (landmarker) => {
        handLandmarkerRef.current = landmarker;
        return landmarker;
      },
      (error) => {
        handLandmarkerPromiseRef.current = null;
        throw error;
      },
    );
    handLandmarkerPromiseRef.current = promise;
    return promise;
  }, [prepareVisionFileset]);

  const prepareFaceLandmarker = useCallback(() => {
    if (faceLandmarkerRef.current) {
      return Promise.resolve(faceLandmarkerRef.current);
    }
    if (faceLandmarkerPromiseRef.current) {
      return faceLandmarkerPromiseRef.current;
    }

    const promise = (async () => {
      const [{ FaceLandmarker }, vision] = await Promise.all([
        import("@mediapipe/tasks-vision"),
        prepareVisionFileset(),
      ]);
      const options = {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU" as const,
        },
        runningMode: "VIDEO" as const,
        numFaces: 1,
        minFaceDetectionConfidence: 0.52,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      };
      try {
        return await FaceLandmarker.createFromOptions(vision, options);
      } catch {
        return FaceLandmarker.createFromOptions(vision, {
          ...options,
          baseOptions: {
            ...options.baseOptions,
            delegate: "CPU",
          },
        });
      }
    })().then(
      (landmarker) => {
        faceLandmarkerRef.current = landmarker;
        return landmarker;
      },
      (error) => {
        faceLandmarkerPromiseRef.current = null;
        throw error;
      },
    );
    faceLandmarkerPromiseRef.current = promise;
    return promise;
  }, [prepareVisionFileset]);

  const stopCamera = useCallback(() => {
    trackingActiveRef.current = false;
    window.cancelAnimationFrame(trackingFrameRef.current);
    videoRef.current?.cancelVideoFrameCallback?.(trackingFrameRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    gestureEngineRef.current.reset();
    previousControlRef.current = null;
    filteredTipRef.current = null;
    filteredGrabRef.current = null;
    depthCalibrationRef.current = null;
    lastHandsSeenAtRef.current = -Infinity;
    noHandsSinceRef.current = -Infinity;
    lastFaceSampleAtRef.current = -Infinity;
    headTurnRef.current = null;
    wristRollRef.current = null;
    lastHandSampleAtRef.current = -Infinity;
    undoSwipeRef.current = null;
    resetSideViewControl();
    releaseObjectGrab();
    hideHoverCursor();
  }, [hideHoverCursor, releaseObjectGrab, resetSideViewControl]);

  useEffect(
    () => () => {
      stopCamera();
      handLandmarkerRef.current?.close();
      handLandmarkerRef.current = null;
      handLandmarkerPromiseRef.current = null;
      faceLandmarkerRef.current?.close();
      faceLandmarkerRef.current = null;
      faceLandmarkerPromiseRef.current = null;
      visionFilesetPromiseRef.current = null;
    },
    [stopCamera],
  );

  const startTrackingLoop = useCallback(() => {
    trackingActiveRef.current = true;

    const handleMissingHands = (timestamp: number) => {
      if (
        timestamp - lastHandsSeenAtRef.current <=
        HAND_TRACKING_GRACE_MS
      ) {
        return;
      }
      gestureEngineRef.current.reset();
      sceneRef.current?.endStroke();
      previousControlRef.current = null;
      filteredTipRef.current = null;
      filteredGrabRef.current = null;
      depthCalibrationRef.current = null;
      lastHandSampleAtRef.current = -Infinity;
      undoSwipeRef.current = null;
      releaseSideViewOrbit();
      releaseObjectGrab();
      hideHoverCursor();
      updateGesture("none");
    };

    const processVideoFrame = (timestamp: number) => {
      if (!trackingActiveRef.current) return;
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
          lastHandsSeenAtRef.current = timestamp;
          noHandsSinceRef.current = -Infinity;
          headTurnRef.current = null;
          processHands(hands, timestamp);
        } else {
          wristRollRef.current = null;
          if (!Number.isFinite(noHandsSinceRef.current)) {
            noHandsSinceRef.current = timestamp;
          }
          if (timestamp - noHandsSinceRef.current >= HEAD_HAND_FREE_MS) {
            processHeadTurn(video, timestamp);
          }
          handleMissingHands(timestamp);
        }
      }
    };

    const scheduleVideoFrame = () => {
      if (!trackingActiveRef.current) return;
      const video = videoRef.current;
      if (video?.requestVideoFrameCallback) {
        trackingFrameRef.current = video.requestVideoFrameCallback(
          (timestamp) => {
            processVideoFrame(timestamp);
            scheduleVideoFrame();
          },
        );
      } else {
        trackingFrameRef.current = window.requestAnimationFrame(
          (timestamp) => {
            processVideoFrame(timestamp);
            scheduleVideoFrame();
          },
        );
      }
    };

    scheduleVideoFrame();
  }, [
    hideHoverCursor,
    processHeadTurn,
    processHands,
    releaseSideViewOrbit,
    releaseObjectGrab,
    updateGesture,
  ]);

  const startCamera = async () => {
    setCameraState("requesting");
    setCameraError("");
    setDemoMode(false);
    try {
      primeAudio();
      const landmarkerPromise = prepareHandLandmarker();
      void prepareFaceLandmarker().catch(() => undefined);
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          "Camera access is unavailable here. Open Airloom in a current browser over HTTPS.",
        );
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 60 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (!videoRef.current) throw new Error("Camera surface unavailable.");
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraState("calibrating");
      await landmarkerPromise;
      if (streamRef.current !== stream) return;
      lastVideoTimeRef.current = -1;
      lastHandSampleAtRef.current = -Infinity;
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

  const pointerPosition = (event: ReactPointerEvent<HTMLElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / rect.width,
      y: (event.clientY - rect.top) / rect.height,
      z: 0,
    };
  };

  const pointerInHandSpace = (pointer: Point3) => ({
    ...pointer,
    x: 1 - pointer.x,
  });

  const handleCanvasPointerDown = (
    event: ReactPointerEvent<HTMLElement>,
  ) => {
    if (!demoMode) return;
    if (menuOpenRef.current) return;
    if (pointerIsOverCanvasUi(event.clientX, event.clientY)) return;
    if (event.target !== event.currentTarget) return;
    const pointer = pointerPosition(event);
    const handPointer = pointerInHandSpace(pointer);
    lastPointerPositionRef.current = pointer;
    pointerHoverActiveRef.current = false;
    hideHoverCursor();
    previousControlRef.current = null;

    if (event.button === 2) {
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      pointerNavigationRef.current = {
        mode: "orbit",
        x: pointer.x,
        y: pointer.y,
      };
      return;
    }
    if (event.button === 1) {
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      pointerNavigationRef.current = {
        mode: "pan",
        x: pointer.x,
        y: pointer.y,
      };
      return;
    }
    if (event.button !== 0) return;

    if (event.shiftKey) {
      releaseObjectGrab();
      const selected = sceneRef.current?.beginObjectGrab(handPointer);
      if (!selected) return;
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      pointerObjectGrabRef.current = true;
      objectGrabActiveRef.current = true;
      pointerGrabDepthRef.current = 0;
      setObjectGrabSelected(true);
      updateSnapKind("none");
      return;
    }

    const now = performance.now();
    const previousClick = lastPointerClickPositionRef.current;
    const repeatedClick =
      now - lastPointerClickAtRef.current < 380 &&
      previousClick !== null &&
      Math.hypot(
        pointer.x - previousClick.x,
        pointer.y - previousClick.y,
      ) < 0.025;
    lastPointerClickAtRef.current = now;
    lastPointerClickPositionRef.current = pointer;
    if (repeatedClick || event.detail > 1) return;

    pointerClickStrokeCountRef.current = 1;
    pointerPressRef.current = {
      x: pointer.x,
      y: pointer.y,
      moved: false,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    pointerDrawingRef.current = true;
    sceneRef.current?.endStroke();
    const point = sceneRef.current?.normalizedToArtwork(
      handPointer,
      0,
    );
    if (point) applyToolAtPoint(point);
  };

  const handleCanvasPointerMove = (
    event: ReactPointerEvent<HTMLElement>,
  ) => {
    if (!demoMode) return;
    if (pointerIsOverCanvasUi(event.clientX, event.clientY)) {
      if (pointerDrawingRef.current) {
        pointerDrawingRef.current = false;
        pointerPressRef.current = null;
        pointerClickStrokeCountRef.current = 0;
        sceneRef.current?.endStroke();
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
      }
      pointerHoverActiveRef.current = false;
      hideHoverCursor();
      return;
    }
    const pointer = pointerPosition(event);
    const handPointer = pointerInHandSpace(pointer);
    lastPointerPositionRef.current = pointer;

    if (pointerObjectGrabRef.current) {
      updateSnapKind(
        sceneRef.current?.moveObjectGrab(
          handPointer,
          pointerGrabDepthRef.current,
        ) ?? "none",
      );
      return;
    }

    const navigation = pointerNavigationRef.current;
    if (navigation) {
      const deltaX = pointer.x - navigation.x;
      const deltaY = pointer.y - navigation.y;
      if (navigation.mode === "pan") {
        sceneRef.current?.pan(deltaX, deltaY);
      } else {
        sceneRef.current?.orbit(deltaX, deltaY, 0);
      }
      navigation.x = pointer.x;
      navigation.y = pointer.y;
      return;
    }

    if (!pointerDrawingRef.current) {
      if (
        event.pointerType !== "touch" &&
        event.target === event.currentTarget &&
        !menuOpenRef.current
      ) {
        pointerHoverActiveRef.current = true;
        showHoverCursor(handPointer);
      } else {
        pointerHoverActiveRef.current = false;
        hideHoverCursor();
      }
      return;
    }

    const press = pointerPressRef.current;
    if (
      press &&
      Math.hypot(pointer.x - press.x, pointer.y - press.y) > 0.008
    ) {
      press.moved = true;
      pointerClickStrokeCountRef.current = 0;
    }
    const point = sceneRef.current?.normalizedToArtwork(
      handPointer,
      0,
    );
    if (point) applyToolAtPoint(point);
  };

  const handleCanvasPointerUp = (
    event: ReactPointerEvent<HTMLElement>,
    cancelled = false,
  ) => {
    if (!demoMode) return;
    if (pointerObjectGrabRef.current) {
      pointerObjectGrabRef.current = false;
      releaseObjectGrab();
    }
    pointerNavigationRef.current = null;
    pointerDrawingRef.current = false;
    pointerPressRef.current = null;
    sceneRef.current?.endStroke();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (
      !cancelled &&
      event.pointerType !== "touch" &&
      !menuOpenRef.current &&
      !pointerIsOverCanvasUi(event.clientX, event.clientY)
    ) {
      const pointer = pointerPosition(event);
      lastPointerPositionRef.current = pointer;
      pointerHoverActiveRef.current = true;
      showHoverCursor(pointerInHandSpace(pointer));
    } else {
      pointerHoverActiveRef.current = false;
      hideHoverCursor();
    }
  };

  const handleCanvasDoubleClick = (
    event: ReactMouseEvent<HTMLElement>,
  ) => {
    if (!demoMode) return;
    if (event.target !== event.currentTarget || menuOpenRef.current) return;
    event.preventDefault();
    if (
      !eraserEnabledRef.current &&
      pointerClickStrokeCountRef.current > 0
    ) {
      sceneRef.current?.undo();
    }
    pointerClickStrokeCountRef.current = 0;
    toggleEraser();
  };

  const handleCanvasWheel = (event: ReactWheelEvent<HTMLElement>) => {
    if (!demoMode) return;
    if (event.target !== event.currentTarget || menuOpenRef.current) return;
    event.preventDefault();

    if (pointerObjectGrabRef.current && lastPointerPositionRef.current) {
      pointerGrabDepthRef.current = clamp(
        pointerGrabDepthRef.current - event.deltaY * 0.0025,
        -2.15,
        2.15,
      );
      updateSnapKind(
        sceneRef.current?.moveObjectGrab(
          pointerInHandSpace(lastPointerPositionRef.current),
          pointerGrabDepthRef.current,
        ) ?? "none",
      );
    } else if (event.ctrlKey) {
      sceneRef.current?.orbit(0, 0, -event.deltaY * 0.0007);
    } else if (event.altKey || event.shiftKey) {
      sceneRef.current?.orbit(
        event.deltaX * 0.0012,
        event.deltaY * 0.0012,
        0,
      );
    } else {
      sceneRef.current?.pan(
        -event.deltaX * 0.0012,
        event.deltaY * 0.0012,
      );
    }
  };

  const handleCanvasPointerLeave = () => {
    if (!demoMode) {
      pointerHoverActiveRef.current = false;
      hideHoverCursor();
      return;
    }
    if (
      pointerDrawingRef.current ||
      pointerObjectGrabRef.current ||
      pointerNavigationRef.current
    ) {
      return;
    }
    pointerHoverActiveRef.current = false;
    hideHoverCursor();
  };

  const handleSideViewPointerDown = (
    event: ReactPointerEvent<HTMLElement>,
  ) => {
    event.stopPropagation();
    if (!demoMode || event.button !== 0 || !hasArtwork) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    sideViewPointerRef.current = {
      x: event.clientX,
      pointerId: event.pointerId,
    };
    sideViewOrbitActiveRef.current = true;
    sceneRef.current?.beginSideViewOrbit();
    setSideViewOrbiting(true);
    setSideViewReturning(false);
  };

  const handleSideViewPointerMove = (
    event: ReactPointerEvent<HTMLElement>,
  ) => {
    event.stopPropagation();
    const pointer = sideViewPointerRef.current;
    if (!demoMode || !pointer || pointer.pointerId !== event.pointerId) {
      return;
    }
    const width = Math.max(1, event.currentTarget.clientWidth);
    sceneRef.current?.rotateSideView(
      ((event.clientX - pointer.x) / width) * 15,
    );
    pointer.x = event.clientX;
  };

  const handleSideViewPointerUp = (
    event: ReactPointerEvent<HTMLElement>,
  ) => {
    event.stopPropagation();
    const pointer = sideViewPointerRef.current;
    if (!pointer || pointer.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    releaseSideViewOrbit();
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
        className={`white-workspace ${demoMode ? "mouse-mode" : "camera-mode"}`}
        onPointerDown={handleCanvasPointerDown}
        onPointerMove={handleCanvasPointerMove}
        onPointerUp={handleCanvasPointerUp}
        onPointerCancel={(event) => handleCanvasPointerUp(event, true)}
        onPointerLeave={handleCanvasPointerLeave}
        onDoubleClick={handleCanvasDoubleClick}
        onWheel={handleCanvasWheel}
        onContextMenu={(event) => {
          if (demoMode && event.target === event.currentTarget) {
            event.preventDefault();
          }
        }}
      >
        <canvas ref={canvasRef} className="air-canvas" />
        <div
          ref={hoverCursorRef}
          className={`brush-hover-cursor ${eraserEnabled ? "is-eraser" : ""}`}
          style={{
            color: eraserEnabled ? "#111111" : selectedColor.value,
          }}
          aria-hidden="true"
        />
        <aside
          data-block-canvas-input
          className={`side-view-inset ${hasArtwork ? "is-visible" : ""} ${sideViewOrbiting ? "is-orbiting" : ""}`}
          aria-label="Right side view, locked ninety degrees from the main view"
          onPointerDown={handleSideViewPointerDown}
          onPointerMove={handleSideViewPointerMove}
          onPointerUp={handleSideViewPointerUp}
          onPointerCancel={handleSideViewPointerUp}
          onLostPointerCapture={() => releaseSideViewOrbit()}
        >
          <header>
            <div>
              <span>SIDE VIEW</span>
              <strong>
                {sideViewOrbiting
                  ? "FINGER ORBIT"
                  : sideViewReturning
                    ? "SNAPPING +90°"
                    : "RIGHT +90°"}
              </strong>
            </div>
            <i />
          </header>
          <div className="side-view-stage">
            <canvas ref={sideCanvasRef} className="side-view-canvas" />
            <span className="side-axis side-axis-horizontal" />
            <span className="side-axis side-axis-vertical" />
            <small className="side-depth-label is-front">FRONT</small>
            <small className="side-depth-label is-back">BACK</small>
          </div>
          <footer>
            <span>DEPTH</span>
            <strong>
              {snapKind === "vertex"
                ? "VERTEX LOCK"
                : snapKind === "edge"
                  ? "EDGE LOCK"
                  : objectGrabSelected
                    ? "OBJECT HELD"
                    : "LIVE"}
            </strong>
          </footer>
        </aside>

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
              : sideViewOrbiting
                ? "Turning side view"
                : sideViewReturning
                  ? "Side view returning"
              : demoMode
                ? eraserEnabled
                  ? "Mouse eraser"
                  : "Mouse drawing"
                : gesture === "grab"
                  ? objectGrabSelected
                    ? snapKind === "none"
                      ? "Moving object"
                      : `${snapKind === "vertex" ? "Vertex" : "Edge"} snap locked`
                    : "Close fist over an object"
                  : eraserEnabled
                    ? `Eraser · ${POSE_LABELS[gesture]}`
                    : POSE_LABELS[gesture]}
        </div>

        <aside
          data-block-canvas-input
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
                  onPointerEnter={() => {
                    void prepareHandLandmarker().catch(() => undefined);
                  }}
                  onFocus={() => {
                    void prepareHandLandmarker().catch(() => undefined);
                  }}
                  onClick={(event) => {
                    event.stopPropagation();
                    void startCamera();
                  }}
                  disabled={cameraState === "requesting"}
                >
                  {cameraState === "error"
                    ? "Try again"
                    : "Enable camera"}
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
                    ? "LIVE HAND + FACE"
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
          data-block-canvas-input
          className={`brush-cartridge-shell ${menuOpen ? "is-open" : ""} ${eraserEnabled ? "is-eraser" : ""}`}
          style={cartridgeStyle}
          onPointerDown={(event) => event.stopPropagation()}
          inert={!demoMode}
          aria-disabled={!demoMode}
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

        <nav
          data-block-canvas-input
          className="minimal-tools"
          aria-label="Artwork controls"
          inert={!demoMode}
          aria-disabled={!demoMode}
        >
          <button
            onClick={undoArtwork}
          >
            Undo
          </button>
          <button
            onClick={() => {
              releaseObjectGrab();
              resetSideViewControl();
              sceneRef.current?.resetView();
            }}
          >
            Reset view
          </button>
          <button
            onClick={() => {
              releaseObjectGrab();
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
          <aside
            data-block-canvas-input
            className="gesture-guide"
            aria-label="Complete Airloom control map"
          >
            <header>
              <div>
                <span>CONTROL MAP</span>
                <strong>ALL AIRLOOM CONTROLS</strong>
              </div>
              <small>CLICK ? TO CLOSE</small>
            </header>
            <div className="gesture-guide-columns" aria-hidden="true">
              <span>ACTION</span>
              <b>HAND + CAMERA</b>
              <strong>KEYBOARD + MOUSE</strong>
            </div>
            {[
              ["CURSOR", "Thumb + index apart", "Hover over canvas"],
              ["DRAW", "Pinch thumb + index; separate to stop", "Left-drag; release to stop"],
              ["PAN", "Two fingers + move", "Two-finger swipe or middle-drag"],
              ["ORBIT", "Three fingers + move", "Right-drag or Option/Shift + swipe"],
              ["ZOOM", "Three fingers + hand in/out", "Trackpad pinch or Ctrl + wheel"],
              ["SIDE VIEW", "Index up + move; release springs back", "Drag viewer; release springs back"],
              ["PALETTE", "No hands; jerk head left to open, right to close", "Click side tab; Escape closes"],
              ["COLOR", "Fist + move inside palette", "Click a swatch"],
              ["TOOL SIZE", "Open palm + move inside palette", "Drag active tool slider"],
              ["ERASER", "Quick palm-to-back wrist roll", "Double-click canvas or click Eraser"],
              ["MOVE", "Fist over object; open to release", "Shift + left-drag object; release"],
              ["DEPTH", "Grab + move hand in/out", "Wheel while holding object"],
              ["SNAP", "Release near an edge or vertex", "Release near an edge or vertex"],
              ["UNDO", "Open-palm swipe left", "Cmd/Ctrl+Z or click Undo"],
              ["RESET VIEW", "Manual button", "Click Reset view"],
              ["CLEAR", "Manual button", "Click Clear"],
              ["EXPORT", "Manual button", "Click Export"],
              ["MODE", "Enable camera", "Exit camera to unlock controls"],
            ].map(([action, hand, combined]) => (
              <div className="gesture-guide-row" key={action}>
                <span>{action}</span>
                <b>{hand}</b>
                <strong>{combined}</strong>
              </div>
            ))}
          </aside>
        )}

        <p className="canvas-hint">
          {cameraState === "active"
            ? "Gesture mode active · Exit camera to use keyboard + mouse"
            : cameraState === "calibrating"
              ? "Loading hand tracking · Keyboard + mouse locked"
              : cameraState === "requesting"
                ? "Requesting camera access · Keyboard + mouse locked"
                : "Left-drag to draw · Double-click toggles eraser · Click ? for every control"}
        </p>
      </section>
    </main>
  );
}
