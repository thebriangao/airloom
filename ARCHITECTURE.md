# Airloom architecture

## Product contract

Airloom is a camera-first 3D painting studio controlled by one hand.

- One raised finger draws a continuous 3D stroke.
- Two raised fingers pan the complete artwork in the screen plane.
- Three raised fingers orbit the complete artwork and use hand depth to zoom.
- A finger snap opens or closes the brush menu.
- While the menu is open, a fist moved left or right selects one of eight colors.
- While the menu is open, an open palm moved left or right selects one of five stroke sizes.

Mouse, keyboard, and visible controls remain available as accessibility and
reliability fallbacks.

## Runtime boundaries

1. `GestureEngine` converts 21 MediaPipe landmarks into stable semantic poses.
   It owns finger classification, temporal debouncing, snap detection, and
   gesture precedence.
2. `AirloomStudio` owns camera permission, the video inference loop, product
   state, menu state, and mapping semantic gestures to actions.
3. `AirScene` owns Three.js resources, 3D curve construction, view transforms,
   undo, clearing, and export rendering.
4. React renders product chrome only. High-frequency landmark updates use refs
   and direct canvas operations to avoid rerendering the interface every frame.

## Gesture precedence

1. Snap is edge-triggered and has the highest priority.
2. When the menu is open, drawing, panning, and orbiting are disabled.
3. In the menu, fist selects color and open palm selects stroke size.
4. Outside the menu, one finger draws, two fingers pan, and three fingers orbit.
5. An unrecognized pose or lost hand ends the active stroke.

## Reliability

- Poses must remain stable before they become active.
- Snap detection requires a thumb-middle-finger contact followed by rapid
  separation and downward middle-finger motion.
- A cooldown prevents one snap from toggling the menu multiple times.
- Camera coordinates are mirrored before mapping into 3D space.
- Stroke points are filtered and distance-throttled before rebuilding geometry.
- Visible controls and keyboard shortcuts remain available if a gesture is
  difficult to perform or detect.
