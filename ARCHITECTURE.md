# Airloom architecture

## Product contract

Airloom is a camera-first 3D painting studio controlled by one hand.

- One raised finger draws a continuous 3D stroke.
- Two raised fingers pan the complete artwork in the screen plane.
- Three raised fingers orbit the complete artwork and use hand depth to zoom.
- A finger snap ejects or holsters the brush cartridge only when a microphone
  transient confirms the visual snap inside the same short time window.
- A two-hand clap plus a microphone transient toggles the eraser.
- While the menu is open, a fist moved in two dimensions selects from a
  five-by-five color grid.
- While the menu is open, an open palm moved left or right continuously adjusts
  brush or eraser thickness.
- Eraser mode removes intersecting stroke segments instead of painting.

Mouse, keyboard, and visible controls remain available as accessibility and
reliability fallbacks.

## Runtime boundaries

1. `GestureEngine` converts 21 MediaPipe landmarks into stable semantic poses.
   It owns finger classification, temporal debouncing, snap detection, and
   gesture precedence.
2. `AirloomStudio` owns camera and microphone permission, video inference,
   transient sound analysis, product state, menu state, and mapping confirmed
   gestures to actions.
3. `AirScene` owns Three.js resources, 3D curve construction, view transforms,
   transformed artwork-space ray projection, segment erasing, undo, clearing,
   and export rendering. Brush and eraser thickness are stored as separate
   continuous normalized values.
4. React renders product chrome only. High-frequency landmark updates use refs
   and direct canvas operations to avoid rerendering the interface every frame.

## Gesture precedence

1. A microphone-confirmed clap has the highest priority and toggles the eraser.
2. A microphone-confirmed snap is edge-triggered and toggles the cartridge.
3. When the cartridge is open, drawing, panning, and orbiting are disabled.
4. In the brush cartridge, fist position selects color in two dimensions and open-palm
   horizontal position controls a continuous thickness value.
5. In the eraser cartridge, colors are removed and open-palm position controls
   only eraser size.
6. Outside the menu, one finger draws or erases, two fingers pan, and three
   fingers orbit.
7. An unrecognized pose or lost hand ends the active stroke.

## Reliability

- Poses must remain stable before they become active.
- Snap detection requires a thumb-middle-finger contact followed by rapid
  separation and downward middle-finger motion.
- A cooldown prevents one snap from toggling the menu multiple times.
- Visual snap and clap candidates do nothing unless a sharp microphone
  transient occurs within their confirmation window.
- The hand tracker accepts two hands so clap confirmation cannot be triggered
  by sound alone.
- Camera coordinates are mirrored before mapping into 3D space.
- Screen rays are transformed into current artwork-local coordinates before
  points are added or removed, so drawing stays beneath the cursor after pans
  and rotations.
- Stroke points are filtered and distance-throttled before rebuilding geometry.
- Visible controls and keyboard shortcuts remain available if a gesture is
  difficult to perform or detect.
- A synthesized Web Audio cue confirms cartridge open, close, color, and thickness
  changes without requiring downloaded sound assets.
