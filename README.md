# Airloom

Airloom is a browser-based 3D painting instrument controlled by hand gestures.
Touching the thumb and index fingertips draws, two fingers pan, three fingers
orbit, a quick head turn opens or closes the brush cartridge, and a fast wrist
roll toggles the eraser.

## Interaction map

- Thumb and index fingertips touching: draw a 3D stroke; separate them to stop immediately
- Two raised fingers: pan the artwork in 2D
- Three raised fingers: orbit and zoom the artwork
- With no hands visible, jerk your head left to open the cartridge and right to close it
- Roll an open hand quickly from palm-facing to back-facing to toggle the eraser
- Fist in the cartridge: move across the camera frame to select from a 5x5 color grid
- Open palm in the cartridge: move left or right for continuous brush or eraser thickness
- One finger in eraser mode: remove intersecting stroke segments
- Optional Shape Assist: refine confident lines, arcs, circles, ellipses, triangles, rectangles, pentagons, and hexagons while preserving their original scale and hand-drawn character

Mouse, keyboard, and visible-button fallbacks are included.

## Development

Requires Node.js `>=22.13.0`.

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

The camera requires HTTPS in production or localhost during development. Hand
and face tracking run locally in the browser. Three.js builds, transforms,
erases, and renders the luminous 3D stroke geometry.
