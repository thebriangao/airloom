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
- Line Smoothing: enabled by default to lightly reduce tracking wobble on longer, visibly rough strokes while preserving short marks, endpoints, and deliberate corners

Mouse, keyboard, and visible-button fallbacks are included.

Export downloads either a clean PNG of the current view, a full-color GLB for
Blender and modern 3D tools, or an STL mesh for Fusion, CAD, and 3D printing.

## Phone and tablet controls

- Draw or Erase: tap the active tool button to switch, then drag one finger
- Move: select Move and drag to pan the canvas
- View: select View and drag to orbit; use the nearby minus and plus controls to zoom
- Object: select Object, drag a stroke group, and release near an edge or vertex to snap
- Brush: open the right-side tab to choose color and thickness
- More: access Line Smoothing, Shape Assist, reset, clear, export, and the complete control guide

The touch layout uses a bottom dock, a full-width phone brush sheet, a wider
landscape tool tray, and tablet-sized controls with safe-area spacing.

## Development

Requires Node.js `>=22.13.0`.

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

The camera requires HTTPS in production or localhost during development. Hand
and face tracking run locally in the browser. Three.js builds, transforms,
erases, renders, and exports the luminous 3D stroke geometry.
