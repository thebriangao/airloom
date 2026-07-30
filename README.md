# Airloom

Airloom is a browser-based 3D painting instrument controlled by hand gestures.
One finger draws, two fingers pan, three fingers orbit, and a finger snap ejects
the brush cartridge.

## Interaction map

- One raised finger: draw a 3D stroke
- Two raised fingers: pan the artwork in 2D
- Three raised fingers: orbit and zoom the artwork
- Finger snap: eject or holster the brush cartridge
- Fist in the cartridge: move across the camera frame to select from a 5x5 color grid
- Open palm in the cartridge: move left or right for continuous stroke thickness

Mouse, keyboard, and visible-button fallbacks are included.

## Development

Requires Node.js `>=22.13.0`.

```bash
npm install
npm run dev
npm test
```

The camera requires HTTPS in production or localhost during development.
Hand-tracking inference runs locally in the browser using MediaPipe. Three.js
builds and renders the luminous 3D stroke geometry.
