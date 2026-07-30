# Airloom

Airloom is a browser-based 3D painting instrument controlled by hand gestures.
One finger draws, two fingers pan, three fingers orbit, an audible finger snap
ejects the brush cartridge, and a clap toggles the eraser.

## Interaction map

- One raised finger: draw a 3D stroke
- Two raised fingers: pan the artwork in 2D
- Three raised fingers: orbit and zoom the artwork
- Finger snap plus its sound: eject or holster the brush cartridge
- Two-hand clap plus its sound: toggle the eraser
- Fist in the cartridge: move across the camera frame to select from a 5x5 color grid
- Open palm in the cartridge: move left or right for continuous brush or eraser thickness
- One finger in eraser mode: remove intersecting stroke segments

Mouse, keyboard, and visible-button fallbacks are included.

## Development

Requires Node.js `>=22.13.0`.

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

The camera and microphone require HTTPS in production or localhost during
development. Hand-tracking inference and transient sound confirmation run
locally in the browser. Three.js builds, transforms, erases, and renders the
luminous 3D stroke geometry.
