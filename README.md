# AlphaFold2 Visually

An interactive 3D visualization that explains the AlphaFold2 algorithm step by step — inspired by [bbycroft.net/llm](https://bbycroft.net/llm).

This is **not** a molecular structure viewer. It visualizes the **algorithm itself**: how Evoformer, Triangle Attention, and IPA turn amino acid sequences into predicted protein structures.

<!-- TODO: add a screenshot/gif here once the visuals are polished -->

## What You'll See

The app walks through the AlphaFold2 pipeline as a series of interactive 3D scenes:

1. **Protein Intro** — What is a protein? Amino acid chains in 3D.
2. **Architecture Overview** — The full AlphaFold2 pipeline at a glance (MSA → Evoformer → Structure Module).
3. **Evoformer Detail** — Deep dive into Triangle Attention: watch how information flows between residue pairs through triangular message passing on the pair representation matrix.
4. **Structure Module Detail** — How the predicted 3D coordinates emerge from the learned representations.

Each scene features:
- Freely rotatable 3D camera (orbit, zoom, pan)
- Step-by-step animation with play/pause controls
- Mathematical formulas rendered with KaTeX
- Plain-language explanations alongside every step

## Tech Stack

| Layer | Choice |
|-------|--------|
| Framework | React 19 + TypeScript + Vite |
| 3D Engine | Three.js + react-three-fiber + drei |
| Post-processing | @react-three/postprocessing (bloom, vignette) |
| Animation | GSAP + useFrame |
| Math Rendering | KaTeX |
| State | Zustand |
| Debug | Leva |

## Getting Started

```bash
# Clone
git clone git@github.com:BriceLucifer/alphafold2-visualization.git
cd alphafold2-visualization

# Install dependencies
npm install

# Start dev server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Build & Deploy

```bash
# Production build
npm run build

# Preview the build locally
npm run preview
```

The project includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) for automatic deployment to GitHub Pages on push to `main`.

## Project Structure

```
src/
  scenes/           # Full-screen 3D scene components (one per pipeline stage)
  primitives/       # Reusable 3D building blocks
    Residue.tsx         # Single amino acid node (sphere + label)
    ResidueChain.tsx    # Chain of residues with entrance animation
    PairMatrix.tsx      # N×N heatmap plane (InstancedMesh + diverging colormap)
    TriangleHighlight.tsx  # Glowing triangle overlay on pair matrix
    AttentionFlow.tsx   # Particle stream showing information flow
  data/             # Synthetic data generators (no real model weights)
  store/            # Zustand animation state
  ui/               # 2D overlay controls (playback bar, info panel, title)
  config.ts         # All magic numbers in one place (colors, sizes, timing)
```

## Roadmap

- [x] **Phase 1** — Triangle Attention MVP: single-module deep dive with full animation
- [ ] **Phase 2** — Complete pipeline: MSA embedding, full Evoformer, Structure Module, Recycling with camera transitions between modules
- [ ] **Phase 3** — Real data: Rust/Wasm inference (ESMFold or simplified), user-supplied sequences

## Acknowledgments

- [AlphaFold2 paper](https://www.nature.com/articles/s41586-021-03819-2) — Jumper et al., 2021
- [bbycroft.net/llm](https://bbycroft.net/llm) — Direct inspiration for the interactive visual style
- [Bartosz Ciechanowski](https://ciechanow.ski) — Gold standard for scroll-driven 3D explainers
- [Distill.pub](https://distill.pub) — Narrative style reference

## License

MIT
