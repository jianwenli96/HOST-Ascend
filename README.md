# HOST: Human-to-robot One-shot Skill Transfer

HOST acquires a novel manipulation skill from a single human demonstration video, without any
parameter update. It does this through two mechanisms:

- **Target coupling** (`alignment/`) aligns the visual demonstration to the robot's own
  trajectory on a shared task-progress manifold, so the robot's prediction target at each moment
  is the corresponding point in the demonstration rather than raw clock time.
- **Self-grounded prediction** (`wam/`) localizes the robot's current progress within that
  coupled demonstration, predicts the robot's own future observations conditioned on the
  localized segment, and derives motor commands from that predicted future — re-expressing the
  demonstrated behaviour in the robot's own embodiment and viewpoint before generating actions.

## Repository layout

```text
HOST/
├── wam/          # Self-grounded prediction: dual-expert video/action diffusion model
├── alignment/    # Target coupling: TCC + Smooth DTW video alignment training
├── coupling/     # Reserved, not yet populated
└── docs/         # Reserved for future paper-mapping documentation
```

Each module is self-contained with its own environment, training scripts, and README:

- [`wam/README.md`](./wam/README.md) ([中文](./wam/README_zh.md))
- [`alignment/README.md`](./alignment/README.md)

Both modules' shipped configs reference this team's internal cluster data paths; see each
module's own README for the exact data format required to train on your own data, and
[`OPEN_SOURCE_PATH_TODOS.md`](./OPEN_SOURCE_PATH_TODOS.md) for the current status of making every
remaining internal path configurable.

## BibTeX

If you find our work helpful, please consider citing:

```bibtex
@article{host2026,
  title={HOST: Human-to-robot One-shot Skill Transfer},
  % TODO: fill in authors, venue, year, DOI/arXiv once available
}
```
