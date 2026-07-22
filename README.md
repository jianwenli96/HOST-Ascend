# HOST: Human-to-robot One-shot Skill Transfer

**Project website:** [https://host-site.host-robotics.workers.dev/](https://host-site.host-robotics.workers.dev/)

HOST acquires a novel manipulation skill from a single human demonstration video, without any
parameter update. It does this through two mechanisms:

- **Target coupling** (`alignment/` + `coupling/`) aligns the visual demonstration to the robot's
  own trajectory on a shared task-progress manifold, so the robot's prediction target at each
  moment is the corresponding point in the demonstration rather than raw clock time.
- **Self-grounded prediction** (`policy_training/`) localizes the robot's current progress within
  that coupled demonstration, predicts the robot's own future observations conditioned on the
  localized segment, and derives motor commands from that predicted future — re-expressing the
  demonstrated behaviour in the robot's own embodiment and viewpoint before generating actions.

## Repository layout

The four directories run in pipeline order:

```text
HOST/
├── data_preprocessing/   # Dataset format + task-grouping preprocessing (feeds alignment/)
├── alignment/            # Target coupling training: TCC + Smooth DTW video alignment
├── coupling/             # Progress alignment: converts alignment/'s DTW output into info_dtw.json
└── policy_training/      # Self-grounded prediction: dual-expert video/action diffusion model
```

Each module is self-contained with its own environment/scripts and README:

- [`data_preprocessing/README.md`](./data_preprocessing/README.md) — the shared dataset schema
  (single source of truth for `alignment/` and `policy_training/`) plus the task-grouping scripts
  that build `task_paths.json`.
- [`alignment/README.md`](./alignment/README.md)
- [`coupling/README.md`](./coupling/README.md) — converts `alignment/`'s own DTW training/eval
  output into the per-episode progress data (`info_dtw.json`) `policy_training/` reads.
- [`policy_training/README.md`](./policy_training/README.md) ([中文](./policy_training/README_zh.md))

`alignment/`'s and `policy_training/`'s shipped configs reference this team's internal cluster data
paths — see [`OPEN_SOURCE_PATH_TODOS.md`](./OPEN_SOURCE_PATH_TODOS.md) for the current status of
making every remaining internal path configurable.

## BibTeX

If you find our work helpful, please consider citing:

```bibtex
@article{host2026,
  title={HOST: Human-to-robot One-shot Skill Transfer},
  % TODO: fill in authors, venue, year, DOI/arXiv once available
}
```
