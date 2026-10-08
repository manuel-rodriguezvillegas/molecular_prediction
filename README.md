# How much 3D geometry does a GNN need? Geometry-aware message passing on QM9

<p align="center">
  <img src="assets/molecule.png" width="300">
  <br>
  <em>Image source: <a href="https://www.researchgate.net/publication/377508996_Mode_Collapse_Alleviation_of_Reinforcement_Learning-based_GANs_in_Drug_Design">ResearchGate</a></em>
</p>

This project measures how much each kind of 3D geometric information helps graph
neural networks predict quantum-chemical properties of small molecules (QM9). Five
models form a ladder in which **each step adds exactly one ingredient**, all with the
same training protocol and a matched parameter budget (~200k):

| Model | Sees | Symmetry of the prediction | What the step isolates |
|---|---|---|---|
| **GIN** | atom features + bond graph | trivially invariant (ignores coordinates) | topology-only baseline |
| **GINEdge** | + bond types | trivially invariant | value of bond types |
| **GINDist** | + bond lengths (RBF) | E(3)-invariant | value of distances |
| **EGNNStatic** | same as GINDist, EGNN-style MPNN | E(3)-invariant | architecture, at equal information |
| **EGNN** | + equivariant coordinate updates | E(3)-invariant output, equivariant coordinates | value of equivariance (angular information) |

Beyond the benchmark, two analyses probe *why* geometry helps:

1. **Noise ablation**: perturb coordinates at inference time (no retraining). GIN and
   GINEdge never read coordinates, so their flat curves are a built-in sanity check.
2. **Over-squashing analysis**: Ollivier-Ricci curvature of every bond identifies
   topological bottlenecks; does the gain over GIN grow with bottleneck severity
   *after controlling for molecule size*?

## Targets

Four QM9 properties, trained jointly (multi-task) and chosen to span different physics:

| Property | Symbol | Unit | Extensive / intensive |
|---|---|---|---|
| Dipole moment | `mu` | D | neither (norm of a vector sum) |
| Isotropic polarisability | `alpha` | a0^3 | extensive |
| HOMO-LUMO gap | `gap` | eV | intensive |
| Heat capacity (298 K) | `Cv` | cal/mol/K | extensive |

## Methodology

- **Random split** 110k / 10k / ~10.8k with a fixed seed (QM9 is stored in GDB
  enumeration order, so contiguous slices are not i.i.d.). Every model and seed is
  evaluated on the same molecules.
- Targets standardised with **train-set statistics only**; loss and model selection in
  normalised units, so all four targets weigh equally. Results reported in both
  normalised and physical units.
- **Shared trunk + per-target heads**: sum- and mean-pooled readout, one small MLP per
  property (sum suits extensive, mean intensive properties).
- **Same protocol for all models**: Adam (lr 5e-4), cosine schedule, 300 epochs,
  batch 96, L1 loss, gradient clipping, early stopping (patience 50); the **best
  validation checkpoint** is restored before testing.
- **3 seeds per model**, reported as mean ± std.
- **Tests** check the properties the design relies on: E(3) invariance of every model
  (including reflections), permutation invariance, EGNN coordinate equivariance,
  coordinate-independence of GIN/GINEdge, curvature on hand-computed graphs, split
  integrity, checkpoint restoration.

## Project structure

```
molecular_prediction/
├── main.py                         # CLI entry point (python main.py <command>)
├── Dockerfile                      # GPU training image (CUDA from the locked wheels)
├── scripts/dgx_train.sh            # multi-GPU launcher: models x seeds, then analyses
├── src/molecular_prediction/
│   ├── config.py                   # typed dataclass configuration
│   ├── cli.py                      # commands: params, data, train, compare, noise, curvature, all
│   ├── data/                       # QM9 loading, random split, normalisation, coordinate noise
│   ├── models/                     # GIN family, EGNN / EGNNStatic, shared RBF and readout heads
│   ├── training/                   # trainer, early stopping, metrics, self-contained checkpoints
│   └── experiments/                # comparison, noise ablation, curvature analysis, plots
├── tests/                          # pytest suite (runs on CPU in seconds)
├── results/                        # committed: JSON summaries, per-run metrics, figures
└── outputs/                        # git-ignored: checkpoints, TensorBoard logs, caches
```

## Usage

```bash
uv sync                                   # install (Python 3.11, locked dependencies)
uv run pytest                             # test-suite
uv run python main.py params              # parameter count of every model
uv run python main.py all --smoke --device cpu   # end-to-end check in ~30 s
```

Full pipeline on a single machine:

```bash
uv run python main.py train --device cuda          # 5 models x 3 seeds
uv run python main.py compare                      # results/comparison.json + figures
uv run python main.py noise --device cuda          # results/noise_ablation.json
uv run python main.py curvature --device cuda      # results/curvature_analysis.json
```

`train` skips runs whose results already exist, so it can be resumed after an
interruption (`--overwrite` forces retraining). TensorBoard: `tensorboard --logdir outputs/runs`.

### On a multi-GPU server (e.g. NVIDIA DGX) with Docker

Requirements on the host: NVIDIA driver (>= 570 recommended for CUDA 12.8) and the
NVIDIA Container Toolkit.

```bash
git clone <repository-url> && cd molecular_prediction
GPUS="0 1 2 3" JOBS_PER_GPU=2 scripts/dgx_train.sh
```

The script builds the image, downloads QM9 once, trains every (model, seed) pair in
parallel across the listed GPUs, then runs the three analyses. Containers run as your
user, with `data/`, `outputs/` and `results/` bind-mounted, so everything persists on
the host. Afterwards commit `results/` to bring the numbers and figures back:

```bash
git add results && git commit -m "Add DGX results" && git push
```

## Results

Generated in `results/` after training: `comparison.json`, `noise_ablation.json`,
`curvature_analysis.json`, per-run `results/runs/<model>/seed_<s>/{metrics,history}.json`
and the figures in `results/figures/`.

## References

- Satorras, Hoogeboom & Welling (2021). *E(n) Equivariant Graph Neural Networks.* ICML.
- Xu, Hu, Leskovec & Jegelka (2019). *How Powerful are Graph Neural Networks?* ICLR.
- Hu et al. (2020). *Strategies for Pre-training Graph Neural Networks.* ICLR (GINE).
- Schütt et al. (2017). *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.* NeurIPS (RBF distance expansion).
- Ramakrishnan, Dral, Rupp & von Lilienfeld (2014). *Quantum chemistry structures and properties of 134 kilo molecules.* Scientific Data.
- Alon & Yahav (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications.* ICLR.
- Topping et al. (2022). *Understanding Over-Squashing and Bottlenecks on Graphs via Curvature.* ICLR.
- Ollivier (2009). *Ricci curvature of Markov chains on metric spaces.* Journal of Functional Analysis.

## Authorship

- Manuel Rodríguez Villegas (manuelrodriguez@alu.comillas.edu)
