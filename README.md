# PI-QGAT: Physics-Informed Quantum Graph Attention Network

A 144-parameter hybrid quantum-classical graph network for **CMS muon transverse momentum (pT) regression**, trained on the Kaggle CMS Trigger dataset (1.18M events). The quantum block gives the best accuracy per parameter we measured, and it converges fast.

## Task

Predict muon pT from hits across four muon stations. Each event is a fully connected 4-node graph (one node per station). Targets are regressed in `log1p(pT)` space and reported in GeV.

- **Split:** 20% train / 20% val / 60% test (707,614 test rows)
- **Loss:** SmoothL1 (β = 1.0) on log1p(pT)

## Architecture

```
node features (7)  ──► node_proj 7→4 ─┐
edge features (8)  ──► pair_proj 8→4 ─┤
physics prior (2) ──► prior 2→1 ──────┤
                                      ▼
            Quantum attention + message block
            4-qubit variational circuit, adaptive RZZ entanglers
            (angles depend on the data, not fixed)
                                      ▼
                    head 4→6→1 ──► log1p(pT)
```

- **Physics prior:** injects known detector geometry into the attention scores.
- **Adaptive RZZ:** entangling angles are data-dependent, so the circuit responds to each event.
- **Budget:** 144 trainable parameters (36 quantum-block, 108 classical).

## Results

**Same architecture, classical interaction block swapped in (ablation, seed 42, test set):**

| Model | Params | Test R² (log1p pT) | Test MAE (GeV) | R² at epoch 4 |
|---|---|---|---|---|
| **PI-QGAT (quantum)** | 144 | **0.840** | **11.85** | **0.818** |
| Classical, 36-wide block | 144 | 0.615 | 14.49 | 0.555 |

**Classical baselines (test set, RMSE):**

| Model | Test RMSE (GeV) | Test RMSE (log) |
|---|---|---|
| **PI-QGAT (quantum), seed 42** | **126.9** | **0.376** |
| GATv2, ~1k params | 126.3 | 0.380 |
| GAT, ~1k params | 128.1 | 0.391 |
| MLP, 6k params | 129.6 | 0.405 |
| XGBoost | 132.0 | 0.597 |

## Why the quantum block wins

- **Higher R²:** 0.840 vs 0.615 at identical parameter count and architecture.
- **Fast convergence:** R² reaches 0.818 by epoch 4. The classical variant is at 0.555 at the same point and is still climbing at epoch 50.
- **Parameter efficiency:** the 36-parameter quantum block carries the accuracy. A classical block distilled from it recovers most of the gain.
- **Compact:** 144 params = 576 bytes in float32, 288 bytes in float16, with negligible accuracy loss.

## Notebooks

| File | Contents |
|---|---|
| `piqgat_classical.ipynb` | Classical baselines: MLP, GNN, GAT (v1/v2), XGBoost, Random Forest, parameter-matched to PI-QGAT |
| `jax_piqgata_ablation_experiments.ipynb` | PI-QGAT model and ablations: physics prior, entanglement topology, adaptive vs fixed RZZ, quantum attention vs message, station, feature, graph-attention, message-passing, qubit, depth, entangler, data re-uploading, distillation |

## Stack

- **Quantum:** PennyLane 0.45.1 (circuit verification), custom closed-form 4-qubit simulator
- **Deep learning:** JAX 0.11.1 (whole-epoch `lax.scan`, JIT), PyTorch 2.11 (classical baselines)
- **Classical ML:** XGBoost, scikit-learn (Random Forest)
- **Data:** pandas, NumPy, kagglehub
- **Dataset:** [CMS Trigger Dataset](https://www.kaggle.com/datasets/prateekagnihotri/cms-trigger-dataset)

## Reproduce

```bash
pip install pennylane kagglehub jax
# Run Cell 0 (installs), then Cell 1 (data), then the model cells in order.
```

Run on CPU or Colab/Kaggle. Each 50-epoch run takes about 4–5 s/epoch on CPU for PI-QGAT.


