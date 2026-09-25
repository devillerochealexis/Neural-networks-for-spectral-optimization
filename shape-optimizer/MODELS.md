# Model-only checkpoint contract

The offline bundler reads only the `.pt` file listed in `model-paths.txt`. A
checkpoint must include the weights **and normalization**. Architecture widths
are inferred from its Linear weight tensors, so tapered networks are supported.
No model code needs to be imported from the checkpoint.

Supported networks: 1–32 Fourier harmonics (2–64 normalized inputs), 1–8 dense
hidden layers, widths up to 2048, ten output heads, and 1–8 ensemble members.
Activations: `relu`, `silu`, or `tanh`. LayerNorm, Dropout layouts, custom heads,
and arbitrary pickled Python model objects are not accepted by this format.

Inputs are `q=[a1/a0,...,am/a0,b1/a0,...,bm/a0]`. Each network produces a
standardized **log(area * eigenvalue)** vector. The bundle restores the saved
normalization before exponentiation. It must not be used for a model trained
to a different target convention without adapting the exporter.

## Notebook ensemble .pt

The project's notebook checkpoints are accepted directly, including
`results/vector10_silu/spectrum_ensemble_192_silu.pt`:

```python
checkpoint = {
    "state_dicts": [model.state_dict() for model in models],
    "x_scale": x_scale,  # length 2*m, positive
    "y_mean": y_mean,    # length 10
    "y_scale": y_scale,  # length 10, positive
    "K": 10,
    "activation": "silu",  # or relu/tanh
}
torch.save(checkpoint, "my_model.pt")
```

State dictionaries contain `layers.0.weight`, `layers.0.bias`, `layers.2.weight`,
etc., from alternating Linear/activation modules, including the final Linear
layer. The older `hidden_dim` and newer `hidden_dims` metadata are not needed:
the actual tensor shapes determine the architecture. Other notebook records,
such as training histories, are not exported to the browser.

## Project model_*.pt and a combined ensemble

Individual project checkpoints are also accepted. These contain `net.0.weight`,
`net.0.bias`, etc., plus `xscale`, `ymean`, `yscale`, `activation_id` and
`faber_krahn=False`. A single file becomes a one-member model. Each selector
entry is independent; listing three member files creates three choices, not
one averaged ensemble.

To combine project member files into one `.pt`, use:

```bash
python web_simple/scripts/pack_ensemble.py \
  results/vector10_relu/model/model_0.pt \
  results/vector10_relu/model/model_1.pt \
  results/vector10_relu/model/model_2.pt \
  --out results/vector10_relu/vector10_relu.pt --name vector10_relu
```

This command was used for the supplied ReLU entry. It copies the state
dictionaries unchanged and records their hashes; it never trains or reads a
dataset. The generated file already exists, so the command refuses to overwrite
it. Use another output path if making a new ensemble. To bundle the existing
file, run `python web_simple/scripts/update_app.py`.

The reader also accepts a single `state_dict` or `model_state_dict` wrapper,
provided that activation and normalization are present in the checkpoint.
Missing normalization is an error: it is never reconstructed from data.

## Optional metadata and defaults

Top-level fields may specify:

```python
"name": "My model",
"eigenvalue_indices": [1,2,3,4,5,6,7,8,9,10],
"rho": 0.75,
"caps": [...],  # one positive value per harmonic, at most rho
```

Without output-index metadata, the contract is outputs 1 through 10 in that
order. A different ordering must be explicitly recorded. Without geometry
metadata, the bundler uses `rho=0.75`, low-mode caps `rho`, and higher caps
`0.1*(3/j)^2`. These are explicit defaults matching this project's shape family;
they cannot be inferred from trained weights. Different training families
should save their intended bounds. The model name defaults to its containing
`vector10_*` directory or checkpoint stem.

Every export is validated against PyTorch on six synthetic shapes, including
the disk, and all ten input Jacobians. These probes use no training samples or
PDE labels. Validation checks conversion correctness, not spectral accuracy.
All candidates must pass before `app-model.json` is replaced. A failed later
build/package step can be retried after fixing its reported error.

The generated browser bundle contains weights, normalization, activation,
output order, geometry bounds, and source/code hashes. It contains no support
cloud, eigenvalue labels, start library, training histories, or dataset paths.
Hash-based filenames prevent stale model caches after changing weights.
