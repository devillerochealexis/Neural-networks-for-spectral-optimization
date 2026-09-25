# Spectral Shape Lab Simple

A separate, browser-only app using **vector10_relu** and **vector10_silu**.
Select a bundled model, enter a functional of the first ten Dirichlet eigenvalues,
and optimize a positive radial Fourier shape at fixed area. Computation, plots,
pause/resume/stop, and JSON/CSV/SVG/PNG downloads run locally in a Web Worker.

**The only model source files are `.pt` checkpoints.** No dataset, split file,
training eigenvalues, previous optimized shapes, or support tree is read by the
bundler or deployed app. The original `web/` app is separate and unchanged.
There is a model selector, with no file-upload control.

## Run the existing app

From the project root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory web_simple
```

Open `http://localhost:8000/`. Any static HTTP server works; this command only
serves files. Direct `file://` opening cannot reliably load module workers.
The default search uses 16 starts and 300 steps per start; these are editable.
One start is the disk, so choose at least two starts to include a random shape.

## Bundle models from .pt

Edit **`web_simple/model-paths.txt`**, with one `.pt` path per line, then run:

```bash
python web_simple/scripts/update_app.py
```

Paths are relative to the path-list file; absolute paths and spaces also work.
Do not quote lines. Blank lines and lines starting with `#` are ignored. The
first model is the default. Models are converted and checked before replacing
the catalog. The script rebuilds `web_simple/dist/` and
`web_simple/spectral-shape-simple.zip`.

For a particular checkpoint, without editing the list:

```bash
python web_simple/scripts/update_app.py --pt path/to/my_model.pt
# Or list several .pt files; these become the new selector entries:
python web_simple/scripts/update_app.py --pt path/to/first.pt path/to/second.pt
```

The `--pt` option replaces the catalog for that invocation; it does not edit the
path list. An alternative list can be supplied with `--config path/to/models.txt`.
Python with PyTorch and NumPy, plus Node.js, is needed **only for bundling**. No
training, support calibration, PDE solve, or npm dependency installation occurs.

The browser consumes the compact `.spectral.json` produced from `.pt`; it does
not run Python or decode PyTorch serialization itself. This offline conversion
preserves the weights and their input gradients. The two bundled ensembles are
about 1.3 MB each, including normalization, with no training samples.

See [MODELS.md](MODELS.md) for accepted checkpoint layouts and metadata.

## Random starts and constraints

The start generator uses **only the seed, harmonic count and geometry bounds**.
It does not read network weights, normalization, predicted scores or stored
shapes. Starts are not ranked or screened for a good objective. For each random
shape, it draws a target amplitude budget uniformly from `[0.1*rho, rho]`, an
amplitude uniformly from `[0, cap[j]]` for each harmonic, and an independent
uniform phase in `[0, 2*pi)`. If the amplitude sum exceeds the drawn budget, all
amplitudes are reduced by a common factor. The random generator is Mulberry32.
This is an explicit reproducible sampling rule, not a uniform distribution over
the full feasible set. Models with the same bounds receive identical starts for
the same seed/count, independent of the functional.

The supplied models use 15 harmonics, total amplitude budget `rho=0.75`, caps
`0.75` for harmonics 1–3, and `0.1*(3/j)^2` for harmonics 4–15. Thus
`r(theta) >= 0.25*a0 > 0` for every angle. Projection retains these bounds.
Area is imposed analytically through
`a0 = sqrt(area / (pi*(1 + sum(q*q)/2)))` and coefficients `[a0, a0*q]`.

There is **no nearest-neighbor support restriction**. The step preconditioner
uses the saved normalization of the first member, symmetrized within each
cosine/sine pair. For the retained paired RMS normalization, this equals the
original preconditioner. No new statistics are estimated from a dataset.

The optimizer uses projected Armijo descent, step 0.2, growth 1.3, maximum step
3, up to 18 backtracks, update norm limited to 0.06, and projected-gradient
tolerance 2e-5. The objective is `F/max(1,abs(F(disk)))`, with a sign change for
maximization. The displayed history contains the actual F. Formula derivatives
are symbolic; network input derivatives use the existing JavaScript reverse
pass. Prediction retains 12 rotations and reflections per member, log averaging
within members and arithmetic averaging across members. All ten outputs remain
in their learned order.

A functional must be defined at the disk. Random starts outside its domain are
reported as skipped, not replaced by favorable starts. Invalid line-search
proposals trigger backtracking. Completed/stopped run JSON exports include every generated start,
its source, seed, sampling-rule version, model hash, endpoints and histories.
Initial points and endpoints compete using the surrogate alone. Predictions are
not PDE validation; the support ablation with training-based starts does not
establish equivalent performance for this different random-start strategy.

## Publish

The update command already builds the site. To rebuild only static assets:

```bash
node web_simple/scripts/build.mjs
python web_simple/scripts/package.py
```

Publish the **contents of `web_simple/dist/`**, or the ZIP contents, on GitHub
Pages or any static host. Relative URLs work under project subdirectories.
Visitors need no Python, Node.js, GPU, FreeFem, or computation backend. The
published model weights are downloadable, but datasets and `.pt` sources are
not included in the build. Only models in the current catalog enter `dist/`.

## Code and tests

- `scripts/checkpoint.py`: `.pt` reader, export and synthetic PyTorch references.
- `scripts/update_app.py`: path list, conversion, validation, build and ZIP.
- `src/model.js`: inference, symmetry averaging and neural input gradients.
- `src/optimizer.js`: random starts, constraints, line search and reproducible records.
- `src/expression.js`, `geometry.js`, `plots.js`: formulas, geometry and visuals.
- `src/app.js`, `worker.js`: interface and background computation.

From the project root:

```bash
python web_simple/tests/test_checkpoint.py
node web_simple/tests/numerics.test.js
```

Browser tests use Playwright, which is a development dependency only:

```bash
cd web_simple
npm install
npx playwright install chromium
npm run test:browser
```

The browser test configuration starts a temporary static server and tests the
built app under a subdirectory, both real models, random starts, all downloads,
pause/stop, custom formulas, recovery from errors and mobile layout.

## Validation of the supplied build

The supplied build passed 8 checkpoint tests, 16 numerical tests and 5 Chromium
browser tests. Numerical checks cover both saved ensembles, all ten outputs and
their 30-component gradients at six synthetic shapes per model. Maximum
relative prediction differences from PyTorch were 3.62e-7 (ReLU) and 3.34e-7
(SiLU); maximum absolute gradient differences were 6.83e-5 and 3.68e-5.
The browser accumulates in float64 using the saved float32 weights; small
rounding differences from float32 PyTorch are expected.

The checks also convert a standalone tapered 6→12→7→10 SiLU checkpoint,
verify that the combined ReLU checkpoint preserves its original weights,
exercise formula derivatives and fixed-area projection, and confirm identical
unranked starts for the two supplied models. Browser tests cover deployment
under a URL subdirectory, mobile layout, switching, downloads and pause/stop.
These checks validate the software; no new PDE accuracy or optimum benchmark
is claimed for the random-start searches.
