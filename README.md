# Neural networks for spectral optimization

On this repository you will find the codes necessary to reproduce the results of the article "Neural networks for spectral optimization"

## Fourier approach

The jupyter notebook `Fourier31_training_and_optimization.ipyng` corresponds to the section 3 of the article and the Fourier based models. It needs Python with NumPy, SciPy, PyTorch, SymPy and Matplotlib to function and defines every thing it uses. Its only other dependencies are the two files in the directory `data/` and the FreeFEM script.

The directory `data/` contains the files that store the coefficients of the training database and the associated eigenvalues, they correspond rwo by row.

The FreeFEM script `eigenvalues.edp` is the PDE solver used to compute the eigenvalues of a given shape. It is best used with the function `ComputeFreeFEMSpectrum` defined in the notebook. It needs a working installation of the [FreeFEM software](https://freefem.org).

In the directory `notebook_outputs/` are saved the pretrained models and the computed shapes and associated coefficients used in the examples of the paper.

## Torsion function based approach
