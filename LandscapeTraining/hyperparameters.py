"""
hyperparameters.py
Architecture and optimization hyperparameters for LandscapeModel, the
surrogate that predicts the first K Dirichlet-Laplacian
eigenvalues and eigenfunctions of a 2D domain from a torsion-based
geometric representation.
"""

# Model architecture (see LandscapeModel ) 

IN_CHANNELS = 4

N_EIG = 10        
BASE_CH = 32      
GRID_SIZE = 64    


# Loss term weights 
ALPHA_RATIO = 1.0
# Weight on the relative error of the predicted eigenvalue ratios

ALPHA_LAM1 = 1.0
# Weight on the lambda_1 loss.

ALPHA_RQ_VAR = 0.1
# Weight on the Rayleigh-quotient 

ALPHA_ORTH = 0.01 # just for Gram-Schmidt


# Training schedule
OPTIMIZER = "AdamW"
LR = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
EPOCHS = 300

LR_SCHEDULER = "CosineAnnealingLR"
LR_ETA_MIN_FACTOR = 1e-2   

GRAD_CLIP_NORM = 2.0

EMA_DECAY = 0.999

SEED = 42
TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.isupper():
            print(f"{name} = {value}")
