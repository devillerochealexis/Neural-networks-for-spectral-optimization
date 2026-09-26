
import torch
import torch.nn.functional as F


#  Discrete Laplacian
LAP_KERNEL = torch.tensor(
    [[0.0, -1.0, 0.0],
     [-1.0, 4.0, -1.0],
     [0.0, -1.0, 0.0]],
    dtype=torch.float32,
).reshape(1, 1, 3, 3)   


def laplacian_pixel(psi, mask):
    B, K, H, W = psi.shape
    psi_m = psi * mask                                #  psi must be = 0 outside Omega
    psi_flat = psi_m.reshape(B * K, 1, H, W)

    kernel = LAP_KERNEL.to(psi.device)
    lap = F.conv2d(psi_flat, kernel, padding=1)       
    lap = lap.reshape(B, K, H, W) * mask              

    return lap 


#  Gram-Schmidt orthonormalization

def gram_schmidt_omega(psi_raw, mask, eps=1e-8):
    """
    Gram-Schmidt orthonormalization 
    """
    mask_2d = mask[:, 0]  
    vecs = []             

    K = psi_raw.shape[1]
    for k in range(K):
        v = psi_raw[:, k] * mask_2d         
        for u in vecs:
            proj = (u * v).sum(dim=(-2, -1), keepdim=True)   
            v = v - proj * u                                  
        norm = (v ** 2).sum(dim=(-2, -1), keepdim=True).sqrt() + eps
        vecs.append(v / norm)

    return torch.stack(vecs, dim=1)  


#  Rayleigh quotients
def rayleigh_quotients_pixel(psi_on, mask):
    """
    Computes R_pixel(psi_hat_k) = <psi_hat_k, -Delta_pixel psi_hat_k> / ||psi_hat_k||^2
    """
    lap_psi = laplacian_pixel(psi_on, mask)           

    num = (psi_on * lap_psi).sum(dim=(-2, -1))        
    den = (psi_on ** 2).sum(dim=(-2, -1)) + 1e-10     

    return num / den   


def rayleigh_ratios(psi_on, mask):
    rq = rayleigh_quotients_pixel(psi_on, mask)    
    rq1 = rq[:, 0:1].clamp(min=1e-6)               
    return rq[:, 1:] / rq1                         


#  Loss terms 
def loss_ratio(ratios_pred, ratios_true):
    """Relative error on the eigenvalue ratios lambda_k / lambda_1, k = 2..K.
    """
    return ((ratios_pred - ratios_true).abs() / (ratios_true + 1e-6)).mean()


def loss_lam1(log_lam1_umax_pred, lam1_true, umax):
    log_target = torch.log((lam1_true * umax).clamp(min=1e-6))
    return ((log_lam1_umax_pred.squeeze(1) - log_target) ** 2).mean()


def loss_variational(ratios_pred, ratios_true_sorted, psi_on, mask):
    GRAM_WEIGHT = 0.1

    excess = F.relu(ratios_pred - ratios_true_sorted)
    variational = excess.mean()

    B, K, H, W = psi_on.shape
    psi_m = (psi_on * mask).reshape(B, K, H * W)
    G = torch.bmm(psi_m, psi_m.transpose(1, 2))
    I = torch.eye(K, device=psi_on.device).unsqueeze(0)
    gram = ((G - I) ** 2).mean()

    return variational + GRAM_WEIGHT * gram
