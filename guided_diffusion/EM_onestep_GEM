import numpy as np
import os
import torch
import torch.fft

def EM_Initial(IR):
    device = IR.device
    k1 = torch.tensor([[1, -1]]).to(device)
    k2 = torch.tensor([[1], [-1]]).to(device)
    fft_k1 = psf2otf(k1, np.shape(IR)).to(device)
    fft_k2 = psf2otf(k2, np.shape(IR)).to(device)
    fft_k1sq = torch.conj(fft_k1) * fft_k1
    fft_k2sq = torch.conj(fft_k2) * fft_k2
    C = torch.ones_like(IR).to(device)
    D = torch.ones_like(IR).to(device)
    F2 = torch.zeros_like(IR).to(device)
    F1 = torch.zeros_like(IR).to(device)
    H = torch.zeros_like(IR).to(device)
    tau_a = 1
    tau_b = 1
    HP = {"C": C, "D": D, "F2": F2, "F1": F1, "H": H, "tau_a": tau_a, "tau_b": tau_b,
          "fft_k1": fft_k1, "fft_k2": fft_k2, "fft_k1sq": fft_k1sq, "fft_k2sq": fft_k2sq}
    return HP

def EM_onestep(f_pre, I, V, HyperP, lamb=0.5, rho=0.01, learning_rate=0.1):
    device = f_pre.device
    # Retrieve hyperparameters and their gradients
    fft_k1, fft_k2, fft_k1sq, fft_k2sq = HyperP['fft_k1'], HyperP['fft_k2'], HyperP['fft_k1sq'], HyperP['fft_k2sq']
    C, D, F2, F1, H = HyperP['C'], HyperP['D'], HyperP['F2'], HyperP['F1'], HyperP['H']
    tau_a, tau_b = HyperP['tau_a'], HyperP['tau_b']

    Y = I - V
    X = f_pre - V

    # E-step: Update D and C based on current estimates
    new_D = torch.sqrt(2 / tau_b / (X**2 + 1e-6))
    new_C = torch.sqrt(2 / tau_a / ((Y - X + 1e-6)**2))
    new_D[new_D > 2 * new_C] = 2 * new_C[new_D > 2 * new_C]
    
    # GEM: Incremental parameter updates
    D += learning_rate * (new_D - D)
    C += learning_rate * (new_C - C)

    # M-step: Update using proximal TV with learning rate adjustments
    H = prox_tv(Y - X, F1, F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq)
    a1 = torch.zeros_like(H)
    a1[:, :, :, :-1] = H[:, :, :, :-1] - H[:, :, :, 1:]
    a1[:, :, :, -1] = H[:, :, :, -1]
    F1 += learning_rate * ((rho / (2 * lamb + rho)) * a1 - F1)

    a2 = torch.zeros_like(H)
    a2[:, :, :-1, :] = H[:, :, :-1, :] - H[:, :, 1:, :]
    a2[:, :, -1, :] = H[:, :, -1, :]
    F2 += learning_rate * ((rho / (2 * lamb + rho)) * a2 - F2)

    X = (2 * C * Y + rho * (Y - H)) / (2 * C + 2 * D + rho)
    F = I - X

    return F, {"C": C, "D": D, "F2": F2, "F1": F1, "H": H, "tau_a": tau_a, "tau_b": tau_b,
               "fft_k1": fft_k1, "fft_k2": fft_k2, "fft_k1sq": fft_k1sq, "fft_k2sq": fft_k2sq}

def prox_tv(X, F1, F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq):
    fft_X = torch.fft.fft2(X)
    fft_F1 = torch.fft.fft2(F1)
    fft_F2 = torch.fft.fft2(F2)
    H = fft_X + torch.conj(fft_k1) * fft_F1 + torch.conj(fft_k2) * fft_F2
    H /= 1 + fft_k1sq + fft_k2sq
    H = torch.real(torch.fft.ifft2(H))
    return H

def psf2otf(psf, outSize):
    psfSize = torch.tensor(psf.shape)
    outSize = torch.tensor(outSize[-2:])
    padSize = outSize - psfSize
    psf = torch.nn.functional.pad(psf, (0, padSize[1], 0, padSize[0]), 'constant')
    for i in range(len(psfSize)):
        psf = torch.roll(psf, -int(psfSize[i] / 2), i)
    otf = torch.fft.fftn(psf)
    if torch.max(torch.abs(torch.imag(otf))) / torch.max(torch.abs(otf)) <= torch.finfo(torch.float32).eps:
        otf = torch.real(otf)
    return otf
