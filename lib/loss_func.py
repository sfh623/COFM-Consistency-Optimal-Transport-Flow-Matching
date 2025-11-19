import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.act_func import ActNormNoLogdet,ActNorm, Activations
from lib.convex_tools import ConvexLinear, PosLinear2, NegExpPositivity,ClippedPositivity, ConvexInitialiser, PosLinear
import time
import ot as pot
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
import torchvision.utils as vutils


class PsiNet(torch.nn.Module):
    def __init__(self, dim, dim_t, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim+dim_t, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1)
        )
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

def transform(Psi, x, t,eps):
    # x.pow(2).sum(dim=1,keepdim=True) => ||x||^2
    return (Psi - 0.5*x.pow(2).sum(dim=1, keepdim=True)) / (1.0 - t)
    #return (0.5*x.pow(2).sum(dim=1, keepdim=True)-Psi)/t
    

def compute_loss_hj(model, x_1, x_0, t, lambda_hj=1.0):  ###scalar
    t = t.clone().detach().requires_grad_(True)
    x_t = (t * x_1 + (1 - t) * x_0)
    x_t = x_t.clone().detach().requires_grad_(True)

    Psi = model(x_t, t)                   # (batch,1)
    psi_val = transform(Psi, x_t, t)    # (batch,1)
    

    #∇_x ψ
    grad_psi_x = torch.autograd.grad(
        outputs=psi_val,
        inputs=x_t,
        grad_outputs=torch.ones_like(psi_val),
        create_graph=True
    )[0]  # (batch, dim)

    grad_psi_t = torch.autograd.grad(
        outputs=psi_val,
        inputs=t,
        grad_outputs=torch.ones_like(psi_val),
        create_graph=True
    )[0]  # (batch,1)

    flow_matching = (x_1 - x_0) - grad_psi_x

    residue_HJ = grad_psi_t + 0.5 * torch.sum(grad_psi_x**2, dim=1, keepdim=True)
    wv = 1.0#(1.0 - t).pow(1)
    wh = (1.0 - t).pow(2) # pow(0): 87
    loss_FM = torch.mean(wv*flow_matching.pow(2))
    loss_HJ = torch.mean(wh * residue_HJ.pow(2))
    total_loss = loss_FM + lambda_hj * loss_HJ

    return total_loss, loss_FM, loss_HJ
    
def compute_loss_l2(model, x_1, x_0, t, t1=0, lambda_hj=1.0):
    # mean flow inspired. ## 1024MB, 16.2128 (non hj), hj (0.5) 14.0, 
    
    x_t = (t * x_1 + (1 - t) * x_0)
    x_t = x_t.detach().requires_grad_(True) 
    t=t.detach().requires_grad_(True)  # (batch,1)

    Psi = model                 
    psi = lambda x,t : (Psi(x,t) - 0.5*x.pow(2).sum(dim=1).view(-1,1)) / (1.0 - t)
    gradx_psi = lambda x,t: torch.autograd.grad(psi(x,t),x,grad_outputs=torch.ones_like(t), create_graph=True)[0]
    grad_psi_x = gradx_psi(x_t,t)
    
    grad_psi_xval1,residue_HJ = torch.autograd.functional.jvp(gradx_psi, (x_t,t), v = (grad_psi_x ,torch.ones_like(t)),create_graph=True) 

    L2 =  grad_psi_xval1 - x_t
    wv = 1.0#(1.0 - t).pow(1)
    wh = (1.0 - t).pow(2) # pow(0): 87
    loss_L2 = torch.mean(wv*L2.pow(2))
    loss_HJ = torch.mean(wh * residue_HJ.pow(2))
    total_loss = loss_L2 + lambda_hj * loss_HJ

    return total_loss, loss_L2, loss_HJ


def compute_loss_hj2(model, x_1, x_0, t, lambda_hj=0.5):  ####scalar
    t = t.clone().detach().requires_grad_(True)
    x_t = (t * x_1 + (1 - t) * x_0)
    x_t = x_t.clone().detach().requires_grad_(True)


    Psi = model(x_t, t)                   # (batch,1)
    psi_val = transform(Psi, x_t, t)    # (batch,1)
    

    #∇_x ψ
    grad_psi_x = torch.autograd.grad(
        outputs=psi_val,
        inputs=x_t,
        grad_outputs=torch.ones_like(psi_val),
        create_graph=True
    )[0]  # (batch, dim)

    grad_psi_t = torch.autograd.grad(
        outputs=psi_val,
        inputs=t,
        grad_outputs=torch.ones_like(psi_val),
        create_graph=True
    )[0]  # (batch,1)

    flow_matching = (x_1 - x_0) - grad_psi_x
    L2 =  grad_psi_x - x_t

    residue_HJ = grad_psi_t + 0.5 * torch.sum(grad_psi_x**2, dim=1, keepdim=True)
    wv = 1.0#(1.0 - t).pow(1)
    wh = (1.0 - t).pow(2) # pow(0): 87
    loss_L2 = torch.mean(wv*L2.pow(2))
    loss_HJ = torch.mean(wh * residue_HJ.pow(2))
    total_loss = loss_L2 + lambda_hj * loss_HJ

    return total_loss, loss_L2, loss_HJ

def compute_loss_back(model, x_1, x_0, t, t1=0, lambda_hj=1):
    # back calculation loss
    
    x_t = (t * x_1 + (1 - t) * x_0)
    x_t = x_t.detach().requires_grad_(True)


    Psi = model(x_t, t)                   # (batch,1)
    psi_val = transform(Psi, x_t, t)    # (batch,1)
    

    #∇_x ψ
    grad_psi_x = torch.autograd.grad(
        outputs=psi_val,
        inputs=x_t,
        grad_outputs=torch.ones_like(psi_val),
        create_graph=True
    )[0]  # (batch, dim)


    t1 = t *t1
    x_t1 = x_t + grad_psi_x * (t1-t)
    x_t1 =x_t1.detach().requires_grad_(True)
    Psi1 = model(x_t1, t1)                   # (batch,1)
    psi_val1 = transform(Psi1, x_t1, t1)    # (batch,1)
    grad_psi_x1 = torch.autograd.grad(
        outputs=psi_val1,
        inputs=x_t1,
        grad_outputs=torch.ones_like(psi_val1),
        create_graph=True
    )[0]  # (batch, dim)
    residue_HJ = grad_psi_x1 - grad_psi_x

    flow_matching = (x_1 - x_0) -grad_psi_x

    wv = 1.0#(1.0 - t).pow(1)
    wh = (1.0 - t).pow(4) # pow(0):33.94, pow(2):20
    loss_FM = torch.mean(wv*flow_matching.pow(2))
    loss_HJ = torch.mean(wh * residue_HJ.pow(2))
    total_loss = loss_FM + lambda_hj * loss_HJ

    return total_loss, loss_FM, loss_HJ

def compute_loss_meanflow_v1(model, x_1, x_0, t, t1=0, lambda_hj=0.5):
    # mean flow inspired. ## 1024MB, 16.2128 (non hj)
    
    x_t = (t * x_1 + (1 - t) * x_0)
    x_t = x_t.detach().requires_grad_(True) 
    t=t.detach().requires_grad_(True)  # (batch,1)

    Psi = model                 
    psi = lambda x,t : (Psi(x,t) - 0.5*x.pow(2).sum(dim=1).view(-1,1)) / (1.0 - t)
    gradx_psi = lambda x,t: torch.autograd.grad(psi(x,t),x,grad_outputs=torch.ones_like(t), create_graph=True)[0]
    grad_psi_x = gradx_psi(x_t,t)

    ## test block
    # Psi1 = model(x_t, t)                   # (batch,1)
    # psi_val = transform(Psi1, x_t, t)
    # grad_psi_x1 = torch.autograd.grad(
    #     outputs=psi_val,
    #     inputs=x_t,
    #     grad_outputs=torch.ones_like(psi_val),
    #     create_graph=True
    # )[0]  # (batch, dim)
    # assert torch.allclose(grad_psi_x,grad_psi_x1)
    ##
    
    grad_psi_xval1,residue_HJ = torch.autograd.functional.jvp(gradx_psi, (x_t,t), v = (grad_psi_x.detach() ,torch.ones_like(t)),create_graph=True) # 13.7

    flow_matching = (x_1 - x_0) -grad_psi_x

    wv = 1.0#(1.0 - t).pow(1)
    wh = (1.0 - t).pow(4)
    loss_FM = torch.mean(wv*flow_matching.pow(2))
    loss_HJ = torch.mean(wh * residue_HJ.pow(2))
    total_loss =  loss_FM+lambda_hj * loss_HJ

    return total_loss, loss_FM, loss_HJ

def compute_loss(model, x_1, x_0, t, t1=0, lambda_hj=1.0):
    # mean flow inspired. ## 1024MB, 16.2128 (non hj), hj (0.5) 14.0, 
    
    x_t = (t * x_1 + (1 - t) * x_0)
    x_t = x_t.detach().requires_grad_(True) 
    t=t.detach().requires_grad_(True)  # (batch,1)

    Psi = model                 
    psi = lambda x,t : (Psi(x,t) - 0.5*x.pow(2).sum(dim=1).view(-1,1)) / (1.0 - t)
    gradx_psi = lambda x,t: torch.autograd.grad(psi(x,t),x,grad_outputs=torch.ones_like(t), create_graph=True)[0]
    grad_psi_x = gradx_psi(x_t,t)

    
    grad_psi_xval1,residue_HJ = torch.autograd.functional.jvp(gradx_psi, (x_t,t), v = (grad_psi_x ,torch.ones_like(t)),create_graph=True) 
    

    flow_matching = (x_1 - x_0) -grad_psi_xval1

    wv = 1.0#(1.0 - t).pow(1)
    wh = (1.0 - t).pow(4) # pow(0): 19 (dim 64)
    # pow(4): 20 (dim 256)
    # pow(2): 28 (dim 256)
    loss_FM = torch.mean(wv*flow_matching.pow(2))
    loss_HJ = torch.mean(wh * residue_HJ.pow(2))
    total_loss = loss_FM+ lambda_hj * loss_HJ

    return total_loss, loss_FM, loss_HJ
def compute_loss_sca(model, x_1, x_0, t, lambda_hj=1.0):
    x_t = (t * x_1 + (1 - t) * x_0)
    x_t = x_t.detach().requires_grad_(True)
    t   = t.detach().requires_grad_(True)

    Psi = model  # Psi(x,t) -> (batch,1)
    # ψ = (Ψ - 1/2||x||^2) / (1 - t)
    psi = lambda x,tt: (Psi(x,tt) - 0.5 * x.pow(2).sum(dim=1, keepdim=True)) / (1.0 - tt)

    psi_val   = psi(x_t, t)                                 # (B,1)
    grad_psi  = torch.autograd.grad(psi_val, x_t,
                                    grad_outputs=torch.ones_like(psi_val),
                                    create_graph=True)[0]   # ∇ψ  (B,d)
    dpsi_dt   = torch.autograd.grad(psi_val, t,
                                    grad_outputs=torch.ones_like(psi_val),
                                    create_graph=True)[0]   # ∂tψ  (B,1)

    flow_matching = (x_1 - x_0) - grad_psi                  # (B,d)
    loss_FM = (flow_matching.pow(2)).mean()

    r_hj = dpsi_dt + 0.5 * (grad_psi.pow(2).sum(dim=1, keepdim=True))  # (B,1)
    wh   = (1.0 - t).pow(4)
    loss_HJ = (wh * r_hj.pow(2)).mean()

    total_loss = loss_FM + lambda_hj * loss_HJ
    return total_loss, loss_FM, loss_HJ