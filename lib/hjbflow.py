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

class TensorSampler:
    def __init__(self, data):
        self.data = data
    def sample(self, n):
        idx = torch.randint(0, len(self.data), (n,), device=self.data.device)
        return self.data[idx]

def transform(Psi, x, t):
    psi = (Psi - 0.5*x.pow(2).sum(dim=1).view(-1,1)) / (1.0 - t)
    return psi
    
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

##################################
def train_flow2d(
    model_Psi,
    optimizer,
    prior_dict,
    target_dict,
    iterations,
    batch_size,
    loss_func=compute_loss,
    minibatch= True
):
    loss_history = []
    model_Psi.train()
    start_time = time.time()

    device = next(model_Psi.parameters()).device

    for i in range(iterations):
        optimizer.zero_grad()

        idx0 = torch.randperm(prior_dict.size(0))[:batch_size]
        idx1 = torch.randperm(target_dict.size(0))[:batch_size]
        x0 = prior_dict[idx0].to(device)
        x1 = target_dict[idx1].to(device)
        if minibatch:
            M = (torch.cdist(x0,x1)**2).detach().cpu().numpy()
            gamma = pot.emd(np.ones((batch_size,)),np.ones((batch_size,)),M)
            x1ind = np.argmax(gamma,axis = 1)
            x1 = x1[x1ind]
        else:
            x1 = x1        

        t = torch.rand(batch_size, 1, device=device) * 0.99

        loss, loss_FM, loss_HJ = loss_func(
            model_Psi,
            x_1=x1,
            x_0=x0,
            t=t)

        # backward
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (i+1) % 1000 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i+1) * (iterations - i - 1)
            msg = (
                f"Iter {i+1}: total {loss:.4f}, "
                f"FM {loss_FM:.4f}, HJ {loss_HJ:.4f}, "
                f"elapsed {elapsed:.1f}s, eta {eta:.1f}s"
            )
            evaluation(model_Psi, x0, x1)
            evaluation_straight(model_Psi, TensorSampler(prior_dict.to(device)))
            plt.show()
            print(msg)

    return loss_history

def train_flow(
    model,
    optimizer,
    prior_sampler,    
    target_sampler,   
    iterations,
    batch_size,
    loss_func=compute_loss,
    log_interval=1000,
    benchmark = None,
    minibatch=False
):
    start_time = time.time() 
    loss_history = []
    model.train()
    device = next(model.parameters()).device
    t0 = 0.0
    pbar = tqdm(range(iterations))

    for it in pbar:
        optimizer.zero_grad()
       
        x0 = prior_sampler.sample(batch_size).to(device)   # [128, 784]
        x1 = target_sampler.sample(batch_size).to(device)

        # minibatch OT coupling
        if minibatch:
            M = torch.cdist(x0, x1).pow(2).cpu().detach().numpy()
            gamma = pot.emd(np.ones(batch_size), np.ones(batch_size), M)
            matched_idx = gamma.argmax(axis=1)
            x1_match = x1[matched_idx]
        else:
            x1_match = x1
        # x0.requires_grad_(True)
        # x1_match.requires_grad_(True)
        # 
        t = torch.rand(batch_size, 1, device=device)*0.99  + t0

        # 
        loss, loss_FM, loss_HJ = loss_func(
            model, x_1=x1_match, x_0=x0, t=t
        )
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (it+1) % log_interval == 0 or it == 0:
            elapsed = time.time() - start_time
            if benchmark is not None:
                x0test = prior_sampler.sample(4096).to(device)
                x0test.requires_grad_(True)
                Y_true = benchmark.map_fwd(x0test, nograd=True).to(device)
                Y_pred = sample_ode_hj(x0test, model, t0=0.0, N=1, return_path=False)
                Y_pred = Y_pred.to(device)
                L2_UVP_fwd = 100 * (((Y_true - Y_pred) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item() 
                pbar.set_description(f"FM {loss_FM:.4f}, HJ {loss_HJ:.4f}, UVP {L2_UVP_fwd:.4f}")
            else:
                pbar.set_description(f" FM {loss_FM:.4f}, HJ {loss_HJ:.4f}")

    return loss_history
    
def train_flow_mnist(
    model,
    optimizer,
    prior_sampler,    # tensor [N0, D], e.g. N(0,I) or learnt latent
    target_sampler,   # tensor [N1, D], e.g. MNIST flattened or latent
    iterations,
    batch_size,
    loss_func=compute_loss,
    log_interval=1000,
    benchmark = None,
    minibatch = True,
):
    start_time = time.time() 
    loss_history = []
    model.train()
    device = next(model.parameters()).device
    t0 = 0.0

    for it in tqdm(range(iterations)):
        optimizer.zero_grad()
       
        x0 = prior_sampler.sample(batch_size).to(device)   # e.g. [128, 784]
        x1 = target_sampler.sample(batch_size).to(device)

        if minibatch:
            M = torch.cdist(x0, x1).pow(2).cpu().detach().numpy()
            gamma = pot.emd(np.ones(batch_size), np.ones(batch_size), M)
            matched_idx = gamma.argmax(axis=1)
            x1_match = x1[matched_idx]
        else:
            x1_match = x1
        # x0.requires_grad_(True)
        # x1_match.requires_grad_(True)
        # 
        t = torch.rand(batch_size, 1, device=device)*0.99 +t0

        # 
        loss, loss_FM, loss_HJ = loss_func(
            model, x_1=x1_match, x_0=x0, t=t
        )
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (it+1) % log_interval == 0 or it ==0:
            elapsed = time.time() - start_time
            if benchmark is not None:
                
                Y_true = benchmark.map_fwd(xtest, nograd=True).to(device)
                Y_pred = sample_ode_hj(xtest, model_Psi, t0=0.0, N=1, return_path=False)
                Y_pred = Y_pred.to(device)
                L2_UVP_fwd = 100 * (((Y_true - Y_pred) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item() 
                print(f"Iter {it}/{iterations} — total {loss:.4f}, FM {loss_FM:.4f}, HJ {loss_HJ:.4f}, UVP {L2_UVP_fwd:.4f}")
            else:
                print(f"Iter {it}/{iterations} — total {loss:.4f}, FM {loss_FM:.4f}, HJ {loss_HJ:.4f}")

    return loss_history

def sample_ode_hj(x0, model, t0=0.0, N=100, return_path=False):
    dt = (1.0 - t0) / N
    x = x0.clone().detach()
    path = [x.clone().detach()] if return_path else None

    for i in range(N):
        x = x.clone().detach()
        x.requires_grad_(True)
        t_current = torch.full((x.shape[0], 1), t0 + i * dt, device=x.device)
        Psi = model(x, t_current)                   # (batch,1)
        psi_val = transform(Psi, x, t_current)
        # 
        grad_psi_x = torch.autograd.grad(
        outputs=psi_val,
        inputs=x,
        grad_outputs=torch.ones_like(psi_val),
        create_graph=True
    )[0]  # (batch, dim)
        v = grad_psi_x
        #v=(x-grad_x)/t_current
        
        x = x + dt * v
        if return_path:
            path.append(x.clone().detach())
    if return_path:
        trajectory = torch.stack(path, dim=0)  # shape: [N+1, batch, dim]
        return x, trajectory
    else:
        return x

def sample_ode_hj2(x0, model, C=None, t1=1.0, N=1,t0=0.0,
                       anchor_strength=0.1, guidance_strength=0.05):
    dt = (t1 - t0) / N
    x = x0.clone().detach()
    for i in range(N):
        x = x.clone().detach().requires_grad_(True)
        t_cur = torch.full((x.shape[0],1), t0 + i*dt, device=x.device)
        psi_val = transform(model(x, t_cur), x, t_cur)
        grad_psi_x = torch.autograd.grad(psi_val.sum(), x)[0]
        v = grad_psi_x - anchor_strength * (x - x0)
        if C is not None:
            score = C(x).sum()
            grad_attr = torch.autograd.grad(score, x)[0]
            v = v + guidance_strength * grad_attr
        x = (x + dt * v).detach()
    return x
#######################################


def evaluation(Psi,
               x0_test,
               x1_test,
               N=[1, 2, 4, 8],
               rescaled=True,
               test_size: int = None,
               plot_traj=True,
               pad_ratio: float = 0.05,
               num_paths: int = 5,
               traj_color: str = "#C44E52",
               highlight_x0_color: str = "#CC79A7",  # 紫（Okabe–Ito）
               highlight_gen_color: str = "#F0E442", # 黄（Okabe–Ito）
               highlight_x0_marker: str = "^",
               highlight_gen_marker: str = "s",
               highlight_alpha: float = 0.95):

    x0_np = x0_test.detach().cpu().numpy()
    x1_np = x1_test.detach().cpu().numpy()
    test_size = test_size or x0_np.shape[0]

    fig, axes = plt.subplots(1, len(N),
                             figsize=(10, 5),
                             constrained_layout=True)
    if len(N) == 1:
        axes = [axes]

    for ax, n_val in zip(axes, N):
        generated, trajectory = sample_ode_hj(
            x0_test, Psi,
            t0=0.0, N=n_val,
            return_path=True
        )
        gen_np = generated.detach().cpu().numpy()

        all_pts = [x0_np, gen_np]

        if plot_traj:
            # 轨迹 shape: [time, batch, 2] -> [batch, time, 2]
            traj = trajectory.detach().cpu().numpy().transpose(1, 0, 2)  # [B, T, 2]

            # ★ 固定选择 10 条（或不足10时取全部），与 test_size 无关
            B = traj.shape[0]
            k = min(num_paths, B)

            sel_idx = np.linspace(0, B - 1, num=k, dtype=int)

            for i in sel_idx:
                pts_i = traj[i]  # [T, 2]
                all_pts.append(pts_i)
                ax.plot(pts_i[:, 0],
                        pts_i[:, 1],
                        color=traj_color,   # ★ 改为红色
                        alpha=0.8,
                        linewidth=1.5,
                        marker = '+',
                        zorder=1)

        all_pts_arr = np.vstack(all_pts)
        xmin, ymin = all_pts_arr.min(axis=0)
        xmax, ymax = all_pts_arr.max(axis=0)

        xpad = (xmax - xmin) * pad_ratio
        ypad = (ymax - ymin) * pad_ratio
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_aspect('equal', 'box')

        ax.scatter(x0_np[:, 0], x0_np[:, 1],
                   s=10,
                   facecolors="#4C72B0",
                   edgecolors="k",
                   linewidths=0.5,
                   alpha=0.6,
                   label=r"$x_0$",
                   zorder=2)
        ax.scatter(gen_np[:, 0], gen_np[:, 1],
                   s=20,
                   facecolors="#55A868",
                   edgecolors="k",
                   linewidths=0.5,
                   alpha=0.6,
                   label=r"$x_{\mathrm{gen}}$",
                   zorder=3)
        ax.scatter(x0_np[sel_idx, 0], x0_np[sel_idx, 1],
                   s=36, facecolors=highlight_x0_color,
                   edgecolors="k", linewidths=0.7, alpha=highlight_alpha,
                   marker=highlight_x0_marker,
                   zorder=4)
        ax.scatter(gen_np[sel_idx, 0], gen_np[sel_idx, 1],
                   s=48, facecolors=highlight_gen_color,
                   edgecolors="k", linewidths=0.7, alpha=highlight_alpha,
                   marker=highlight_gen_marker,
                   zorder=5)

        ax.set_title(f"$N={n_val}$", pad=4)
        ax.legend(loc='upper right', fontsize='small')

    return fig, trajectory



def evaluation_benchmark_ode(
    model, 
    prior_sampler,    #[N_vis, dim]
    target_sampler,   #[N_vis, dim] 
    log_dir, 
    benchmark,
    num_display=512,
    N_steps=100
):
    """
    sample_ode_hj as Y_pred, comparing with OT(Y_true)
    """
    device = next(model.parameters()).device

    X_vis = prior_sampler.sample(num_display).to(device)
    X_vis.requires_grad_(True)
    Y_vis = target_sampler.sample(num_display).to(device)

    Y_true = benchmark.map_fwd(X_vis, nograd=True).to(device)

    # sample_ode_hj(x0, model, t0, N, return_path=False) 返回 [num_display, dim]
    Y_pred = sample_ode_hj(X_vis, model, t0=0.0, N=N_steps, return_path=False)
    Y_pred = Y_pred.to(device)
    L2_UVP_fwd = 100 * (((Y_true - Y_pred) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()

    #  PCA 降到二维
    B = X_vis.size(0)
    combined = torch.cat([X_vis, Y_true, Y_pred], dim=0).cpu().detach().numpy()
    Z = PCA(n_components=2).fit_transform(combined)
    Xp, Yt, Yp = Z[:B], Z[B:2*B], Z[2*B:3*B]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.scatter(Xp[:, 0], Xp[:, 1], s=5, alpha=0.3, label='X')
    ax1.scatter(Yt[:, 0], Yt[:, 1], s=5, alpha=0.6, label='Y_true')
    ax1.set_title("Ground Truth OT Map")
    ax1.legend()

    ax2.scatter(Xp[:, 0], Xp[:, 1], s=5, alpha=0.3, label='X')
    ax2.scatter(Yp[:, 0], Yp[:, 1], s=5, alpha=0.6, label='Y_pred (ODE)')
    ax2.set_title(f"Model ODE Map (N={N_steps})")
    ax2.legend()

    plt.tight_layout()
    return fig, L2_UVP_fwd

def evaluation_mnist(
    model_Psi,
    prior_samples,
    target_samples,
    num_display=64,    
    N_steps=100        
):

    device = next(model_Psi.parameters()).device

    # small prior
    x0 = prior_samples[:num_display].to(device)
   
    x_gen = sample_ode_hj(x0, model_Psi, t0=0.0, N=N_steps, return_path=False)
    x_gen = x_gen.cpu()
    # choose real target
    x_true = target_samples[:num_display].cpu()

    # reshape - image
    side = int(np.sqrt(x_gen.size(1)))  #dim=28
    x_gen_img  = x_gen.view(-1, 1, side, side)
    x_true_img = x_true.view(-1, 1, side, side)

    grid_gen  = vutils.make_grid(x_gen_img,  nrow=8, normalize=True, pad_value=1)
    grid_true = vutils.make_grid(x_true_img, nrow=8, normalize=True, pad_value=1)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Generated")
    plt.axis('off')
    plt.imshow(grid_gen.permute(1,2,0), cmap='gray')

    plt.subplot(1,2,2)
    plt.title("Real MNIST")
    plt.axis('off')
    plt.imshow(grid_true.permute(1,2,0), cmap='gray')

    plt.tight_layout()

def evaluation_straight_1(model,prior_sampler,N=[1,2,4,8],num_display=40,rescaled=True,plot_traj=True):
    device = next(model.parameters()).device
    x0_test = prior_sampler.sample(num_display).to(device)
    x0_test.requires_grad_(True)
    fig = plt.figure(layout='constrained', figsize=(6, 6))
    subfigs = fig.subfigures(1, len(N), wspace=0.07)
    for isubfig in range(len(N)):
        generated, trajectory = sample_ode_hj(x0_test, model, t0=0.0, N=N[isubfig], return_path=True)
        generated = generated.cpu().detach().numpy()
        ax2 = subfigs[isubfig].subplots(1,1)
        if plot_traj:
            trajectory = trajectory.cpu().detach().numpy()
            trajectory = trajectory.transpose(1,0,2)
            for i in range(40):
                ax2.plot(
                    trajectory[i, :, 0],  #x
                    trajectory[i, :, 1],  #y
                    color="gray", alpha=0.5, linewidth=0.5,
                    zorder=1, marker = '+'
                )
            
        # generated
        ax2.scatter(
            generated[:, 0],
            generated[:, 1],
            s=10, color="blue", alpha=0.2,
            label="x1gen",
            zorder=3
        )
        ax2.legend()
    return trajectory

def evaluation_straight(model,prior_sampler,N=[1,2,4,8],num_display=10,rescaled=True,plot_traj=True):
    device = next(model.parameters()).device
    if hasattr(prior_sampler, "sample"):
        x0_test = prior_sampler.sample(num_display).to(device)
    elif isinstance(prior_sampler, torch.Tensor):
        x0_test = prior_sampler[:num_display].to(device)
    else:
        raise TypeError(f"Unsupported prior_sampler type: {type(prior_sampler)}")
    x0_test = prior_sampler.sample(num_display).to(device)
    x0_test.requires_grad_(True)
    fig = plt.figure(layout='constrained', figsize=(6, 6))
    subfigs = fig.subfigures(1, 1)
    ax2 = subfigs.subplots(1,1)
    for isubfig in range(len(N)):
        generated, trajectory = sample_ode_hj(x0_test, model, t0=0.0, N=N[isubfig], return_path=True)
        generated = generated.cpu().detach().numpy()
        if plot_traj:
            trajectory = trajectory.cpu().detach().numpy()
            trajectory = trajectory.transpose(1,0,2)
            for i in range(num_display):
                ax2.plot(
                    trajectory[i, :, 0],  #x
                    trajectory[i, :, 1],  #y
                    color="gray", alpha=0.5, linewidth=0.5,
                    zorder=1, marker = '+'
                )
            
        # generated
        ax2.scatter(
            generated[:, 0],
            generated[:, 1],
            s=10, color="blue", alpha=0.2,
            label="x1gen",
            zorder=3
        )
    return trajectory

class ICNN(nn.Module):
    def __init__(self, dim=2, dim_t=1, dimh=64, num_hidden_layers=5, act_fn=F.elu, batch_size=1024,
                actnorm_initialized=False):
        super().__init__()
        self.act = act_fn
        self.num_hidden_layers = num_hidden_layers+1
        # Wzs
        self.Wzs = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            layer = PosLinear(dimh, dimh, bias=False)
            self.Wzs.append(layer)
        # last layer
        self.Wzs.append(PosLinear(dimh, 1, bias=False))
        
        # skip Wxs
        self.Wxs = nn.ModuleList()
        self.Wxs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            self.Wxs.append(nn.Linear(dim, dimh))
        self.Wxs.append(nn.Linear(dim, 1))

    # Normalization type activation (linear)
        actnorms = list()
        for _ in range(num_hidden_layers):
            actnorms.append(ActNormNoLogdet(dimh, initialized=actnorm_initialized))
        actnorms.append(ActNormNoLogdet(1, initialized=actnorm_initialized))
        self.actnorms = torch.nn.ModuleList(actnorms)
        
        # Time-bias networks S_l
        self.Snets = nn.ModuleList()
        for l in range(self.num_hidden_layers):
            out_dim = dimh if l < num_hidden_layers else 1
            in_dim = dim_t
            # simple MLP for S_l
            self.Snets.append(nn.Sequential(nn.Linear(in_dim, dimh), nn.ReLU(),nn.Linear(dimh, out_dim)))

    def forward(self, x, t):
        # x: (batch, dim), t: (batch, 1)
        z = None 

        for l in range(self.num_hidden_layers):
            # last output: Wz * N_l
            wz = self.Wzs[l-1](z) if l > 0 else 0
            # skip layer: Wx * x
            wx = self.Wxs[l](x)
            # time bias: S_l(t) - S_l(1)
            s_t = self.Snets[l](t) 
            bias_t = s_t
            pre = wz + wx + bias_t
            
            # last layer is not activated and output directly
            if l < self.num_hidden_layers - 1:
                # add ActNorm to first layer
                pre = self.actnorms[l](pre)
                z = self.act(pre)
            else:
                z = pre

        # initial condition, Psi(1,x)=0.5||x||^2
        # x2 = 0.5 * x.pow(2).sum(dim=1, keepdim=True)
        # Psi = z # * (1 - t) + x2
        Psi = z * (1.0 - t)
        return Psi

class ICNN1(nn.Module):
    def __init__(self, dim=2, dim_t=1, dimh=64, num_hidden_layers=5, act_fn=F.elu, batch_size=1024,
                actnorm_initialized=False):
        super().__init__()
        self.act = act_fn
        self.num_hidden_layers = num_hidden_layers+1
        # Wzs
        self.Wzs = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            layer = PosLinear(dimh, dimh, bias=False)
            self.Wzs.append(layer)
        # last layer
        self.Wzs.append(PosLinear(dimh, 1, bias=False))
        
        # skip Wxs
        self.Wxs = nn.ModuleList()
        self.Wxs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            self.Wxs.append(nn.Linear(dim, dimh))
        self.Wxs.append(nn.Linear(dim, 1))

    # Normalization type activation (linear)
        actnorms = list()
        for _ in range(num_hidden_layers):
            actnorms.append(ActNormNoLogdet(dimh, initialized=actnorm_initialized))
        actnorms.append(ActNormNoLogdet(1, initialized=actnorm_initialized))
        self.actnorms = torch.nn.ModuleList(actnorms)
        
        # Time-bias networks S_l
        self.Snets = nn.ModuleList()
        for l in range(self.num_hidden_layers):
            out_dim = dimh if l < num_hidden_layers else 1
            in_dim = dim_t
            # simple MLP for S_l
            self.Snets.append(nn.Sequential(nn.Linear(in_dim, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, out_dim)))
        
        self.rnet = nn.Sequential(nn.Linear(dim_t, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, 1),nn.Softplus())
        
        

    def forward(self, x, t):
        # x: (batch, dim), t: (batch, 1)
        z = None 

        for l in range(self.num_hidden_layers):
            # last output: Wz * N_l
            wz = self.Wzs[l-1](z) if l > 0 else 0
            # skip layer: Wx * x
            wx = self.Wxs[l](x)
            # time bias: S_l(t) - S_l(1)
            s_t = self.Snets[l](t) 
            bias_t = s_t
            pre = wz + wx + bias_t
            
            # last layer is not activated and output directly
            if l < self.num_hidden_layers - 1:
                # add ActNorm to first layer
                pre = self.actnorms[l](pre)
                z = self.act(pre)
            else:
                z = pre

        # initial condition, Psi(1,x)=0.5||x||^2
        x2 = F.sigmoid(self.rnet(t)*(1.0-t))*(x.pow(2).sum(dim=1, keepdim=True))
        Psi = z * (1.0 - t) + x2
        return Psi

class ICNN2(nn.Module):
    def __init__(self, dim=2, dim_t=1, dimh=64, num_hidden_layers=5, act_fn=F.elu, batch_size=1024,
                actnorm_initialized=False):
        super().__init__()
        self.act = act_fn
        self.num_hidden_layers = num_hidden_layers+1
        # Wzs
        self.Wzs = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            layer = PosLinear2(dimh, dimh, bias=False)
            self.Wzs.append(layer)
        # last layer
        self.Wzs.append(PosLinear2(dimh, 1, bias=False))
        
        # skip Wxs
        self.Wxs = nn.ModuleList()
        self.Wxs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            self.Wxs.append(nn.Linear(dim, dimh))
        self.Wxs.append(nn.Linear(dim, 1))

    # Normalization type activation (linear)
        actnorms = list()
        for _ in range(num_hidden_layers):
            actnorms.append(ActNormNoLogdet(dimh, initialized=actnorm_initialized))
        actnorms.append(ActNormNoLogdet(1, initialized=actnorm_initialized))
        self.actnorms = torch.nn.ModuleList(actnorms)
        
        # Time-bias networks S_l
        self.Snets = nn.ModuleList()
        for l in range(self.num_hidden_layers):
            out_dim = dimh if l < num_hidden_layers else 1
            in_dim = dim_t
            # simple MLP for S_l
            self.Snets.append(nn.Sequential(nn.Linear(in_dim, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, out_dim)))
        
        self.rnet = nn.Sequential(nn.Linear(dim_t, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, 1),nn.Softplus())
        
        

    def forward(self, x, t):
        # x: (batch, dim), t: (batch, 1)
        z = None 

        for l in range(self.num_hidden_layers):
            # last output: Wz * N_l
            wz = self.Wzs[l-1](z) if l > 0 else 0
            # skip layer: Wx * x
            wx = self.Wxs[l](x)
            # time bias: S_l(t) - S_l(1)
            s_t = self.Snets[l](t) 
            bias_t = s_t
            pre = wz + wx + bias_t
            
            # last layer is not activated and output directly
            if l < self.num_hidden_layers - 1:
                # add ActNorm to first layer
                pre = self.actnorms[l](pre)
                z = self.act(pre)
            else:
                z = pre

        # initial condition, Psi(1,x)=0.5||x||^2
        x2 = F.sigmoid(self.rnet(t)*(1.0-t))*(x.pow(2).sum(dim=1, keepdim=True))
        Psi = z * (1.0 - t) + x2
                # add 0.5 * epsilon * ||x||^2
        strong_convex_reg = 0.5 * 1e-4 * x2
        Psi = z * (1.0 - t) + x2 + strong_convex_reg
        return Psi

class ICNN3(nn.Module):
    def __init__(self, dim=2, dim_t=1, dimh=64, num_hidden_layers=5, act_fn=F.elu, batch_size=1024,eps=1e-4,
                actnorm_initialized=False):
        super().__init__()
        self.act = act_fn
        self.eps = eps
        self.num_hidden_layers = num_hidden_layers+1
        # Wzs
        self.Wzs = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            layer = PosLinear2(dimh, dimh, bias=False)
            self.Wzs.append(layer)
        # last layer
        self.Wzs.append(PosLinear2(dimh, 1, bias=False))
        
        # skip Wxs
        self.Wxs = nn.ModuleList()
        self.Wxs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            self.Wxs.append(nn.Linear(dim, dimh))
        self.Wxs.append(nn.Linear(dim, 1))

    # Normalization type activation (linear)
        actnorms = list()
        for _ in range(num_hidden_layers):
            actnorms.append(ActNormNoLogdet(dimh, initialized=actnorm_initialized))
        actnorms.append(ActNormNoLogdet(1, initialized=actnorm_initialized))
        self.actnorms = torch.nn.ModuleList(actnorms)
        
        # Time-bias networks S_l
        self.Snets = nn.ModuleList()
        for l in range(self.num_hidden_layers):
            out_dim = dimh if l < num_hidden_layers else 1
            in_dim = dim_t
            # simple MLP for S_l
            self.Snets.append(nn.Sequential(nn.Linear(in_dim, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, out_dim)))
        
        # self.rnet = nn.Sequential(nn.Linear(dim_t, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, 1),nn.Softplus())
        self.rnet = nn.Sequential(nn.Linear(dim_t, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, dimh), nn.Sigmoid(),nn.Linear(dimh, 1)) # no softplus
        
        

    def forward(self, x, t):
        # x: (batch, dim), t: (batch, 1)
        z = None 

        for l in range(self.num_hidden_layers):
            # last output: Wz * N_l
            wz = self.Wzs[l-1](z) if l > 0 else 0
            # skip layer: Wx * x
            wx = self.Wxs[l](x)
            # time bias: S_l(t) - S_l(1)
            s_t = self.Snets[l](t) 
            bias_t = s_t
            pre = wz + wx + bias_t
            
            # last layer is not activated and output directly
            if l < self.num_hidden_layers - 1:
                # add ActNorm to first layer
                pre = self.actnorms[l](pre)
                z = self.act(pre)
            else:
                z = pre

        # initial condition
        x2 = F.sigmoid(self.rnet(t)*(1.0-t))*(x.pow(2).sum(dim=1, keepdim=True))
                # add 0.5 * epsilon * ||x||^2

        strong_convex_reg = 0.5 * self.eps * x2
        Psi = z * (1.0 - t) + x2 + strong_convex_reg

        # quad = (x**2).sum(dim=1, keepdim=True)           #  ||x||^2
        # alpha = F.sigmoid(self.rnet(t))                 # α(t) ≥ 0 
        # Psi = 0.5 * quad + (1.0 - t) * ( z + alpha * quad )
        return Psi
        
        
    def gradx(self,x,t):
        grad = torch.autograd.grad(torch.sum(self.forward(x,t)), x, \
        retain_graph=True, create_graph=True)[0]
        # assert x.size()==grad.size()
        return grad