import torch
import logging
import numpy as np
import copy
import csv

def normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y

def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size())==1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min=sorted_x[int(d * min_ratio)]
        max=sorted_x[int(d * (1-max_ratio)-1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min=sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max=sorted_x[:, int(d * (1-max_ratio)-1)].unsqueeze(1)
    clamped_x= torch.clamp(x, min, max)
    return clamped_x

def act(x):
    y = torch.tanh(x)  # torch.relu(x)
    return y

def PCB_merge(flat_task_checks, pcb_ratio=0.1):
    all_checks = flat_task_checks.clone()
    n, d = all_checks.shape   
    all_checks_abs = clamp(torch.abs(all_checks), min_ratio=0.0001, max_ratio=0.0001)
    clamped_all_checks = torch.sign(all_checks)*all_checks_abs
    self_pcb = normalize(all_checks_abs, 1)**2
    self_pcb_act = torch.exp(n*self_pcb)
    cross_pcb = all_checks * torch.sum(all_checks, dim=0)
    cross_pcb_act = act(cross_pcb)
    task_pcb = self_pcb_act * cross_pcb_act

    scale = normalize(clamp(task_pcb, 1-pcb_ratio, 0), dim=1)
    tvs = clamped_all_checks
    merged_tv = torch.sum(tvs * scale, dim=0) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
    return merged_tv, clamped_all_checks, scale


### With Evolutionary Strategies

n = len(datasets)
reference_state_dict=ft_checks[0]
lams = [1.2]*n
pcb_ratio = 0.05
clamped_all_checks, scale = PCB_merge(tv_flat_checks, pcb_ratio=pcb_ratio)

from functools import partial
import nevergrad as ng
get_acc = partial(compute_acc, all_checks=clamped_all_checks, scale=scale, flat_ptm=flat_ptm, 
                            model=model, datasets=datasets, args=args, split='validation', 
                            reference_state_dict=ft_checks[0], remove_keys=remove_keys)
def opp_get_acc(*args, **kwargs):
    return -get_acc(*args, **kwargs)[0]

instrum = ng.p.Array(init=[1.0]*n, upper=[1.3]*n, lower=[0.8]*n,)
max_inference_step = 100
optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
print(">>> Begin to perform gradient-free optimization ...")
# Please make sure that your function returns a float, and that you indeed want to 
# perform minimization and not maximization
recommendation = optimizer.minimize(opp_get_acc, verbosity=1)
lams = recommendation.value

print('lams:', lams)
split = 'test'
acc, acc_lst = compute_acc(lams, clamped_all_checks, scale, flat_ptm, model, datasets, args, split, 
            reference_state_dict, remove_keys)