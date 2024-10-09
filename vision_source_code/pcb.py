import torch
import logging
import numpy as np
import copy
import csv
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.eval import eval_single_dataset
from utils.tpa_utils import load_vit, loadCheckpoint_intoModel, vector_to_state_dict
# from utils.merge_utils import *
from args import parse_arguments
logger = logging.getLogger("root")

model = 'ViT-B-32'  # 'ViT-B-32', 'ViT-L-14'
args = parse_arguments()
args.save = f'{args.model_location}/{model}'
args.data_location = '/data/yourpath/merging/soup_data'
model_location = '/data/yourpath/merging'
load_dir = f'{model_location}/{model}'

pretrained_checkpoint = f'{args.model_location}/{model}/zeroshot.pt'
model = torch.load(pretrained_checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD',]

tv_flat_checks, flat_ptm, flat_ft, \
    ft_checks, ptm_check, remove_keys = load_vit(datasets, load_dir)

def compute_acc(reference_state_dict, flat_ptm, model, datasets, args):
    merged_check = flat_ptm + merged_tv
    merged_checkpoint = vector_to_state_dict(
        merged_check, reference_state_dict, remove_keys=remove_keys
    )
    model = loadCheckpoint_intoModel(merged_checkpoint, model)
    acc_lst = []
    for dataset in datasets:
        top1 = eval_single_dataset(model, dataset, args)
        acc_lst.append(top1)
    avg_acc = sum(acc_lst)/len(acc_lst)
    return round(100*avg_acc, 2), acc_lst

def piecewise_function(x, ratios=[0.039], scores=[1, 0]):
    assert len(ratios)+1 == len(scores)
    sorted_x, _ = torch.sort(x, descending=True)
    ratio_idx = [int(len(x) * ratio) for ratio in ratios]
    values = [sorted_x[idx] for idx in ratio_idx]
    values.insert(0, sorted_x[0])
    values.append(sorted_x[-1])
    y = torch.zeros_like(x)
    for i in range(len(scores)):
        y += ((x <= values[i]) & (x > values[i+1])).float() * scores[i]
    return y

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
    y = torch.tanh(x)  # x**7; torch.relu(x)
    return y

if __name__ == "__main__":
    all_checks = tv_flat_checks
    n, d = all_checks.shape    # d = 113448705
    all_checks_abs = clamp(torch.abs(all_checks), min_ratio=0.01, max_ratio=0.01)
    clamped_all_checks = torch.sign(all_checks)*all_checks_abs
    att_ratio = 0.05
    all_checks_normalized = torch.sign(all_checks) * normalize(all_checks_abs, dim=1)
    intra = normalize(all_checks_abs, 1)**2
    intra = torch.exp(n*intra)
    inter = all_checks * torch.sum(all_checks_normalized, dim=0)
    inter = act(inter)
    balancing = intra * inter

    """
    If there is a validation dataset, we apply a mask ratio to help reduce redundancy.
    """
    # mask_ratio = 0.05
    # kth_values, _ = all_checks.abs().kthvalue(int(d * (1-mask_ratio)), dim=1, keepdim=True)
    # mask = all_checks.abs() >= kth_values
    # balancing = mask * balancing

    scale = normalize(clamp(balancing, 1-att_ratio, 0), dim=1)
    # lams = (torch.ones(n)*1.2).unsqueeze(1)
    lams = torch.tensor([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]).unsqueeze(1)
    tvs = clamped_all_checks * lams
    merged_tv = torch.sum(tvs * scale, dim=0) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
    reference_state_dict = ft_checks[0]
    args.validate = False
    acc, acc_lst = compute_acc(reference_state_dict, flat_ptm, model, datasets, args)
    acc_lst = [round(i*100, 2) for i in acc_lst]
    print('acc_avg:', acc, acc_lst, 'lams:', [round(i, 3) for i in lams.squeeze().tolist()], \
          'att_ratio:', att_ratio)
    



