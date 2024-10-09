import json
import torch
from torch.nn.functional import normalize
from collections import defaultdict
import json
import numpy as np
import math
import argparse
from collections import defaultdict


## compute instance value
def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores


def combine_gradients(gradient_path, chunk_num=8):
    all_gradients = []
    for idx in range(chunk_num):
        g = torch.load(f"{gradient_path}/output_{idx}", map_location='cpu')
        all_gradients.extend(torch.cat(g, dim=0)) 
    return all_gradients


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", add_help=True)
    parser.add_argument('--chunk_num', type=int, default=8)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--p', type=float, default=0.15)
    parser.add_argument('--q', type=float, default=0.001)
    parser.add_argument("--gradient_path", type=str)
    args = parser.parse_args()
    
    grads_path = args.gradient_path
    
    grads = []
    for idx in range(args.chunk_num):
        grads.extend(json.load(open(f'{grads_path}/output_norm_{idx}.json', 'r')))
    
    data = json.load(open(args.data_path, 'r'))
    hyper_q, hyper_p = args.q, args.p

    ## compute task value
    ds2si, ds2data = defaultdict(float), defaultdict(list)

    for idx, d in enumerate(data):
        # exclude caption data since it's been use during pre-training
        if d['dataset'] == 'textcaps':
            continue
        ds2si[d['dataset']] += grads[idx]
        ds2data[d['dataset']].append(d)

    total_si = 0
    ## print task value
    for k, v in ds2si.items():
        print(f"{k}: {v / len(ds2data[k])}")
        total_si += v / len(ds2data[k])
    
    ## initializing gradient features
    grads = combine_gradients(grads_path, args.chunk_num)

    grads = torch.cat([sg.unsqueeze(0) for sg in grads], dim = 0)
    grads_normalized = normalize(grads, dim=1)

    ## record the gradient features and corrsponding index in original data within each super category
    ds2grad = defaultdict(list)
    ds2idx = defaultdict(list)

    for idx, d in enumerate(data):
        if d['dataset'] == 'textcaps':
            continue
        ds2grad[d['dataset']].append(grads_normalized[idx])
        ds2idx[d['dataset']].append(idx)
        
    
    selected_data_size = hyper_p * len(data)
    selected_idx = []
    
    ## select data for each task dataset
    for ds in ds2grad.keys():
        print(ds)
        ds_size = ds2si[ds] / len(ds2data[ds]) / total_si * selected_data_size
        print(ds_size)
        
        target_grads_sub_tasks = torch.cat(ds2grad[ds])
        target_grads_sub_tasks = target_grads_sub_tasks.reshape([-1, 8192])

        ## in case the data samples of one task is too large, causing out-of-memory
        if len(ds2grad[ds]) > 300000:
            chunk_size = 80000
            chunk_data = [target_grads_sub_tasks[i:i+chunk_size] for i in range(0, len(target_grads_sub_tasks), chunk_size)]
            iscores = []
            for d in chunk_data:
                influence_score = calculate_influence_score(d, target_grads_sub_tasks)
                influence_score = influence_score.reshape(
                            influence_score.shape[0], 1, -1).mean(-1).max(-1)[0]
                iscores.append(influence_score)
            influence_score = torch.cat(iscores)

        else:
            influence_score = calculate_influence_score(target_grads_sub_tasks, target_grads_sub_tasks)
            influence_score = influence_score.reshape(
                        influence_score.shape[0], 1, -1).mean(-1).max(-1)[0]

        print(influence_score.size())
        
        ## soft sampling new implementation
        q = hyper_q
        instance_weight = torch.nn.functional.softmax((influence_score * q).clone().detach())
        instance_weight = instance_weight.cpu().numpy()
        instance_weight = instance_weight / np.sum(instance_weight)

        ## enough samples for selection
        if ds_size < len(ds2idx[ds]):
            selected_idx.extend(np.random.choice(ds2idx[ds], p = instance_weight, size=int(ds_size), replace=False))
            
        ## else, oversampling
        else:
            oversample_times = int(ds_size // len(ds2idx[ds]))
            sampled_numbers = int(ds_size % len(ds2idx[ds]))
            for i in range(oversample_times):
                selected_idx.extend(ds2idx[ds])
            selected_idx.extend(np.random.choice(ds2idx[ds], p = instance_weight, size=sampled_numbers, replace=False))
        
        
    ## combine selected data
    new_data = []
    for idx in selected_idx:
        new_data.append(data[idx])

    ## save selected_data
    json.dump(new_data, open(f"{args.save_path}/Tive_q_{hyper_q}_p_{hyper_p}", 'w'), indent=4)
