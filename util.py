# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 19:35
@Author  : Zekun Cai
@File    : util.py
@Software: PyCharm
'''
import pickle
import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score

# def focal_loss(log_prob, target, weight=None, gamma=2.0):
#     """
#     log_prob: log softmax output (N,C)
#     target: label (N)
#     """

#     prob = torch.exp(log_prob)

#     ce_loss = F.nll_loss(log_prob, target, weight=weight, reduction='none')

#     pt = prob.gather(1, target.unsqueeze(1)).squeeze()

#     loss = ((1 - pt) ** gamma) * ce_loss

#     return loss.mean()

def load_datasets(filename="dataset.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["datasets"], data["time_points"]


def _bce_binary_targets(y: torch.Tensor) -> torch.Tensor:
    """
    F.binary_cross_entropy (CUDA) 要求 target 落在 [0, 1]。
    若数据是两类的任意编码（如 -1/1、0/2），映射为 0/1。
    """
    y = y.float()
    lo, hi = float(y.min()), float(y.max())
    if lo >= 0.0 and hi <= 1.0:
        return y
    u = torch.unique(y)
    if u.numel() != 2:
        raise ValueError(
            f'BCE需要标签在 [0,1] 内，或恰好两个类别；当前 unique={u.tolist()}'
        )
    return (y == u.max()).float()


def _safe_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Use eps large enough for float32 so upper clamp is below 1.0 in machine precision.
    eps = 1e-6
    pred = pred.clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(pred, target)


def dataset_preparation(dataset, path, device):
    datasets, time_points = load_datasets(path)

    if dataset == 'Moons':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        Y = [_bce_binary_targets(torch.Tensor(item[1])[:, None]).to(device) for item in datasets]
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'MNIST':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        X = [torch.transpose(x, 1, 3) for x in X]
        Y = [torch.Tensor(item[1]).long().to(device) for item in datasets]
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'YearBook':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        X = [torch.transpose(x, 1, 3) for x in X]
        Y = [_bce_binary_targets(torch.Tensor(item[1])[:, None]).to(device) for item in datasets]
        # Scale the timeline to match the rk step size.
        # We set the maximum time to 50 so that it is not too large (too many integral steps) or too small (not enough precision).
        # Note that this value does not must be set to 50.
        time_points = time_points / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'Twitter':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        Y = [_bce_binary_targets(torch.Tensor(item[1])[:, None]).to(device) for item in datasets]
        time_points = time_points / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'Cyclone':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        X = [torch.transpose(x, 1, 3) for x in X]
        Y = [torch.Tensor(item[1])[:, None].to(device) for item in datasets]
        time_points = time_points / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'House':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        Y = [torch.Tensor(item[1])[:, None].to(device) for item in datasets]
        time_points = time_points / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'Complaints8':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        Y = [torch.Tensor(item[1]).long().to(device) for item in datasets]
        time_points = time_points / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    else:
        raise ValueError('Not defined dataset!')

    return X, Y, time_points


def nll_loss_maybe_focal(
    log_probs,
    target,
    class_weight=None,
    focal_gamma=0.0,
    tail_class_ids=None,
    tail_boost=1.0,
):
    """
    log_probs: log-softmax。可选 class_weight、focal、以及对 tail_class_ids 内标签的逐样本 CE 放大（针对难尾类）。
    """
    ce = F.nll_loss(log_probs, target, weight=class_weight, reduction='none')
    fg = float(focal_gamma or 0.0)
    if fg > 0.0:
        pt = log_probs.exp().gather(1, target.unsqueeze(1)).squeeze(1).clamp(
            min=1e-8, max=1.0 - 1e-8
        )
        ce = ce * (1.0 - pt).pow(fg)
    tb = float(tail_boost or 1.0)
    if tail_class_ids is not None and len(tail_class_ids) > 0 and tb > 1.0:
        if isinstance(tail_class_ids, torch.Tensor):
            tbuf = tail_class_ids.to(device=target.device, dtype=torch.long)
        else:
            tbuf = torch.as_tensor(tail_class_ids, device=target.device, dtype=torch.long)
        m = torch.isin(target, tbuf)
        ce = ce * torch.where(m, torch.full_like(ce, tb), torch.ones_like(ce))
    return ce.mean()


def build_nll_class_weight(Y_train, n_class, device, power=0.25):
    """按训练域标签频次构造 NLL 权重：freq^{-power} 后归一化到均值 1（缓解 Complaints8 等类别不均衡）。"""
    ys = []
    for y in Y_train:
        arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        ys.append(arr.reshape(-1))
    ys = np.concatenate(ys)
    freq = np.bincount(ys.astype(np.int64), minlength=n_class).astype(np.float32)
    weight = 1.0 / np.power(np.maximum(freq, 1.0), power)
    weight = weight / weight.mean()
    return torch.tensor(weight, dtype=torch.float32, device=device)


def get_cia_aux_loss(
    dataset,
    Y,
    cia_aux,
    mix_weight=0.5,
    spu_entropy_weight=0.05,
    class_weight=None,
    focal_gamma=0.0,
    tail_class_ids=None,
    tail_boost=1.0,
    skip_causal_aux=False,
):
    """Auxiliary CIA-style losses: causal & mix supervision; spurious branch entropy maximization.
    当主任务已仅用因果特征 x_c 时，设 skip_causal_aux=True，避免 init_c/gene_c 与主路 NLL/BCE 重复加权。
    """
    if cia_aux is None:
        return torch.tensor(0.0, device=Y[0].device)

    from model.cia_tabular import entropy_from_log_softmax, binary_entropy_from_probs

    y = torch.cat(Y)

    if dataset in ['MNIST', 'Complaints8', 'Complaints8_M']:
        w = class_weight
        fg = focal_gamma
        tid, tb = tail_class_ids, tail_boost
        if skip_causal_aux:
            z = torch.zeros((), device=y.device, dtype=torch.float32)
            loss_ic = loss_gc = z
        else:
            loss_ic = nll_loss_maybe_focal(torch.cat(cia_aux['init_c']), y, w, fg, tid, tb)
            loss_gc = nll_loss_maybe_focal(torch.cat(cia_aux['gene_c']), y, w, fg, tid, tb)
        loss_im = nll_loss_maybe_focal(torch.cat(cia_aux['init_m']), y, w, fg, tid, tb)
        loss_gm = nll_loss_maybe_focal(torch.cat(cia_aux['gene_m']), y, w, fg, tid, tb)
        ent_i = entropy_from_log_softmax(torch.cat(cia_aux['init_s']))
        ent_g = entropy_from_log_softmax(torch.cat(cia_aux['gene_s']))
    elif dataset in ['YearBook', 'Moons', 'Twitter']:
        y = _bce_binary_targets(y)
        if skip_causal_aux:
            z = torch.zeros((), device=y.device, dtype=torch.float32)
            loss_ic = loss_gc = z
        else:
            loss_ic = _safe_bce(torch.cat(cia_aux['init_c']), y)
            loss_gc = _safe_bce(torch.cat(cia_aux['gene_c']), y)
        loss_im = _safe_bce(torch.cat(cia_aux['init_m']), y)
        loss_gm = _safe_bce(torch.cat(cia_aux['gene_m']), y)
        ent_i = binary_entropy_from_probs(torch.cat(cia_aux['init_s']))
        ent_g = binary_entropy_from_probs(torch.cat(cia_aux['gene_s']))
    elif dataset in ['Cyclone', 'House']:
        # 回归：与主任务同一套 regression_loss（l1/smooth_l1/l1_mse）；伪相关仍用 |y_s| 惩罚
        from param import SETTING
        cfg = SETTING.get(dataset, {})
        yv = torch.cat(Y).float()
        if skip_causal_aux:
            z = torch.zeros((), device=yv.device, dtype=torch.float32)
            loss_ic = loss_gc = z
        else:
            loss_ic = _regression_supervised_loss(torch.cat(cia_aux['init_c']), yv, cfg)
            loss_gc = _regression_supervised_loss(torch.cat(cia_aux['gene_c']), yv, cfg)
        loss_im = _regression_supervised_loss(torch.cat(cia_aux['init_m']), yv, cfg)
        loss_gm = _regression_supervised_loss(torch.cat(cia_aux['gene_m']), yv, cfg)
        ps_i = torch.cat(cia_aux['init_s'])
        ps_g = torch.cat(cia_aux['gene_s'])
        loss_spu = spu_entropy_weight * (ps_i.abs().mean() + ps_g.abs().mean())
        return loss_ic + loss_gc + mix_weight * (loss_im + loss_gm) + loss_spu
    else:
        return torch.tensor(0.0, device=Y[0].device)

    # Minimize negative entropy => push spurious predictions toward high entropy (uninformative)
    loss_spu = -spu_entropy_weight * (ent_i + ent_g)

    return loss_ic + loss_gc + mix_weight * (loss_im + loss_gm) + loss_spu


def pred_to_label_np(dataset: str, pred: torch.Tensor) -> np.ndarray:
    """
    将模型输出转为离散类别，供 classification_report / sklearn 使用。
    二分类（sigmoid，形状 [N,1] 或 [N]）：阈值0.5，不用 argmax。
    多分类（log-softmax，形状 [N,C]）：argmax(dim=1)。
    """
    if dataset in ['Moons', 'YearBook', 'Twitter']:
        return (pred.detach() > 0.5).long().squeeze(-1).cpu().numpy()
    if dataset in ['MNIST', 'Complaints8', 'Complaints8_M']:
        return pred.argmax(dim=1).detach().cpu().numpy()
    raise ValueError(f'pred_to_label_np: unsupported dataset {dataset}')


def y_to_label_np(dataset: str, y: torch.Tensor) -> np.ndarray:
    """标签张量展平为 int 向量，与 pred_to_label_np 对齐。"""
    if dataset in ['Moons', 'YearBook', 'Twitter', 'MNIST', 'Complaints8', 'Complaints8_M']:
        return y.view(-1).long().cpu().numpy()
    raise ValueError(f'y_to_label_np: unsupported dataset {dataset}')


def _regression_supervised_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict) -> torch.Tensor:
    """House/Cyclone：主任务回归监督，可由 param 选择 l1 / smooth_l1 / l1+mse。"""
    kind = str(cfg.get('regression_loss', 'l1')).lower()
    pred = pred.float()
    target = target.float()
    if kind == 'smooth_l1':
        beta = float(cfg.get('regression_smooth_beta', 1.0))
        return F.smooth_l1_loss(pred, target, beta=beta)
    if kind in ('l1_mse', 'l1+mse', 'mix'):
        w1 = float(cfg.get('regression_l1_mse_l1_weight', 0.8))
        w2 = float(cfg.get('regression_l1_mse_mse_weight', 0.2))
        return w1 * F.l1_loss(pred, target) + w2 * F.mse_loss(pred, target)
    return F.l1_loss(pred, target)


def get_task_loss(
    dataset,
    Y,
    init_pred,
    gene_pred,
    class_weight=None,
    focal_gamma=0.0,
    tail_class_ids=None,
    tail_boost=1.0,
):
    init_pred, gene_pred, Y = torch.cat(init_pred), torch.cat(gene_pred), torch.cat(Y)

    if dataset in ['Moons', 'YearBook', 'Twitter']:
        Y = _bce_binary_targets(Y)
        loss_intri = _safe_bce(init_pred, Y)
        loss_integ = _safe_bce(gene_pred, Y)
        return loss_intri, loss_integ

    elif dataset in ['MNIST', 'Complaints8', 'Complaints8_M']:
        w = class_weight
        fg = focal_gamma
        tid, tb = tail_class_ids, tail_boost
        loss_intri = nll_loss_maybe_focal(init_pred, Y, w, fg, tid, tb)
        loss_integ = nll_loss_maybe_focal(gene_pred, Y, w, fg, tid, tb)
        return loss_intri, loss_integ

    elif dataset in ['Cyclone', 'House']:
        from param import SETTING
        cfg = SETTING.get(dataset, {})
        loss_intri = _regression_supervised_loss(init_pred, Y, cfg)
        loss_integ = _regression_supervised_loss(gene_pred, Y, cfg)
        return loss_intri, loss_integ

    else:
        raise ValueError('Not defined dataset!')


def twitter_accuracy_auc(Y: torch.Tensor, pred: torch.Tensor):
    """
    Twitter 二分类：准确率（预测 > 0.5 为 1）与 ROC-AUC（用连续分数）。
    若该批 y 仅含单一类别，AUC 无定义，返回 nan。
    """
    y_l = y_to_label_np('Twitter', Y)
    pred_l = pred_to_label_np('Twitter', pred)
    acc = accuracy_score(y_l, pred_l)
    prob = pred.detach().cpu().numpy().reshape(-1)
    if len(np.unique(y_l)) < 2:
        return acc, float('nan')
    try:
        auc = float(roc_auc_score(y_l, prob, labels=[0, 1]))
    except ValueError:
        auc = float('nan')
    return acc, auc


def get_task_score(dataset, Y, pred):
    if dataset in ['Moons', 'YearBook']:
        pred_l = pred_to_label_np(dataset, pred)
        y_l = y_to_label_np(dataset, Y)
        return accuracy_score(y_l, pred_l)

    elif dataset in ['MNIST', 'Complaints8', 'Complaints8_M']:
        pred_l = pred_to_label_np(dataset, pred)
        y_l = y_to_label_np(dataset, Y)
        return accuracy_score(y_l, pred_l)

    elif dataset in ['Twitter']:
        y_l = y_to_label_np(dataset, Y)
        prob = pred.detach().cpu().numpy().reshape(-1)
        return roc_auc_score(y_l, prob, labels=[0, 1])

    elif dataset in ['Cyclone', 'House']:
        mae = F.l1_loss(Y, pred).item()
        return mae

    else:
        raise ValueError('Not defined dataset!')
