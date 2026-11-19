# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 19:35
@Author  : Zekun Cai
@File    : main.py
@Software: PyCharm
'''
import sys
import shutil
import argparse
import random
from datetime import datetime

from koodos import *
from util import *
from param import *

import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
torch.cuda.empty_cache()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trainModel(X_train, Y_train, time_train):
    # To avoid time consumption and cumulative errors in integrating long sequences,
    # we divide the domain sequence into segments by sliding window with no longer than seg_len and integrate each segment separately.
    train_domain_seg = []
    class_weight = None
    if nll_class_weight and data_set in ['MNIST', 'Complaints8', 'Complaints8_M']:
        class_weight = build_nll_class_weight(
            Y_train, nll_num_classes, device, power=nll_class_weight_power
        )
        print('[nll_class_weight]', class_weight.detach().cpu().numpy())
    if nll_tail_class_ids is not None and nll_tail_boost_factor > 1.0:
        print('[nll_tail_boost] classes', list(nll_tail_class_ids), 'factor', nll_tail_boost_factor)
    for idx in range(n_train - 1):
        l = min(n_train - idx, seg_len)
        seg_time, seg_X, seg_Y = time_train[idx:idx + l], X_train[idx:idx + l], Y_train[idx:idx + l]
        train_domain_seg.append([seg_time, seg_X, seg_Y])

    # The last few segments are used for validation
    n_seg, n_train_seg = len(train_domain_seg), len(train_domain_seg) - n_val_seg
    idx_train_seg = np.arange(n_train_seg, dtype=np.int64)
    idx_val_seg = np.arange(n_train_seg, n_seg, dtype=np.int64)

    # Initialization model-related
    model = Koodos(data_set, time_train).to(device)
    opt_params = [
        {'params': model.shared_model.parameters(), 'lr': pred_learning_rate, 'weight_decay': weight_decay},
        {'params': model.pred_model.parameters(), 'lr': pred_learning_rate, 'weight_decay': weight_decay},
        {'params': model.encoder.parameters(), 'lr': coder_learning_rate, 'weight_decay': weight_decay},
        {'params': model.decoder.parameters(), 'lr': coder_learning_rate, 'weight_decay': weight_decay},
        {'params': model.dynamic.parameters(), 'lr': dyn_learning_rate, 'weight_decay': weight_decay},
    ]
    if getattr(model, 'cia_module', None) is not None:
        opt_params.append({'params': model.cia_module.parameters(), 'lr': pred_learning_rate, 'weight_decay': weight_decay})
    optimizer = torch.optim.Adam(opt_params)

    # Training
    min_val_loss = np.inf
    best_ckpt_epoch = -1
    best_ckpt_val_loss = float('nan')
    model.train()
    for e in range(epoch):
        epoch_loss = 0.0
        # component meters (train)
        m_intri = 0.0
        m_integ = 0.0
        m_recon = 0.0
        m_dyna = 0.0
        m_consis = 0.0
        m_cia = 0.0
        n_seg_seen = 0

        n_splits = max(1, n_train_seg // batch)
        for batch_idx in np.array_split(idx_train_seg, n_splits):
            loss = 0.0
            for idx in batch_idx:
                seg_time, seg_X, seg_Y = train_domain_seg[idx]
                init_pred, gene_pred, init_param, init_embed, init_debed, gene_param, gene_embed, cia_aux = model(seg_X, seg_time, idx)

                # Calculate the loss
                loss_intri, loss_integ = get_task_loss(
                    data_set, seg_Y, init_pred, gene_pred,
                    class_weight=class_weight,
                    focal_gamma=nll_focal_gamma,
                    tail_class_ids=nll_tail_class_ids,
                    tail_boost=nll_tail_boost_factor,
                )
                loss_recon = F.mse_loss(init_param, init_debed)
                # loss_dyna = F.mse_loss(init_embed, gene_embed)

                weights = torch.linspace(1.0, 1.2, steps=gene_embed.size(0), device=gene_embed.device)
                loss_dyna_vec = ((gene_embed - init_embed) ** 2).mean(dim=1)
                loss_dyna = (weights * loss_dyna_vec).mean()
                loss_param_smooth = F.mse_loss(gene_param[1:], gene_param[:-1])
                lambda_smooth = 1e-4

                loss_consis = F.mse_loss(init_param, gene_param)
                # base = alpha * loss_intri + alpha * loss_integ + beta * loss_recon + gamma * loss_dyna + beta * loss_consis
                base = alpha * loss_intri + alpha * loss_integ + beta * loss_recon + gamma * loss_dyna + beta * loss_consis + lambda_smooth * loss_param_smooth
                loss = loss + base

                # meters
                m_intri += float(loss_intri.item())
                m_integ += float(loss_integ.item())
                m_recon += float(loss_recon.item())
                m_dyna += float(loss_dyna.item())
                m_consis += float(loss_consis.item())
                n_seg_seen += 1
                if cia_lambda > 0 and cia_aux is not None:
                    loss_cia = get_cia_aux_loss(
                        data_set, seg_Y, cia_aux,
                        mix_weight=cia_mix_weight,
                        spu_entropy_weight=cia_spu_entropy_weight,
                        class_weight=class_weight,
                        focal_gamma=nll_focal_gamma,
                        tail_class_ids=nll_tail_class_ids,
                        tail_boost=nll_tail_boost_factor,
                        skip_causal_aux=cia_skip_causal_aux,
                    )
                    loss = loss + cia_lambda * loss_cia
                    m_cia += float(loss_cia.item())

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.item())

        # Validate generalization on the last few domain sequences
        model.eval()
        val_loss = 0.0
        val_plain_integ = 0.0
        val_cia = 0.0
        with torch.no_grad():
            for idx in idx_val_seg:
                seg_time, seg_X, seg_Y = train_domain_seg[idx]
                init_pred, gene_pred, _, _, _, _, _, cia_aux = model(seg_X, seg_time, idx)
                _, loss_integ = get_task_loss(
                    data_set, seg_Y, init_pred, gene_pred,
                    class_weight=class_weight,
                    focal_gamma=nll_focal_gamma,
                    tail_class_ids=nll_tail_class_ids,
                    tail_boost=nll_tail_boost_factor,
                )
                val_loss = val_loss + float(loss_integ.item())
                if ckpt_val_plain_nll and data_set in ['MNIST', 'Complaints8', 'Complaints8_M']:
                    _, li_plain = get_task_loss(
                        data_set, seg_Y, init_pred, gene_pred,
                        class_weight=None,
                        focal_gamma=0.0,
                        tail_class_ids=None,
                        tail_boost=1.0,
                    )
                    val_plain_integ += float(li_plain.item())
                if cia_lambda > 0 and cia_aux is not None:
                    val_cia = val_cia + float(get_cia_aux_loss(
                        data_set, seg_Y, cia_aux,
                        mix_weight=cia_mix_weight,
                        spu_entropy_weight=cia_spu_entropy_weight,
                        class_weight=class_weight,
                        focal_gamma=nll_focal_gamma,
                        tail_class_ids=nll_tail_class_ids,
                        tail_boost=nll_tail_boost_factor,
                        skip_causal_aux=cia_skip_causal_aux,
                    ).item())
        model.train()

        val_for_ckpt = val_plain_integ if (
            ckpt_val_plain_nll and data_set in ['MNIST', 'Complaints8', 'Complaints8_M']
        ) else val_loss
        if val_for_ckpt < min_val_loss:
            # For saving IO time
            if e > epoch * 0.8:
                min_val_loss = val_for_ckpt
                best_ckpt_epoch = e
                best_ckpt_val_loss = val_loss
                torch.save(model.state_dict(), save_path + '/model.pt')
                print(
                    f'[checkpoint] saved model.pt @ epoch {e}: '
                    f'val_for_ckpt={val_for_ckpt:.6f}, Val Loss (log)={val_loss:.6f}'
                )
                with open(save_path + '/log.txt', 'a') as f:
                    f.write(
                        f'# ckpt epoch={e} val_for_ckpt={val_for_ckpt:.6f} val_loss={val_loss:.6f}\n'
                    )

        # Log
        log_every = data_setting.get('log_every', 1)
        if (e % log_every) == 0 or e == (epoch - 1):
            denom = max(1, n_seg_seen)
            avg_intri = m_intri / denom
            avg_integ = m_integ / denom
            avg_recon = m_recon / denom
            avg_dyna = m_dyna / denom
            avg_consis = m_consis / denom
            avg_cia = m_cia / denom if denom > 0 else 0.0
            extra_val = (
                f", val_plain_integ {val_plain_integ:.5f}" if (
                    ckpt_val_plain_nll and data_set in ['MNIST', 'Complaints8', 'Complaints8_M']
                ) else ""
            )
            msg = (
                f"Epoch: {e}, "
                f"Train Loss: {epoch_loss:.5f}, Val Loss: {val_loss:.5f}{extra_val} | "
                f"intri {avg_intri:.4f}, integ {avg_integ:.4f}, "
                f"recon {avg_recon:.4f}, dyna {avg_dyna:.4f}, consis {avg_consis:.4f}, "
                f"cia_aux {avg_cia:.4f}, val_cia_aux {val_cia:.4f}"
            )
            print(msg)
        else:
            print('Epoch: {}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(e, epoch_loss, val_loss))
        with open(save_path + '/log.txt', 'a') as f:
            f.write("{},{},{}\n".format(e, epoch_loss, val_loss))

    _rs = globals().get('run_seed')
    summary_lines = [
        f'epoch\t{best_ckpt_epoch}',
        f'val_for_ckpt\t{min_val_loss}',
        f'val_loss_at_ckpt\t{best_ckpt_val_loss}',
    ]
    if _rs is not None:
        summary_lines.append(f'seed\t{_rs}')
    with open(save_path + '/checkpoint_best.txt', 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(
        f'[checkpoint] loading model.pt from best: epoch {best_ckpt_epoch}, '
        f'val_for_ckpt={min_val_loss:.6f}, Val Loss (at save)={best_ckpt_val_loss:.6f}'
    )

    model.load_state_dict(torch.load(save_path + '/model.pt'.format(data_set),weights_only=True))
    return model


def testModel(model, X_train, Y_train, X_text, Y_test, time_test):
    model.eval()

    train_preds = []
    train_labels = []
    with torch.no_grad():
        for i in range(n_train):
            x = model.shared_model(X_train[i])
            pred = model.pred_model[i](x)
            train_preds.append(pred)
            train_labels.append(Y_train[i])
    print('---- Train Domains ----')
    for i in range(n_train):
        if data_set == 'Twitter':
            acc, auc = twitter_accuracy_auc(train_labels[i], train_preds[i])
            print('Step {}, Train Acc {:.4f}, AUC {:.4f}'.format(i + 1, acc, auc))
        else:
            score = get_task_score(data_set, train_labels[i], train_preds[i])
            print('Step {}, Train Metric {:.4}'.format(i + 1, score))
    # 回归任务（House/Cyclone）：指标为 MAE，不做 classification_report
    if data_set in ['Cyclone', 'House']:
        with torch.no_grad():
            train_mae = F.l1_loss(torch.cat(train_preds), torch.cat(train_labels)).item()
        print('Train overall MAE {:.4f}'.format(train_mae))
    else:
        # ===== Confusion Matrix: Train domains (optional) =====
        train_true = y_to_label_np(data_set, torch.cat(train_labels))
        train_pred = pred_to_label_np(data_set, torch.cat(train_preds))

        # cm_train = confusion_matrix(train_true, train_pred)
        # plt.figure(figsize=(8, 8))
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm_train)
        # disp.plot(values_format="d")
        # plt.title(f"Train Confusion Matrix - {data_set} (n_train={n_train})")
        # plt.tight_layout()
        # plt.savefig(save_path + '/cm_train.png', dpi=200)
        # plt.close()

        # print("Saved train CM to:", save_path + "/cm_train.png")
        print("Train classification report:\n", classification_report(train_true, train_pred, digits=4))
        if data_set == 'Twitter':
            tr_acc, tr_auc = twitter_accuracy_auc(torch.cat(train_labels), torch.cat(train_preds))
            print('Train overall Acc {:.4f}, AUC {:.4f}'.format(tr_acc, tr_auc))
    with torch.no_grad():
        last_param = torch.cat([p.flatten() for p in model.pred_model[-1].parameters()])
        last_time = model.time_point[-1:]
        final_time_test = torch.cat([last_time, time_test])

        last_embed = model.encoder(last_param)
        test_embed = odeint(model.dynamic, last_embed, final_time_test, method=model.method, options={'step_size': model.step})[1:]
        test_param = model.decoder(test_embed)
        X_text = [model.shared_model(x) for x in X_text]
        test_pred = model.generalized_model_pred(X_text, test_param)
    if data_set in ['Cyclone', 'House']:
        with torch.no_grad():
            test_mae = F.l1_loss(torch.cat(test_pred), torch.cat(Y_test)).item()
        print('Test overall MAE {:.4f}'.format(test_mae))
    else:
        # ===== Confusion Matrix: Test (extrapolation) overall =====
        y_true_all = y_to_label_np(data_set, torch.cat(Y_test))
        y_pred_all = pred_to_label_np(data_set, torch.cat(test_pred))

        # cm_test = confusion_matrix(y_true_all, y_pred_all)
        # plt.figure(figsize=(8, 8))
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
        # disp.plot(values_format="d")
        # plt.title(f"Test Confusion Matrix - {data_set}")
        # plt.tight_layout()
        # plt.savefig(save_path + '/cm_test.png', dpi=200)
        # plt.close()

        # print("Saved test CM to:", save_path + "/cm_test.png")
        print("Test classification report:\n", classification_report(y_true_all, y_pred_all, digits=4))
        if data_set == 'Twitter':
            te_acc, te_auc = twitter_accuracy_auc(torch.cat(Y_test), torch.cat(test_pred))
            print('Test overall Acc {:.4f}, AUC {:.4f}'.format(te_acc, te_auc))

    # Test the performance
    f = open(save_path + '/scores.txt', 'a')
    if data_set == 'Twitter':
        acc_all, auc_all = twitter_accuracy_auc(torch.cat(Y_test), torch.cat(test_pred))
        print('\nAll Acc {:.4f}, All AUC {:.4f}\n'.format(acc_all, auc_all))
        f.write('All Acc {}\n'.format(acc_all))
        f.write('All AUC {}\n'.format(auc_all))
        for i in range(len(Y_test)):
            acc_s, auc_s = twitter_accuracy_auc(Y_test[i], test_pred[i])
            print('Step {}, Acc {:.4f}, AUC {:.4f}'.format(i + 1, acc_s, auc_s))
            f.write('Step {}, Acc {}\n'.format(i + 1, acc_s))
            f.write('Step {}, AUC {}\n'.format(i + 1, auc_s))
    else:
        score = get_task_score(data_set, torch.cat(Y_test), torch.cat(test_pred))
        print('\nAll Metric {:.4}\n'.format(score))
        f.write('All Metric {}\n'.format(score))

        for i in range(len(Y_test)):
            Y_test_step, test_pred_step = Y_test[i], test_pred[i]
            score = get_task_score(data_set, Y_test_step, test_pred_step)
            print('Step {}, Metric {:.4}'.format(i + 1, score))
            f.write('Step {}, Metric {}\n'.format(i + 1, score))
    f.close()




def main():
    X, Y, time_points = dataset_preparation(data_set, data_path, device)
    X_train, Y_train, time_train, = X[:n_train], Y[:n_train], time_points[:n_train]
    X_text, Y_test, time_test = X[n_train:], Y[n_train:], time_points[n_train:]

    model = trainModel(X_train, Y_train, time_train)
    testModel(model,X_train, Y_train, X_text, Y_test, time_test)


# Parameters Loading
##################################################################################################
parser = argparse.ArgumentParser(description='Set dataset and CUDA device.')
parser.add_argument('--dataset', type=str, default='Moons', help='Name of the dataset.')
parser.add_argument('--cuda', type=int, default=0, help='CUDA device number.')
parser.add_argument('--seed', type=int, default=None, help='Random seed (default: param seed or 42).')
args = parser.parse_args()

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
data_set = args.dataset

data_setting = SETTING[data_set]
data_path = data_setting['data_file']

n_train = data_setting['n_train']
seg_len = data_setting['seg_len']
n_val_seg = data_setting['n_val_seg']

epoch = data_setting['epoch']
batch = data_setting['batch']
pred_learning_rate = data_setting['pred_learning_rate']
coder_learning_rate = data_setting['coder_learning_rate']
dyn_learning_rate = data_setting['dyn_learning_rate']
weight_decay = data_setting['weight_decay']

alpha = data_setting['alpha']
beta = data_setting['beta']
gamma = data_setting['gamma']

cia_lambda = data_setting.get('cia_lambda', 0.0)
cia_mix_weight = data_setting.get('cia_mix_weight', 0.5)
cia_spu_entropy_weight = data_setting.get('cia_spu_entropy_weight', 0.05)
cia_skip_causal_aux = bool(data_setting.get(
    'cia_skip_causal_aux',
    data_setting.get('cia_main_use_invariant', False),
))

nll_class_weight = data_setting.get('nll_class_weight', False)
nll_class_weight_power = data_setting.get('nll_class_weight_power', 0.25)
_nll_default_classes = {'MNIST': 10, 'Complaints8': 8, 'Complaints8_M': 8}
nll_num_classes = data_setting.get('nll_num_classes', _nll_default_classes.get(data_set, 10))
grad_clip_norm = float(data_setting.get('grad_clip_norm', 0.0) or 0.0)
nll_focal_gamma = float(data_setting.get('nll_focal_gamma', 0.0) or 0.0)
ckpt_val_plain_nll = bool(data_setting.get('ckpt_val_plain_nll', False))

_nll_tail = data_setting.get('nll_tail_boost_classes')
if _nll_tail is None:
    _nll_tail = []
nll_tail_boost_factor = float(data_setting.get('nll_tail_boost_factor', 1.0) or 1.0)
nll_tail_class_ids = (
    list(_nll_tail) if (_nll_tail and nll_tail_boost_factor > 1.0) else None
)

keyword = 'pred_' + data_set + '_' + datetime.now().strftime("%y%m%d%H%M%S")
save_path = './save/' + keyword
run_seed = args.seed if args.seed is not None else data_setting.get('seed', 42)
##################################################################################################

if __name__ == '__main__':
    set_seed(run_seed)
    print(f'[seed] {run_seed}')
    currentPython = sys.argv[0]
    shutil.copytree('model', save_path + '/model', dirs_exist_ok=True)
    shutil.copy2('koodos.py', save_path)
    shutil.copy2('param.py', save_path)
    shutil.copy2('util.py', save_path)
    shutil.copy2(currentPython, save_path)

    main()
