import torch
import torch.optim as optim
import time
import atexit
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from geoopt.optim import RiemannianAdam
from models.geometric_models import build_model, build_pvnn_frechet_sweep
from lib.data_loader import get_dataset

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train(model, train_loader, optimizer, criterion, epoch, print_interval=50):
    model.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if batch_idx % print_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')
    try:
        if TIME_TEST:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapse = time.perf_counter() - start_time
            warmup_epochs = globals().get('WARMUP_EPOCHS', 0)
            if epoch <= warmup_epochs:
                print(f'[TIME][WARMUP] Epoch {epoch} time: {elapse*1000:.2f}ms')
            else:
                print(f'[TIME] Epoch {epoch} time: {elapse*1000:.2f}ms')
                try:
                    EPOCH_TIMES.append(elapse)
                except NameError:
                    pass
    except NameError:
        pass

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set on {model.model_type.upper()}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def run_single_experiment(model_type, config, train_loader, test_loader, input_dim, n_classes, seed):
    set_seed(seed)          
    
    print(f"--- Training {model_type.upper()} with seed {seed} ---")
    
    model = build_model(
        model_type,
        dim=input_dim,
        hidden_dim=config['hidden_dim'],
        n_classes=n_classes,
        p_drop=config['dropout'],
        c=config['curvature'],
        inner_act=config.get('inner_act', 'none'),
        outer_act=config.get('outer_act', 'tangent'),
        linear_type=config.get('linear_type', 'pvfc')
    )
    if model_type.lower() == 'pvnn' and hasattr(model, 'no_proj_exp'):
        model.no_proj_exp = bool(config.get('no_proj_exp', False))
        if not model.no_proj_exp:
            if bool(config.get('use_mid_bn', False)):
                model.use_mid_bn = True
                model.use_mid_log_euc_bn = False
            elif bool(config.get('use_mid_log_euc_bn', False)):
                model.use_mid_bn = False
                model.use_mid_log_euc_bn = True

            gyrobn_cfg = config.get('gyrobn', None)
            if isinstance(gyrobn_cfg, dict) and hasattr(model, "bn_mid"):
                bn = getattr(model, "bn_mid", None)
                if bn is not None:
                    for k, v in gyrobn_cfg.items():
                        if hasattr(bn, k):
                            setattr(bn, k, v)
    if model_type != 'fc':
        optimizer = RiemannianAdam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, config['epochs'] + 1):
        train(model, train_loader, optimizer, criterion, epoch, print_interval=config['print_interval'])
        
        if epoch % 10 == 0 or epoch == config['epochs']:
            current_accuracy = evaluate(model, test_loader)
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch} - no improvement for {config['patience']} eval intervals")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    accuracy = test(model, test_loader, criterion)
    return accuracy
def run_model_instance(model, config, train_loader, test_loader, seed):
    set_seed(seed)
    if hasattr(model, 'model_type') and model.model_type != 'fc':
        optimizer = RiemannianAdam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, config['epochs'] + 1):
        train(model, train_loader, optimizer, criterion, epoch, print_interval=config['print_interval'])
        if epoch % 10 == 0 or epoch == config['epochs']:
            current_accuracy = evaluate(model, test_loader)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch} - no improvement for {config['patience']} eval intervals")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return test(model, test_loader, criterion)
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def main():
    global TIME_TEST, WARMUP_EPOCHS
    TIME_TEST = False
    WARMUP_EPOCHS = 0
    global EPOCH_TIMES
    EPOCH_TIMES = []
    global DATASET_AVG_TIMES
    DATASET_AVG_TIMES = {}
    global ITER_AVG_TIMES
    ITER_AVG_TIMES = {}
    def _print_avg_epoch_time():
        try:
            if TIME_TEST and len(EPOCH_TIMES) > 0:
                avg_epoch_time = sum(EPOCH_TIMES) / len(EPOCH_TIMES)
                print(f'[TIME] Avg epoch time over {len(EPOCH_TIMES)} epochs: {avg_epoch_time*1000:.2f}ms')
        except Exception:
            pass
    PRINT_GLOBAL_TIME_SUMMARY = False
    if PRINT_GLOBAL_TIME_SUMMARY:
        atexit.register(_print_avg_epoch_time)
    base_config = {
        'lr': 0.015,
        'epochs': 2000,
        'batch_size': 128,
        'dataset': 'disease',
        'dropout': 0.0,
        'weight_decay': 0.000,
        'hidden_dim': 16,
        'curvature': 0.5,
        'num_runs': 5,
        'print_interval': 100,
        'use_bias': True,
        'patience': 200,
        'inner_act': 'none',
        'outer_act': 'tangent'
    }

    pvnn_template = {
        'lr': 0.005,
        'epochs': 2000,
        'batch_size': 128,
        'dataset': 'cora',
        'dropout': 0.2,
        'weight_decay': 0.001,
        'hidden_dim': 16,
        'curvature': 0.5,
        'num_runs': 5,
        'print_interval': 100,
        'use_bias': True,  
        'patience': 200,
        'inner_act': 'none',                                              # options: none/relu/tanh/softplus (PVFC v_k activation)
        'outer_act': 'direct_tanh',                           # options: none/tangent/direct/direct_tanh
        'linear_type': 'pvfc',                            # options: pvfc/pv/pv_lfc/euc_tangent_fc
        'no_proj_exp': False,
        'use_mid_bn': False,
        'gyrobn': {'use_euclid_stats': False,'use_gyro_midpoint': True,'clamp_factor': 3.0, 'var_floor': 1e-1, 'max_tan_norm': 20.0, 'scalar_sinh_clip': 20.0, 'use_post_gain': False},
    }
    if TIME_TEST:
        pvnn_template['epochs'] = 1000
        pvnn_template['num_runs'] = 1

    dataset_overrides = {
        'pubmed':  {'lr': 0.05,  'dropout': 0.6, 'weight_decay': 0.0005, 'curvature': 1.0},
        'disease': {'lr': 0.01, 'dropout': 0.4, 'weight_decay': 0.0005, 'curvature': 0.3},
        #'cora':    {'lr': 0.05, 'dropout': 0.6, 'weight_decay': 0.0005, 'curvature': 1.0}, 
        'cora':    {'lr': 0.01, 'dropout': 0.6, 'weight_decay': 0.0005, 'curvature': 1.5}, #用于directact消融实验
        'airport': {'lr': 0.01, 'dropout': 0.4, 'weight_decay': 0.0005, 'curvature': 0.3},
    }

    RUN_FRECHET_SWEEP = False                             
    RUN_ALL_DATASETS = False
    SINGLE_DATASET = 'cora'                                



    datasets_to_run = list(dataset_overrides.keys()) if RUN_ALL_DATASETS else [SINGLE_DATASET]

    model_types = ['pvnn']

    PRINT_IMMEDIATE_SUMMARY = False                                       

    all_results = {}

    for dataset_name in datasets_to_run:
        print(f"\n========== Dataset: {dataset_name.upper()} ==========")

        pvnn_config = dict(pvnn_template)
        pvnn_config.update(dataset_overrides[dataset_name])
        pvnn_config['dataset'] = dataset_name

        print("\nPVNN config:")
        print(f"lr={pvnn_config['lr']}, dropout={pvnn_config['dropout']}, weight_decay={pvnn_config['weight_decay']}")
        print(f"hidden_dim={pvnn_config['hidden_dim']}, curvature={pvnn_config['curvature']}")
        try:
            dataset_prev_len = len(EPOCH_TIMES) if TIME_TEST else None
        except NameError:
            dataset_prev_len = None

        X_train, y_train, X_test, y_test = get_dataset(dataset_name)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=pvnn_config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=pvnn_config['batch_size'], shuffle=False)

        input_dim = X_train.shape[1]
        n_classes = len(torch.unique(y_train))

        if RUN_FRECHET_SWEEP:
            sweep_iters = [1]
            results = {str(it): [] for it in sweep_iters}
        else:
            results = {model_type: [] for model_type in model_types}
        base_seed = 40
        for run in range(pvnn_config['num_runs']):
            seed = base_seed + run

            

            print(f"\n=========================================================")
            print(f"--- Run {run+1}/{pvnn_config['num_runs']} (seed: {seed}) ---")
            print(f"=========================================================")

            if RUN_FRECHET_SWEEP:
                models = build_pvnn_frechet_sweep(
                    dim=input_dim, hidden_dim=pvnn_config['hidden_dim'], n_classes=n_classes,
                    p_drop=pvnn_config['dropout'], c=pvnn_config['curvature'],
                    iters_list=[1], inner_act=pvnn_config['inner_act'], outer_act=pvnn_config['outer_act'],
                    bn_mode='gyro'
                )
                for it_key, model in models.items():
                    model.use_mid_bn = True
                    model.use_mid_log_euc_bn = False
                    if hasattr(model, "bn_mid"):
                        model.bn_mid.use_euclid_stats = bool(pvnn_config.get('bn_use_euclid_stats', False))
                        model.bn_mid.use_gyro_midpoint = bool(pvnn_config.get('bn_use_gyro_midpoint', False))
                        max_iter = -1 if (isinstance(it_key, str) and str(it_key).lower() in ("inf", "-1")) else int(it_key)
                        model.bn_mid.max_iter = max_iter
                        gyrobn_cfg = pvnn_config.get('gyrobn', None)
                        if isinstance(gyrobn_cfg, dict):
                            for k, v in gyrobn_cfg.items():
                                if hasattr(model.bn_mid, k):
                                    setattr(model.bn_mid, k, v)
                for it_key, model in models.items():
                    label = 'inf' if it_key == -1 else str(it_key)
                    print(f"\n--- Training PVNN (Frechet iters={label}) ---")
                    print(f"config: lr={pvnn_config['lr']}, dropout={pvnn_config['dropout']}, weight_decay={pvnn_config['weight_decay']}, curvature={pvnn_config['curvature']}")
                    try:
                        iter_prev_len = len(EPOCH_TIMES) if TIME_TEST else None
                    except NameError:
                        iter_prev_len = None

                    acc = run_model_instance(model, pvnn_config, train_loader, test_loader, seed)
                    results[label].append(acc)

                    try:
                        if TIME_TEST and iter_prev_len is not None:
                            seg_times = EPOCH_TIMES[iter_prev_len:]
                            if len(seg_times) > 0:
                                avg_t = sum(seg_times) / len(seg_times)
                                ITER_AVG_TIMES.setdefault(dataset_name.upper(), {}).setdefault(label, []).append((avg_t, len(seg_times)))
                    except NameError:
                        pass
            else:
                for model_type in model_types:
                    print(f"\n--- Training and evaluating {model_type.upper()} ---")
                    config = pvnn_config if model_type == 'pvnn' else base_config
                    print(f"config: lr={config['lr']}, dropout={config['dropout']}, weight_decay={config['weight_decay']}, curvature={config['curvature']}")
                    accuracy = run_single_experiment(
                        model_type, config, train_loader, test_loader,
                        input_dim, n_classes, seed
                    )
                    results[model_type].append(accuracy)

        all_results[dataset_name] = {
            'accuracies': results,
            'config': pvnn_config,
            'frechet_sweep': RUN_FRECHET_SWEEP
        }

        if PRINT_IMMEDIATE_SUMMARY:
            print("\n\n=========================================================")
            print(f"--- {dataset_name.upper()} | {pvnn_config['num_runs']} runs summary ---")
            print("=========================================================")
            if RUN_FRECHET_SWEEP:
                for label, accs in results.items():
                    mean_acc = np.mean(accs) if len(accs) else float('nan')
                    std_acc = np.std(accs) if len(accs) else float('nan')
                    print(f"Frechet iters={label:>3} | mean: {mean_acc:.2f}% ± {std_acc:.2f}%  | runs: {', '.join([f'{a:.2f}%' for a in accs])}")
            else:
                accuracies = results['pvnn']
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                var_acc = np.var(accuracies)
                print(f"Model: PVNN       | mean acc: {mean_acc:.2f}% ± {std_acc:.2f}%")
                print(f"std: {std_acc:.4f} | var: {var_acc:.4f}")
                print(f"runs: {', '.join([f'{acc:.2f}%' for acc in accuracies])}")
            print(f"config: lr={pvnn_config['lr']}, dropout={pvnn_config['dropout']}, weight_decay={pvnn_config['weight_decay']}, curvature={pvnn_config['curvature']}")
        try:
            if TIME_TEST and dataset_prev_len is not None:
                seg_times = EPOCH_TIMES[dataset_prev_len:]
                if len(seg_times) > 0:
                    avg_t = sum(seg_times) / len(seg_times)
                    DATASET_AVG_TIMES[dataset_name.upper()] = (avg_t, len(seg_times))
        except NameError:
            pass

    print("\n===== Final summary (mean ± std) =====")
    for ds in datasets_to_run:
        rec = all_results[ds]
        if rec.get('frechet_sweep', False):
            print(f"\n{ds.upper()} (Frechet sweep):")
            for label, accs in rec['accuracies'].items():
                mean_acc = np.mean(accs) if len(accs) else float('nan')
                std_acc = np.std(accs) if len(accs) else float('nan')
                print(f"  iters={label:>3} | {mean_acc:.2f}% ± {std_acc:.2f}%")
            if TIME_TEST and isinstance(globals().get('ITER_AVG_TIMES', None), dict):
                ds_key = ds.upper()
                if ds_key in ITER_AVG_TIMES:
                    for label, entries in ITER_AVG_TIMES[ds_key].items():
                        total_epochs = sum(n for _, n in entries)
                        total_time = sum(avg * n for avg, n in entries)
                        if total_epochs > 0:
                            print(f"  [TIME] 迭代={label:>3} avg epoch time over {total_epochs} epochs: {(total_time/total_epochs)*1000:.2f}ms")
        else:
            accs = rec['accuracies']['pvnn'] if isinstance(rec['accuracies'], dict) else rec['accuracies']
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"{ds.upper()}: {mean_acc:.2f}% ± {std_acc:.2f}%")
        if TIME_TEST and isinstance(globals().get('DATASET_AVG_TIMES', None), dict):
            if ds.upper() in DATASET_AVG_TIMES:
                avg_t, n_epochs = DATASET_AVG_TIMES[ds.upper()]
                print(f"[TIME] {ds.upper()} avg epoch time over {n_epochs} epochs: {avg_t*1000:.2f}ms")

if __name__ == '__main__':
    main()
