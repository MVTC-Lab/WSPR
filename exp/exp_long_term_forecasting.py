from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, true):
        loss_mse = self.mse(pred, true)
        loss_mae = self.mae(pred, true)
        return self.alpha * loss_mse + (1 - self.alpha) * loss_mae


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self._printed_future_covariate_safety = False

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model ID: {self.args.model_id} | Total Params: {total_params}")
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_name = self.args.loss.lower()
        if loss_name == 'combo':
            alpha_val = getattr(self.args, 'alpha', 0.5)
            print(f"[Loss Config] Using ComboLoss (alpha={alpha_val})")
            return ComboLoss(alpha=alpha_val)
        elif loss_name == 'mae':
            return nn.L1Loss()
        elif loss_name == 'huber':
            return nn.HuberLoss(delta=self.args.delta)
        else:
            return nn.MSELoss()

    def _build_decoder_input(self, batch_y):
        future = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        use_future_covariates = bool(getattr(self.args, 'use_future_covariates', 0))
        if use_future_covariates and self.args.features == 'MS' and future.shape[-1] > 1:
            available_dim = future.shape[-1] - 1
            configured_dim = int(getattr(self.args, 'known_future_dim', -1))
            reliability_idx = int(getattr(self.args, 'reliability_target_idx', -1))
            if configured_dim < 0:
                if reliability_idx >= 0:
                    raise ValueError(
                        "known_future_dim must be set when reliability_target_idx is enabled; "
                        "otherwise the future reliability label could be copied into x_dec."
                    )
                copy_dim = available_dim
            else:
                copy_dim = min(configured_dim, available_dim)
            if reliability_idx >= 0 and copy_dim > reliability_idx:
                raise ValueError(
                    f"Unsafe future covariate setup: known_future_dim={configured_dim} would copy "
                    f"reliability_target_idx={reliability_idx}. Use known_future_dim <= reliability_target_idx."
                )
            if copy_dim > 0:
                future[:, :, :copy_dim] = batch_y[:, -self.args.pred_len:, :copy_dim]
            if 0 <= reliability_idx < future.shape[-1]:
                future[:, :, reliability_idx:reliability_idx + 1] = 0.0
            future[:, :, -1:] = 0.0
            if not self._printed_future_covariate_safety:
                kept = f"0..{copy_dim - 1}" if copy_dim > 0 else "none"
                print("NWP-assisted baseline safety check:")
                print(f"enc_in = {getattr(self.args, 'enc_in', 'NA')}")
                print(f"dec_in = {getattr(self.args, 'dec_in', 'NA')}")
                print(f"c_out = {getattr(self.args, 'c_out', 'NA')}")
                print(f"known_future_dim = {configured_dim}")
                print(f"future kept channels = {kept}")
                if reliability_idx >= 0:
                    print(f"future reliability channel {reliability_idx} = zeroed")
                print(f"future OT channel {future.shape[-1] - 1} = zeroed")
                self._printed_future_covariate_safety = True
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], future], dim=1)
        self._set_reliability_target(batch_y, enabled=self.model.training)
        return dec_inp.float().to(self.device)

    def _model_core(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _set_reliability_target(self, batch_y, enabled):
        core = self._model_core()
        setter = getattr(core, 'set_reliability_target', None)
        if setter is None:
            return
        target_idx = int(getattr(self.args, 'reliability_target_idx', -1))
        use_future = bool(getattr(self.args, 'use_future_covariates', 0))
        if enabled and use_future and 0 <= target_idx < batch_y.shape[-1] - 1:
            target = batch_y[:, -self.args.pred_len:, target_idx:target_idx + 1].to(self.device)
            setter(target)
        else:
            setter(None)

    def _auxiliary_loss(self):
        return getattr(self._model_core(), 'auxiliary_loss', None)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._build_decoder_input(batch_y)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._build_decoder_input(batch_y)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # Forecast loss plus optional model-specific auxiliary objectives.
                        loss = criterion(outputs, batch_y_target)
                        auxiliary_loss = self._auxiliary_loss()
                        if auxiliary_loss is not None:
                            loss = loss + auxiliary_loss
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # Forecast loss plus optional model-specific auxiliary objectives.
                    loss = criterion(outputs, batch_y_target)
                    auxiliary_loss = self._auxiliary_loss()
                    if auxiliary_loss is not None:
                        loss = loss + auxiliary_loss
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"Loaded best validation checkpoint: {best_model_path}")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        # In test-only mode, load the saved checkpoint first.
        if test:
            print('loading model')
            checkpoint_dir = os.path.join(self.args.checkpoints, setting)
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                print(f"Loaded checkpoint: {checkpoint_path}")
            else:
                # Release safeguard: never silently evaluate a similar experiment.
                raise FileNotFoundError(f"Exact checkpoint required: {checkpoint_path}")

        preds = []
        trues = []

        pic_folder_path = os.path.join('outputs', 'results_pic', setting)
        if not os.path.exists(pic_folder_path):
            os.makedirs(pic_folder_path)

        npy_folder_path = os.path.join('outputs', 'results_npy', setting)
        if not os.path.exists(npy_folder_path):
            os.makedirs(npy_folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._build_decoder_input(batch_y)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                # Keep raw test outputs for metric calculation.
                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        import matplotlib.pyplot as plt
        plot_preds = preds.copy()
        plot_trues = trues.copy()

        if test_data.scale:
            scaler = test_data.scaler
            mean = scaler.mean_[-1]
            std = scaler.scale_[-1]
            plot_preds = plot_preds * std + mean
            plot_trues = plot_trues * std + mean

        plot_preds = np.maximum(plot_preds, 0)
        plot_trues = np.maximum(plot_trues, 0)

        flat_trues = plot_trues[:, :, -1].reshape(-1)
        flat_preds = plot_preds[:, :, -1].reshape(-1)

        unit_map = {
            'PVOD-PV.csv': 'Power (MW)',
            'AS-PV.csv': 'Power (kW)',
            'CC-PV.csv': 'Power (kW)',
            'Comp-PV.csv': 'Normalized Power (pu)'
        }
        current_data_name = os.path.basename(self.args.data_path) if self.args.data_path else 'Unknown'
        y_label_unit = unit_map.get(current_data_name, 'Power Value')

        freq_arg = str(self.args.freq)
        x_label_unit = 'Time (min)' if ('t' in freq_arg or 'min' in freq_arg) else 'Time (h)'

        plot_len = min(len(flat_trues), 2880)

        plt.figure(figsize=(24, 6))
        plt.plot(flat_trues[:plot_len], label='Actual', color='steelblue', linewidth=1.5)
        plt.plot(flat_preds[:plot_len], label='Prediction', color='tomato', linestyle='--', linewidth=1.5)

        plt.ylabel(y_label_unit, fontsize=16, fontweight='bold')
        plt.xlabel(x_label_unit, fontsize=16, fontweight='bold')
        plt.title(f'Actual vs Prediction on {current_data_name}', fontsize=18)
        plt.legend(loc='upper right', fontsize=14, frameon=True)
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(pic_folder_path, 'forecast_curve_real_scale.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved forecast curve: {save_path}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        r2 = r2_score(trues.reshape(-1), preds.reshape(-1))

        print('mse:{}, mae:{}, r2:{}, dtw:{}'.format(mse, mae, r2, dtw))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, r2:{}, dtw:{}'.format(mse, mae, r2, dtw))
        f.write('\n\n')
        f.close()

        np.save(npy_folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(npy_folder_path + 'pred.npy', preds)
        np.save(npy_folder_path + 'true.npy', trues)
        print(f"Saved prediction arrays: {npy_folder_path}")

        return mse, mae, r2




