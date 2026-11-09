import argparse
import os
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSPR photovoltaic forecasting runner')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='WSPR',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # 随机种子参数
    parser.add_argument('--fix_seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../datasets/PV/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock and DFT_series_decomp')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')

    # 🟢 [关键修改] 注册新参数：分解方法、KAN参数、DWT参数
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, support: moving_avg, dft_decomp, dwt_decomp')
    parser.add_argument('--grid_size', type=int, default=3, help='KAN grid size')
    parser.add_argument('--w_kan', type=float, default=0.05, help='KAN initial weight')
    # 👇 这里是你报错缺失的参数 👇
    parser.add_argument('--wavelet_name', type=str, default='db1', help='wavelet name for DWT (db1, sym4)')
    parser.add_argument('--dwt_level', type=int, default=2, help='decomposition level for DWT')
    # Release entry point: run.py.
    parser.add_argument('--kan_weight', type=float, default=0.5, help='Weight for KAN in hybrid layer')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')
    # === Optuna Tuning Params (新增) ===
    parser.add_argument('--kan_degree', type=int, default=4, help='Degree for ChebyKAN')
    parser.add_argument('--wavelet_num', type=int, default=8, help='Wavelet number for WavKAN')
    parser.add_argument('--kan_grid_size', type=int, default=5, help='Grid size for LCB Mixer')
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for Combo Loss: alpha * MSE + (1-alpha) * MAE')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_future_covariates', type=int, default=0, help='pass non-target future covariates to compatible models')
    parser.add_argument('--known_future_dim', type=int, default=-1,
                        help='number of leading non-target channels known at inference; -1 keeps legacy behavior')
    parser.add_argument('--reliability_target_idx', type=int, default=-1,
                        help='channel index of the train-only NWP reliability label; -1 disables supervision')
    parser.add_argument('--unit_interval_cols', type=str, default='',
                        help='comma-separated feature columns kept in raw [0,1] scale by Dataset_Custom')
    parser.add_argument('--reliability_loss_weight', type=float, default=0.15,
                        help='weight of supervised NWP reliability loss for WSPR v4')
    parser.add_argument('--physics_loss_weight', type=float, default=0.02,
                        help='weight of solar-geometry soft constraint loss for WSPR v4')
    parser.add_argument('--physics_min_pred_len', type=int, default=0,
                        help='minimum prediction length for enabling WSPR adaptive physics; 0 keeps legacy behavior')
    parser.add_argument('--physics_horizon_gamma', type=float, default=0.0,
                        help='lead-time weighting exponent for WSPR adaptive physics; 0 keeps uniform physics weighting')
    parser.add_argument('--physics_reliability_gamma', type=float, default=0.0,
                        help='NWP reliability weighting exponent for WSPR adaptive physics; 0 disables reliability weighting')
    parser.add_argument('--residual_l1_weight', type=float, default=0.0,
                        help='L1 regularization weight for WSPR NWP residual correction magnitude')
    parser.add_argument('--site_latitude', type=float, default=39.0, help='PV station latitude in degrees')
    parser.add_argument('--site_longitude', type=float, default=116.0, help='PV station longitude in degrees')
    parser.add_argument('--site_timezone', type=float, default=8.0, help='local UTC offset used by solar geometry')
    parser.add_argument('--wspr_ablation', type=str, default='none',
                        choices=[
                            'none',
                            'no_reliability',
                            'no_physics',
                            'no_routing',
                            'no_state_routing',
                            'no_nwp_residual',
                        ],
                        help='WSPR v4 ablation mode')
    parser.add_argument('--skip_test_after_train', action='store_true',
                        help='train and load the best validation checkpoint, but do not run the test split')
    parser.add_argument('--save_wspr_diagnostics', action='store_true',
                        help='save WSPR internal diagnostic arrays such as gate and NWP residual during test')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type') # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    # --- 消融实验开关 ---
    parser.add_argument('--use_cheby_kan', type=int, default=1, help='是否使用 ChebyKAN 分支 (1:是, 0:否)')
    parser.add_argument('--use_wav_kan', type=int, default=1, help='是否使用 WavKAN 分支 (1:是, 0:否)')
    parser.add_argument('--use_tri_gate', type=int, default=1, help='是否使用三向门控融合 (1:是, 0:否)')
    parser.add_argument('--use_nwp_gate', type=int, default=1, help='是否使用 NWP 气象门控 (1:是, 0:否)')
    parser.add_argument('--use_comb_loss', type=int, default=1, help='是否使用复合损失函数 (1:CombLoss, 0:仅MSE)')


    parser.add_argument('--allow_training', action='store_true',
                        help='explicitly acknowledge that training will start')
    args = parser.parse_args()
    if args.task_name != 'long_term_forecast' or args.data != 'custom':
        parser.error('This paper release supports long_term_forecast with custom data only.')
    if args.is_training and not args.allow_training:
        parser.error('Training requires explicit --allow_training acknowledgement.')

    # Import the numerical/training stack only after argument parsing. This keeps
    # `python run.py --help` usable before the research environment is installed;
    # real training and evaluation still fail fast when dependencies are missing.
    import numpy as np
    import torch
    import torch.backends
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.print_args import print_args

    # 🟢 应用 parser 里的随机种子
    fix_seed = args.fix_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # 训练模式
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_grid{}_w{}_decomp{}_wav{}_lvl{}_kdeg{}_wnum{}_kgrid{}_{}'.format(
                args.task_name, args.model_id, args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
                args.e_layers, args.d_layers, args.d_ff, args.expand, args.d_conv,
                args.factor, args.embed, args.distil, args.des, args.grid_size,
                args.w_kan, args.decomp_method, args.wavelet_name, args.dwt_level,
                args.kan_degree, args.wavelet_num, args.kan_grid_size, ii)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if args.skip_test_after_train:
                print('>>>>>>>skip testing after training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            else:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
            torch.cuda.empty_cache()
    else:
        # 🟢 关键修改：这是独立的测试分支
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_grid{}_w{}_decomp{}_wav{}_lvl{}_kdeg{}_wnum{}_kgrid{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_layers, args.d_ff, args.expand, args.d_conv,
            args.factor, args.embed, args.distil, args.des, args.grid_size,
            args.w_kan, args.decomp_method, args.wavelet_name, args.dwt_level,
            args.kan_degree, args.wavelet_num, args.kan_grid_size, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1) # 🟢 确保这里调用了 test 方法
        torch.cuda.empty_cache()
