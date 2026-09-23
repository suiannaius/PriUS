import os
import torch
import logging
import time
import argparse
import shutil
import json
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model.models import Unet, ProbabilisticUnet3D, Udrop3D
from training.train import train_whs
from utilities.weights_init import weights_init_kaiming
from utilities.utils import save_args, print_pairs_info, adjust_learning_rate, check_training_statistics
from training.dataset import WHS_Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from run_inference import run_inference_WHS


def getArgs():
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    parser = argparse.ArgumentParser()
    # Basic Information
    parser.add_argument('--dataset', type=str, default='WHS')
    parser.add_argument('--user', default='user', type=str)
    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
    parser.add_argument('--task_id', required=True, type=int)
    parser.add_argument('--seed', default=42, type=int)
    # Training detalis
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs to train [default: 200]')
    parser.add_argument('--annealing_steps', default=20, type=int, help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--early_stop_steps', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--beta', default=0.0, type=float, help='coefficient of gradient loss [default: 0.0]')
    parser.add_argument('--gamma', default=0.0, type=float, help='coefficient of noise loss [default: 0.0]')
    parser.add_argument('--coef_sigma', default=0.0, type=float, help='coefficient of noise_loss_sigma [default: 0.0]')
    parser.add_argument('--coef_d', default=0.0, type=float, help='coefficient of noise_loss_d [default: 0.0]')
    parser.add_argument('--coef_far', default=0.0, type=float, help='coefficient of noise_loss_far [default: 0.0]')
    parser.add_argument('--coef_cu', default=0.0, type=float, help='coefficient of CU Loss [default: 0.0]')
    parser.add_argument('--coef_kl', default=0.1, type=float, help='coefficient of CU Loss [default: 0.0]')
    parser.add_argument('--threshold_sigma', default=4, type=int)
    parser.add_argument('--threshold_d', default=4, type=int)
    parser.add_argument('--threshold_far', default=4, type=int)
    parser.add_argument('--sigma_upper_bound', default=50, type=int)
    parser.add_argument('--sigma_blur', default=0., type=float)
    parser.add_argument('--intensity', default=5., type=float)
    
    parser.add_argument('--num_classes', default=8, type=int)
    parser.add_argument('--num_modalities', default=1, type=int, help='number of modalities [default: 1]')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--patch_size', default=(64, 64, 64), nargs=3, type=int)
    parser.add_argument('--target_size', default=(150, 150, 150), nargs=3, type=int)
    parser.add_argument('--new_spacing', default=(2.0, 2.0, 2.0), nargs=3, type=float)
    parser.add_argument('--method', default='Ours', type=str, help="Ours/DEviS/PU/TTA/UDrop/EU")
    parser.add_argument('--base_channels', default=16, type=int)
    parser.add_argument('--loss_type', default='digamma', type=str, help="digamma/log/mse")
    parser.add_argument('--modality_filter', default='ct', type=str, help="ct/mr/all")
    parser.add_argument('--center_filter', default='A', type=str, help="A/B/C/D/E/F/G/all")
    parser.add_argument('--num_train_samples', default=None, type=int, help="Number of training samples. If None, all samples will be used for training [default: None]")
    parser.add_argument('--num_val_samples', default=None, type=int, help="Number of validating samples. If None, all samples will be used for validating [default: None]")
    parser.add_argument('--shuffle', type=int, default=1, help='1 for shuffle, 0 for no shuffle')
    parser.add_argument('--good_model_step', default=50, type=int)

    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--skip_test', action='store_true', help='Skip automatic post-training test evaluation')
    parser.add_argument('--use_pretrain', action='store_true', help="Whether load parameters from pretrained models.")
    parser.add_argument('--use_prior', action='store_true')
    parser.add_argument('--use_noise_aug', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = getArgs()
    check_training_statistics(args.use_prior)
    args_name = f'./configs/Task_{args.task_id}_{args.date}_config.json'
    folder_path = os.path.dirname(args_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_args(args, args_name)
    project_name = f'Task_{args.task_id}_{args.date}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = f'./logging/{project_name}_{args.method}.txt'
    folder_path = os.path.dirname(log_file)
    
    with open("./zscore_stats_ct_centerA.json") as f:
        stats = json.load(f)

    low  = stats["low"]
    high = stats["high"]
    mean = stats["mean"]
    std  = stats["std"]
    
    data_stats = (low, high, mean, std)
    print("\n========== Z-Score Normalization Statistics ==========")
    print(f"low   (0.5% percentile) : {low:.6f}")
    print(f"high  (99.5% percentile): {high:.6f}")
    print(f"mean  (after clipping)  : {mean:.6f}")
    print(f"std   (after clipping)  : {std:.6f}")
    print("======================================================\n")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.isfile(log_file):
        open(log_file, 'w').close()

    logging.basicConfig(filename=log_file,
                        format = '%(asctime)s - %(name)s - %(message)s',
                        level=logging.INFO,
                        filemode='w')
    logging.info('Date: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logging.info('Dataset: {}'.format(args.dataset))
    logging.info('Project name: {}'.format(project_name))
    logging.info('Writer: tensorboard --logdir=./runs/{}/{}_{}'.format(args.user, project_name, args.method))

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    print('Dataset: ', args.dataset)
    train_datapath = os.environ["WHS_DATA_ROOT"]

    if args.method in ['Ours', 'DEviS', 'PureEvidential', 'PlainUNet', 'TTA', 'UDrop', 'PU']:
        log_dir = f'./runs/{project_name}_{args.method}'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        else:
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        
        if args.method in ['Ours', 'DEviS', 'PureEvidential']:
            print('Using evidential method.')
            model = Unet(in_channels=args.num_modalities, base_channels=args.base_channels, num_classes=args.num_classes).to(device)
        
        elif args.method in ['PlainUNet', 'TTA']:
            print('Using Plain UNet.')
            model = Unet(in_channels=args.num_modalities, base_channels=args.base_channels, num_classes=args.num_classes).to(device)
        
        elif args.method == 'UDrop':
            print('Using UDrop.')
            model = Udrop3D(in_channels=args.num_modalities, out_channels=args.num_classes).to(device)
        
        elif args.method == 'PU':
            print('Using Probabilistic UNet.')
            model = ProbabilisticUnet3D(input_channels=args.num_modalities, num_classes=args.num_classes, num_filters=[32, 64, 128, 256], latent_dim=2, # num_filters=[64, 128, 256, 512]
                                        no_convs_fcomb=4, beta=10.0).to(device)
        if not args.use_pretrain:
            model.apply(weights_init_kaiming)
            print('Use kaiming_init.')
 
        else:
            args.good_model_step = 0
            load_name = args.pretrain_path
            if not load_name:
                raise ValueError('--use_pretrain requires --pretrain_path')
            model.load_state_dict(torch.load(load_name, weights_only=True))
            print(f'Using pretrained weights loaded from {load_name}...')
            print(f'Good model step is reset to {args.good_model_step}.')

        pytorch_total_params = []
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total parameters: ', pytorch_total_params)
        print(f'Writer: tensorboard --logdir=./runs/{project_name}_{args.method}')
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        save_name = f'./saved_models/{project_name}.pth'
        folder_path = os.path.dirname(save_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        supervised_dataset = WHS_Dataset(train_datapath, 
                                         num_classes=8, 
                                         mode='train', 
                                         patch_size=args.patch_size,
                                         modality_filter=args.modality_filter,
                                         center_filter=args.center_filter,
                                         target_size=args.target_size,
                                         new_spacing=args.new_spacing,
                                         normalize=False)
        print_pairs_info(supervised_dataset)
        
        train_loader = DataLoader(supervised_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                  pin_memory=True, drop_last=True, num_workers=0)
        print(f"Train dataset size: {len(supervised_dataset)}")

        for epoch in tqdm(range(args.num_epochs), desc='Epochs'):
            adjust_learning_rate(optimizer, epoch, args.num_epochs, args.lr)
            train_loss, train_LV, train_RV, train_LA, train_RA, train_Myo, train_AO, train_PA = train_whs(model=model,
                                                                                                dataloader=train_loader, 
                                                                                                optimizer=optimizer,
                                                                                                args=args,
                                                                                                current_epoch=epoch,
                                                                                                device=device,
                                                                                                writer=writer,
                                                                                                sample_size=int(1e7),
                                                                                                data_stats=data_stats)
            print(f'''
                Writer: tensorboard --logdir=./runs/{project_name}_{args.method}
                Epoch {epoch+1}/{args.num_epochs}, project: {project_name}, lr: {optimizer.param_groups[0]['lr']}
                beta: {args.beta}, gamma: {args.gamma}, coef_sigma: {args.coef_sigma}, coef_d: {args.coef_d}, coef_far: {args.coef_far}
                (Train) Loss: {train_loss:.3f}, 
                (Train) LV: {train_LV:.3f}, RV: {train_RV:.3f}, LA: {train_LA:.3f}, RA: {train_RA:.3f}, Myo: {train_Myo:.3f}, AO: {train_AO:.3f}, PA: {train_PA:.3f}''')
            logging.info(
                "[Epoch {:d}]  lr: {:.7f}\n"
                "(Train) Loss: {:.3f}  Dice: LV: {:.3f}  RV: {:.3f}  LA: {:.3f}  RA: {:.3f}  Myo: {:.3f}  AO: {:.3f}  PA: {:.3f}\n".format(
                    epoch + 1, 
                    optimizer.param_groups[0]['lr'],
                    train_loss, train_LV, train_RV, train_LA, train_RA, train_Myo, train_AO, train_PA))
                
            if epoch == args.num_epochs - 1 or epoch % 10 == 0:
                torch.save(model.state_dict(), save_name)
                print(f"New model is saved as {save_name}")
                logging.info(f"New model is saved as {save_name}")
        writer.close()
    
    elif args.method == 'EU':
        print('Using Ensemble UNet.')
        supervised_list = [
            WHS_Dataset(train_datapath, 
                        num_classes=8, 
                        mode='train', 
                        patch_size=args.patch_size,
                        modality_filter=args.modality_filter,
                        center_filter=args.center_filter,
                        target_size=args.target_size,
                        new_spacing=args.new_spacing,
                        normalize=False,
                        ensemble_index=i,
                        ensemble_parts=4)
                        for i in range(4)
                        ]
        
        for i in range(4):
            log_dir = f'./runs/{project_name}_{i}'
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            else:
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
            
            model = Unet(in_channels=args.num_modalities, base_channels=args.base_channels, num_classes=args.num_classes).to(device)
            model.apply(weights_init_kaiming)
            print('Use kaiming_init.')
            
            pytorch_total_params = []
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Total parameters: ', pytorch_total_params)
            print(f'Writer: tensorboard --logdir=./runs/{project_name}_{i}')
            
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            save_name = f'./saved_models/{project_name}_ensemble_{i}.pth'
            folder_path = os.path.dirname(save_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            train_loader = DataLoader(supervised_list[i], batch_size=args.batch_size, shuffle=args.shuffle,
                                      pin_memory=True, drop_last=True, num_workers=0)
            print(f"Train dataset size: {len(supervised_list[i])}")

            for epoch in tqdm(range(args.num_epochs), desc='Epochs'):
                adjust_learning_rate(optimizer, epoch, args.num_epochs, args.lr)
                train_loss, train_LV, train_RV, train_LA, train_RA, train_Myo, train_AO, train_PA = train_whs(model=model,
                                                                                                    dataloader=train_loader, 
                                                                                                    optimizer=optimizer,
                                                                                                    args=args,
                                                                                                    current_epoch=epoch,
                                                                                                    device=device,
                                                                                                    writer=writer,
                                                                                                    sample_size=int(1e7),
                                                                                                    data_stats=data_stats)
                print(f'''
                    Writer: tensorboard --logdir=./runs/{project_name}_{i}
                    Epoch {epoch+1}/{args.num_epochs}, project: {project_name}, lr: {optimizer.param_groups[0]['lr']}
                    (Train) Loss: {train_loss:.3f}, 
                    (Train) LV: {train_LV:.3f}, RV: {train_RV:.3f}, LA: {train_LA:.3f}, RA: {train_RA:.3f}, Myo: {train_Myo:.3f}, AO: {train_AO:.3f}, PA: {train_PA:.3f}''')
                logging.info(
                    "[Epoch {:d}]  lr: {:.7f}\n"
                    "(Train) Loss: {:.3f}  Dice: LV: {:.3f}  RV: {:.3f}  LA: {:.3f}  RA: {:.3f}  Myo: {:.3f}  AO: {:.3f}  PA: {:.3f}\n".format(
                        epoch + 1, 
                        optimizer.param_groups[0]['lr'],
                        train_loss, train_LV, train_RV, train_LA, train_RA, train_Myo, train_AO, train_PA))
                    
                if epoch == args.num_epochs - 1 or epoch % 10 == 0:
                    torch.save(model.state_dict(), save_name)
                    print(f"New model is saved as {save_name}")
                    logging.info(f"New model is saved as {save_name}")
            writer.close()
    else:
        raise NotImplementedError(f'Method {args.method} is not implemented.')
    
    if args.skip_test:
        return
    print('Training finished. Inference is starting...')
    use_lcc = True
    lcc_classes = None
    lcc_classes = [1, 2, 3, 4, 5, 6, 7]  # LV, RV, LA, RA, Myo, AO
    target_size = None  # None意味着不在inference阶段进行pad_to_size操作
    target_size=(160, 160, 160)

    run_inference_WHS(project_name, consider_spacing=True, noised_image=False,  mode='test', sliding_mode='gaussian', target_size=target_size, use_lcc=use_lcc, lcc_classes=lcc_classes)

if __name__ == "__main__":
    main()
