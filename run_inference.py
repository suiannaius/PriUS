import os
import torch
import numpy as np
import json
import shutil
from torch.utils.data import DataLoader
from model.models import Unet, ProbabilisticUnet3D, Udrop3D
from utilities.utils import load_args, check_training_statistics
from training.inference import inference_WHS
from training.dataset import WHS_Dataset, whs_collate_test
from utilities.utils import print_pairs_info
from torch.utils.tensorboard import SummaryWriter


def run_inference_WHS(project_name, consider_spacing=True, mode=None, target_size=None, **kwargs):
    CLASS_NAME_MAP = {
        0: "LV",
        1: "RV",
        2: "LA",
        3: "RA",
        4: "Myo",
        5: "AO",
        6: "PA"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = load_args(f'./configs/{project_name}_config.json')
    check_training_statistics(args.use_prior)
    test_datapath = os.environ["WHS_DATA_ROOT"]
    log_dir=f"./results/writer/inference_{project_name}_{kwargs.get('sliding_mode', 'avg')}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    with open("./zscore_stats_ct_centerA.json") as f:
        stats = json.load(f)
    low  = stats["low"]
    high = stats["high"]
    mean = stats["mean"]
    std  = stats["std"]
    data_stats = (low, high, mean, std)
    
    print(f'Using {args.method} method.')
    # ----------- Load model weights -----------
    if args.method in ['Ours', 'DEviS', 'PureEvidential', 'PlainUNet', 'TTA', 'UDrop', 'PU']:
        if args.method in ['Ours', 'DEviS', 'PureEvidential', 'PlainUNet', 'TTA']:
            model = Unet(in_channels=args.num_modalities, base_channels=args.base_channels, num_classes=args.num_classes).to(device)
        
        elif args.method == 'UDrop':
            model = Udrop3D(in_channels=args.num_modalities, out_channels=args.num_classes).to(device)
        
        elif args.method == 'PU':
            model = ProbabilisticUnet3D(input_channels=args.num_modalities, num_classes=args.num_classes, num_filters=[32, 64, 128, 256], latent_dim=2,
                                        no_convs_fcomb=4, beta=10.0).to(device)
        
        load_name = f'./saved_models/{project_name}.pth'
        model.load_state_dict(torch.load(load_name, weights_only=True))
        print(f'Weights loaded from {load_name}...')

        test_dataset = WHS_Dataset(test_datapath, 
                                   num_classes=8, 
                                   mode=mode, 
                                   patch_size=args.patch_size,
                                   num_patches_per_volume=None, 
                                   modality_filter=args.modality_filter,
                                   center_filter=args.center_filter,
                                   new_spacing=args.new_spacing,
                                   target_size=target_size,
                                   normalize=False,
                                   **kwargs)
        print(f"Test dataset size: {len(test_dataset)}")
        print_pairs_info(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                pin_memory=True, num_workers=0, collate_fn=whs_collate_test)
        
        results = inference_WHS(model=model,
                                dataloader=test_loader, 
                                device=device,
                                writer=writer,
                                args=args,
                                consider_spacing=consider_spacing,
                                data_stats=data_stats,
                                **kwargs)                            
        
    elif args.method == 'EU':
        print('Using Ensemble UNet.')
        test_list = [
            WHS_Dataset(test_datapath, 
                        num_classes=8, 
                        mode='test', 
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
        models = []
        for i in range(4):
            model = Unet(in_channels=args.num_modalities, base_channels=args.base_channels, num_classes=args.num_classes).to(device)
            load_name = f'./saved_models/{project_name}_ensemble_{i}.pth'
            model.load_state_dict(torch.load(load_name, weights_only=True))
            print(f'Weights loaded from {load_name}...')
            models.append(model)
            
        print(f"Test dataset size: {len(test_list[i])}")
        print_pairs_info(test_list[i])
        test_loader = DataLoader(test_list[i], batch_size=1, shuffle=False, 
                                 pin_memory=True, num_workers=0, collate_fn=whs_collate_test)
        results = inference_WHS(model=models,
                                dataloader=test_loader, 
                                device=device,
                                writer=writer,
                                args=args,
                                consider_spacing=consider_spacing,
                                data_stats=data_stats,
                                **kwargs)
            
    dice, hd95 = results['dice'], results['hd95']
    ucc_sigma, ur_sigma, ucc_d, ur_d, ucc_grad, ur_grad = results['ucc_sigma'], results['ur_sigma'], results['ucc_d'], results['ur_d'], results['ucc_grad'], results['ur_grad']
    print('---- Inference Results ----')
    for cls in range(0, args.num_classes - 1):
        cls_name = CLASS_NAME_MAP.get(cls, f"Class_{cls}")
        print(
            f'{cls_name}: '
            f'Dice: {dice[cls]:.4f},'
            f'HD95: {hd95[cls]:.4f}'
        )

    avg_dice = np.mean(dice)
    avg_hd95 = np.mean(hd95)

    print(
        f'Average: Dice: {avg_dice:.4f}, '
        f'HD95: {avg_hd95:.4f}, '
        f'ucc_grad: {ucc_grad:.4f}, ucc_sigma: {ucc_sigma:.4f}, ucc_d: {ucc_d:.4f}, '
        f'ur_grad: {ur_grad:.4f}, ur_sigma: {ur_sigma:.4f}, ur_d: {ur_d:.4f}'
    )

    results_dict = {
        "project_name": project_name,
        "Writer": f"tensorboard --logdir={log_dir}",
        "metrics_per_class": {},
        "average_metrics": {
            "Dice": float(avg_dice),
            "HD95": float(avg_hd95),
            "ucc_grad": float(ucc_grad),
            "ucc_sigma": float(ucc_sigma),
            "ucc_d": float(ucc_d),
            "ur_grad": float(ur_grad),
            "ur_sigma": float(ur_sigma),
            "ur_d": float(ur_d)
        }
    }

    for cls in range(0, args.num_classes - 1):
        cls_name = CLASS_NAME_MAP.get(cls, f"Class_{cls}")
        results_dict["metrics_per_class"][cls_name] = {
            "class_id": cls,
            "Dice": float(dice[cls]),
            "HD95": float(hd95[cls])
        }
    save_dir = "./results/inference/WHS"
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{project_name}_inference_{kwargs.get('sliding_mode', 'avg')}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    print(f"Results saved to: {json_path}")
    print(f'Writer: tensorboard --logdir={log_dir}')
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained WHS model")
    parser.add_argument('--project', required=True, help='Task_ID_DATE matching config and checkpoint')
    parser.add_argument('--no_lcc', action='store_true')
    cli = parser.parse_args()
    run_inference_WHS(cli.project, consider_spacing=True, noised_image=False,
                      mode='test', sliding_mode='gaussian', target_size=(160, 160, 160),
                      use_lcc=not cli.no_lcc, lcc_classes=[1, 2, 3, 4, 5, 6, 7])
