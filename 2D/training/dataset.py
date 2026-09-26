import os
import numpy as np
import torch
import SimpleITK as sitk
from PIL import Image
from torch.utils.data import Dataset
from utilities.geometry import compute_distance_map_numpy, compute_gradient_numpy
from utilities.resampling import resample_data_or_seg


class ACDC2017_Dataset(Dataset):
    def __init__(self, ImgFiles, num_classes: int = 4, name_flg: bool = True):

        self.ImgFiles = ImgFiles
        self.ori_size = 128
        self.num_classes = num_classes
        self.name_flg = name_flg
        imgs = np.zeros((1, 128, 128))
        labs = np.zeros((1, 128, 128))
        spacings = np.zeros((1, 2))

        sitk.ProcessObject_SetGlobalWarningDisplay(False)

        self.name_list = []
        for imgfile in self.ImgFiles:
            itkimg = sitk.ReadImage(imgfile)

            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,
            # npimg = max_min_norm(npimg)
            npimg = np.pad(npimg, ((0, 0), (50, 50), (50, 50)), "minimum")

            itklab = sitk.ReadImage(imgfile.replace(".nii", "_gt.nii"))
            nplab = sitk.GetArrayFromImage(itklab)  # (H, W, D)
            nplab = np.pad(nplab, ((0, 0), (50, 50), (50, 50)), "minimum")

            index = np.where(nplab != 0)

            npimg = npimg[
                :,
                (np.min(index[1]) + np.max(index[1])) // 2 - 64 : (
                    np.min(index[1]) + np.max(index[1])
                )
                // 2
                + 64,
                (np.min(index[2]) + np.max(index[2])) // 2 - 64 : (
                    np.min(index[2]) + np.max(index[2])
                )
                // 2
                + 64,
            ]
            nplab = nplab[
                :,
                (np.min(index[1]) + np.max(index[1])) // 2 - 64 : (
                    np.min(index[1]) + np.max(index[1])
                )
                // 2
                + 64,
                (np.min(index[2]) + np.max(index[2])) // 2 - 64 : (
                    np.min(index[2]) + np.max(index[2])
                )
                // 2
                + 64,
            ]

            spacing = np.array(itkimg.GetSpacing()).reshape(1, 3)[:, :-1]
            spacing = np.repeat(spacing, npimg.shape[0], axis=0)

            imgs = np.concatenate((imgs, npimg), axis=0)
            labs = np.concatenate((labs, nplab), axis=0)
            spacings = np.concatenate((spacings, spacing), axis=0)

            for i in range(npimg.shape[0]):
                self.name_list.append(imgfile + str("_slice_%03d" % i))

        self.imgs = imgs[1:, :, :]
        self.labs = labs[1:, :, :]
        self.spacings = spacings[1:, :]
        self.imgs = self.imgs.astype(np.float32)  # [366, 128, 128]
        self.labs = self.labs.astype(np.uint8)
        self.spacings = self.spacings.astype(np.float32)
        self.case_cache = []
        self.distance_cache = {}
        self.gradient_cache = {}

    def __len__(self):

        return self.imgs.shape[0]

    def __getitem__(self, item):
        img = self.imgs[item]
        lab = self.labs[item]
        spacing = self.spacings[item]
        img_name = self.name_list[item]

        img = img.copy()  # (128, 128)
        lab = lab.copy()  # (128, 128)

        if img_name not in self.case_cache:
            self.case_cache.append(img_name)
            distance_map = compute_distance_map_numpy(lab)
            gradient = compute_gradient_numpy(img, sigma_blur=0.5)
            self.distance_cache[img_name] = distance_map
            self.gradient_cache[img_name] = gradient

        img = (
            torch.from_numpy(img).unsqueeze(0).type(dtype=torch.FloatTensor)
        )  # (1, 128, 128)
        one_hot_lab = np.eye(self.num_classes)[lab]  # (128, 128) → (128, 128, 4)
        one_hot_lab = np.transpose(one_hot_lab, (2, 0, 1))  # (4, 128, 128)
        lab = torch.from_numpy(one_hot_lab).type(
            dtype=torch.LongTensor
        )  # (4, 128, 128)

        spacing = spacing.copy()  # (2,)
        spacing = torch.from_numpy(spacing).type(dtype=torch.FloatTensor)  # (1, 2)

        distance_map = self.distance_cache[img_name]
        distance_map = torch.from_numpy(distance_map).type(
            dtype=torch.FloatTensor
        )  # (128, 128)

        gradient = self.gradient_cache[img_name]
        gradient = torch.from_numpy(gradient).type(dtype=torch.FloatTensor).unsqueeze(0)

        return img, lab, spacing, distance_map, gradient


class ISIC2018_Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes=2, index=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.new_shape = (256, 256)

        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.mask_files = sorted(
            [f for f in os.listdir(mask_dir) if f.endswith(".png")]
        )

        if index is not None:
            self.img_files = self.img_files[
                index * (len(self.img_files) // 4) : (index + 1)
                * (len(self.img_files) // 4)
                if index < 4 - 1
                else None
            ]
            self.mask_files = self.mask_files[
                index * (len(self.mask_files) // 4) : (index + 1)
                * (len(self.mask_files) // 4)
                if index < 4 - 1
                else None
            ]

        assert len(self.img_files) == len(self.mask_files), (
            f"The number of images and masks do not match! Got {len(self.img_files)} and {len(self.mask_files)}."
        )

        if not self.img_files:
            raise ValueError("No ISIC .jpg images found")
        for image, mask in zip(self.img_files, self.mask_files):
            if os.path.splitext(image)[0] != os.path.splitext(mask)[0].removesuffix(
                "_segmentation"
            ):
                raise ValueError(f"Image/mask ID mismatch: {image} versus {mask}")

        self.case_cache = []
        self.distance_cache = {}
        self.gradient_cache = {}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # img = self.transform(img)
        # img = resample_data_or_seg(img.numpy(), new_shape=self.new_shape)
        img = np.array(img).astype(np.float32)  # H * W * C
        img = img.transpose((2, 0, 1))  # C * H * W
        img = resample_data_or_seg(img, new_shape=self.new_shape)  # [3,512,512]

        mask = np.array(mask)  # [oH,oW]
        mask = (mask > 0).astype(np.uint8)
        mask = resample_data_or_seg(
            mask[None, :, :], new_shape=self.new_shape, is_seg=True
        )

        if img_name not in self.case_cache:
            self.case_cache.append(img_name)
            distance_map = compute_distance_map_numpy(mask[0])
            gradient = compute_gradient_numpy(img[0], sigma_blur=0.5)
            self.distance_cache[img_name] = distance_map
            self.gradient_cache[img_name] = gradient

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask).long()

        one_hot_mask = torch.nn.functional.one_hot(
            mask.squeeze(0), num_classes=self.num_classes
        )
        one_hot_mask = one_hot_mask.permute(2, 0, 1).float()  # (H, W, C) → (C, H, W)
        spacing = (1.0, 1.0)

        # 直接从缓存获取，不检查（因为已经在上面确保存在）
        distance_map = torch.from_numpy(self.distance_cache[img_name]).float()
        gradient = torch.from_numpy(self.gradient_cache[img_name]).float().unsqueeze(0)

        return img, one_hot_mask, spacing, distance_map, gradient


def acdc_split(root, fold):
    """Original diagnostic-group-stratified split, seed 1; sorted file order."""
    from pathlib import Path

    if fold not in range(5):
        raise ValueError("ACDC fold must be 0..4")
    rng = np.random.RandomState(1)
    ids = rng.permutation(np.arange(1, 101).reshape(5, 20).transpose())
    heldout = set(ids[4 * fold : 4 * (fold + 1)].reshape(-1).tolist())
    train, test = [], []
    for patient in range(1, 101):
        paths = sorted(
            (Path(root) / f"patient{patient:03d}").glob("*_frame[0-9][0-9].nii*")
        )
        if len(paths) != 2:
            raise ValueError(
                f"Expected two annotated frames for patient{patient:03d}, found {len(paths)}"
            )
        for path in paths:
            if not Path(str(path).replace(".nii", "_gt.nii")).is_file():
                raise FileNotFoundError(f"Missing annotation for {path.name}")
        (test if patient in heldout else train).extend(map(str, paths))
    return train, test


def make_dataset(args, fold, training=False):
    if args.dataset == "ACDC":
        train, test = acdc_split(args.data_dir, fold)
        return ACDC2017_Dataset(train if training else test)
    return ISIC2018_Dataset(args.image_dir, args.mask_dir)


def sample_names(dataset):
    return (
        [os.path.basename(n) for n in dataset.name_list]
        if hasattr(dataset, "name_list")
        else dataset.img_files
    )
