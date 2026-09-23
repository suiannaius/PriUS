import os
import glob
import torch
import random
import re
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
    
    
class WHS_Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_classes: int,
        mode: str = None,
        patch_size: Tuple[int, int, int] = None,
        target_size: Optional[Tuple[int, int, int]] = None,
        new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0), 
        normalize: bool = False,
        augment_prob: float = 0.5,
        modality_filter: str = 'all',
        center_filter: str = 'all',
        ensemble_index: int = 0,
        ensemble_parts: int = 1,
        **kwargs):

        assert mode in ['train', 'validate', 'test']
        assert modality_filter in ['ct', 'mr', 'all'], "modality_filter must be in: ['ct','mr','all']"
        assert center_filter in ['A','B','C','D','E','F','G','all'], "center_filter must be in: ['A','B','C','D','E','F','G','all']"

        self.mode = mode
        self.num_classes = num_classes
        self.root_dir = os.path.join(root_dir, mode)
        self.image_dir = os.path.join(self.root_dir, 'image')
        self.label_dir = os.path.join(self.root_dir, 'label')
        self.modality_filter = modality_filter
        self.center_filter = center_filter
        self.center_map = {'1':'A', '2':'B', '3':'C', '5':'E', '6':'F', '7':'G'}
        if self.mode == 'train' and ensemble_parts > 1:
            assert 0 <= ensemble_index < ensemble_parts
            all_pairs = self._get_pairs()
            self.pairs = self._split_pairs_for_ensemble(all_pairs, ensemble_index, ensemble_parts)
        else:
            self.pairs = self._get_pairs()
        self.patch_size = patch_size
        self.new_spacing = new_spacing
        self.normalize = normalize
        self.augment_prob = augment_prob
        self.target_size = target_size
        self.case_cache = {}

        print(f"Modality filter: {self.modality_filter}")
        print(f"Center filter: {self.center_filter}")
        print(f"Patch size: {self.patch_size}")

        if self.mode == 'train':
            self.patch_index = []  # [(case_idx, z, y, x), ...]

            for case_idx, (img_path, lbl_path) in enumerate(self.pairs):
                img_itk = sitk.ReadImage(img_path)
                lbl_itk = sitk.ReadImage(lbl_path)

                img_rs = self._resample_to_spacing(img_itk, self.new_spacing, False)
                lbl_rs = self._resample_to_spacing(lbl_itk, self.new_spacing, True)

                img = sitk.GetArrayFromImage(img_rs)
                lbl = self._map_label(sitk.GetArrayFromImage(lbl_rs))
                if self.target_size is not None:
                    img, lbl, _ = self._pad_to_size(img, lbl, self.target_size, pad_value=None, background_label=0)
                else:
                    img, lbl, _ = WHS_Dataset.pad_if(img, lbl, self.patch_size, pad_value=None, background_label=0)
                
                coords = self._get_sliding_coords(img.shape)

                for (z, y, x) in coords:
                    self.patch_index.append((case_idx, z, y, x))

    @staticmethod
    def axis_coords(L, p, s):
            coords = list(range(0, L - p + 1, s))
            if coords[-1] != L - p:
                coords.append(L - p)
            return coords

    @staticmethod
    def pad_if(img, lbl, patch_size, pad_value=None, background_label=0):
        pd, ph, pw = patch_size
        D, H, W = img.shape

        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)

        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return img, lbl, None

        pd1, pd2 = pad_d // 2, pad_d - pad_d // 2
        ph1, ph2 = pad_h // 2, pad_h - pad_h // 2
        pw1, pw2 = pad_w // 2, pad_w - pad_w // 2

        if pad_value is None:
            pad_value = img.min()

        img = np.pad(
            img,
            ((pd1, pd2), (ph1, ph2), (pw1, pw2)),
            mode="constant",
            constant_values=pad_value)

        lbl = np.pad(
            lbl,
            ((pd1, pd2), (ph1, ph2), (pw1, pw2)),
            mode="constant",
            constant_values=background_label)

        pad_info = (pd1, pd2, ph1, ph2, pw1, pw2)
        return img, lbl, pad_info

    def _get_pairs(self) -> List[Tuple[str, str]]:
        image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.nii*')))
        pairs = []
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            # ----------------- 模态筛选 -----------------
            if self.modality_filter in ['ct', 'mr']:
                if not img_name.lower().startswith(self.modality_filter):
                    continue
            # ----------------- 提取病例ID,中心筛选 -----------------
            if 'image' in img_name:
                case_id = img_name.replace('_image.nii.gz', '')
            elif '_train_' in img_name:
                case_id = img_name.split('.')[0]
            else:
                case_id = img_name.split('_')[0]

            match = re.search(r'(\d{4})', case_id)
            if match:
                num = match.group(1)
                center_digit = num[0]
                center = self.center_map.get(center_digit, None)
                if self.center_filter != 'all':
                    if center != self.center_filter:
                        continue
            else:
                # 如果没有找到四位数字，跳过
                continue
            # ----------------- 配对 label -----------------
            label_candidates = glob.glob(os.path.join(self.label_dir, f"{case_id}_label*.nii*"))
            if label_candidates:
                label_path = sorted(label_candidates, key=lambda x: ('_1mm' in x, x))[0]
                pairs.append((img_path, label_path))
        return pairs

    def _split_pairs_for_ensemble(self, pairs, index, num_parts):
        num_cases = len(pairs)
        assert num_cases >= num_parts, \
            f"Number of cases {num_cases} < ensemble_parts {num_parts}"

        pairs = sorted(pairs, key=lambda x: os.path.basename(x[0]))
        splits = np.array_split(pairs, num_parts)
        selected_pairs = list(splits[index])

        print(f"[Ensemble] split {index}/{num_parts}, "
               f"cases={len(selected_pairs)}, "
               f"first={os.path.basename(selected_pairs[0][0])}, "
               f"last={os.path.basename(selected_pairs[-1][0])}")
        return selected_pairs

    def _resample_to_spacing(self, itk_img, new_spacing, is_label=False, target_size=None):
        original_spacing = itk_img.GetSpacing()
        original_size = itk_img.GetSize()

        if target_size is not None:
            new_size = target_size
        else:
            new_size = [
                int(np.round(osz * ospc / nspc))
                for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(itk_img.GetDirection())
        resampler.SetOutputOrigin(itk_img.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)

        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)

        return resampler.Execute(itk_img)

    def _normalize(self, image):
        low, high = np.percentile(image, (0.5, 99.5))
        image = np.clip(image, low, high)
        image = (image - image.mean()) / (image.std() + 1e-8)
        return image

    def _map_label(self, label):
        value2class = {
            0: 0, 500: 1, 600: 2, 420: 3,
            550: 4, 205: 5, 820: 6, 850: 7,
        }
        mapped = np.zeros_like(label, dtype=np.int64)
        for v, c in value2class.items():
            mapped[label == v] = c
        return mapped
    
    def _pad_to_size(self, img, label, target_size, pad_value, background_label=0, pad_mode="constant"):
        target_d, target_h, target_w = target_size
        D, H, W = img.shape
        assert target_d >= D, f"target_d {target_d} smaller than image depth {D}"
        assert target_h >= H, f"target_h {target_h} smaller than image height {H}"
        assert target_w >= W, f"target_w {target_w} smaller than image width {W}"
        # ----------- pad if image is smaller than target size -----------
        pad_d = max(0, target_d - D)
        pad_h = max(0, target_h - H)
        pad_w = max(0, target_w - W)

        # symmetric padding
        pd1, pd2 = pad_d // 2, pad_d - pad_d // 2
        ph1, ph2 = pad_h // 2, pad_h - pad_h // 2
        pw1, pw2 = pad_w // 2, pad_w - pad_w // 2

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            if pad_value is None:
                pad_value = img.min()
            img = np.pad(
                img,
                ((pd1, pd2), (ph1, ph2), (pw1, pw2)),
                mode=pad_mode,
                constant_values=pad_value)
            label = np.pad(
                label,
                ((pd1, pd2), (ph1, ph2), (pw1, pw2)),
                mode=pad_mode,
                constant_values=background_label)
            pad_info = (pd1, pd2, ph1, ph2, pw1, pw2)
        else:
            pad_info = None
        return img, label, pad_info

    def _get_sliding_coords(self, shape):
        D, H, W = shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = (pd // 2, ph // 2, pw // 2)

        assert D >= pd and H >= ph and W >= pw, \
            f"Image size {shape} smaller than patch_size {self.patch_size}"

        zs = WHS_Dataset.axis_coords(D, pd, sd)
        ys = WHS_Dataset.axis_coords(H, ph, sh)
        xs = WHS_Dataset.axis_coords(W, pw, sw)

        return [(z, y, x) for z in zs for y in ys for x in xs]
    
    def _random_augment(self, img, lbl):
        if random.random() > self.augment_prob:
            return img, lbl
        # Flip
        for axis in range(3):
            if random.random() < 0.5:
                img = np.flip(img, axis=axis).copy()
                lbl = np.flip(lbl, axis=axis).copy()
        # Rotation 90°
        k = random.randint(0, 3)
        axes = random.choice([(0, 1), (1, 2), (0, 2)])
        img = np.rot90(img, k, axes=axes).copy()
        lbl = np.rot90(lbl, k, axes=axes).copy()
        # Intensity jitter
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.1, 0.1)
            img = img * scale + shift
        # Gaussian noise
        return img, lbl

    def __getitem__(self, idx):

        if self.mode == 'train':
            case_idx, z, y, x = self.patch_index[idx]
            img_path, lbl_path = self.pairs[case_idx]

            if case_idx not in self.case_cache:
                img_itk = sitk.ReadImage(img_path)
                lbl_itk = sitk.ReadImage(lbl_path)

                img_rs = self._resample_to_spacing(img_itk, self.new_spacing, False)
                lbl_rs = self._resample_to_spacing(lbl_itk, self.new_spacing, is_label=True, target_size=img_rs.GetSize())

                img = sitk.GetArrayFromImage(img_rs)
                lbl = self._map_label(sitk.GetArrayFromImage(lbl_rs))

                if self.normalize and self.modality_filter == 'ct':
                    img = self._normalize(img)
                if self.target_size is not None:
                    img, lbl, _ = self._pad_to_size(img, lbl, self.target_size, pad_value=None, background_label=0)
                else:
                    img, lbl, _ = WHS_Dataset.pad_if(img, lbl, self.patch_size, pad_value=None, background_label=0)
                self.case_cache[case_idx] = (img, lbl)

            else:
                img, lbl = self.case_cache[case_idx]

            pd, ph, pw = self.patch_size
            img_patch = img[z:z+pd, y:y+ph, x:x+pw]
            lbl_patch = lbl[z:z+pd, y:y+ph, x:x+pw]

            img_patch, lbl_patch = self._random_augment(img_patch, lbl_patch)

            img_patch = torch.from_numpy(img_patch).float().unsqueeze(0)  # [1, D, H, W]

            lbl_patch = torch.from_numpy(lbl_patch).long()                # [D, H, W]
            lbl_patch = F.one_hot(lbl_patch, self.num_classes)            # [D, H, W, C]
            lbl_patch = lbl_patch.permute(3, 0, 1, 2).float()             # [C, D, H, W]

            spacing = torch.tensor(self.new_spacing, dtype=torch.float32)

            return img_patch, lbl_patch, spacing, img_path

        elif self.mode == 'test' or self.mode == 'validate':
            img_path, lbl_path = self.pairs[idx]
        
            img_itk = sitk.ReadImage(img_path)
            lbl_itk = sitk.ReadImage(lbl_path)
            
            img_rs = self._resample_to_spacing(img_itk, self.new_spacing, is_label=False)
            lbl_rs = self._resample_to_spacing(lbl_itk, self.new_spacing, is_label=True, target_size=img_rs.GetSize())

            img = sitk.GetArrayFromImage(img_rs)  # [D,H,W] [58, 92, 92]
            lbl = sitk.GetArrayFromImage(lbl_rs)
            lbl = self._map_label(lbl)

            if self.normalize and self.modality_filter == 'ct':
                img = self._normalize(img)

            if self.target_size is not None:
                img, lbl, pad_info = self._pad_to_size(img, lbl, target_size=self.target_size, pad_value=None, background_label=0, pad_mode="constant")
            else:
                img, lbl, pad_info = self.pad_if(img, lbl, self.patch_size, pad_value=None, background_label=0)
            imgs = torch.from_numpy(img).float().unsqueeze(0)  # [1,D,H,W]
            lbls = torch.from_numpy(lbl).long()
            lbls = F.one_hot(lbls, num_classes=self.num_classes).permute(3, 0, 1, 2).float()  # [C,D,H,W]
            spacing = torch.tensor(self.new_spacing, dtype=torch.float32)
            return imgs, lbls, spacing, img_path, pad_info
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __len__(self):
        if self.mode == 'train':
            return len(self.patch_index)
        return len(self.pairs)


def whs_collate_train(batch):
    all_imgs, all_labels, all_spacings, all_img_paths = [], [], [], []

    for img_list, lbl_list, spacing, img_path in batch:
        for img_t, lbl_t in zip(img_list, lbl_list):
            all_imgs.append(img_t)
            all_labels.append(lbl_t)
            all_spacings.append(spacing)
            all_img_paths.append(img_path)

    imgs = torch.stack(all_imgs, dim=0)
    labels = torch.stack(all_labels, dim=0)
    spacings = torch.stack(all_spacings, dim=0)

    return imgs, labels, spacings, all_img_paths


def whs_collate_test(batch):
    imgs, labels, spacings, img_paths, pad_infos = [], [], [], [], []

    for img, lbl, spacing, img_path, pad_info in batch:
        imgs.append(img)
        labels.append(lbl)
        spacings.append(spacing)
        img_paths.append(img_path)
        pad_infos.append(pad_info)

    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    spacings = torch.stack(spacings, dim=0)

    return imgs, labels, spacings, img_paths, pad_infos
