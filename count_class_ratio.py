import os
import json
import numpy as np
import SimpleITK as sitk

label_root = os.path.join(os.environ["WHS_DATA_ROOT"], "train", "label")

VALUE_TO_CLASS = {
    0: 0,     # Background
    500: 1,   # LV
    600: 2,   # RV
    420: 3,   # LA
    550: 4,   # RA
    205: 5,   # Myocardium
    820: 6,   # Aorta
    850: 7    # Pulmonary Artery
}

CENTER_CODE_TO_NAME = {
    '1': 'A',
    '2': 'B',
    '3': 'C',
    '5': 'E',
    '6': 'F',
    '7': 'G',
}

def get_center_from_filename(filename: str):
    digits = [c for c in filename if c.isdigit()]
    if len(digits) < 4:
        return None
    center_code = digits[0]
    return CENTER_CODE_TO_NAME.get(center_code, None)


def resample_label_itk(itk_img, new_spacing):
    """
    使用最近邻插值对 label 进行重采样
    """
    original_spacing = itk_img.GetSpacing()
    original_size = itk_img.GetSize()

    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    resampler.SetOutputDirection(itk_img.GetDirection())
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(itk_img)


# def collect_files(prefix):
#     files = []
#     for f in os.listdir(label_root):
#         if not f.endswith(".nii.gz"):
#             continue
#         if prefix == "ct" and f.startswith("ct") and f.endswith("label.nii.gz"):
#             files.append(os.path.join(label_root, f))
#         if prefix == "mr" and f.startswith("mr"):
#             files.append(os.path.join(label_root, f))
#     return files

def collect_files(prefix, center=None):
    files = []
    for f in os.listdir(label_root):
        if not f.endswith(".nii.gz"):
            continue

        if prefix == "ct":
            if not (f.startswith("ct") and f.endswith("label.nii.gz")):
                continue
        elif prefix == "mr":
            if not f.startswith("mr"):
                continue

        if center is not None:
            file_center = get_center_from_filename(f)
            if file_center != center:
                continue

        files.append(os.path.join(label_root, f))

    return files

def crop_to_non_bg(data):
    """返回裁剪到包含所有非背景区域的最小立方体"""
    mask = data > 0

    if not np.any(mask):
        return data

    coords = np.where(mask)
    z_min, y_min, x_min = np.min(coords, axis=1)
    z_max, y_max, x_max = np.max(coords, axis=1)

    return data[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

def compute_single_mode(prefix, use_crop, center=None):
    files = collect_files(prefix, center=center)
    num_files = len(files)

    print(f"\n========== 开始处理 {prefix.upper()} | Center = {center if center else 'ALL'} ==========")
    print(f"共找到 {num_files} 个图像文件。")

    class_counts = {cls: 0 for cls in VALUE_TO_CLASS.values()}
    total_voxels = 0

    for idx, file_path in enumerate(files, start=1):
        print(f"[{idx}/{num_files}] 正在处理: {os.path.basename(file_path)}")

        # nii = nib.load(file_path)
        # data = nii.get_fdata().astype(np.int32)

        itk_lbl = sitk.ReadImage(file_path)
        itk_lbl_rs = resample_label_itk(itk_lbl, new_spacing=(2.0, 2.0, 2.0))
        data = sitk.GetArrayFromImage(itk_lbl_rs).astype(np.int32)

        if use_crop:
            cropped = crop_to_non_bg(data)
        else:
            cropped = data

        for val, cls in VALUE_TO_CLASS.items():
            count = int(np.sum(cropped == val))
            class_counts[cls] += count
            total_voxels += count

    class_ratios = {
        cls: (class_counts[cls] / total_voxels if total_voxels > 0 else 0.0)
        for cls in class_counts
    }

    class_counts = {int(k): int(v) for k, v in class_counts.items()}
    total_voxels = int(total_voxels)

    print(f"=== {prefix.upper()} 统计完成 ===\n")

    return {
        "class_counts": class_counts,
        "total_voxels": total_voxels,
        "class_ratios": class_ratios,
    }

def merge_ct_mr_as_all(ct_result, mr_result):
    all_counts = {
        cls: ct_result["class_counts"][cls] + mr_result["class_counts"][cls]
        for cls in ct_result["class_counts"]
    }

    total_voxels = ct_result["total_voxels"] + mr_result["total_voxels"]

    all_ratios = {
        cls: all_counts[cls] / total_voxels if total_voxels > 0 else 0.0
        for cls in all_counts
    }

    return {
        "class_counts": all_counts,
        "total_voxels": total_voxels,
        "class_ratios": all_ratios,
    }

def compute_all(output_json, use_crop, center=None):
    ct_result = compute_single_mode("ct", use_crop=use_crop, center=center)
    mr_result = compute_single_mode("mr", use_crop=use_crop, center=center)
    all_result = merge_ct_mr_as_all(ct_result, mr_result)

    results = {
        "center": center if center else "ALL",
        "ct": ct_result,
        "mr": mr_result,
        "all": all_result,
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print("所有模式统计完成!")
    print(f"结果已保存到: {output_json}")


if __name__ == "__main__":
    use_crop = False
    if use_crop:
        a = 'foreground'
    else:
        a = 'volume'
    center = 'A'
    output_json = "label_statistics.json"
    compute_all(output_json, use_crop, center=center)
