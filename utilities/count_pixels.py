import torch
import numpy as np


def check_ratio(ratio):
    if not (0 <= ratio <= 1):
        raise ValueError(f"ratio must be between 0 and 1 (inclusive), but got {ratio}")
    
def spearman_correlation(x, y):
    x_rank = x.argsort().argsort()
    y_rank = y.argsort().argsort()
    corr_matrix = np.corrcoef(x_rank, y_rank)
    return corr_matrix[0, 1]

def count_pixels_sigma(u_noised_sigma1, u_noised_sigma2, sigma1, sigma2, d, device, threshold=None, du=False):
    # ---- to tensor & device ----
    u_noised_sigma1, u_noised_sigma2 = map(lambda x: torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x.to(device), [u_noised_sigma1, u_noised_sigma2])
    
    # ---- ratio ----
    delta_u = u_noised_sigma2[d <= threshold] - u_noised_sigma1[d <= threshold]   # [N]
    delta_sigma = sigma2 - sigma1
    mul = delta_u * delta_sigma
    num = (mul > 0).sum().item()
    denom = np.sum(d <= threshold)
    ratio = num / denom if denom > 0 else 0.0
    check_ratio(ratio)

    # ---- corr ----
    u1 = u_noised_sigma1[d <= threshold]
    u2 = u_noised_sigma2[d <= threshold]
    sigma_np = np.array([sigma1, sigma2])
    corr = 0.0

    for i in range(len(u1)):
        u_np = np.array([u1[i].cpu().float().item(), u2[i].cpu().float().item()])
        corr += spearman_correlation(u_np, sigma_np)
    corr = corr / len(u1) if len(u1) > 0 else 0.0

    if du:
        return ratio, corr, delta_u.cpu().numpy()
    else:
        return ratio, corr

def count_pixels_grad(u, grad, d, labels, batch_size, device, threshold=0, interval=0.04, epsilon=1e-4):
    u, grad, d, labels = map(
        lambda x: torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x.to(device),
        [u, grad, d, labels]
    )
    u = u.view(batch_size, -1)  # [N, HW]
    grad = grad.view(batch_size, -1)  # [N, HW]
    d = d.view(batch_size, -1)  # [N, HW]
    labels = labels.view(batch_size, labels.shape[1], -1)  # [N, C, HW]

    ratio = 0.0
    corr = 0.0
    valid_batch_count = 0

    for i in range(batch_size):
        boundary_mask = d[i, :] <= threshold
        if boundary_mask.sum() == 0:
            continue

        class_probs = labels[i, :, boundary_mask]
        _, class_indices = torch.max(class_probs, dim=0)
        unique_classes = torch.unique(class_indices)

        sample_ratio = 0.0
        sample_corr = 0.0
        valid_class_count = 0

        for class_idx in unique_classes:
            class_mask = class_indices == class_idx
            if class_mask.sum() < 2:
                continue

            u_class = u[i, boundary_mask][class_mask]
            grad_class = grad[i, boundary_mask][class_mask]

            # 排序 + 等间隔采样
            grad_sorted, idx_sorted = torch.sort(grad_class)
            u_sorted = u_class[idx_sorted]

            g_min, g_max = grad_sorted[0], grad_sorted[-1]
            target_vals = torch.arange(g_min, g_max + interval, interval, device=device)
            diff = torch.abs(grad_sorted.unsqueeze(1) - target_vals.unsqueeze(0))
            idx_nearest = torch.argmin(diff, dim=0)
            idx_sample = torch.unique(idx_nearest)

            u_sampled = u_sorted[idx_sample]
            grad_sampled = grad_sorted[idx_sample]
            L_sampled = len(u_sampled)

            if L_sampled < 2:
                continue

            # Spearman
            corr_class = spearman_correlation(u_sampled.cpu().numpy(), grad_sampled.cpu().numpy())

            # pairwise difference
            u_diff = u_sampled.unsqueeze(0) - u_sampled.unsqueeze(0).t()
            grad_diff = grad_sampled.unsqueeze(0) - grad_sampled.unsqueeze(0).t()

            # 只取上三角 (i<j)
            mask = torch.triu(torch.ones(L_sampled, L_sampled, device=device), diagonal=1).bool()
            u_diff = u_diff[mask]
            grad_diff = grad_diff[mask]

            mul = u_diff * grad_diff
            num = (mul < 0).sum() + ((grad_diff == 0) & (abs(u_diff) < epsilon)).sum()
            total_pairs = L_sampled * (L_sampled - 1) // 2

            ratio_class = num.item() / total_pairs if total_pairs > 0 else 0.0

            sample_ratio += ratio_class
            sample_corr += corr_class
            valid_class_count += 1

        if valid_class_count > 0:
            ratio += sample_ratio / valid_class_count
            corr += sample_corr / valid_class_count
            valid_batch_count += 1

    return (ratio / valid_batch_count if valid_batch_count > 0 else 0.0,
            corr / valid_batch_count if valid_batch_count > 0 else 0.0)

def count_pixels_d_chunk(u, d, labels, batch_size, device, threshold=None, chunk_size=100, epsilon=1e-4):
    u, d, labels = map(
        lambda x: torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x.to(device),
        [u, d, labels]
    )

    u = u.view(batch_size, -1)           # [N, HW]
    d = d.view(batch_size, -1)           # [N, HW]
    labels = labels.view(batch_size, labels.shape[1], -1)  # [N, C, HW]

    ratio = 0.0
    corr = 0.0
    valid_batch_count = 0

    for i in range(batch_size):
        # 边界像素掩码
        boundary_mask = d[i, :] <= threshold  # [HW]
        if boundary_mask.sum() == 0:
            continue

        # 获取类别概率 [C, L_boundary]
        class_probs = labels[i, :, boundary_mask]  
        _, class_indices = torch.max(class_probs, dim=0)  # [L_boundary]
        unique_classes = torch.unique(class_indices)

        sample_ratio = 0.0
        sample_corr = 0.0
        valid_class_count = 0

        for class_idx in unique_classes:
            class_mask = class_indices == class_idx
            if class_mask.sum() < 2:
                continue

            # 取当前类别的 u, d
            u_class = u[i, boundary_mask][class_mask]  # [L_class]
            d_class = d[i, boundary_mask][class_mask]  # [L_class]
            L_class = len(d_class)

            # Spearman 相关性
            corr_class = spearman_correlation(d_class.cpu().numpy(), u_class.cpu().numpy())

            # ratio 统计
            num_conflict = 0
            total_pairs = 0

            for start in range(0, L_class, chunk_size):
                end = min(start + chunk_size, L_class)
                N = end - start
                if N < 2:
                    continue

                mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
                d_chunk = d_class[start:end].unsqueeze(0) - d_class[start:end].unsqueeze(0).t()
                u_chunk = u_class[start:end].unsqueeze(0) - u_class[start:end].unsqueeze(0).t()
                
                d_diff = d_chunk[mask]
                u_diff = u_chunk[mask]
                
                mul = d_diff * u_diff
                num = (mul < 0).sum() + ((d_diff == 0) & (abs(u_diff) < epsilon)).sum()


                num_conflict += num.item()
                total_pairs += N * (N - 1) // 2

            ratio_class = num_conflict / total_pairs if total_pairs > 0 else 0.0

            sample_ratio += ratio_class
            sample_corr += corr_class
            valid_class_count += 1

        if valid_class_count > 0:
            ratio += sample_ratio / valid_class_count
            corr += sample_corr / valid_class_count
            valid_batch_count += 1

    return (ratio / valid_batch_count if valid_batch_count > 0 else 0.0,
            corr / valid_batch_count if valid_batch_count > 0 else 0.0)

def main():
    u = np.array([1, 1])
    sigma = np.array([10, 50])
    corr = spearman_correlation(u, sigma)
    print(corr)

if __name__ == "__main__":
    main()
    