from collections import OrderedDict
from typing import Union, Tuple, List
import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize


def resample_data_or_seg(
    data: np.ndarray,
    new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    order: int = 3,
):

    assert data.ndim == 3, "Data must be (M, X, Y)"
    assert len(new_shape) == data.ndim - 1

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()

    else:
        resize_fn = resize
        kwargs = {"mode": "edge", "anti_aliasing": False}

    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        reshaped = []
        for c in range(data.shape[0]):
            reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None])
        reshaped_final_data = np.vstack(reshaped)

        return reshaped_final_data.astype(dtype_data)
    else:
        return data
