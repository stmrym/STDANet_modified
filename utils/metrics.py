import cv2
import numpy as np
import math
import os
import lpips
from skimage.metrics import structural_similarity as compare_ssim
from scipy.ndimage.filters import convolve
from scipy.special import gamma
import torch
from utils.network_utils import AverageMeter

import cpbd
from skimage.transform import resize
from utils.utils_cuda import util
from utils.utils_cuda import AnisoSetEst 
# from utils.utils_cuda.denoise_cuda import Denoise
from utils.utils_cuda.compute_ncc import compute_ncc
from utils.utils.compute_ncc import compute_ncc as compute_ncc_cpu

from utils.utils.pyr_ring import align, grad_ring



class PSNR(AverageMeter):
    def __init__(self, crop_border=0, max_order=255.0):
        super(PSNR, self).__init__()
        self.crop_border = crop_border
        self.max_order = max_order

    def calculate(self, img1, img2, **kwargs):
        """Calculate PSNR (Peak Signal-to-Noise Ratio).

        Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            img (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
            test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

        Returns:
            float: PSNR result.
        """

        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')

        if self.crop_border != 0:
            img1 = img1[self.crop_border:-self.crop_border, self.crop_border:-self.crop_border, ...]
            img2 = img2[self.crop_border:-self.crop_border, self.crop_border:-self.crop_border, ...]

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 10. * np.log10(self.max_order * self.max_order / mse)


class SSIM(AverageMeter):
    def __init__(self, data_range=255.0, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False):
        super(SSIM, self).__init__()
        self.data_range = data_range
        self.channel_axis = channel_axis
        self.gaussian_weights = gaussian_weights
        self.sigma = sigma
        self.use_sample_covariance = use_sample_covariance

    def calculate(self, img1, img2, **kwargs):
        ssim = compare_ssim(img2, img1, **vars(self))
        return ssim

class LPIPS(AverageMeter):
    def __init__(self, val_range=255.0):
        super(LPIPS, self).__init__()
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        self.val_range = val_range

    def calculate(self, img1, img2, **kwargs):
        img1 = self._convert_to_tensor(img1, self.val_range) 
        img2 = self._convert_to_tensor(img2, self.val_range) 
        loss = self.loss_fn_alex(img1.permute(2,0,1).unsqueeze(0), img2.permute(2,0,1).unsqueeze(0)).mean().detach().cpu()
        return loss

    def _convert_to_tensor(self, img, val_range):
        if not isinstance(img, torch.Tensor):
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.astype(np.float32) / val_range).to(self.device)
            else:
                print('unsupported format')
                exit()
        return img

class NIQE(AverageMeter):
    def __init__(self, crop_border, input_order='HWC', convert_to='y', **kwargs):
        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to = convert_to

    def calculate(self, img1, **kwargs):
        """Calculate NIQE (Natural Image Quality Evaluator) metric.

        Ref: Making a "Completely Blind" Image Quality Analyzer.
        This implementation could produce almost the same results as the official
        MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

        > MATLAB R2021a result for tests/data/baboon.png: 5.72957338 (5.7296)
        > Our re-implementation result for tests/data/baboon.png: 5.7295763 (5.7296)

        We use the official params estimated from the pristine dataset.
        We use the recommended block size (96, 96) without overlaps.

        Args:
            img (ndarray): Input image whose quality needs to be computed.
                The input image must be in range [0, 255] with float/int type.
                The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
                If the input order is 'HWC' or 'CHW', it will be converted to gray
                or Y (of YCbCr) image according to the ``convert_to`` argument.
            crop_border (int): Cropped pixels in each edge of an image. These
                pixels are not involved in the metric calculation.
            input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
                Default: 'HWC'.
            convert_to (str): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
                Default: 'y'.

        Returns:
            float: NIQE result.
        """
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        # we use the official params estimated from the pristine dataset.
        niqe_pris_params = np.load(os.path.join(ROOT_DIR, 'niqe_pris_params.npz'))
        mu_pris_param = niqe_pris_params['mu_pris_param']
        cov_pris_param = niqe_pris_params['cov_pris_param']
        gaussian_window = niqe_pris_params['gaussian_window']

        img = img1.astype(np.float32)
        if self.input_order != 'HW':
            img = reorder_image(img, input_order=self.input_order)
            if self.convert_to == 'y':
                img = to_y_channel(img)
            elif self.convert_to == 'gray':
                img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
            img = np.squeeze(img)

        if self.crop_border != 0:
            img = img[self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        # round is necessary for being consistent with MATLAB's result
        img = img.round()

        niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)

        return niqe_result


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img

def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def cubic(x):
    """cubic function used for calculate_weights_indices."""
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (((absx > 1) *
                                                                                    (absx <= 2)).type_as(absx))

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    """Calculate weights and indices, used for imresize function.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialisaing (bool): Whether to apply anti-aliasing when downsampling.
    """

    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialias
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

@torch.no_grad()
def imresize(img, scale, antialiasing=True):
    """imresize function same as MATLAB.

    It now only supports bicubic.
    The same scale applies for both height and width.

    Args:
        img (Tensor | Numpy array):
            Tensor: Input image with shape (c, h, w), [0, 1] range.
            Numpy: Input image with shape (h, w, c), [0, 1] range.
        scale (float): Scale factor. The same scale applies for both height
            and width.
        antialisaing (bool): Whether to apply anti-aliasing when downsampling.
            Default: True.

    Returns:
        Tensor: Output image with shape (c, h, w), [0, 1] range, w/o round.
    """
    squeeze_flag = False
    if type(img).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if img.ndim == 2:
            img = img[:, :, None]
            squeeze_flag = True
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if img.ndim == 2:
            img = img.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = img.size()
    out_h, out_w = math.ceil(in_h * scale), math.ceil(in_w * scale)
    kernel_width = 4
    kernel = 'cubic'

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(in_h, out_h, scale, kernel, kernel_width,
                                                                            antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(in_w, out_w, scale, kernel, kernel_width,
                                                                            antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(img)

    sym_patch = img[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2

def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)

def compute_feature(block):
    """Compute features.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        list: Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        # Eq. 8
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat

def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Note that we do not include block overlap height and width, since they are
    always 0 in the official implementation.

    For good performance, it is advisable by the official implementation to
    divide the distorted image in to the same size patched as used for the
    construction of multivariate Gaussian model.

    Args:
        img (ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    assert img.ndim == 2, ('Input image must be a gray or Y (of YCbCr) image with shape (h, w).')
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                                    idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        if scale == 1:
            img = imresize(img / 255., scale=0.5, antialiasing=True)
            img = img * 255.

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    # use nancov. ref: https://ww2.mathworks.cn/help/stats/nancov.html
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param), np.transpose((mu_pris_param - mu_distparam)))

    quality = np.sqrt(quality)
    quality = float(np.squeeze(quality))
    return quality







class LR(AverageMeter):
    def __init__(self, device, **kwargs):
        super(LR, self).__init__()
        self.device = device

    def _img2tensor(self, img):
        '''
        ndarray (BGR) with shape(H, W, C) -> tensor (RGB) with shape (1, C, H, W)
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img, device=self.device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        return img_tensor


    def _tensor2img(self, img_tensor):
        '''
        tensor (RGB) with shape (1, C, H, W) -> ndarray (BGR) with shape (H, W, C)
        '''
        img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


    def _tensor_rgb2gray(self, img_tensor):
        '''
        tensor (RGB) with shape (..., C, H, W) -> tensor (Gray) with shape (..., H, W)
        '''
        weights = torch.tensor([0.299, 0.5870, 0.1140], device=img_tensor.device)
        gray_tensor = torch.tensordot(img_tensor, weights, dims=([-3], [0]))
        return gray_tensor

    # @stop_watch
    def calculate(self, img1, img2, **kwargs):
        '''
        img1: deblurred image: ndarray (BGR) [0, 255] with shape (H, W, C)
        img2: blurred image: ndarray (BGR) [0, 255] with shape (H, W, C)
        '''
        img1_tensor = self._img2tensor((img1/255).astype(np.float32))
        img2_tensor = self._img2tensor((img2/255).astype(np.float32))

        score, features = self._measure(deblurred=img1_tensor, blurred=img2_tensor)
        # print(score, features)
        return score
    

    def _measure(self, deblurred, blurred):
        '''
        deblurred: torch.Tensor (RGB) [0,1] with shape (B, C, H, W)
        blurred: torch.Tensor (RGB) [0,1] with shape (B, C, H, W)
        '''
        features = {}

        features['sparsity'] = self._sparsity(deblurred)
        features['smallgrad'] = self._smallgrad(deblurred)
        features['metric_q'] = self._metric_q(deblurred)

        # denoise = Denoise(self.device)
        # denoised = denoise.denoise(deblurred)
        denoised = deblurred

        # denoised_np = self._tensor2img(denoised)
        # cv2.imwrite('denoised.png', np.clip(denoised_np*255, 0, 255).astype(np.uint8))
        
        features['auto_corr'] = self._auto_corr(denoised)

        # print(denoised.shape)
        # features['auto_corr'] = self._auto_corr_cpu(denoised)

        features['norm_sps'] = self._norm_sparsity(denoised)

        denoised = self._tensor2img(denoised)
        blurred = self._tensor2img(blurred)
        deblurred = self._tensor2img(deblurred)


        features['cpbd'] = self._calc_cpbd(denoised)
        features['pyr_ring'] = self._pyr_ring(denoised, blurred)
        features['saturation'] = self._saturation(deblurred)
        
        score = (features['sparsity']   * -8.70515   +
                features['smallgrad']  * -62.23820  +
                features['metric_q']   * -0.04109   +
                features['auto_corr']  * -0.82738   +
                features['norm_sps']   * -13.90913  +
                features['cpbd']       * -2.20373   +
                features['pyr_ring']   * -149.19139 +
                features['saturation'] * -6.62421)

        return score, features
    

    # @stop_watch
    def _sparsity(self, img):
        '''
        img: torch.Tensor (RGB) with shape (B, C, H, W)
        '''
        dx, dy = util.gradient_cuda(img)
        d = torch.sqrt(dx**2 + dy**2)
        
        norm_l = torch.stack([util.mean_norm_cuda(d[:,c], 0.66) for c in range(d.shape[1])])
        result = torch.sum(norm_l)
        return result.cpu().item()


    # @stop_watch
    def _smallgrad(self, img):
        '''
        img: torch.Tensor (RGB) with shape (B, C, H, W)
        '''
        d = torch.zeros_like(img[:, 0, :, :])

        for c in range(img.shape[1]):
            dx, dy = util.gradient_cuda(img[:, c, :, :])
            d += torch.sqrt(dx**2 + dy**2)
        d /= 3
        
        sorted_d, _ = torch.sort(d.reshape(-1))
        n = max(int(sorted_d.numel() * 0.3), 10)
        result = util.my_sd_cuda(sorted_d[:n], 0.1)
        
        return result.cpu().item()
    

    # @stop_watch
    def _metric_q(self, img):
        '''
        img: torch.Tensor (RGB) with shape (B, C, H, W)
        '''
        PATCH_SIZE = 8
        img = self._tensor_rgb2gray(img) * 255
        result = -AnisoSetEst.MetricQ_cuda(img, PATCH_SIZE)
        return result.cpu().item()
    

    # @stop_watch
    def _auto_corr(self, img):
        '''
        img: torch.Tensor (RGB) with shape (B, C, H, W)
        '''
        img = self._tensor_rgb2gray(img)

        MARGIN = 50

        ncc_orig = compute_ncc(img, img, MARGIN)

        sizes = ncc_orig.size()
        assert sizes[0] == sizes[1]
        assert sizes[0] % 2 == 1

        # 半径を計算
        radius = sizes[0] // 2

        # 距離行列を計算
        y_dists, x_dists = torch.meshgrid(torch.arange(sizes[0], device=img.device), torch.arange(sizes[1], device=img.device))
        dists = torch.sqrt((y_dists - radius).float() ** 2 + (x_dists - radius).float() ** 2)

        # ncc の絶対値を取得
        ncc = torch.abs(ncc_orig)

        # max_m の初期化
        max_m = torch.zeros(1 + radius, device=img.device)

        # 各半径に対して計算
        for r in range(0, radius + 1):
            w = torch.abs(dists - r)
            w = torch.min(w, torch.tensor(1.0, device=img.device))
            w = 1 - w
            max_m[r] = torch.max(ncc[w > 0])

        # max_m の最初の要素を 0 に設定
        max_m[0] = 0

        # 結果を計算
        result = torch.sum(max_m)

        return result.cpu().item()



    # @stop_watch
    def _auto_corr_cpu(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        MARGIN = 50

        ncc_orig = compute_ncc_cpu(img, img, MARGIN)


        sizes = ncc_orig.shape 
        assert sizes[0] == sizes[1] 
        assert sizes[0] % 2 == 1 
        radius = sizes[0] // 2 
        y_dists, x_dists = np.meshgrid(np.arange(sizes[0]), np.arange(sizes[1]), indexing='ij') 
        dists = np.sqrt((y_dists - radius) ** 2 + (x_dists - radius) ** 2) 
        ncc = np.abs(ncc_orig) 
        max_m = np.zeros(radius + 1) 
        for r in range(radius + 1): 
            w = np.abs(dists - r) 
            w = np.minimum(w, 1) 
            w = 1 - w 
            max_m[r] = np.max(ncc[w > 0]) 

        max_m[0] = 0 
        result = np.sum(max_m)

        return result


    # @stop_watch
    def _norm_sparsity(self, img):
        '''
        img: torch.Tensor (RGB) with shape (B, C, H, W)
        '''
        img = self._tensor_rgb2gray(img)


        dx, dy = util.gradient_cuda(img)
        d = torch.sqrt(dx**2 + dy**2)
        
        result = util.mean_norm_cuda(d, 1.0) / util.mean_norm_cuda(d, 2.0)
        return result.cpu().item()        
        


    # @stop_watch
    def _norm_sparsity_cpu(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dx, dy = np.gradient(img)
        d = np.sqrt(dx**2 + dy**2)

        result = util_cpu.mean_norm(d, 1.0) / util_cpu.mean_norm(d, 2.0)
        return result

    # @stop_watch
    def _calc_cpbd(self, img):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return -cpbd.compute(img)

    # @stop_watch
    def _pyr_ring(self, img, blurred):

        img, blurred = align(img, blurred, True)
        height, width, color_count = img.shape

        result = 0.0
        sizes = []
        j = 0
        while True:
            coef = 0.5 ** j
            cur_height = round(height * coef)
            cur_width = round(width * coef)
            if min(cur_height, cur_width) < 16:
                break
            sizes.append([j, cur_width, cur_height])

            cur_img = resize(img, (cur_height, cur_width), order=1)
            cur_blurred = resize(blurred, (cur_height, cur_width), order=1)

            diff = grad_ring(cur_img, cur_blurred)
            if j > 0:
                result += np.mean(diff)

            j += 1

        return result

    # @stop_watch
    def _saturation(self, img):
        # 各ピクセルの最大値を計算
        max_values = np.max(img, axis=2)
        
        # 最大値が10/255以下のマスクを作成
        mask_low = (max_values <= 10.0 / 255.0)
        result_low = np.sum(mask_low.astype(np.float64)) / max_values.size

        # 各ピクセルの最小値を計算
        min_values = np.min(img, axis=2)
        
        # 最小値が1 - 10/255以上のマスクを作成
        mask_high = (min_values >= 1.0 - (10.0 / 255.0))
        result_high = np.sum(mask_high.astype(np.float64)) / min_values.size

        # 結果を計算
        result = result_low + result_high
        
        return result

