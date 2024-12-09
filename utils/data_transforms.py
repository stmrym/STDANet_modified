#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
'''ref: http://pytorch.org/docs/master/torchvision/transforms.html'''


import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random
import io
import logging
from utils.mmedit import blur_kernels
from fractions import Fraction
try:
    import av
    has_av = True
except ImportError:
    has_av = False

class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq_blur, seq_clear):
        for t in self.transforms:
            seq_blur, seq_clear = t(seq_blur, seq_clear)
        return seq_blur, seq_clear


class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, seq_blur, seq_clear):
        seq_blur  = [Image.fromarray(np.uint8(img)) for img in seq_blur]
        seq_clear = [Image.fromarray(np.uint8(img)) for img in seq_clear]
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            seq_blur  = [F.adjust_brightness(img, brightness_factor) for img in seq_blur]
            seq_clear = [F.adjust_brightness(img, brightness_factor) for img in seq_clear]

        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            seq_blur  = [F.adjust_contrast(img, contrast_factor) for img in seq_blur]
            seq_clear = [F.adjust_contrast(img, contrast_factor) for img in seq_clear]

        if self.saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            seq_blur  = [F.adjust_saturation(img, saturation_factor) for img in seq_blur]
            seq_clear = [F.adjust_saturation(img, saturation_factor) for img in seq_clear]

        if self.hue > 0:
            hue_factor = np.random.uniform(-self.hue, self.hue)
            seq_blur  = [F.adjust_hue(img, hue_factor) for img in seq_blur]
            seq_clear = [F.adjust_hue(img, hue_factor) for img in seq_clear]

        seq_blur  = [np.asarray(img) for img in seq_blur]
        seq_clear = [np.asarray(img) for img in seq_clear]

        seq_blur  = [img.clip(0,255) for img in seq_blur]
        seq_clear = [img.clip(0,255) for img in seq_clear]

        return seq_blur, seq_clear

class RandomColorChannel(object):
    def __call__(self, seq_blur, seq_clear):
        random_order = np.random.permutation(3)

        seq_blur  = [img[:,:,random_order] for img in seq_blur]
        seq_clear = [img[:,:,random_order] for img in seq_clear]

        return seq_blur, seq_clear

class RandomGaussianNoise(object):
    def __init__(self, gaussian_para):
        self.mu = gaussian_para[0]
        self.std_var = gaussian_para[1]

    def __call__(self, seq_blur, seq_clear):

        shape = seq_blur[0].shape
        gaussian_noise = np.random.normal(self.mu, self.std_var, shape)
        # only apply to blurry images
        seq_blur = [img + gaussian_noise for img in seq_blur]
        # [YNU] added np.float32
        seq_blur = [img.clip(0, 1).astype(np.float32) for img in seq_blur]

        return seq_blur, seq_clear

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
    def __call__(self, seq_blur, seq_clear):
        # [YNU] added np.float32
        seq_blur  = [(img/self.std -self.mean).astype(np.float32) for img in seq_blur]
        seq_clear = [(img/self.std -self.mean).astype(np.float32) for img in seq_clear]

        return seq_blur, seq_clear

class CenterCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""

        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, seq_blur, seq_clear):
        input_size_h, input_size_w, _ = seq_blur[0].shape
        x_start = int(round((input_size_w - self.crop_size_w) / 2.))
        y_start = int(round((input_size_h - self.crop_size_h) / 2.))

        seq_blur  = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_blur]
        seq_clear = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_clear]

        return seq_blur, seq_clear

class RandomCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, seq_blur, seq_clear):
        input_size_h, input_size_w, _ = seq_blur[0].shape
        x_start = random.randint(0, input_size_w - self.crop_size_w)
        y_start = random.randint(0, input_size_h - self.crop_size_h)

        seq_blur  = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_blur]
        seq_clear = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_clear]

        return seq_blur, seq_clear

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""

    def __call__(self, seq_blur, seq_clear):
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            seq_blur  = [np.copy(np.fliplr(img)) for img in seq_blur]
            seq_clear = [np.copy(np.fliplr(img)) for img in seq_clear]

        return seq_blur, seq_clear


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""

    def __call__(self, seq_blur, seq_clear):
        if random.random() < 0.5:
            seq_blur  = [np.copy(np.flipud(img)) for img in seq_blur]
            seq_clear = [np.copy(np.flipud(img)) for img in seq_clear]

        return seq_blur, seq_clear


class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, seq_blur, seq_clear):
        seq_blur  = [np.transpose(img, (2, 0, 1)) for img in seq_blur]
        seq_clear = [np.transpose(img, (2, 0, 1)) for img in seq_clear]
        # handle numpy array
        seq_blur_tensor  = [torch.from_numpy(img).float() for img in seq_blur]
        seq_clear_tensor = [torch.from_numpy(img).float() for img in seq_clear]
        
        return seq_blur_tensor, seq_clear_tensor


# for LQ only transforms by mmedit

class UnsharpMasking:
    """Apply unsharp masking to an image or a sequence of images.

    Args:
        kernel_size (int): The kernel_size of the Gaussian kernel.
        sigma (float): The standard deviation of the Gaussian.
        weight (float): The weight of the "details" in the final output.
        threshold (float): Pixel differences larger than this value are
            regarded as "details".
        keys (list[str]): The keys whose values are processed.

    Added keys are "xxx_unsharp", where "xxx" are the attributes specified
    in "keys".
    """

    def __init__(self, kernel_size_l, sigma, weight_prob, threshold):

        self.kernel_size_l = kernel_size_l
        self.sigma = sigma
        self.weight_prob = weight_prob
        self.threshold = threshold

    def _unsharp_masking(self, imgs):

        kernel_size = random.choice(self.kernel_size_l)
        kernel = cv2.getGaussianKernel(kernel_size, self.sigma)
        self.kernel = np.matmul(kernel, kernel.transpose())
        self.weight = np.random.uniform(self.weight_prob[0], self.weight_prob[1])
        
        
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        outputs = []
        for img in imgs:
            residue = img - cv2.filter2D(img, -1, self.kernel)

            smoothing = cv2.getGaussianKernel(3, 0)
            self.smoothing_kernel = np.matmul(smoothing, smoothing.transpose())

            smoothed =  cv2.filter2D(img, -1, self.smoothing_kernel)
            laplacian = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3)

            # mask = np.float32(np.abs(residue) * 255 > self.threshold)    
            mask = np.float32(np.abs(laplacian) * 255 > self.threshold)    
            soft_mask = cv2.filter2D(mask, -1, self.kernel)

            sharpened = np.clip(img + self.weight * residue, 0, 1)

            outputs.append(soft_mask * sharpened + (1 - soft_mask) * img)

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def __call__(self, seq_blur, seq_clear):
        seq_blur = self._unsharp_masking(seq_blur)
        return seq_blur, seq_clear

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(kernel_size={self.kernel_size_l}, '
                     f'sigma={self.sigma}, weight={self.weight_prob}, '
                     f'threshold={self.threshold})')
        return repr_str



class RandomBlur:
    """Apply random blur to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params):
        self.params = params

    def get_kernel(self, num_kernels):
        kernel_type = np.random.choice(
            self.params['kernel_list'], p=self.params['kernel_prob'])
        kernel_size = random.choice(self.params['kernel_size'])

        sigma_x_range = self.params.get('sigma_x', [0, 0])
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        sigma_x_step = self.params.get('sigma_x_step', 0)

        sigma_y_range = self.params.get('sigma_y', [0, 0])
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        sigma_y_step = self.params.get('sigma_y_step', 0)

        rotate_angle_range = self.params.get('rotate_angle', [-np.pi, np.pi])
        rotate_angle = np.random.uniform(rotate_angle_range[0],
                                         rotate_angle_range[1])
        rotate_angle_step = self.params.get('rotate_angle_step', 0)

        beta_gau_range = self.params.get('beta_gaussian', [0.5, 4])
        beta_gau = np.random.uniform(beta_gau_range[0], beta_gau_range[1])
        beta_gau_step = self.params.get('beta_gaussian_step', 0)

        beta_pla_range = self.params.get('beta_plateau', [1, 2])
        beta_pla = np.random.uniform(beta_pla_range[0], beta_pla_range[1])
        beta_pla_step = self.params.get('beta_plateau_step', 0)

        omega_range = self.params.get('omega', None)
        omega_step = self.params.get('omega_step', 0)
        if omega_range is None:  # follow Real-ESRGAN settings if not specified
            if kernel_size < 13:
                omega_range = [np.pi / 3., np.pi]
            else:
                omega_range = [np.pi / 5., np.pi]
        omega = np.random.uniform(omega_range[0], omega_range[1])

        # determine blurring kernel
        kernels = []
        for _ in range(0, num_kernels):
            kernel = blur_kernels.random_mixed_kernels(
                [kernel_type],
                [1],
                kernel_size,
                [sigma_x, sigma_x],
                [sigma_y, sigma_y],
                [rotate_angle, rotate_angle],
                [beta_gau, beta_gau],
                [beta_pla, beta_pla],
                [omega, omega],
                None,
            )
            kernels.append(kernel)

            # update kernel parameters
            sigma_x += np.random.uniform(-sigma_x_step, sigma_x_step)
            sigma_y += np.random.uniform(-sigma_y_step, sigma_y_step)
            rotate_angle += np.random.uniform(-rotate_angle_step,
                                              rotate_angle_step)
            beta_gau += np.random.uniform(-beta_gau_step, beta_gau_step)
            beta_pla += np.random.uniform(-beta_pla_step, beta_pla_step)
            omega += np.random.uniform(-omega_step, omega_step)

            sigma_x = np.clip(sigma_x, sigma_x_range[0], sigma_x_range[1])
            sigma_y = np.clip(sigma_y, sigma_y_range[0], sigma_y_range[1])
            rotate_angle = np.clip(rotate_angle, rotate_angle_range[0],
                                   rotate_angle_range[1])
            beta_gau = np.clip(beta_gau, beta_gau_range[0], beta_gau_range[1])
            beta_pla = np.clip(beta_pla, beta_pla_range[0], beta_pla_range[1])
            omega = np.clip(omega, omega_range[0], omega_range[1])

        return kernels

    def _apply_random_blur(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # get kernel and blur the input
        kernels = self.get_kernel(num_kernels=len(imgs))
        imgs = [
            cv2.filter2D(img, -1, kernel)
            for img, kernel in zip(imgs, kernels)
        ]

        if is_single_image:
            imgs = imgs[0]

        return imgs

    def __call__(self, seq_blur, seq_clear):
        if np.random.uniform() <= self.params.get('prob', 1):
            seq_blur = self._apply_random_blur(seq_blur)
        return seq_blur, seq_clear

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params})')
        return repr_str


class RandomNoise:
    """Apply random noise to the input.

    Currently support Gaussian noise and Poisson noise.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params):
        self.params = params

    def _apply_gaussian_noise(self, imgs):
        sigma_range = self.params['gaussian_sigma']
        sigma = np.random.uniform(sigma_range[0], sigma_range[1]) / 255.

        sigma_step = self.params.get('gaussian_sigma_step', 0)

        gray_noise_prob = self.params['gaussian_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = np.float32(np.random.randn(*(img.shape))) * sigma
            if is_gray_noise:
                noise = noise[:, :, :1]
            outputs.append(img + noise)

            # update noise level
            sigma += np.random.uniform(-sigma_step, sigma_step) / 255.
            sigma = np.clip(sigma, sigma_range[0] / 255.,
                            sigma_range[1] / 255.)

        return outputs

    def _apply_poisson_noise(self, imgs):
        scale_range = self.params['poisson_scale']
        scale = np.random.uniform(scale_range[0], scale_range[1])

        scale_step = self.params.get('poisson_scale_step', 0)

        gray_noise_prob = self.params['poisson_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = img.copy()
            if is_gray_noise:
                noise = cv2.cvtColor(noise[..., [2, 1, 0]], cv2.COLOR_BGR2GRAY)
                noise = noise[..., np.newaxis]
            noise = np.clip((noise * 255.0).round(), 0, 255) / 255.
            unique_val = 2**np.ceil(np.log2(len(np.unique(noise))))
            noise = np.random.poisson(noise * unique_val) / unique_val - noise

            outputs.append(img + noise * scale)

            # update noise level
            scale += np.random.uniform(-scale_step, scale_step)
            scale = np.clip(scale, scale_range[0], scale_range[1])

        return outputs

    def _apply_random_noise(self, imgs):
        noise_type = np.random.choice(
            self.params['noise_type'], p=self.params['noise_prob'])

        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        if noise_type.lower() == 'gaussian':
            imgs = self._apply_gaussian_noise(imgs)
        elif noise_type.lower() == 'poisson':
            imgs = self._apply_poisson_noise(imgs)
        else:
            raise NotImplementedError(f'"noise_type" [{noise_type}] is '
                                      'not implemented.')

        if is_single_image:
            imgs = imgs[0]

        return imgs

    def __call__(self, seq_blur, seq_clear):
        if np.random.uniform() <= self.params.get('prob', 1):
            seq_blur = self._apply_random_noise(seq_blur)
        return seq_blur, seq_clear

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params})')
        return repr_str


class RandomJPEGCompression:
    """Apply random JPEG compression to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params):
        self.params = params

    def _apply_random_compression(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # determine initial compression level and the step size
        quality = self.params['quality']
        quality_step = self.params.get('quality_step', 0)
        jpeg_param = round(np.random.uniform(quality[0], quality[1]))

        # apply jpeg compression
        outputs = []
        for img in imgs:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_param]
            _, img_encoded = cv2.imencode('.jpg', img * 255., encode_param)
            outputs.append(np.float32(cv2.imdecode(img_encoded, 1)) / 255.)

            # update compression level
            jpeg_param += np.random.uniform(-quality_step, quality_step)
            jpeg_param = round(np.clip(jpeg_param, quality[0], quality[1]))

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def __call__(self, seq_blur, seq_clear):
        if np.random.uniform() <= self.params.get('prob', 1):
            seq_blur = self._apply_random_compression(seq_blur)
        return seq_blur, seq_clear

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params})')
        return repr_str


class RandomVideoCompression:
    """Apply random video compression to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params):
        assert has_av, 'Please install av to use video compression.'
        self.params = params
        logging.getLogger('libav').setLevel(50)

    def _apply_random_compression(self, imgs):
        codec_prob = [Fraction(prob) for prob in self.params['codec_prob']]
        codec = random.choices(self.params['codec'],
                               weights=codec_prob)[0]
        bitrate = self.params['bitrate']
        bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = imgs[0].shape[0]
            stream.width = imgs[0].shape[1]
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = bitrate

            for img in imgs:
                img = (255 * img).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame.pict_type = 'NONE'
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    outputs.append(
                        frame.to_rgb().to_ndarray().astype(np.float32) / 255.)

        return outputs

    def __call__(self, seq_blur, seq_clear):
        if np.random.uniform() <= self.params.get('prob', 1):
            seq_blur = self._apply_random_compression(seq_blur)
        return seq_blur, seq_clear

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params})')
        return repr_str