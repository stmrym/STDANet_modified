from skimage.metrics import structural_similarity as compare_ssim
from utils.network_utils import AverageMeter

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