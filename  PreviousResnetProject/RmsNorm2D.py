import torch
import torch.nn as nn

class RMSNorm2D(nn.Module):
    def __init__(self, num_features, p=-1., eps=1e-8, bias=False):
        """
        Root Mean Square Layer Normalization for 2D inputs

        :param num_features: number of features (channels)
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps: epsilon value, default 1e-8
        :param bias: whether to use bias term for RMSNorm, disabled by default
        """
        super(RMSNorm2D, self).__init__()

        self.eps = eps
        self.num_features = num_features
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(num_features, 1, 1))
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(num_features, 1, 1))

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=(1, 2, 3), keepdim=True)
            d_x = self.num_features * x.shape[2] * x.shape[3]
        else:
            partial_size = int(self.num_features * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.num_features - partial_size], dim=1)

            norm_x = partial_x.norm(2, dim=(1, 2, 3), keepdim=True)
            d_x = partial_size * x.shape[2] * x.shape[3]

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed