import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableSincConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, padding=0, sample_rate=16000):
        super(DeformableSincConv1d, self).__init__()
        self.out_channels = out_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sample_rate = sample_rate

        if kernel_size % 2 == 0:
            self.kernel_size += 1

        self.low_hz = 0
        self.high_hz = sample_rate / 2

        hz = torch.linspace(self.low_hz, self.high_hz, self.out_channels + 1)
        self.band = nn.Parameter((hz[1:] - hz[:-1]).view(-1, 1))
        self.hz = nn.Parameter(hz[:-1].view(-1, 1))

        self.offset_conv = nn.Conv1d(in_channels, self.kernel_size, kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def sinc(self, t):
        return torch.where(t == 0, torch.ones_like(t), torch.sin(t) / t)

    def forward(self, x):
        x = x.unsqueeze(1)
        device = x.device
        self.hz.data = self.hz.data.clamp(self.low_hz, self.high_hz)
        self.band.data = self.band.data.clamp(3.0, self.high_hz - self.low_hz)

        N = self.kernel_size
        t_right = torch.linspace(1, (N - 1) / 2, steps=(N - 1) // 2, device=device) / self.sample_rate

        filters = []
        for i in range(self.out_channels):
            low = self.hz[i] - self.band[i] / 2
            high = self.hz[i] + self.band[i] / 2

            band_pass_left = (2 * high * self.sinc(2 * high * t_right)) - (2 * low * self.sinc(2 * low * t_right))
            band_pass_center = torch.ones(1, device=device)
            band_pass_right = band_pass_left.flip(dims=[0])

            band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right])
            band_pass = band_pass / (2 * self.band[i])
            filters.append(band_pass)

        filters = torch.stack(filters).view(self.out_channels, 1, self.kernel_size)

        offset = self.offset_conv(x)
        offset = offset.unsqueeze(0).chunk(1, dim=2)  # batch_size x offset_groups x length x kernel_size
        offset = torch.vstack(offset).moveaxis((0, 2), (1, 3))

        kernel_rfield = self.dilation * (self.kernel_size - 1) + 1
        dilated_positions = torch.linspace(0, kernel_rfield - 1, self.kernel_size, device=offset.device, dtype=offset.dtype)

        max_t0 = (offset.shape[-2] - 1) * self.stride
        t0s = torch.linspace(0, max_t0, offset.shape[-2], device=offset.device, dtype=offset.dtype).unsqueeze(-1)

        dilated_offsets_repeated = dilated_positions + offset

        T = t0s + dilated_offsets_repeated
        T = torch.max(T, t0s)
        T = torch.min(T, t0s + torch.max(dilated_positions))

        with torch.no_grad():
            U = torch.floor(T).to(torch.long)
            U = torch.clamp(U, min=0, max=x.shape[2] - 2)

            U = torch.stack([U, U + 1], dim=-1)
            if U.shape[1] < x.shape[1]:
                U = U.repeat(1, x.shape[1], 1, 1, 1)

        x = x.unsqueeze(-1).repeat(1, 1, 1, U.shape[-1])
        x = torch.stack([x.gather(index=U[:, :, :, i, :], dim=-2) for i in range(U.shape[-2])], dim=-1)

        G = torch.max(torch.zeros(U.shape, device=device), 1 - torch.abs(U - T.unsqueeze(-1)))

        mx = torch.multiply(G, x.moveaxis(-2, -1))

        deformed_output = torch.sum(mx, axis=-1)
        deformed_output = deformed_output.flatten(-2, -1)

        return F.conv1d(deformed_output, filters, stride=self.stride, padding=self.padding, dilation=1, bias=None)



class SincConv1d(nn.Module):
    def __init__(self, out_channels, kernel_size, dilation=1, stride=1, padding=0, sample_rate=16000):
        super(SincConv1d, self).__init__()
        self.out_channels = out_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sample_rate = sample_rate

        if kernel_size % 2 == 0:
            self.kernel_size += 1

        self.low_hz = 0
        self.high_hz = sample_rate / 2

        hz = torch.linspace(self.low_hz, self.high_hz, self.out_channels + 1)
        self.band = nn.Parameter((hz[1:] - hz[:-1]).view(-1, 1))
        self.hz = nn.Parameter(hz[:-1].view(-1, 1))

    def sinc(self, t):
        return torch.where(t == 0, torch.ones_like(t), torch.sin(t) / t)

    def forward(self, x):
        device = x.device
        x= x.unsqueeze(1)
        self.hz.data = self.hz.data.clamp(self.low_hz, self.high_hz)
        self.band.data = self.band.data.clamp(3.0, self.high_hz - self.low_hz)

        N = self.kernel_size
        t_right = torch.linspace(1, (N - 1) / 2, steps=(N - 1) // 2, device=device) / self.sample_rate

        filters = []
        for i in range(self.out_channels):
            low = self.hz[i] - self.band[i] / 2
            high = self.hz[i] + self.band[i] / 2

            band_pass_left = (2 * high * self.sinc(2 * high * t_right)) - (2 * low * self.sinc(2 * low * t_right))
            band_pass_center = torch.ones(1, device=device)
            band_pass_right = band_pass_left.flip(dims=[0])

            band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right])
            band_pass = band_pass / (2 * self.band[i])
            filters.append(band_pass)

        filters = torch.stack(filters).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, filters, stride=self.stride, padding=self.padding, dilation=1, bias=None)
