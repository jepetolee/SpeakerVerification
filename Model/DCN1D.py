import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Callable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _reverse_repeat_tuple
import math
from torch.nn import init
from typing import Optional

def efficient_linterpolate(
        x,
        offsets,
        kernel_size,
        dilation,
        stride,
        dilated_positions=None,
        device="cpu",
        _test=False,
        unconstrained=False
):
    assert x.device == offsets.device, "x and offsets must be on same device"
    kernel_rfield = dilation * (kernel_size - 1) + 1
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, kernel_rfield - 1, kernel_size, device=offsets.device,
                                           dtype=offsets.dtype)  # kernel_size

    max_t0 = (offsets.shape[-2] - 1) * stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2], device=offsets.device, dtype=offsets.dtype).unsqueeze(
        -1)  # out_length x 1
    dilated_offsets_repeated = dilated_positions + offsets

    T = t0s + dilated_offsets_repeated  # batch_size x channels x out_length x kernel_size
    if not unconstrained:
        T = torch.max(T, t0s)
        T = torch.min(T, t0s + torch.max(dilated_positions))
    else:
        T = torch.clamp(T, 0.0, float(x.shape[-1]))

    if _test:
        print("x:", x.shape)  # batch_size x in_channels x input_length
        print("offsets:", offsets.shape)  # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:", t0s.shape)  # out_lengths x 1
        print("dilated positions:", dilated_positions.shape)  # kernel_size
        print("dilated_offsets_repeated:", dilated_offsets_repeated.shape)
        print("T:", T.shape)  # batch_size x groups x out_length x kernel_rfield

    with torch.no_grad():
        U = torch.floor(T).to(torch.long)  # 1 x 1 x length x kernel_rfield
        U = torch.clamp(U, min=0, max=x.shape[2] - 2)

        if _test:
            print("U:", U.shape)

        U = torch.stack([U, U + 1], dim=-1)
        if U.shape[1] < x.shape[1]:
            U = U.repeat(1, x.shape[1], 1, 1, 1)
        if _test:
            print("U:", U.shape)

    x = x.unsqueeze(-1).repeat(1, 1, 1, U.shape[-1])
    x = torch.stack([x.gather(index=U[:, :, :, i, :], dim=-2) for i in range(U.shape[-2])], dim=-1)

    G = torch.max(torch.zeros(U.shape, device=device),
                  1 - torch.abs(U - T.unsqueeze(-1)))  # batch_size x groups x out_length x kernel_rfield x kernel_size

    if _test:
        print("G:", G.shape)

    mx = torch.multiply(G, x.moveaxis(-2, -1))

    return torch.sum(mx, axis=-1)  # .float()  # batch_size x channels x output_length x kernel size


class DeformConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = "valid",
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "reflect",
                 device: str = "cpu",
                 interpolation_function: Callable = efficient_linterpolate,
                 unconstrained: str = None,  # default None to maintain backwards compatibility
                 *args,
                 **kwargs
                 ) -> None:
        """
        1D Deformable convolution kernel layer
        Args:
            in_channels (int): Value of convolution kernel size
            out_channels (int): Value of convolution kernel dilation factor
            kernel_size (int): Value of convolution kernel size
            stride (int): Value convolution kernel stride
            padding (int): See torch.nn.Conv1d for details. Default "valid". Still experimental beware of unexpected behaviour.
            dilation (int): Value of convolution kernel dilation factor
            groups (int) = 1
            bias (bool) = True
            padding_mode: See torch.nn.Conv1d for details. Default "reflect". Still experimental beware of unexpected behaviour.
            device: Device to operate function on. Default: torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        """

        self.device = device
        self.interpolation_function = interpolation_function
        padding_ = padding if isinstance(padding, str) else _single(padding)
        stride_ = _single(stride)
        dilation_ = _single(dilation)
        kernel_size_ = _single(kernel_size)

        super(DeformConv1d, self).__init__(*args, **kwargs)
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride_):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding_  # note this is tuple-like for compatibility
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)
            if padding == 'same':
                for d, k, i in zip(dilation_, kernel_size_,
                                   range(len(kernel_size_) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                            total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight = Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )

        self.dilated_positions = torch.linspace(0,
                                                dilation * kernel_size - dilation,
                                                kernel_size,
                                                )  # automatically store dilation offsets

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        if not unconstrained == None:
            self.unconstrained = unconstrained

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DeformConv1d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def forward(
            self,
            input: Tensor,
            offsets: Tensor,
            mask: Optional[Tensor] = None  # TODO
    ) -> Tensor:
        """
        Forward pass of 1D deformable convolution layer
        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            offset (Tensor[batch_size, offset_groups, output length, kernel_size]):
                offsets to be applied for each position in the convolution kernel. Offset groups can be 1 or such that (in_channels%offset_groups == 0) is satisfied.
            mask (Tensor[batch_size, offset_groups, kernel_width, 1, out_width]): To be implemented

        Returns:
            output (Tensor[batch_size, in_channels, length]): output tensor
        """
        in_shape = input.shape
        if self.padding_mode != 'zeros':
            input = F.pad(
                input,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode
            )
        elif self.padding == 'same':
            input = F.pad(
                input,
                self._reversed_padding_repeated_twice,
                mode='constant',
                value=0
            )

        if not self.device == offsets.device:  # naive assumption
            self.device = offsets.device
        if self.dilated_positions.device != self.device:
            self.dilated_positions = self.dilated_positions.to(offsets.device)

        if "unconstrained" in self.__dict__.keys():
            input = self.interpolation_function(
                input,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                offsets=offsets,
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                device=self.device,
                unconstrained=self.unconstrained
            )
        else:
            input = self.interpolation_function(
                input,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                offsets=offsets,
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                device=self.device
            )
        input = input.flatten(-2, -1)
        output = F.conv1d(input,
                          self.weight,
                          self.bias,
                          stride=self.kernel_size,
                          groups=self.groups
                          )
        if self.padding == 'same':
            assert in_shape[-1] == output.shape[
                -1], f"input length {in_shape} and output length {output.shape} do not match."
        return output