# adapted from https://github.com/hzwer/Practical-RIFE/
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warper import Warper

__all__ = ["IFNet"]


class Head(nn.Module):
    """
    Head module for the IFNet
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_hidden_channels: int = 16,
        out_channels: int = 4,
        relu_slope: float = 0.2,
    ) -> None:
        """
        :param num_channels: number of input channels
        :param num_hidden_channels: number of hidden channels
        :param out_channels: number of output channels
        :param relu_slope: slope of the LeakyReLU activation function
        """
        super().__init__()
        self.cnn0 = nn.Conv2d(num_channels, num_hidden_channels, 3, 2, 1)
        self.cnn1 = nn.Conv2d(num_hidden_channels, num_hidden_channels, 3, 1, 1)
        self.cnn2 = nn.Conv2d(num_hidden_channels, num_hidden_channels, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(num_hidden_channels, out_channels, 4, 2, 1)
        self.relu = nn.LeakyReLU(relu_slope, inplace=True)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        :param x: input tensor
        :param return_features: whether to return intermediate features
        :return: output tensor or intermediate features and output tensor
        """
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)

        if return_features:
            return x0, x1, x2, x3

        return x3  # type: ignore[no-any-return]


class ResConv(nn.Module):
    """
    A residual convolutional block
    """

    def __init__(
        self, num_channels: int, dilation: int = 1, relu_slope: float = 0.2
    ) -> None:
        """
        :param num_channels: number of input and output channels
        :param dilation: dilation factor of the convolutional layer
        :param relu_slope: slope of the LeakyReLU activation function
        """
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )
        self.beta = nn.Parameter(
            torch.ones((1, num_channels, 1, 1)), requires_grad=True
        )
        self.relu = nn.LeakyReLU(relu_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :return: output tensor
        """
        return self.relu(self.conv(x) * self.beta + x)  # type: ignore[no-any-return]


class IFBlock(nn.Module):
    """
    An Image Flow block
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_convs: int = 8,
        last_conv_channels: int = 52,
        relu_slope: float = 0.2,
    ) -> None:
        """
        :param in_channels: number of input channels
        :param hidden_channels: number of hidden channels
        :param relu_slope: slope of the LeakyReLU activation function
        """
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels // 2, 3, 2, 1),
                nn.LeakyReLU(relu_slope, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_channels // 2, hidden_channels, 3, 2, 1),
                nn.LeakyReLU(relu_slope, inplace=True),
            ),
        )
        self.convblock = nn.Sequential(
            *[ResConv(hidden_channels) for _ in range(num_convs)]
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, last_conv_channels, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(
        self, x: torch.Tensor, flow: Optional[torch.Tensor] = None, scale: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor
        :param flow: flow tensor
        :param scale: scaling factor
        :return: flow, mask, and feature tensors
        """
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                )
                * 1.0
                / scale
            )
            x = torch.cat([x, flow], dim=1)  # type: ignore[list-item]

        feat = self.conv0(x)
        feat = self.convblock(feat)

        tmp = self.lastconv(feat)
        tmp = F.interpolate(
            tmp, scale_factor=scale, mode="bilinear", align_corners=False
        )

        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]

        return flow, mask, feat  # type: ignore[return-value]


class IFNet(nn.Module):
    """
    An Image Flow network
    """

    def __init__(
        self,
        head_channels: int = 3,
        head_hidden_channels: int = 16,
        head_out_channels: int = 4,
        block_channels: Tuple[int, ...] = (15, 28, 28, 28, 28),
        block_hidden_channels: Tuple[int, ...] = (192, 128, 96, 64, 32),
    ) -> None:
        super().__init__()
        self.warper = Warper()
        self.encode = Head(
            num_channels=head_channels,
            num_hidden_channels=head_hidden_channels,
            out_channels=head_out_channels,
        )

        assert len(block_channels) == len(
            block_hidden_channels
        ), "The number of block channels and hidden channels must be the same"

        self.num_blocks = len(block_channels)

        for i, (block_channel, block_hidden_channel) in enumerate(
            zip(block_channels, block_hidden_channels)
        ):
            block = IFBlock(
                in_channels=block_channel, hidden_channels=block_hidden_channel
            )
            self.add_module(f"block{i}", block)

    @property
    def block_list(self) -> List[IFBlock]:
        """
        :return: list of IFBlocks
        """
        return [getattr(self, f"block{i}") for i in range(self.num_blocks)]

    @classmethod
    def edge_map(cls, x: torch.Tensor) -> torch.Tensor:
        # cheap Sobel-ish gradient magnitude
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return (gx.abs() + gy.abs()).mean(dim=1, keepdim=True)

    @classmethod
    def confidence_from_warp_agreement(
        cls, w0: torch.Tensor, w1: torch.Tensor
    ) -> torch.Tensor:
        # w0,w1: [B,3,H,W]
        phot = (w0 - w1).abs().mean(dim=1, keepdim=True)
        edg = (cls.edge_map(w0) - cls.edge_map(w1)).abs()
        err = phot + 0.5 * edg
        # map to confidence in [0,1] (tunable)
        conf = torch.exp(-10.0 * err)
        return conf.clamp(0, 1)

    def estimate_pair(
        self,
        x: torch.Tensor,
        timestep: Union[float, torch.Tensor] = 0.5,
        scale_list: Tuple[int, ...] = (16, 8, 4, 2, 1),
        return_feat: bool = False,
        return_logits: bool = True,
        return_confidence: bool = False,
    ) -> Tuple[
        torch.Tensor,  # warped0
        torch.Tensor,  # warped1
        torch.Tensor,  # flow
        torch.Tensor,  # mask_logits
        torch.Tensor,  # mask
        Optional[torch.Tensor],  # feat
        Dict[str, torch.Tensor],  # extras
    ]:
        """
        x: [B, 2C, H, W] where C includes any extra channels (rgb+aux ok)
        timestep: float or [B,1] tensor
        """
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        if not torch.is_tensor(timestep):
            timestep_map = (x[:, :1].clone() * 0 + 1) * float(timestep)
        else:
            # allow [,] [B,1] or [B,1,H,W]
            if timestep.ndim == 0:
                timestep_map = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
            elif timestep.ndim == 2:
                timestep_map = timestep[:, :, None, None].repeat(
                    1, 1, img0.shape[2], img0.shape[3]
                )
            else:
                timestep_map = timestep

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        warped_img0 = img0
        warped_img1 = img1

        flow: Optional[torch.Tensor] = None
        mask: Optional[torch.Tensor] = None
        feat: Optional[torch.Tensor] = None

        assert len(scale_list) == self.num_blocks

        for block, scale in zip(self.block_list, scale_list):
            if flow is None or mask is None:
                flow, mask, feat = block(
                    torch.cat([img0[:, :3], img1[:, :3], f0, f1, timestep_map], dim=1),
                    flow=None,
                    scale=scale,
                )
            else:
                wf0 = self.warper(f0, flow[:, :2])
                wf1 = self.warper(f1, flow[:, 2:4])
                flow_d, mask_d, feat_d = block(
                    torch.cat(
                        [
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            wf0,
                            wf1,
                            timestep_map,
                            mask,
                            feat,
                        ],
                        dim=1,
                    ),
                    flow=flow,
                    scale=scale,
                )
                mask = mask_d
                flow = flow + flow_d
                feat = feat_d

            warped_img0 = self.warper(img0, flow[:, :2])
            warped_img1 = self.warper(img1, flow[:, 2:4])

        mask_logits = mask
        mask_sig = torch.sigmoid(mask_logits)
        extras: Dict[str, torch.Tensor] = {}

        if return_confidence:
            extras["confidence"] = self.confidence_from_warp_agreement(
                warped_img0[:, :3], warped_img1[:, :3]
            )

        return (
            warped_img0,
            warped_img1,
            flow,
            mask_logits if return_logits else torch.empty(0, device=x.device),
            mask_sig,
            feat if return_feat else None,
            extras,
        )

    def compose(
        self,
        warped0: torch.Tensor,
        warped1: torch.Tensor,
        mask_logits: torch.Tensor,
        mask: torch.Tensor,
        mode: Literal["rife_mask", "alpha", "alpha_x_mask"] = "rife_mask",
        alpha: Optional[torch.Tensor] = None,  # [B,1,1,1] or [B,1,H,W]
        use_sigmoid_mask: bool = True,
        clamp_alpha: Tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        if not use_sigmoid_mask:
            mask = mask_logits

        if mode == "rife_mask":
            m = mask
            return warped0 * m + warped1 * (1 - m)

        if alpha is None:
            raise ValueError("alpha required for mode != 'rife_mask'")

        a = alpha
        if a.ndim == 1:
            a = a[:, None, None, None]
        a = a.clamp(*clamp_alpha)

        if mode == "alpha":
            return (
                warped0 * (1 - a) + warped1 * a
            )  # define alpha as weight toward warped1

        if mode == "alpha_x_mask":
            # useful if you want temporal ramp but also let rife avoid occlusion artifacts
            m = mask
            w = a * (1 - m) + (1 - a) * m  # one reasonable hybrid; feel free to change
            # Simpler: just multiply the mask strength by alpha: m2 = (1-a)*m + a*(1-m)
            return warped0 * w + warped1 * (1 - w)

        raise ValueError(f"unknown mode {mode}")

    def forward(
        self,
        x: torch.Tensor,
        timestep: Union[float, torch.Tensor] = 0.5,
        scale_list: Tuple[int, ...] = (16, 8, 4, 2, 1),
    ) -> torch.Tensor:
        """
        :param x: input tensor
        :param timestep: time step
        :param scale_list: list of scaling factors
        :return: output tensor
        """
        (
            warped0,
            warped1,
            _,
            mask_logits,
            mask,
            _,
            _,
        ) = self.estimate_pair(
            x, timestep=timestep, scale_list=scale_list, return_feat=False
        )
        return self.compose(warped0, warped1, mask_logits, mask, mode="rife_mask")
