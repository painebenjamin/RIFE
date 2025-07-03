# MIT License
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from .ifnet import IFNet

__all__ = [
    "RIFEInterpolator",
]


class RIFEInterpolator(torch.nn.Module):
    """
    Frame Interpolation with RIFE (Real-Time Intermediate Flow Estimation)

    https://arxiv.org/abs/2011.06294
    Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, Shuchang Zhou
    ECCV 2022
    Megvii Research, NERCVT, School of Computer Science, Peking University,
    Institute for Artificial Intelligence, Peking University, Beijing Academy of Artificial Intelligence
    """

    def __init__(
        self,
        module: IFNet,
    ) -> None:
        """
        Initializes the RIFEInterpolator with a given IFNet module.

        :param module: An instance of IFNet, which is the frame interpolation network.
        """
        super().__init__()
        self.module = module

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "benjamin-paine/taproot-common",
        filename: str = "image-interpolation-rife-flownet.safetensors",
        subfolder: str | None = None,
        revision: str | None = None,
        device: str | torch.device | int = "cpu",
    ) -> "RIFEInterpolator":
        """
        Loads a pretrained RIFE model from a specified repository.
        :param repo_id: The repository ID where the model is hosted.
        :param repo_filename: The filename of the model weights.
        :param device: The device to load the model onto (default is "cpu").
        :return: An instance of RIFEInterpolator with the loaded model.
        """
        device = (
            torch.device(device)
            if isinstance(device, str)
            else torch.device(f"cuda:{device}") if isinstance(device, int) else device
        )
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
        )

        _, ext = os.path.splitext(path)
        if ext == ".safetensors":
            # Load the state dict from a safetensors file
            state_dict = {}
            with safe_open(path, framework="pt", device=device.type) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            # Load the state dict from a regular PyTorch file
            state_dict = torch.load(path, map_location=device)

        # Initialize the IFNet module with the loaded state dict
        module = IFNet()
        module.load_state_dict(state_dict)
        module.eval()  # Set the module to evaluation mode
        module.requires_grad_(False)
        # Ensure the module is on the correct device
        module.to(device)

        # Create an instance of RIFEInterpolator with the IFNet module
        return cls(module)

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the module's parameters are located.
        This is useful for ensuring that inputs are on the same device as the model.
        """
        return next(self.module.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the module's parameters.
        This is useful for ensuring that inputs are of the same type as the model.
        """
        return next(self.module.parameters()).dtype

    def pad_image(
        self, image: torch.Tensor, nearest: int = 128
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """
        Pad the image tensor to ensure it has dimensions divisible by 128.
        :param image: The input image tensor ([C,H,W] or [B,C,H,W]).
        :param nearest: The nearest multiple to pad to (default is 128).
        :return: A tuple containing the padded image and the padding sizes.
        """

        squeeze = False
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            squeeze = True

        h, w = image.shape[2:]
        pad_h = (nearest - h % nearest) % nearest
        pad_w = (nearest - w % nearest) % nearest
        pad_l = pad_w // 2
        pad_t = pad_h // 2
        pad_r = pad_w - pad_l
        pad_b = pad_h - pad_t
        padding = (pad_l, pad_r, pad_t, pad_b)
        padded_image = F.pad(image, padding)

        if squeeze:
            padded_image = padded_image.squeeze(0)

        return padded_image, padding

    def unpad_image(
        self, image: torch.Tensor, padding: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        Unpad the image tensor to remove the padding.
        :param image: The padded image tensor ([C,H,W] or [B,C,H,W]).
        :param padding: The padding sizes (top, bottom, left, right).
        :return: The unpadded image tensor.
        """
        squeeze = False
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            squeeze = True

        h, w = image.shape[2:]
        pad_l, pad_r, pad_t, pad_b = padding
        crop_l = pad_l
        crop_r = w - pad_r
        crop_t = pad_t
        crop_b = h - pad_b
        cropped_image = image[:, :, crop_t:crop_b, crop_l:crop_r]

        if squeeze:
            cropped_image = cropped_image.squeeze(0)

        return cropped_image

    def prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Prepare a single tensor by ensuring it has the correct shape and type.
        :param tensor: The input tensor ([C,H,W] or [B,C,H,W]).
        :return: The prepared tensor.
        """
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] == 4:
            tensor = tensor[:, :3, :, :]
        elif tensor.shape[1] != 3:
            raise ValueError("Tensor must have 1, 3, or 4 channels.")

        if tensor.dtype is torch.uint8:
            tensor = tensor.float() / 255.0

        return tensor.to(self.device, dtype=self.dtype)

    def prepare_tensors(
        self,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Prepare multiple tensors by ensuring they have the correct shape and type.
        """
        prepared_tensors = [self.prepare_tensor(tensor) for tensor in tensors]
        first_shape = prepared_tensors[0].shape

        assert all(
            tensor.shape == first_shape for tensor in prepared_tensors
        ), "All tensors must have the same shape."

        return tuple(prepared_tensors)

    def forward(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        num_frames: int = 1,
    ) -> torch.Tensor:
        """
        Runs the frame interpolation network, returning all frames including
        the start and end frames.
        """
        b, c, h, w = start.shape
        timesteps = torch.linspace(0, 1, num_frames + 2)[1:-1]
        timesteps = timesteps.to(self.device, dtype=self.dtype)

        middle = [
            self.module(
                torch.cat([start, end], dim=1),
                timestep=t,
            )
            for t in timesteps
        ]

        results = torch.cat(middle, dim=0)
        return results

    @torch.inference_mode()
    def interpolate(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        num_frames: int = 1,
        include_start: bool = False,
        include_end: bool = False,
    ) -> torch.Tensor:
        """
        Interpolate frames between two images.
        :param start: The starting image tensor ([C,H,W] or [B,C,H,W]).
        :param end: The ending image tensor ([C,H,W] or [B,C,H,W]).
        :param num_frames: The number of frames to interpolate between start and end.
        :param include_start: Whether to include the start frame in the output.
        :param include_end: Whether to include the end frame in the output.
        :return: A tensor containing the interpolated frames ([B,C,H,W], fp32, cpu).
        """
        start, end = self.prepare_tensors(start, end)
        start, padding = self.pad_image(start)
        end, _ = self.pad_image(end)

        interpolated_frames = self.forward(
            start,
            end,
            num_frames=num_frames,
        )

        return_frames = []
        if include_start:
            return_frames.append(start)
        return_frames.append(interpolated_frames)
        if include_end:
            return_frames.append(end)

        frames = torch.cat(return_frames, dim=0)
        frames = self.unpad_image(frames, padding)
        frames = frames.clamp(0, 1)
        frames = frames.float()
        frames = frames.detach().cpu()

        return frames

    @torch.inference_mode()
    def interpolate_video(
        self,
        video: torch.Tensor,
        num_frames: int = 1,
        loop: bool = False,
        use_tqdm: bool = False,
        use_scene_detection: bool = False,
    ) -> torch.Tensor:
        """
        Interpolate frames in a video tensor.
        :param video: The video tensor ([B,C,H,W]).
        :param num_frames: The number of frames to interpolate.
        :param loop: Whether to loop the video.
        :param use_tqdm: Whether to show progress bar.
        :param use_scene_detection: Whether to use scene detection to preserve hard cuts.
        :return: A tensor containing the interpolated frames ([B,C,H,W], fp32, cpu).
        """
        video = self.prepare_tensor(video)
        video, padding = self.pad_image(video)

        b, c, h, w = video.shape

        assert b >= 2, "Video must have at least 2 frames for interpolation."

        if use_scene_detection:
            return self.interpolate_video_with_scene_detection(
                video, num_frames, loop, use_tqdm, padding
            )
        else:
            return self.interpolate_video_standard(
                video, num_frames, loop, use_tqdm, padding
            )

    def detect_scenes(self, video: torch.Tensor) -> list[tuple[int, int]]:
        """
        Detect scenes in the video using PySceneDetect.
        :param video: The video tensor ([B,C,H,W]).
        :return: list of (start_frame, end_frame) tuples for each scene.
        """
        try:
            import imageio
            from scenedetect import (  # type: ignore[import-untyped]
                ContentDetector,
                detect,
            )
        except ImportError as e:
            raise ImportError(
                "PySceneDetect and ImageIO are required for scene detection. "
                "Install them with: pip install rife[scene-detection]"
            ) from e

        # Convert video tensor to temporary video file for PySceneDetect
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)  # Close the file descriptor to avoid locking issues
        video_np = video.permute(0, 2, 3, 1).cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)

        writer = imageio.get_writer(temp_path, fps=16, codec="libx264")
        for frame in video_np:
            writer.append_data(frame)

        writer.close()

        if not os.path.exists(temp_path):
            raise RuntimeError(
                "Failed to create temporary video file for scene detection."
            )

        try:

            # Use PySceneDetect to detect scenes
            scenes = detect(temp_path, ContentDetector(threshold=27.0))

            # Convert to frame indices
            scene_list = []
            for start_timecode, end_timecode in scenes:
                scene_list.append((start_timecode.frame_num, end_timecode.frame_num))

            # If no scenes detected or only one scene, return the entire video as one scene
            if len(scene_list) <= 1:
                return [(0, video.shape[0] - 1)]

            return scene_list

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def interpolate_video_with_scene_detection(
        self,
        video: torch.Tensor,
        num_frames: int,
        loop: bool,
        use_tqdm: bool,
        padding: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        Interpolate video with scene detection to preserve hard cuts.

        :param video: The video tensor ([B,C,H,W]), fp32, 0 <= video <= 1.
        :param num_frames: The number of frames to interpolate between each pair of frames.
        :param loop: Whether to loop the video.
        :param use_tqdm: Whether to show a progress bar.
        :param padding: Padding sizes to apply to the video.
        :return: A tensor containing the interpolated frames ([B,C,H,W], fp32, cpu).
        """
        scenes = self.detect_scenes(video)

        # If only one scene, use standard interpolation
        if len(scenes) == 1:
            return self.interpolate_video_standard(
                video, num_frames, loop, use_tqdm, padding
            )

        # Interpolate each scene separately
        interpolated_scenes = []
        total_scenes = len(scenes)

        iterator = iter(scenes)
        if use_tqdm:
            from tqdm import tqdm

            iterator = tqdm(  # type: ignore[assignment]
                iterator, desc="Interpolating scenes", unit="scene", total=total_scenes
            )

        for i, (start_frame, end_frame) in enumerate(iterator):
            if i == total_scenes - 1:
                end_frame += 1  # Include the last frame in the last scene

            # Extract scene
            scene_video = video[start_frame:end_frame]

            # Interpolate this scene
            interpolated_scene = self.interpolate_video_standard(
                scene_video, num_frames, False, use_tqdm, padding
            )

            interpolated_scenes.append(interpolated_scene)

        # Concatenate all scenes without interpolation between them
        result = torch.cat(interpolated_scenes, dim=0)
        result = self.unpad_image(result, padding)
        return result

    def interpolate_video_standard(
        self,
        video: torch.Tensor,
        num_frames: int,
        loop: bool,
        use_tqdm: bool,
        padding: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        Interpolate frames in a video tensor.
        :param video: The video tensor ([B,C,H,W]).
        :param num_frames: The number of frames to interpolate.
        :param loop: Whether to loop the video.
        :return: A tensor containing the interpolated frames ([B,C,H,W], fp32, cpu).
        """
        b, c, h, w = video.shape

        num_interpolated_frames = b * num_frames
        if loop:
            num_interpolated_frames += 1
        else:
            num_interpolated_frames -= num_frames

        num_output_frames = b + num_interpolated_frames
        results = torch.zeros(
            (num_output_frames, 3, h, w),
            dtype=torch.float32,
            device="cpu",
        )

        iterator = range(b)
        if use_tqdm:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Interpolating", unit="frame", total=b - 1)  # type: ignore[assignment]

        for i in iterator:
            left = video[i]
            if i == b - 1:
                if not loop:
                    break
                right = video[0]
            else:
                right = video[i + 1]

            start_i = i * (num_frames + 1)
            results[start_i] = left

            interpolated_frames = self.forward(
                left[None],
                right[None],
                num_frames=num_frames,
            )
            interpolated_frames = interpolated_frames.clamp(0, 1)
            interpolated_frames = interpolated_frames.float()
            interpolated_frames = interpolated_frames.detach().cpu()

            results[start_i : start_i + num_frames] = interpolated_frames
            results[start_i + num_frames] = right
        if loop:
            results = results[:-1]
        else:
            results[-1] = video[-1]

        results = self.unpad_image(results, padding)
        return results
