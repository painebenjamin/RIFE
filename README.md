# RIFE: Real-Time Intermediate Flow Estimation

## Installation

```sh
pip install rife@git+https://github.com/painebenjamin/RIFE.git@main
```

## Inference

All images should be floating-point tensors in the shape `[c, h, w]` or `[1, c, h, w]` and range `[0.0, 1.0]`.
Videos must be in the shape `[b, c, h, w]`, and same range as above.

```py
from rife import RIFEInterpolator

interpolator = RIFEInterpolator.from_pretrained(device="cuda")  # or torch.device

# Image Examples
interpolated_1 = interpolator.interpolate(start, end)  # [c, h, w] if start is 3-dim, else [1, c, h, w]
interpolated_2 = interpolator.interpolate(start, end, include_start = True)  # [2, c, h, w]
interpolated_3 = interpolator.interpolate(start, end, include_start = True, include_end = True)  # [3, c, h, w]
interpolated_4 = interpolator.interpolate(start, end, include_start = True, include_end = True, num_frames = 2)  # [4, c, h, w]

# Video Examples
interpolated_v1 = interpolator.interpolate_video(interpolated_4)  # [7, c, h, w]
interpolated_v2 = interpolator.interpolate_video(interpolated_4, num_frames=2)  # [10, c, h, w]
```
