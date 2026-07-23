# pipeline/ort_backend.py
# ========================
# ONNX Runtime drop-in replacements for the heavy Fast-SAM-3D-Body
# components. Used when USE_ORT=1 is set in save_mhr_data.py.
#
# Phase 3 of the BlendCap installer plan: cross-platform GPU support
# via ORT's execution provider abstraction. Same .onnx file runs on:
#   NVIDIA Win/Linux  -> CUDAExecutionProvider   (tested)
#   AMD Windows       -> DmlExecutionProvider    (untested, best-effort)
#   Intel Windows     -> DmlExecutionProvider    (untested, best-effort)
#   AMD Linux         -> ROCMExecutionProvider   (untested, best-effort)
#   Apple Silicon Mac -> CoreMLExecutionProvider (untested, best-effort)
#   Anything else     -> CPUExecutionProvider    (slow fallback)

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn


# Priority order — ORT picks the first available EP from this list.
# CUDA first because it's the only path we can validate; the rest are
# best-effort. Honest UI labeling lives at the call site.
_EP_PRIORITY = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


def select_providers(prefer: Optional[List[str]] = None) -> List[str]:
    """Return providers in install-priority order, filtered to what's available."""
    # [BlendCap] Full-CPU escape hatch for memory-constrained / unstable GPUs
    # (e.g. shared-memory iGPUs that OOM-reboot under the DirectML backbone
    # arena). BLENDCAP_ORT_CPU=1 forces every ORT session onto the CPU EP,
    # regardless of which GPU EPs are installed. Off by default; opt-in only.
    # This is the ORT half of a full-CPU run; the torch half is _get_device()
    # in save_mhr_data.py.
    if os.environ.get("BLENDCAP_ORT_CPU", "0") == "1":
        return ["CPUExecutionProvider"]
    import onnxruntime as ort
    available = set(ort.get_available_providers())
    priority = prefer or _EP_PRIORITY
    chosen = [p for p in priority if p in available]
    if not chosen:
        chosen = ["CPUExecutionProvider"]
    return chosen


def describe_ep(ep: str) -> str:
    """Human-readable label for the chosen EP. Used in UI."""
    return {
        "CUDAExecutionProvider": "NVIDIA CUDA",
        "DmlExecutionProvider": "DirectML (Experimental)",
        "ROCMExecutionProvider": "AMD ROCm (Experimental)",
        "CoreMLExecutionProvider": "Apple CoreML (Experimental)",
        "CPUExecutionProvider": "CPU (slow fallback)",
    }.get(ep, ep)


class OrtDinov3Backbone(nn.Module):
    """
    Drop-in replacement for Fast-SAM-3D-Body's Dinov3Backbone using ONNX Runtime.

    Mirrors the interface attributes (embed_dim, patch_size, etc.) and the
    forward(x, extra_embed=None) signature so it can swap in via
        model.backbone = OrtDinov3Backbone(...)
    after load_sam_3d_body() returns.

    Input/output kept on GPU when CUDA EP is active via io_binding. On
    other EPs we round-trip through numpy/CPU — measurably slower but
    correct everywhere.
    """

    def __init__(
        self,
        onnx_path: str,
        embed_dim: int = 1280,
        patch_size: int = 16,
        providers: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.onnx_path = str(onnx_path)
        self.embed_dim = embed_dim
        self.embed_dims = embed_dim  # Fast lib reads both names
        self.patch_size = patch_size
        # Sentinels for the parent's compile / TRT flags; we are neither.
        self._compiled = False
        self._use_tensorrt = False

        if not Path(self.onnx_path).exists():
            raise FileNotFoundError(f"ONNX backbone not found: {self.onnx_path}")

        import onnxruntime as ort

        self._providers = providers or select_providers()
        if verbose:
            print(f"[OrtDinov3Backbone] Loading {Path(self.onnx_path).name}")
            print(f"[OrtDinov3Backbone] Provider priority: {self._providers}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Quieten the "Memcpy nodes added" warnings — we know.
        sess_opts.log_severity_level = 3

        self.session = ort.InferenceSession(
            self.onnx_path, sess_options=sess_opts, providers=self._providers,
        )
        self._active_ep = self.session.get_providers()[0]
        self._on_cuda = self._active_ep == "CUDAExecutionProvider"

        # Input/output names from the actual graph
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

        # Backbone exported FP16, returns FP16
        self._input_dtype = np.float16
        self._output_dtype = np.float16

        if verbose:
            print(f"[OrtDinov3Backbone] Active EP: {self._active_ep} "
                  f"({describe_ep(self._active_ep)})")
            print(f"[OrtDinov3Backbone] I/O: {self._input_name} -> "
                  f"{self._output_name}")

    @property
    def device(self):
        # Match nn.Module-style expectation; ORT doesn't expose a torch device
        return torch.device("cuda" if self._on_cuda else "cpu")

    def forward(self, x: torch.Tensor, extra_embed=None) -> torch.Tensor:
        if extra_embed is not None:
            raise NotImplementedError("extra_embed not supported in ORT path")

        # Input prep — cast to FP16 contiguous to match the exported graph
        x_in = x.detach().to(torch.float16).contiguous()

        if self._on_cuda:
            # GPU stays GPU: use io_binding to avoid CPU round-trip
            return self._forward_iobinding(x_in)
        # Other EPs: simple numpy round-trip
        return self._forward_numpy(x_in)

    def _forward_numpy(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.cpu().numpy()
        y_np = self.session.run([self._output_name], {self._input_name: x_np})[0]
        return torch.from_numpy(y_np).to(x.device)

    def _forward_iobinding(self, x: torch.Tensor) -> torch.Tensor:
        # ORT IO binding lets us pass GPU pointers directly.
        # Output shape: (B, embed_dim, H/patch, W/patch) for a ViT.
        B, _, H, W = x.shape
        H_tok = H // self.patch_size
        W_tok = W // self.patch_size
        out_shape = (B, self.embed_dim, H_tok, W_tok)

        # Allocate output on GPU
        out_tensor = torch.empty(
            out_shape, device=x.device, dtype=torch.float16
        ).contiguous()

        io = self.session.io_binding()
        io.bind_input(
            name=self._input_name,
            device_type="cuda",
            device_id=x.device.index or 0,
            element_type=self._input_dtype,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )
        io.bind_output(
            name=self._output_name,
            device_type="cuda",
            device_id=x.device.index or 0,
            element_type=self._output_dtype,
            shape=out_shape,
            buffer_ptr=out_tensor.data_ptr(),
        )
        self.session.run_with_iobinding(io)
        return out_tensor


def load_ort_backbone_for_sam3dbody(
    sam_model: nn.Module,
    onnx_path: str,
    providers: Optional[List[str]] = None,
) -> str:
    """
    Replace `sam_model.backbone` with an ORT-backed equivalent.

    Returns a short status string describing which EP got picked, useful for
    surfacing in logs / Blender UI ("GPU: NVIDIA CUDA" etc.).
    """
    # Don't bind the old backbone to a local — that reference keeps the
    # ~1.7 GB of PyTorch weights alive past empty_cache() and the VRAM
    # never gets reclaimed.
    embed_dim = getattr(sam_model.backbone, "embed_dim", 1280)
    patch_size = getattr(sam_model.backbone, "patch_size", 16)

    ort_bb = OrtDinov3Backbone(
        onnx_path=onnx_path,
        embed_dim=embed_dim,
        patch_size=patch_size,
        providers=providers,
    )

    sam_model.backbone = ort_bb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ort_bb._active_ep


class OrtMoGeEncoder(nn.Module):
    """
    Drop-in replacement for MoGe-s's DINOv2Encoder, calling an ORT session
    in place of the PyTorch encoder. Matches the call convention from
    moge/model/v2.py:150:
        self.encoder(image, base_h, base_w, return_class_token=True)

    The exported .onnx has both `features` and `cls_token` outputs so this
    wrapper always returns the (features, cls_token) tuple when the caller
    asks for it. When return_class_token=False, returns just features.

    The TensorRT analogue in build_fov_estimator.py (TRTEncoderWrapper)
    does the same job for TRT — we follow that pattern with ORT.
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.onnx_path = str(onnx_path)
        if not Path(self.onnx_path).exists():
            raise FileNotFoundError(f"MoGe encoder ONNX not found: {self.onnx_path}")

        import onnxruntime as ort

        self._providers = providers or select_providers()
        if verbose:
            print(f"[OrtMoGeEncoder] Loading {Path(self.onnx_path).name}")
            print(f"[OrtMoGeEncoder] Provider priority: {self._providers}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.log_severity_level = 3  # Quieten Memcpy warnings

        self.session = ort.InferenceSession(
            self.onnx_path, sess_options=sess_opts, providers=self._providers,
        )
        self._active_ep = self.session.get_providers()[0]
        self._on_cuda = self._active_ep == "CUDAExecutionProvider"

        # Discover graph I/O names
        inputs = {i.name for i in self.session.get_inputs()}
        outputs = [o.name for o in self.session.get_outputs()]
        self._has_features = "features" in outputs
        self._has_cls_token = "cls_token" in outputs

        if verbose:
            print(f"[OrtMoGeEncoder] Active EP: {self._active_ep} "
                  f"({describe_ep(self._active_ep)})")
            print(f"[OrtMoGeEncoder] Inputs: {sorted(inputs)}, "
                  f"Outputs: {outputs}")

    def forward(self, image, token_rows, token_cols, return_class_token=False):
        # Match MoGe call signature exactly
        image_fp16 = image.detach().to(torch.float16).contiguous()

        # Coerce scalar ints to np.int64 — the export gave token_rows/cols as
        # int64 scalar graph inputs
        if isinstance(token_rows, torch.Tensor):
            tr = int(token_rows.item())
        else:
            tr = int(token_rows)
        if isinstance(token_cols, torch.Tensor):
            tc = int(token_cols.item())
        else:
            tc = int(token_cols)

        feed = {
            "image": image_fp16.cpu().numpy(),
            "token_rows": np.array(tr, dtype=np.int64),
            "token_cols": np.array(tc, dtype=np.int64),
        }
        out_names = ["features", "cls_token"] if return_class_token else ["features"]
        outs = self.session.run(out_names, feed)

        # Move back to image's device
        features = torch.from_numpy(outs[0]).to(image.device)
        if return_class_token:
            cls_token = torch.from_numpy(outs[1]).to(image.device)
            return features, cls_token
        return features


def load_ort_moge_encoder(
    moge_model,
    onnx_path: str,
    providers: Optional[List[str]] = None,
) -> str:
    """
    Swap moge_model.encoder for an ORT-backed equivalent. Same pattern as
    the TRT wrapper in build_fov_estimator.py — bypass the nn.Module
    attribute check via object.__setattr__ since MoGeModel registers its
    encoder as a submodule.

    Returns the active EP string.
    """
    ort_enc = OrtMoGeEncoder(onnx_path=onnx_path, providers=providers)
    # Preserve the original so we could swap back if needed
    moge_model._orig_encoder = moge_model.encoder
    object.__setattr__(moge_model, "encoder", ort_enc)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ort_enc._active_ep
