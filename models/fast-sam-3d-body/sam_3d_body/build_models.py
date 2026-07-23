# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch

from .models.meta_arch import SAM3DBody
from .utils.config import get_config
from .utils.checkpoint import load_state_dict


def load_sam_3d_body(checkpoint_path: str = "", device: str = "cuda", mhr_path: str = ""):
    print("Loading SAM 3D Body model...")
    
    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path

    # Support overriding IMAGE_SIZE config via IMG_SIZE environment variable
    img_size_env = os.environ.get("IMG_SIZE", "0")
    if img_size_env and int(img_size_env) > 0:
        size = int(img_size_env)
        model_cfg.MODEL.IMAGE_SIZE = [size, size]
        print(f"[build_models] IMAGE_SIZE overridden by IMG_SIZE env: {size}x{size}")

    # [BlendCap non-CUDA dtype patch - keep when re-vendoring the Fast lib]
    # DirectML (privateuseone) cannot allocate bfloat16 - the bf16 torso
    # selected by FP16_TYPE: bfloat16 aborts the process at `model.to(device)`
    # below with "[F... dml_util.cc] Invalid or unsupported data type BFloat16."
    # MPS bf16 support is also incomplete. So on those two back-ends fall back
    # to fp16. CUDA and CPU are LEFT ON bf16: CUDA is the validated NVIDIA path,
    # and CPU supports bf16 natively and is FASTER with it than emulated fp16
    # (the v79 CPU-only test showed forced fp16 running slow emulation + tripping
    # the "CPU autocast target dtype not supported" warning). After BUG 3a,
    # _get_device() returns "cpu" on a DirectML box, so 'device' is rarely
    # literally privateuseone here - this stays a defensive guard for anyone
    # routing torch back to the raw DML device.
    _dev = str(device).lower()
    _non_native = _dev.startswith("privateuseone") or _dev.startswith("mps")
    if (_non_native
            and model_cfg.TRAIN.get("USE_FP16", False)
            and model_cfg.TRAIN.get("FP16_TYPE", "float16") == "bfloat16"):
        model_cfg.TRAIN.FP16_TYPE = "float16"
        print(f"[build_models] device '{device}': FP16_TYPE bfloat16 -> "
              f"float16 (bf16 unsupported on DirectML/MPS)")

    model_cfg.freeze()

    # Initialze the model
    model = SAM3DBody(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)

    # [BlendCap non-CUDA .to() patch - keep when re-vendoring the Fast lib]
    # torch-directml (privateuseone) / MPS can't shallow-copy a CPU tensor onto
    # the device, so nn.Module._apply REBUILDS every parameter and asserts each
    # _parameters entry is an nn.Parameter. A few Fast-lib / MHR submodules keep
    # raw Tensors in _parameters (fine on CUDA and on CPU->CPU, which take the
    # in-place set_data path), tripping `assert isinstance(param, Parameter)` in
    # torch/nn/modules/module.py during model.to(device) below. Promote them to
    # Parameters first. requires_grad=False keeps them inference-only and lets
    # int tensors wrap without error. Scoped to the SAME DirectML/MPS back-ends
    # as the dtype patch above (CUDA and CPU->CPU never hit the assertion).
    if _non_native:
        for _mod in model.modules():
            for _pname, _pval in list(_mod._parameters.items()):
                if _pval is not None and not isinstance(_pval, torch.nn.Parameter):
                    _mod._parameters[_pname] = torch.nn.Parameter(
                        _pval, requires_grad=False)

    model = model.to(device)
    model.eval()

    # Parse LAYER_DTYPE environment variable (only supports fp16/bf16, used for autocast)
    layer_dtype_str = os.environ.get("LAYER_DTYPE", "").lower()
    if layer_dtype_str in ("fp16", "float16"):
        layer_dtype = torch.float16
    elif layer_dtype_str in ("bf16", "bfloat16"):
        layer_dtype = torch.bfloat16
    elif layer_dtype_str and layer_dtype_str not in ("", "none", "fp32", "float32"):
        print(f"[build_models] WARNING: LAYER_DTYPE='{layer_dtype_str}' is not supported, only fp16/bf16 are supported. Ignoring this setting.")
        layer_dtype = None
    else:
        layer_dtype = None

    # Decide whether to apply torch.compile based on environment variable
    use_compile = os.environ.get("USE_COMPILE", "0")
    if use_compile.lower() in ("1", "true", "yes"):
        compile_mode = os.environ.get("COMPILE_MODE", "reduce-overhead")
        model.apply_compile(mode=compile_mode, dtype=layer_dtype)
    elif layer_dtype is not None:
        # Only convert precision, do not compile
        model.convert_decoder_dtype(layer_dtype)

    return model, model_cfg


def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "model.ckpt"), os.path.join(local_dir, "assets", "mhr_model.pt")


def load_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path)
