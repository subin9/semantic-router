import pytest
import torch
from vllm_sr_runtime import runtime
from vllm_sr_runtime.accel.cuda import CUDAAccelerator
from vllm_sr_runtime.accel.rocm import ROCmAccelerator
from vllm_sr_runtime.config import ServeConfig
from vllm_sr_runtime.errors import PlacementError
from vllm_sr_runtime.placement import parse_device, place

GPU = torch.cuda.is_available()


def test_parse_device():
    assert parse_device("auto") == ("auto", None)
    assert parse_device("rocm:3") == ("rocm", 3)
    assert parse_device("CPU") == ("cpu", None)
    for bad in ("gpu", "cuda:x", "rocm:-1", ""):
        with pytest.raises(PlacementError):
            parse_device(bad)


def test_cpu_placement_and_budget():
    placement = place("tiny", "cpu", 1_000_000)
    assert placement.device.accelerator == "cpu" and placement.accelerator.validated
    with pytest.raises(PlacementError, match="memory-budget"):
        place("tiny", "cpu", 10_000_000_000, memory_budget_gib=1)


@pytest.mark.skipif(GPU, reason="checks the CPU-only fallback")
def test_auto_falls_back_to_cpu_without_gpus():
    assert place("tiny", "auto", 1000).device.accelerator == "cpu"
    with pytest.raises(PlacementError, match="not available"):
        place("tiny", "rocm:0", 1000)


def test_unusable_device_fails_before_the_download(monkeypatch):
    def download(*args, **kwargs):
        raise AssertionError("the model was resolved before the device was checked")

    monkeypatch.setattr(runtime, "resolve", download)
    config = ServeConfig(model="vllm-sr/Decision-1.0-Kai-0.6B", device="rocm:99")
    with pytest.raises(PlacementError, match="no device can serve vllm-sr/Decision"):
        runtime.Runtime(config).load()


def test_cuda_is_marked_unvalidated_and_rocm_validated():
    assert CUDAAccelerator.validated is False
    assert ROCmAccelerator.validated is True


@pytest.mark.gpu
@pytest.mark.skipif(not GPU, reason="needs a CUDA or ROCm device")
def test_gpu_devices_report_memory_and_bf16():
    accelerator = ROCmAccelerator() if torch.version.hip else CUDAAccelerator()
    devices = accelerator.devices()
    assert devices and devices[0].total_memory and devices[0].index == 0
