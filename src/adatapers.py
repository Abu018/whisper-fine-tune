import nemo.collections.asr as nemo_asr
import torch


def get_gpu_config():
    if not torch.cuda.is_available():
        print("No GPU available, using CPU")
        return "cpu", 1
    print(f"Available GPUs: {torch.cuda.device_count()}")
    return "gpu", torch.cuda.device_count()


def get_base_model(model_path):
    return nemo_asr.models.ASRModel.restore_from(model_path, map_location="cpu")
