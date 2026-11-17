import lightning.pytorch as pl
from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf

from src.adatapers import get_base_model, get_gpu_config
from src.config import config
from src.data.dataset import manifest_file, preprocess
from src.model.model import freeze_parameter

CONFIG_PATH = "speech_to_text_finetune.yml"


def setup_dataloaders(asr_model: ASRModel, cfg: DictConfig) -> ASRModel:
    """
    Sets up the training, validation and test dataloaders for the model.
    Args:
        asr_model: ASRModel instance
        cfg: config
    Returns:
        asr_model: ASRModel instance with updated dataloaders
    """
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_multiple_validation_data(cfg.model.validation_ds)
    if cfg.model.get("test_ds") is not None and cfg.model.test_ds.manifest_filepath is not None:
        asr_model.setup_multiple_test_data(cfg.model.test_ds)

    return asr_model


def main():
    cfg = OmegaConf.merge(OmegaConf.load(CONFIG_PATH), OmegaConf.from_cli())
    manifest_file(config.TRAIN_SPLIT, config.TRAIN_SAMPLES, True)
    manifest_file(config.EVAL_SPLIT, config.EVAL_SAMPLES, False)
    preprocess()

    model = get_base_model(config.CONVERTED_MODEL_PATH)
    freeze_parameter(model)

    accelerator, devices = get_gpu_config()
    precision = "16-mixed" if accelerator == "gpu" else 32
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        **OmegaConf.to_container(cfg.trainer),
    )
    exp_manager(trainer, cfg.get("exp_manager", None))

    # The optimizer schedule needs the trainer and the training data to know the total steps.
    model.set_trainer(trainer)
    model = setup_dataloaders(model, cfg)
    model.setup_optimization(cfg.model.optim)

    if cfg.model.get("spec_augment") is not None:
        model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    trainer.fit(model)
    model.save_to(f"{config.OUTPUT_DIR}/finetuned_hi.nemo")


if __name__ == "__main__":
    main()
