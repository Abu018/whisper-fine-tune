.PHONY: help install data train smoke lint format clean

help:
	@echo "make install  Install dependencies from uv.lock"
	@echo "make data     Download Kathbath clips and write the manifests"
	@echo "make train    Fine-tune the model (needs a GPU for a full run)"
	@echo "make smoke    Two-batch CPU run to check the pipeline end to end"
	@echo "make lint     Check code with ruff"
	@echo "make format   Format code with ruff"
	@echo "make clean    Delete caches and experiment outputs"

install:
	uv sync

data:
	uv run python -c "from src.config import config; from src.data.dataset import manifest_file, preprocess; \
	manifest_file(config.TRAIN_SPLIT, config.TRAIN_SAMPLES, True); \
	manifest_file(config.EVAL_SPLIT, config.EVAL_SAMPLES, False); \
	preprocess()"

train:
	uv run main.py $(ARGS)

smoke:
	uv run main.py trainer.max_epochs=1 trainer.limit_train_batches=2 \
		trainer.limit_val_batches=2 trainer.accumulate_grad_batches=1 \
		model.train_ds.batch_size=2 model.validation_ds.batch_size=2 \
		exp_manager.exp_dir=./.smoke_experiments

lint:
	uv run ruff check main.py src

format:
	uv run ruff format main.py src

clean:
	rm -rf .smoke_experiments experiments
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
