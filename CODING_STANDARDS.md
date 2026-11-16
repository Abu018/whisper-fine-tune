# Coding standards

How code in this repository is written, checked, tested, and reviewed. These are standard practices for production Python and ML projects, adapted to this one (Whisper fine-tuning with `uv`, PyTorch, and Hugging Face). See `ARCHITECTURE.md` for what the code does; this file covers how it is written.

## 1. Tooling

Every rule here that a tool can enforce is enforced by a tool, not by reviewers.

| Purpose | Tool | Command |
|---|---|---|
| Environment and dependencies | `uv` | `uv sync`, `uv add <pkg>` |
| Formatting | `ruff format` | `uv run ruff format .` |
| Linting and import sorting | `ruff check` | `uv run ruff check --fix .` |
| Type checking | `mypy` | `uv run mypy src` |
| Tests | `pytest` | `uv run pytest` |
| Run all checks before each commit | `pre-commit` | `uv run pre-commit install` (once) |

Add the tools as development dependencies:

```bash
uv add --dev ruff mypy pytest pytest-cov pre-commit
```

Configuration in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = [
    "E", "W",  # pycodestyle
    "F",       # pyflakes: unused imports, undefined names
    "I",       # import sorting
    "N",       # naming conventions
    "UP",      # modern Python syntax
    "B",       # likely bugs (mutable defaults, etc.)
    "SIM",     # simplifications
    "RUF",     # ruff-specific checks
]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: takes more than a few seconds",
    "gpu: needs a CUDA GPU",
]
addopts = "-m 'not slow and not gpu'"
```

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.x.y  # pin to the latest release
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.x.y  # pin to the latest release
    hooks:
      - id: check-added-large-files
        args: [--maxkb=1024]
      - id: end-of-file-fixer
      - id: trailing-whitespace
```

`check-added-large-files` stops model weights and datasets from being committed by accident.

## 2. Project structure

- All importable code lives in `src/poc/`. Top-level scripts are not used; entry points are declared in `pyproject.toml` under `[project.scripts]`.
- One module per responsibility (`data.py`, `collator.py`, `metrics.py`, ...). If a module passes about 400 lines or does two unrelated things, split it.
- Tests mirror the package: `src/poc/text.py` is tested by `tests/test_text.py`.
- Configs live in `configs/`. Generated files (checkpoints, caches, logs) go in `outputs/`, which is git-ignored.

## 3. Style

### General

- Follow PEP 8; `ruff format` decides formatting, so there are no style debates in review.
- Line length is 100.
- Use f-strings for formatting, except in logging calls (see section 7).
- Use `pathlib.Path`, not string paths or `os.path`.
- Prefer small functions that do one thing. A function that needs a comment to separate its sections should usually be several functions.

### Naming

| Kind | Convention | Example |
|---|---|---|
| Modules, functions, variables | `snake_case` | `build_dataset`, `max_label_tokens` |
| Classes | `PascalCase` | `WhisperCollator`, `TrainConfig` |
| Constants | `UPPER_SNAKE_CASE` | `SAMPLING_RATE`, `MAX_AUDIO_SECONDS` |
| Private helpers | leading underscore | `_prepare`, `_is_usable` |
| Booleans | read as a question | `is_usable`, `freeze_encoder`, `has_labels` |

Names say what something is, including units where it matters: `max_audio_seconds`, not `max_len`; `learning_rate`, not `lr2`. Avoid single-letter names outside short comprehensions and math.

### Imports

- Absolute imports within the package: `from poc.text import normalize`.
- No wildcard imports.
- `ruff` sorts imports into standard library, third-party, then local groups.

## 4. Type hints

All functions in `src/` have type hints for parameters and return values, and `mypy --strict` passes.

```python
def compute_wer(references: list[str], predictions: list[str]) -> float:
    ...
```

- Use built-in generics (`list[str]`, `dict[str, int]`) and `X | None`, not `typing.List` or `Optional`.
- Use `dataclasses` (or `pydantic` if validation is needed) for structured data instead of loose dicts.
- When a third-party library has no type information, add a narrow `# type: ignore[<code>]` on that line rather than loosening the global config.

## 5. Docstrings and comments

- Every public module, class, and function has a docstring in Google style. Private helpers get one when their behavior isn't obvious from the name and signature.

```python
def build_dataset(
    splits: dict[str, str],
    processor: WhisperProcessor,
    cache_dir: Path,
    num_proc: int = 16,
) -> DatasetDict:
    """Load, filter, and featurize Kathbath splits, caching the result on disk.

    Args:
        splits: Maps a name ("train", "eval") to a Hugging Face split string ("train[:2%]").
        processor: Whisper processor configured with the target language and task.
        cache_dir: Where the processed dataset is saved and reloaded from.
        num_proc: Number of worker processes for filtering and featurizing.

    Returns:
        The processed splits with `input_features` and `labels` columns.
    """
```

- Comments explain **why**, never **what**. Write one only when the code cannot show a constraint by itself:

```python
# Good: states a constraint the code can't express
# Whisper's pad token is also its end-of-text token, so mask by attention_mask
labels = padded.input_ids.masked_fill(padded.attention_mask.ne(1), -100)

# Bad: repeats the code
# move labels to the device
labels = labels.to(device)
```

- No commented-out code; git keeps history.
- `TODO` comments include an owner or issue link: `# TODO(hai): ...` or `# TODO(#42): ...`.

## 6. Configuration

- No hard-coded hyperparameters, paths, model IDs, or dataset splits in code. They come from a config file in `configs/`, with command-line overrides.
- Configs are typed dataclasses, validated when loaded, so a typo fails at startup instead of hours into a run.
- Secrets (Hugging Face tokens, W&B keys) come from environment variables, never from code or committed files. Keep a `.env.example` listing the variables without values.
- The resolved config is saved into every run's output directory.

## 7. Logging

- Use the `logging` module, never `print`, in `src/`. Each module creates its own logger:

```python
import logging

logger = logging.getLogger(__name__)
```

- Configure logging once, in the entry point (`train.py`, `evaluate.py`), not in library modules.
- Use lazy `%` formatting in log calls so messages aren't built when the level is disabled:

```python
logger.info("Kept %d of %d clips", kept, total)  # good
logger.info(f"Kept {kept} of {total} clips")      # avoid
```

- Levels: `DEBUG` for detail useful while developing, `INFO` for progress and results, `WARNING` for recoverable surprises (clips dropped, fallbacks used), `ERROR` for failures.
- Metrics go to the experiment tracker (TensorBoard or W&B), not only to logs.
- In multi-GPU runs, only the main process logs and saves.

## 8. Error handling

- Fail fast with a clear message at the boundary where input enters (config loading, command-line arguments, dataset loading):

```python
if not torch.cuda.is_available():
    raise RuntimeError("Training requires a CUDA GPU; none was found.")
```

- Raise specific exceptions (`ValueError`, `FileNotFoundError`, custom ones in `poc/errors.py`) rather than bare `Exception`.
- Never use a bare `except:` or `except Exception: pass`. Catch only what you can handle, and log or re-raise the rest.
- Don't use `assert` for runtime checks; it is removed when Python runs with `-O`. Use it only in tests.

## 9. Functions and side effects

- No work at import time. Modules define functions, classes, and constants; nothing loads models, datasets, or GPUs when imported. Entry points use:

```python
def main() -> None:
    ...


if __name__ == "__main__":
    main()
```

- Pass dependencies in as arguments (processor, config, paths) instead of reading globals, so functions can be tested in isolation.
- Avoid mutable default arguments (`def f(x: list[str] = [])`); use `None` and create the value inside.

## 10. ML-specific practices

### Training loop and batching

- Train with `Seq2SeqTrainer`. Do not write your own training loop (`for epoch ...: for batch ...: loss.backward()`) or slice batches by hand (`train[start:start + batch_size]`).
- Batching, shuffling, mixed precision, gradient accumulation and clipping, the learning-rate schedule, evaluation, checkpointing, and multi-GPU are handled by the trainer and configured through `Seq2SeqTrainingArguments` in the config file.
- Custom behavior plugs into the trainer instead of replacing it:
  - how examples are padded into a batch: a data collator (`collator.py`)
  - how scores are computed: `compute_metrics` (`metrics.py`)
  - extra actions during training (logging, early stopping): a `TrainerCallback`
  - a different loss: subclass `Seq2SeqTrainer` and override `compute_loss`
- A custom loop (or Hugging Face Accelerate) needs a written reason in the pull request, such as training with several models or losses the trainer can't express.

### Reproducibility

- Set and log a seed for every run (`transformers.set_seed(seed)` covers Python, NumPy, and PyTorch).
- Dependencies are pinned through `uv.lock`, which is committed.
- Save the config, git commit hash, and dataset split definitions with every checkpoint.

### Data

- Training, validation, and test splits never overlap. Validation picks checkpoints and hyperparameters; the test split is used only for final numbers.
- Preprocessing is deterministic and lives in `data.py`, shared by training and evaluation.
- Text normalization for metrics lives in one place (`text.py`) and is used everywhere scores are computed.
- Log how many examples each filter drops.

### Devices and precision

- Never hard-code `"cuda"` deep in the code. Decide the device once in the entry point and pass it down (the trainer does this for you).
- Use `torch.no_grad()` or `torch.inference_mode()` for evaluation and inference.
- Free GPU memory between models in evaluation scripts (`del model; torch.cuda.empty_cache()`).

### Artifacts

- Never commit model weights, datasets, or large outputs. Store models in a model registry or a private Hugging Face Hub repo, tagged with their metrics and config.

## 11. Testing

- Every module in `src/` has unit tests for its logic: text normalization, label masking in the collator, filtering rules, metric calculation.
- Tests are fast and run on CPU by default. Tests that need a GPU or take more than a few seconds are marked `@pytest.mark.gpu` or `@pytest.mark.slow`.
- A smoke test trains a few steps with `openai/whisper-tiny` on a handful of clips to catch wiring errors without a full run.
- Tests use small fixed inputs, not the full dataset.
- Name tests after the behavior they check:

```python
def test_normalize_keeps_devanagari_vowel_signs() -> None:
    assert normalize("नमस्ते!") == "नमस्ते"


def test_collator_masks_padding_but_keeps_end_of_text() -> None:
    ...
```

- Aim for high coverage of `src/poc/` logic (`uv run pytest --cov=poc`). Coverage is a guide, not a target to game.

## 12. Git workflow

- `main` is always working. Work happens on short-lived branches named `<type>/<short-description>`, for example `feat/trainer-pipeline` or `fix/label-masking`.
- Commits are small and focused, with messages in Conventional Commits style:

```
feat(data): cache preprocessed features on disk
fix(collator): keep end-of-text token when masking padding
docs: add evaluation instructions to README
```

  Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`.

- Every change goes through a pull request that:
  - passes `ruff`, `mypy`, and `pytest` in CI
  - explains what changed and why
  - for changes that affect training or evaluation, reports the metrics before and after
- Don't commit generated files: checkpoints, `outputs/`, `.venv/`, caches, notebooks with outputs.

## 13. Continuous integration

CI (for example GitHub Actions) runs on every pull request:

1. `uv sync --frozen` (fails if `uv.lock` is out of date)
2. `uv run ruff format --check .`
3. `uv run ruff check .`
4. `uv run mypy src`
5. `uv run pytest`

GPU and slow tests run on a schedule or manually on a GPU runner, not on every pull request.

## 14. Review checklist

Before requesting review, check:

- [ ] `pre-commit` passes; `mypy` and `pytest` pass.
- [ ] New logic has tests.
- [ ] No hard-coded paths, hyperparameters, or secrets.
- [ ] Training goes through `Seq2SeqTrainer`; no hand-written training loop or manual batch slicing.
- [ ] No `print`, commented-out code, or unexplained `# type: ignore`.
- [ ] Public functions have type hints and docstrings.
- [ ] Training or evaluation changes include before and after metrics.
- [ ] `README.md`, `ARCHITECTURE.md`, or `RESULTS.md` are updated if behavior changed.
