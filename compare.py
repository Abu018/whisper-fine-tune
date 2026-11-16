import sys
import unicodedata

import jiwer
import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODELS = ["openai/whisper-large-v3-turbo", "whisper-hi/epoch-1", "whisper-hi/epoch-2"]
NUM_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
BATCH_SIZE = 16
NUM_EXAMPLES = 5

valid = load_dataset("ai4bharat/Kathbath", "hindi", split=f"valid[:{NUM_CLIPS}]")
valid = valid.rename_column("audio_filepath", "audio")
valid = valid.cast_column("audio", Audio(sampling_rate=16000))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Whisper's BasicTextNormalizer drops Devanagari vowel signs, so only strip punctuation and symbols
def normalize(text):
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def transcribe(model_id):
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16).to(device)
    model.eval()

    predictions = []
    for start in range(0, len(valid), BATCH_SIZE):
        batch = valid[start:start + BATCH_SIZE]
        input_features = processor.feature_extractor(
            [audio["array"] for audio in batch["audio"]],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device, torch.float16)

        with torch.no_grad():
            ids = model.generate(input_features, language="hi", task="transcribe", use_cache=True)
        predictions += processor.batch_decode(ids, skip_special_tokens=True)

    del model
    torch.cuda.empty_cache()
    return predictions


references = [normalize(t) for t in valid["text"]]
results = {}

for model_id in MODELS:
    predictions = [normalize(p) for p in transcribe(model_id)]
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    results[model_id] = (wer, cer, predictions)
    print(f"{model_id}: WER {wer:.2%}, CER {cer:.2%}", flush=True)

print(f"\nResults on {len(valid)} validation clips")
print(f"{'model':<35} {'WER':>8} {'CER':>8}")
for model_id, (wer, cer, _) in results.items():
    print(f"{model_id:<35} {wer:>8.2%} {cer:>8.2%}")

print("\nExamples")
for i in range(NUM_EXAMPLES):
    print(f"\nreference: {references[i]}")
    for model_id, (_, _, predictions) in results.items():
        print(f"{model_id.split('/')[-1]:>23}: {predictions[i]}")
