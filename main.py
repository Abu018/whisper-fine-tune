
import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_ID = "openai/whisper-large-v3-turbo"

# Load dataset
train = load_dataset(
    "ai4bharat/Kathbath",
    "hindi",
    split="train[:2%]"
)

valid = load_dataset(
    "ai4bharat/Kathbath",
    "hindi",
    split="valid[:500]"
)

# Rename and decode audio at 16 kHz
train = train.rename_column("audio_filepath", "audio")
valid = valid.rename_column("audio_filepath", "audio")

train = train.cast_column("audio", Audio(sampling_rate=16000))
valid = valid.cast_column("audio", Audio(sampling_rate=16000))

# Load processor and model
processor = WhisperProcessor.from_pretrained(
    MODEL_ID,
    language="hindi",
    task="transcribe"
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float32)

# No need the cache as of now it can be used for the Inference Script
model.config.use_cache = False

# this is intended to reduce GPU memory consumption
model.activation_checkpointing_enable()

# Training all ~800M weights doesn't fit in a T4's 15 GB; train only the decoder
model.freeze_encoder()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5
)
scaler = torch.amp.GradScaler("cuda")

# Training configuration
batch_size = 2
epochs = 2

model.train()

for epoch in range(epochs):

    total_loss = 0

    for start in range(0, len(train), batch_size):

        # Get only the current batch
        batch = train[start:start + batch_size]

        audio_batch = batch["audio"]
        text_batch = batch["text"]

        # Convert audio dictionaries into waveform arrays
        audio_arrays = [audio["array"] for audio in audio_batch]

        # Convert audio into Whisper input features Encoder
        input_features = processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)

        # Convert transcripts into token IDs  ,Decoder
        tokens = processor.tokenizer(
            text_batch,
            padding=True,
            return_tensors="pt"
        )

        # Whisper's pad token is also its end-of-text token, so mask by attention_mask
        labels = tokens.input_ids.masked_fill(tokens.attention_mask.ne(1), -100)
        # The model prepends the start token itself
        if (labels[:, 0] == model.config.decoder_start_token_id).all():
            labels = labels[:, 1:]
        labels = labels.to(device)

        # Train on this batch
        optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(
                input_features=input_features,
                labels=labels
            )

        loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        step = start // batch_size
        if step % 100 == 0:
            print(f"Epoch {epoch + 1}, step {step}, loss {loss.item():.4f}", flush=True)

    average_loss = total_loss / ((len(train) + batch_size - 1) // batch_size)

    print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")

    model.save_pretrained(f"whisper-hi/epoch-{epoch + 1}")
    processor.save_pretrained(f"whisper-hi/epoch-{epoch + 1}")
