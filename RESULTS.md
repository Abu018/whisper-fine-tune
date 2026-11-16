# Whisper Hindi fine-tuning results

Fine-tuning `openai/whisper-large-v3-turbo` on a small slice of Kathbath Hindi lowered word error rate on held-out validation audio from **27.22% to 23.89%** after one epoch, about 12% fewer word errors. The second epoch was slightly worse, so `whisper-hi/epoch-1` is the best checkpoint.

## Results

Scored on the first 500 clips of the Kathbath Hindi `valid` split. None of these clips were used in training.

| Model | Word error rate | Character error rate |
|---|---|---|
| `openai/whisper-large-v3-turbo` (base) | 27.22% | 8.88% |
| `whisper-hi/epoch-1` | **23.89%** | **7.66%** |
| `whisper-hi/epoch-2` | 24.66% | 7.85% |

Training loss per epoch:

| Epoch | Average training loss |
|---|---|
| 1 | 0.2216 |
| 2 | 0.1498 |

Training loss kept falling in epoch 2 while validation error went up. With only 1,835 training clips, the second pass mostly fit the training set rather than improving on new audio.

## Training setup

| Setting | Value |
|---|---|
| Base model | `openai/whisper-large-v3-turbo` |
| Dataset | `ai4bharat/Kathbath`, Hindi config |
| Training data | `train[:2%]`, 1,835 clips (full split is 91,752) |
| Trainable weights | Decoder only (encoder frozen to fit in T4 memory) |
| Optimizer | AdamW, learning rate 1e-5 |
| Precision | float16 mixed precision with gradient scaling |
| Batch size | 2 |
| Epochs | 2 |
| Hardware | 1 × Tesla T4 (15 GB) |
| Training time | About 27 minutes per epoch |

## How it was scored

`compare.py` transcribes each validation clip with each model (Hindi, transcribe task, greedy decoding, float16), then compares the output with the reference transcript using `jiwer`.

- **Word error rate** is the share of words that are substituted, deleted, or inserted. Lower is better.
- **Character error rate** is the same measure at the character level, so a single wrong matra counts as a small error instead of a whole wrong word.

Before scoring, both the reference and the prediction have punctuation and symbols removed and their whitespace collapsed. Whisper's built-in `BasicTextNormalizer` wasn't used because it strips Devanagari vowel signs (for example, it turns "नमस्ते" into "नमस त"), which would make the scores misleading.

## Example transcriptions

**Reference:** हमने ओबीसी तबके के बच्चों के लिए उच्च शिक्षा में आरक्षण दिया

| Model | Output |
|---|---|
| base | हमने OBC तबके के बच्चों के लिए उच्छ शिक्षा में आरच्छन दिया |
| epoch-1 | हमने ओबीसी तबके के बच्चों के लिए उच्च शिक्षा में आरचन दिया |
| epoch-2 | हमने ओबीसी तबके के बच्चों के लिए उच्च शिक्षा में आरचन दिया |

**Reference:** उसको छोड़कर लोग इस प्रकार के मुद्दो पर चर्चा कर रहे है

| Model | Output |
|---|---|
| base | उसको छोड़कर लोग इस प्रकार के मुद्दों पर चर्चा कर रही हैं |
| epoch-1 | उसको छोड़कर लोग इस प्रकार के मुद्दों पर चर्चा कर रहे हैं |
| epoch-2 | उसको छोड़कर लोग इस प्रकार के मुद्दों पर चर्चा कर रहे हैं |

**Reference:** उन्होंने कहा कि अतिथि अध्यापको को हटाना न्यायसंगत नहीं है

| Model | Output |
|---|---|
| base | उन्होंने कहा कि अथिती अधिवापको कोटाना नया संगत नहीं है |
| epoch-1 | उन्होंने कहा कि अथिति अधिवाप्रों को कोटाना न्याय संगत नही है |
| epoch-2 | उन्होंने कहा कि अथिति अधिवाप्रों को कोटाना न्य संगत नहीं है |

**Reference:** चीन में मोबाइल फोन की लत एक युवती को बड़ी मुश्किल में डाल गई

| Model | Output |
|---|---|
| base | चीन में मुबाइल फोन की लत एक यूवदी को बड़ी मुश्किल में डाल गई |
| epoch-1 | चीन में मोबाइल फोन की लत एक युवद्धी को बड़ी मुश्किल में डाल गई |
| epoch-2 | चीन में मोबायल फोन की लत एक युवदी को बड़ी मुश्किल में डाल गई |

**Reference:** मगर साथ में मनपसंद साथी हो तो मेहनत भी रास आती है

| Model | Output |
|---|---|
| base | मगर साथ मेमन पसंद साथी हो तो मेहनत भी रास आती है |
| epoch-1 | मग़ साथ मेमन पसंद साथी हो तो महनत भी रास आती है |
| epoch-2 | मग़ साथ मेमन पसंद साथी हो तो महनत भी रास आती है |

## Observations

- The fine-tuned model writes English acronyms in Devanagari ("ओबीसी" rather than "OBC"), matching how Kathbath transcripts are written.
- Common words and grammar improved, for example "उच्च" for "उच्छ", "मोबाइल" for "मुबाइल", and "रहे हैं" for "रही हैं".
- Less common words are still wrong in every model, such as "अध्यापको" and "आरक्षण".
- Some outputs got worse. In the last example, the base model had "मगर" and "मेहनत" right and the fine-tuned models did not.

## Limitations

- Training used only 2% of the data, about 2–3 hours of audio.
- Validation used 500 clips. The 0.77-point gap between epoch 1 and epoch 2 is small enough that it could partly be noise.
- Only the decoder was trained; the encoder was frozen.
- There was one run with one learning rate and no hyperparameter search.

## Reproducing

```bash
# Train (writes whisper-hi/epoch-N after each epoch)
HF_DATASETS_OFFLINE=1 uv run main.py

# Compare the base model with the checkpoints on N validation clips (default 500)
HF_DATASETS_OFFLINE=1 uv run compare.py 500
```

The 500-clip comparison of three models takes about 15 minutes on a T4.
