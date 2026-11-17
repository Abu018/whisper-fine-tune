from src.config import config
from datasets import Audio, load_dataset
import json
from pathlib import Path
import soundfile as sf
import re 



def manifest_file(split, num_samples,is_train : bool):
    ds= load_dataset(
        config.DATASET_ID, 
        config.DATASET_CONFIG, 
        split=split,
         streaming=True
        ).cast_column("audio_filepath", Audio(sampling_rate=config.SAMPLING_RATE)).take(num_samples)
    if is_train:
        manifest_name = "train.json"
        wav_dir = Path("data/wavs/train")
    else:
        manifest_name = "valid.json"
        wav_dir=Path("data/wavs/valid")
    if Path(manifest_name).exists():
        print(f"{manifest_name} already exists. Skipping generation.")
        return
    wav_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_name, 'w',encoding="utf-8") as file:
        for i, val in enumerate(ds):
            audio = val["audio_filepath"]
            output_name = wav_dir / f"{i:06d}.wav"
            sf.write(output_name, audio["array"], audio["sampling_rate"])
            record={
                "audio_filepath":str(output_name.resolve()),
                "text":val["text"],
                "duration":val["duration"]
                } 
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
def preprocess():
    manifest_name = "preproc_train.json"
    if Path(manifest_name).exists():
        print(f"{manifest_name} already exists. Skipping generation.")
        return
    with open ("train.json",'r',encoding="utf-8")as input_file,\
        open(manifest_name,"w",encoding="utf-8")as outputfile: 
        for i,val in enumerate (input_file):
            val = json.loads(val)
            # result = re.sub(r'[^0-9]', ' ', val["text"])
            result = re.sub(r'\s+', ' ', val["text"]).strip()
            record={
                    "audio_filepath":val["audio_filepath"],
                    "text":result,
                    "duration":val["duration"]
                    } 
            outputfile.write(json.dumps(record, ensure_ascii=False) + "\n")



        
        

    

