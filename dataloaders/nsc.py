# Currently, attention mask ignores all padded tokens
# Padded tokens are being decoded as EOT tokens - is it a problem that that we do not have true EOT tokens
# Tokenisers read text differently if they are at front of sentence - is this accounted for here? 

import glob 
import os 
import re 
import numpy as np 

import torch 
from torch.utils.data import DataLoader
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F

import json 
import matplotlib.pyplot as plt
# from playsound import playsound
from tqdm import tqdm 
import whisper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NationalSpeechCorpusDataset(Dataset):
    def __init__(self, root, tokenizer, device = DEVICE): 
        self.file_locations = glob.glob(f"{root}/wave/**/*.WAV", recursive= True) 
        scripts = glob.glob(f"{root}/labels/**/*.TXT", recursive= True) 
        
        self.scripts = {}
        for element in scripts: 
            match = re.search(r'(?<=/)[^/]+\.TXT', element)
            file_name = match.group()
            key = file_name.split('.')[0] 
            # first digit represents channel number 
            # next 4 digits represent speaker ID 
            # last digit represents session number 
            self.scripts[key] = element
      
        self.device = device
        self.tokenizer = tokenizer
    
    def __len__(self): 
        return len(self.file_locations)

    def __getitem__(self, idx):
        """
        Arguments: 
            idx: index of item 
        Returns: 
            Dictionary of {
                mels: log-mel spectogram of audio recording (trimmed/padded to 30 seconds)
                sentence: corresponding text of audio file
                labels: labels in tokenized form. Does not include 1st BOS token and includes EOS token. Used for CE loss later. 
                dec_input_ids: Similar to labels but includes all BOS token and does not include EOS token.  
            }

        """
        file_path = self.file_locations[idx]

        # Speaker Number
        match = re.search(r'(?<=wave/SPEAKER)[^/]+', file_path)
        speaker = match.group() # 4 digit number

        # Full file name
        match = re.search(r'(?<=/)[^/]+\.WAV', file_path)
        file_name = match.group()
        file_name = file_name.split('.')[0]
        session = file_name[-4]

        # Channel 
        match = re.search(r'(?<=channel)[^/]+', file_path)
        channel = match.group().replace(" ", "") # 4 digit number

        script_filename = channel + speaker + session
        script_path = self.scripts[script_filename]

        ### Extract using new format 
        with open(script_path, "r") as f: 
            data = json.load(f)
            sentence = data[file_name]

        waveform, sample_rate = torchaudio.load(file_path)

        assert sample_rate ==16000
        audio = whisper.pad_or_trim(waveform.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)

        dec_input_ids = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(sentence)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]

        return {
            "mels": mel,
            "sentence": sentence, 
            "labels": labels, 
            "dec_input_ids": dec_input_ids 
        }
    


class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        mels, sentences, labels, dec_input_ids =  [], [], [], []
        
        for f in features:
            mels.append(f["mels"])
            sentences.append(f["sentence"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        mels = torch.stack(mels)
        labels = torch.tensor(np.array(labels))
        dec_input_ids = torch.tensor(np.array(dec_input_ids))

        batch = {
            "mels": mels, 
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "sentences": sentences
        }

        return batch


if __name__ == "__main__": 
    train_path = "/hdd/NationalSpeechCorpus/part1/data/train/channel0"
    test_path = "/hdd/NationalSpeechCorpus/part1/data/test/channel0"
    woptions = whisper.DecodingOptions(language="en", without_timestamps=False)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=woptions.task)

    train_dataset = NationalSpeechCorpusDataset(train_path, wtokenizer)
    test_dataset = NationalSpeechCorpusDataset(test_path, wtokenizer)
    print(len(train_dataset), len(test_dataset))

    trainloader = DataLoader(train_dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())
    testloader = DataLoader(test_dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

    sample_batch = next(iter(trainloader))
    print(sample_batch["labels"][0])
    print(sample_batch["dec_input_ids"][0])
    print(sample_batch["sentences"])
    print("\n")

    for b in trainloader:
        print(b["labels"].shape)
        print(b["mels"].shape)
        print(b["dec_input_ids"].shape)

        for token, dec in zip(b["labels"], b["dec_input_ids"]):
            token[token == -100] = wtokenizer.eot
            text = wtokenizer.decode(token, skip_special_tokens=False)
            print(text)

            dec[dec == -100] = wtokenizer.eot
            text = wtokenizer.decode(dec, skip_special_tokens=False)
            print(text)
    
        break
