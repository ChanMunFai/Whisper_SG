# Try without text Normalizer first 

import glob 
from pathlib import Path

import os
import numpy as np

import torch
from torch import nn
import pandas as pd
import whisper
import torchaudio
import torchaudio.transforms as at

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm import tqdm
import pyopenjtalk
import evaluate

DATASET_DIR = "/home/munfai98/Documents/Whisper_SG/jvs" # may need to change directory 
SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
SEED = 3407
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)

woptions = whisper.DecodingOptions(language="en", without_timestamps=True)
wmodel = whisper.load_model("base")
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=woptions.task)

from torch.utils.data import DataLoader
from dataloaders.nsc import NationalSpeechCorpusDataset, WhisperDataCollatorWhithPadding

train_path = "/hdd/NationalSpeechCorpus/part1/data/train/channel0"
val_path = "/hdd/NationalSpeechCorpus/part1/data/test/channel0"

class Config:
    learning_rate = 0.0005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 16
    num_worker = 2
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    sample_rate = SAMPLE_RATE

class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name, lang, train_dataset, val_dataset) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=self.options.task)

        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["mels"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_id):
        input_ids = batch["mels"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()


        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, collate_fn=WhisperDataCollatorWhithPadding(), num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, collate_fn=WhisperDataCollatorWhithPadding(), num_workers=0)



cfg = Config()
model_name = "base"
lang = "en"

train_dataset = NationalSpeechCorpusDataset(train_path, wtokenizer, DEVICE)
val_dataset = NationalSpeechCorpusDataset(val_path, wtokenizer, DEVICE)
whisper_model = WhisperModelModule(cfg, model_name, lang, train_dataset, val_dataset)

loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, collate_fn=WhisperDataCollatorWhithPadding(), num_workers=0)

from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()

refs = []
res = []
for b in tqdm(loader):
    input_ids = b["mels"].half().cuda()
    labels = b["labels"].long().cuda()
    with torch.no_grad():
        results = whisper_model.model.decode(input_ids, woptions)
        for r in results:
            if normalizer: 
                res.append(normalizer(r.text))
            else: 
                res.append(r.text)
        
        for l in labels:
            l[l == -100] = wtokenizer.eot
            ref = wtokenizer.decode(l, skip_special_tokens=True)
            if normalizer: 
                refs.append(normalizer(ref))
            else: 
                refs.append(ref) 

cer_metrics = evaluate.load("cer")
cer_score = cer_metrics.compute(references=refs, predictions=res)

wer_metrics = evaluate.load("wer")
wer_score = wer_metrics.compute(references=refs, predictions=res)

print("Initial Model - CER: ", cer_score) #  0.10137379582075763 or 0.06021966471660497
print("Initial Model - WER: ", wer_score) # 0.2999281144191108 or 0.1204662995806523

checkpoint_path = "/home/munfai98/Documents/Whisper_SG/jap/artifacts/checkpoint/checkpoint-epoch=0004-v1.ckpt"
state_dict = torch.load(checkpoint_path)
state_dict = state_dict['state_dict']
whisper_model.load_state_dict(state_dict)

refs = []
res = []

for b in tqdm(loader):
    input_ids = b["mels"].half().cuda()
    labels = b["labels"].long().cuda()
    with torch.no_grad():
        results = whisper_model.model.decode(input_ids, woptions)
        for r in results:
            if normalizer: 
                res.append(normalizer(r.text))
            else: 
                res.append(r.text)
        
        for l in labels:
            l[l == -100] = wtokenizer.eot
            ref = wtokenizer.decode(l, skip_special_tokens=True)
            if normalizer: 
                refs.append(normalizer(ref))
            else: 
                refs.append(ref) 

cer_metrics = evaluate.load("cer")
cer_score = cer_metrics.compute(references=refs, predictions=res)

wer_metrics = evaluate.load("wer")
wer_score = wer_metrics.compute(references=refs, predictions=res)

print("Finetuned Model - CER: ", cer_score) # 0.04613077655197783 or 0.03992734013174651
print("Finetuned Model - WER: ", wer_score) # 0.09351335250983822 or 0.07339838866799621
