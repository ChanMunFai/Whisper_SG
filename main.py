import os 
import numpy as np
import configargparse
import whisper

import torch
from torch import nn

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm import tqdm
import evaluate

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from torch.utils.data import DataLoader
from dataloaders.nsc import NationalSpeechCorpusDataset, WhisperDataCollatorWhithPadding

# import warnings
# 
import warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
warnings.filterwarnings(
    "ignore", "Detected call of", UserWarning
)


SEED = 3407
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)


class WhisperModelModule(LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = kwargs['args']
        
        self.options = whisper.DecodingOptions(language='en', without_timestamps=True)
        self.model = whisper.load_model(self.args.model)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=self.options.task)

        # only decoder training
        if self.args.decoder_only == "True": 
            for p in self.model.encoder.parameters():
                p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
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
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.args.learning_rate, 
                          eps=self.args.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.train_dataset) // (self.args.batch_size))
                // self.args.gradient_accumulation_steps
                * float(self.args.epochs)
            )
    
    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=32, collate_fn=WhisperDataCollatorWhithPadding(), num_workers=0)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=32, collate_fn=WhisperDataCollatorWhithPadding(), num_workers=0)

if __name__ == "__main__": 
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', required=True, is_config_file=True, help='config file path')
    parser.add('-m', '--model', choices=['tiny', 'base', 'small', 'medium', 'large'], help='Model Size')
    parser.add('--exp_id', type=str, help='Experiment ID')

    parser.add('-e', '--epochs', type=int, help='Number of training epochs')
    parser.add('--decoder_only', type=str, choices=['True', 'False'], help='If True, encoder will be frozen during training.')
    parser.add('-bs', '--batch_size', type=int, required=False, help='Batch Size')
    parser.add('-lr', '--learning_rate', type=float, required=False, help='Learning Rate')

    parser.add('--weight_decay', type=float, help='Weight Decay')
    parser.add('--adam_epsilon', type=float, help='Adam Epsilon')
    parser.add('--warmup_steps', type=int, help='Number of Warmup Steps')
    parser.add('--gradient_accumulation_steps', type=int, help='Gradient Accumulation Steps')

    # Deprecated arguments 
    parser.add('--wandb_on', choices=["True", "None"], help='DEPRECATED: Flag for wandb')
    parser.add('--save_every', type=int, help='DEPRECATED: Number of training steps to save model')
    parser.add('--log_every', type=int, required=False, help='DEPRECATED: Number of training steps to output results in log file')
    args = parser.parse_args()

    woptions = whisper.DecodingOptions(language="en", without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=woptions.task)

    train_path = "/hdd/NationalSpeechCorpus/part1/data/train/channel0"
    val_path = "/hdd/NationalSpeechCorpus/part1/data/test/channel0"
    
    train_dataset = NationalSpeechCorpusDataset(train_path, wtokenizer, "cpu")
    val_dataset = NationalSpeechCorpusDataset(val_path, wtokenizer, "cpu")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=WhisperDataCollatorWhithPadding(), num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=WhisperDataCollatorWhithPadding(), num_workers=12)

    log_output_dir = f"logs/{args.model}/{args.exp_id}/"
    check_output_dir = f"saved_models/{args.model}/{args.exp_id}/"

    if not os.path.exists(log_output_dir):
        os.makedirs(log_output_dir)

    if not os.path.exists(check_output_dir):
        os.makedirs(check_output_dir)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=args.model,
        version=args.exp_id
    )
    
    # How to save more frequently 
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1 # all model save
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(args=args) 

    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader) # Move train and test dataloaders here 