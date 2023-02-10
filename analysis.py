import configargparse
import torch
import whisper
from whisper.normalizers import EnglishTextNormalizer

from tqdm import tqdm
import evaluate
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from dataloaders.nsc import NationalSpeechCorpusDataset, WhisperDataCollatorWhithPadding
from main import WhisperModelModule

SEED = 3407
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)

def evaluate_scores(model, options, loader, normalizer):
    """ Evaluates CER and WER for the entire loader. 
    """
    refs = []
    res = []
    for b in tqdm(loader):
        input_ids = b["mels"].half().cuda()
        labels = b["labels"].long().cuda()
        with torch.no_grad():
            results = model.model.decode(input_ids, options)
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

    return cer_score, wer_score


if __name__ == "__main__": 
    val_path = "/hdd/NationalSpeechCorpus/part1/data/test/channel0"

    woptions = whisper.DecodingOptions(language="en", without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=woptions.task)

    val_dataset = NationalSpeechCorpusDataset(val_path, wtokenizer, "cpu")
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=WhisperDataCollatorWhithPadding(), num_workers=12)

    normalizer = EnglishTextNormalizer()

    parser = configargparse.ArgParser()
    parser.add('-c', '--config', default='config/base_config.yaml', is_config_file=True, help='config file path')
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

    wmodel = WhisperModelModule(args=args)

    # cer_score, wer_score = evaluate_scores(wmodel, woptions, val_loader, normalizer)

    # print("Initial Model - CER: ", cer_score) 
    # print("Initial Model - WER: ", wer_score) 

    checkpoint_path = "/home/munfai98/Documents/Whisper_SG/saved_models/base/00003/checkpoint/checkpoint-epoch=0004.ckpt"
    state_dict = torch.load(checkpoint_path)
    state_dict = state_dict['state_dict']
    wmodel.load_state_dict(state_dict)

    cer_score, wer_score = evaluate_scores(wmodel, woptions, val_loader, normalizer)
    print("Finetuned Model - CER: ", cer_score) 
    print("Finetuned Model - WER: ", wer_score) 
