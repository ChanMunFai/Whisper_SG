# WhisperSG

This repository seeks to finetune [OpenAI Whisper](https://openai.com/blog/whisper/) on [Singaporean English](https://en.m.wikipedia.org/wiki/Singapore_English).

The data comes from the [National Speech Corpus(NSC)](https://www.imda.gov.sg/nationalspeechcorpus) spearheaded by the 
Info-communications and Media Development Authorithy(IMDA) of Singapore

Whisper by OpenAI is an Automatic Speech Recognition(ASR) model trained on 680,000 hours of supervised data coming from multiple languages and accents. Due to this large and diverse training data, Whisper is robust and attains good zero-shot performance across different accents and languages. However, finetuning Whisper will often generate better in-domain performance than its initial pretrained model.

# Model Versions
### v1.1
I have used a subset(~150GB) of the NSC data, with a 80/20 training and testing split. Currently, all train and test data are single-speaker, studio-recorded audio files.
Performance after finetuning for 5 epochs on the 'base' model improved performance from a Wrong Error Rate(WER) of 12% to 7%.
Training on my RTX 3090Ti took about 3-4 hours.

# What this Model is NOT about
This model has not been fine tuned on any data containing [Singlish](https://eresources.nlb.gov.sg/infopedia/articles/SIP_1745_2010-12-29.html). 
I am unable to find any large-scale labelled audio datasets, and would have to figure out how to train the tokenizer/embedding layer for new Singlish words. 
Kindly get in touch if you would like to collaborate on this :) 

# Roadmap 
- [ ] Finetune Whisper on different model sizes on the entire NSC dataset. Other parts of the NSC dataset include multiple-speaker and slightly noisier recordings. 
- [ ] Data augmentation (i.e. adding noise) to improve real-world performance 
- [ ] App
    - [ ] I will be publishing a Gradio demo soon but I hope to learn some ML Ops to develop an app that performs automatic transcription of Singaporean videos.
