# Automatic Speech Recognition with Whisper: Tagalog

##### Table of Contents  
[Background](#background)  

[Method](#method)

[Results](#results) 

[Discussion](#discussion)  

[References](#references)



## Background

### Motivation

- For Automatic Speech Recognition (ASR), there is the problem of the "limiting size of existing high-quality supervised datasets" (Radford et al., 2022, p.1).
- Another problem is generalization for models when evaluating datasets not including those they were trained on; these models may be state-of-the-art (SOTA) for their respective datasets, but can be much less robust when tested using outside datasets
- In image recognition, using much larger but weakly supervised datasets compared to high-quality datasets has greatly improved generalizability. This could be applied to ASR
- Attempts to address the limiting size of high-quality datasets include simply adding more training hours (e.g., GigaSpeech with 10k hours of audio data) or using self-supervised learning (e.g., wav2vec is SOTA with 100x less labeled data); however, this does not address the generalization problem
- For generalization, one attempt was using as much data as possible of different datasets to train model (SpeechStew); this is more like a precursor that achieves SOTA performance even without a language model (e.g., beating HMMs with language models)
- Prior to Whisper, scaling weakly-supervised datasets has scarcely been attempted, if at all

### Problem Addressed
- Whisper builds on previous works by addresses both problems: 1) It scales existing high-quality datasets to 680k hours of weakly-supervised labeled audio data and 2) uses a diverse corpus of data training simultaneously on multiple tasks (e.g., transcription, translation, speech recognition) to address the generalization problem
- However, the model only uses 117,000 hours for the rest of 96 languages. This means low-resource languages have much less training time in comparison to their counterparts
- I set out to answer if the current transcription metrics can be improved by further fine-tuning a checkpoint on a specific language--Tagalog--and to demonstrate this newly-tuned Whisper model using this language


## Method

### Data
- Fleurs Dataset: ~1.36 GB of data in Tagalog, or about 10 hours of audio data in the training set. There are 1884 rows in the training set, 418 rows in the validation set, and 964 rows in the test set. This means that the dataset is small; we will 
Dataset link:
[](https://huggingface.co/datasets/google/fleurs/tree/main)

- Bloom-Speech used primarily for testing; we used the test set, which had 50 rows, for the test set. The training set has 352 rows, so this dataset is tiny in comparison.
Dataset link:
[](https://huggingface.co/datasets/sil-ai/bloom-speech)

### Preparation
- Prepared data by using the prepare_dataset function, which uses the Whisper processor for feature extraction to resample audio to 16k frequency and convert amplitude array to log-Mel spectrogram using the transformation of the Fast Fourier Transform, as well as tokenization using Byte-pair encoder which adds the required special tokens. Whisper takes the "minimalist" approach by not doing significant standardization in the hopes of further improving generalization.
- A data collator is needed to pad input features and labels to the length of the longest entry in the batch (NOT the entire dataset), reminiscent of the masked softmax we learned in lecture for Seq2Seq. According to the huggingface transformers docs, there is no data collator for ASR, so we needed to adapt the DataCollatorWithPadding class.

### Model
![Transformer Architecture used by Radford et al.,2022](https://github.com/user-attachments/assets/8964ee8f-036a-46c2-9b79-9aeedeae60a6)
*Transformer Architecture used by Radford et al.,2022*
- The Whisper model follows the transformer architecture. I note here of the model’s use of learned positional encoding in the decoder and the encoder processes small stem consisting of two convolutional layers with kernel width of 3 and GELU activation. Note as well that the input of the encoder is the Log-Mel Spectrogram. These are additions to the transformer architecture from lecture.
- We chose to use the Whisper-small model due to the limited size of our dataset.

### Experimental Design
- We set the Seq2SeqTrainingArguments to be able to save checkpoints, specify the learning rate as small (1e-5), and set to report metrics. We defined the arguments following the docs: Trainer
- We trained using the Seq2SeqTrainer on the train and validation splits pre-set by the Fleurs dataset; we did not touch the test split until the very end, where we made sure to set with no gradient for a proper testing
- The trainer allowed us to push the intermediate results to the Hub and save steps as we went



## Results

- The WER achieved from the Fleurs test set was 16.08%, while the WER from the bloom-speech test set was 16.63%. This seemed to indicate that our training did handle generalization well as the discrepancy between both WER scores were not significant.
![image](https://github.com/user-attachments/assets/7eab8915-4485-4e0a-baff-c31f25f5c33c)
- You might notice overfitting. This is likely the result of such a small dataset and using the Whisper-small model.
- Compare the WER from the Whisper model. As you can see, our results beat the results of the Transcription from the Whisper original paper by over 10%. This indicates that we were able to see improvements by fine-tuning the model on more data of this low-resource language.

### Live Demo
We were able to create a live demo using Gradio. Here are the results for a locally-run website:

![image](https://github.com/user-attachments/assets/28affd06-8d9b-4378-aeaf-8262f2f33a47)


And for mobile:


![image](https://github.com/user-attachments/assets/68fd3a46-ed06-4a38-8e5a-5cae73fac0a6)


![image](https://github.com/user-attachments/assets/84f51420-9658-4b50-a007-c9b28c333072)



## Discussion

- We need a lot more data if we want the WER to get below 10%. Even English transcription doesn't get below 5% for most datasets evaluated in the Whisper paper, and that is the most high resource language.
- We need more data even still for translation from Tagalog; we could not find a single one on HuggingFace
- There were only 2 datasets usable on HuggingFace for the task of transcription of Tagalog
- Using basic-normalize for the correct calculation of the WER rate helped produce correct results
- When switching between Colab and Wulver, difference between library versions made a big difference and would prevent certain versions from running or not (e.g., NumPy had to be less than 2.0.0)



## References
  Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023, July). Robust speech recognition via large-scale weak supervision. In International conference on machine learning (pp. 28492-28518). PMLR.
