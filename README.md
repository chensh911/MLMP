# MLMP: Metapath-Enhanced Language Model Pretraining on Text-Attributed Heterogeneous Graphs

This repository contains the source code and datasets for MLMP: Metapath-enhanced Language Model Pretraining on Text-Attributed Heterogeneous Graphs.

## Links

- [Datasets](#datasets)
- [Preprocess](#preprocess)
- [Pretraining](#pretraining)
- [Finetuning](#finetuning)


## Datasets
**Download processed data.** To reproduce the results in our paper, you need to first download the processed [datasets](https://github.com/Hope-Rita/THLM). You need to also download [bert-base-cased](https://huggingface.co/bert-base-cased) and put them into ```./data```.

## Preprocess
You need to execute ```./data/data_process.ipynb``` for OAG-Venue dataset and ```./data/data_process_googreads.ipynb``` for GoodReads dataset.

## Pretraining
Pretraining in ```./pretrain```.
```
sh run.sh
```


## Finetuning

### Node Classification

Run node classification in ```./downstream/node-classification```.
```
sh run.sh
```


### Link Prediction
Run link prediction in ```./downstream/link-predict```.
```
sh run.sh
```
