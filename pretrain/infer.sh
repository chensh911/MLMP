DATASET=GoodReads
FOLDER=/home/chenshangheng/Desktop/MELON/data

# export CUDA_VISIBLE_DEVICES=1,2,3

# train
# python -m torch.distributed.launch --nproc_per_node=3 --master_port=25600  main.py --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed

# infer
# CUDA_VISIBLE_DEVICES=1 python main.py --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed --mode infer --load True --load_ckpt_name ckpt/OAG_Venue/text-True-0.0001-100-768-tfidf-best-nograph.pt

CUDA_VISIBLE_DEVICES=4 python main.py --feats 4 --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed --mode infer --load True --load_ckpt_name ckpt/OAG_Venue/text-True-6e-05-100-768-tfidf-best-nograph.pt
