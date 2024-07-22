# DATASET=Patents
DATASET=OAG_Venue
FOLDER=/root/TKDE/data
# export CUDA_VISIBLE_DEVICES=0,1,2

# train
# CUDA_VISIBLE_DEVICES=0,1,4 python main.py --feats 4 --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed
# CUDA_VISIBLE_DEVICES=1 python main.py --feats 4 --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed --mode infer --load True --load_ckpt_name ckpt/OAG_Venue/text-True-6e-05-100-768-tfidf-best-nograph.pt

# CUDA_VISIBLE_DEVICES=3,4,5 python main.py --feats 2 --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed
# CUDA_VISIBLE_DEVICES=5 python main.py --feats 2 --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed --mode infer --load True --load_ckpt_name ckpt/OAG_Venue/text-True-6e-05-100-768-tfidf-best-nograph.pt

CUDA_VISIBLE_DEVICES=3 python main.py --feats 3 --model_dir ckpt/$DATASET --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed
CUDA_VISIBLE_DEVICES=3 python main.py --feats 3 --model_dir ckpt/$DATASET --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed --mode infer --load True --load_ckpt_name ckpt/$DATASET/text-True-1e-05-100-768-tfidf-best-feat3.pt

# CUDA_VISIBLE_DEVICES=0,1,4 python main.py --feats 5 --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed
# CUDA_VISIBLE_DEVICES=1 python main.py --feats 5 --data_path $FOLDER/$DATASET --pretrain_embed True --pretrain_dir $FOLDER/$DATASET/pretrain_embed --mode infer --load True --load_ckpt_name ckpt/OAG_Venue/text-True-6e-05-100-768-tfidf-best-nograph.pt
