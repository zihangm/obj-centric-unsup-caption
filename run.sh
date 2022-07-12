#!/bin/bash

# for COCO-> GCC
#: '
# Train
python tools/train.py --language_eval 0 --id coco_GCC_exp \
--caption_model transformer --input_json data/coco_GCC.json --input_label_h5 data/GCC_label.h5 --batch_size 100 \
--learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3 \
--save_checkpoint_every 5000 --max_epochs 30 --max_length=20 --seq_per_img=1 --num_layers=1 --use_box 1 \
--input_att_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/feats_only \
--input_box_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/box_only  \
--input_vg_ix_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/vg_obj_to_GCC_word_ix \
--obj_ix_to_img_ix data/coco_GCC_obj_ix_to_img_ix.npy \
--useful_sentence_ixs data/coco_GCC_useful_sentence_ixs.npy \
--beam_size=1 --val_images_use=-1 --save_every_epoch >> logs/try67_coco_GCC_sentence_multi_img_half_objs_replace.txt &&

# Test
python tools/train.py --debug 1 --language_eval 1 --id coco_GCC_exp \
--caption_model transformer --input_json data/coco_GCC.json --input_label_h5 data/GCC_label.h5 --batch_size 4 \
--learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3 \
--save_checkpoint_every 5000 --max_epochs 31 --max_length=20 --seq_per_img=1 --num_layers=1 --use_box 1 \
--input_att_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/feats_only \
--input_box_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/box_only  \
--input_vg_ix_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/vg_obj_to_GCC_word_ix \
--obj_ix_to_img_ix data/coco_GCC_obj_ix_to_img_ix.npy \
--useful_sentence_ixs data/coco_GCC_useful_sentence_ixs.npy \
--beam_size=5 --val_images_use=-1 --save_every_epoch  >> logs/try67_coco_GCC_sentence_multi_img_half_objs_replace_beam5.txt
#'

# for COCO-> SS
: '
# Train
python tools/train.py --language_eval 0 --id coco_SS_exp \
--caption_model transformer --input_json data/coco_SS.json --input_label_h5 data/SS_label.h5 --batch_size 100 \
--learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3 \
--save_checkpoint_every 5000 --max_epochs 30 --max_length=20 --seq_per_img=1 --num_layers=1 --use_box 1 \
--input_att_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/feats_only \
--input_box_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/box_only  \
--input_vg_ix_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/vg_obj_to_SS_word_ix \
--obj_ix_to_img_ix data/coco_SS_obj_ix_to_img_ix.npy \
--useful_sentence_ixs data/coco_SS_useful_sentence_ixs.npy \
--beam_size=1 --val_images_use=-1 --save_every_epoch  >> logs/try68_coco_SS_sentence_multi_img_half_objs_replace.txt &&

# Test
python tools/train.py --debug 1 --language_eval 1 --id coco_SS_exp \
--caption_model transformer --input_json data/coco_SS.json --input_label_h5 data/SS_label.h5 --batch_size 8 \
--learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3 \
--save_checkpoint_every 5000 --max_epochs 31 --max_length=20 --seq_per_img=1 --num_layers=1 --use_box 1 \
--input_att_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/feats_only \
--input_box_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/box_only  \
--input_vg_ix_dir /mnt/m2/Datasets/COCO/COCO_feats_detectron2/vg_obj_to_SS_word_ix \
--obj_ix_to_img_ix data/coco_SS_obj_ix_to_img_ix.npy \
--useful_sentence_ixs data/coco_SS_useful_sentence_ixs.npy \
--beam_size=5 --val_images_use=-1 --save_every_epoch >> logs/try68_coco_SS_sentence_multi_img_half_objs_replace_beam5.txt
'


