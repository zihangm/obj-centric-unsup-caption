from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
from captioning.utils import misc as utils
# from .local_optimal_transport import local_OT

# load coco-caption if available
try:
    sys.path.append("coco-caption")
    sys.path.append("/home/zihang/Research/FB2021/ImageCaptioning.pytorch")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap #COCOEvalCap_spice
except:
    print('Warning: coco-caption not available')

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def getCOCO(dataset):
    if 'coco' in dataset:
        # annFile = 'coco-caption/annotations/captions_val2014.json' # this is coco original caption

        #annFile = 'coco-caption/annotations/captions_LN_val2014_norepeat.json' # load 8k LN validation set
         
        annFile = 'coco-caption/annotations/captions_coco_Chinese_test.json' # coco Chinese test set
        
        #annFile = 'coco-caption/annotations/captions_French-multi30k_coco_test.json' # French-multi30k on coco test

    # elif 'flickr30k' in dataset or 'f30k' in dataset or 'flk30k' in dataset:
    #     annFile = 'coco-caption/annotations/captions_flk30k_LN_test.json'
    # elif 'ade20k' in dataset:
    #     annFile = 'coco-caption/annotations/captions_ade20k_LN_test.json'
    # elif 'openimg' in dataset:
    #     annFile = 'coco-caption/annotations/captions_openimg_LN_test.json'
    print(annFile)
    return COCO(annFile)


#'/home/zihang/Research/FB2021/ImageCaptioning.pytorch/eval_results/saved_captions_try9_coco_GCC_sentence_multi_img_val.json'
#'/home/zihang/Downloads/Feng2019_coco_test.json'

#cache_path = '/home/zihang/Research/FB2021/ImageCaptioning.pytorch/eval_results/unsup_try17_coco_OPEN_LN_SG_sentence_multi_img_rel_attr_val.json' #'/home/zihang/Research/FB2021/RemovingSpuriousAlignment/saved_models/saved/output.json'

cache_path = '/home/zihang/Research/FB2021/RemovingSpuriousAlignment/saved_models/saved_cocoCN_30epoch/output_int_id.json'

#cache_path = '/home/zihang/Research/FB2021/ImageCaptioning.pytorch/eval_results/.cache_try76_coco_coco-caption-CN-split_translate_into_Chinese_val.json' 
#'/home/zihang/Downloads/Feng2019_coco_test.json'

loadres = json.load(open(cache_path))
'''
if 'COCO' in str(loadres[0]['image_id']):
    # rewrite it
    for line in loadres:
        line['image_id'] = int(line['image_id'].split('_')[-1].split('.')[0])
    with open('/home/zihang/Research/FB2021/RemovingSpuriousAlignment/saved_models/saved/output_int_id.json', 'w') as outfile:
        json.dump(loadres, outfile)
    cache_path = '/home/zihang/Research/FB2021/RemovingSpuriousAlignment/saved_models/saved/output_int_id.json'
'''

score_list = []
size_per_split = 100000000
l = len(json.load(open(cache_path)))
num_splits = (l//size_per_split) + (1 if (l%size_per_split)!=0 else 0)
for i in range(num_splits):
    coco = getCOCO('coco')
    valids = coco.getImgIds()

    cocoRes = coco.loadRes(cache_path)#, split=i, size_per_split = size_per_split)
    cocoEval = COCOEvalCap(coco, cocoRes) #_spice
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    try:
        cocoEval.evaluate()
    except:
        print('this split fail: #', i)
        continue

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
        # score_list.append(score)
    # print(i, '-th current_split:', score, 'Overall ave:', sum(score_list) / len(score_list))
# print(score_list)
# print(sum(score_list) / len(score_list))

# # Add mean perplexity
# out['perplexity'] = mean_perplexity
# out['entropy'] = mean_entropy

imgToEval = cocoEval.imgToEval

# for k in list(imgToEval.values())[0]['SPICE'].keys():
#     if k != 'All':
#         out['SPICE_' + k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
#         out['SPICE_' + k] = (out['SPICE_' + k][out['SPICE_' + k] == out['SPICE_' + k]]).mean()
# for p in preds_filt:
#     image_id, caption = p['image_id'], p['caption']
#     imgToEval[image_id]['caption'] = caption
#
# out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
# outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
# with open(outfile_path, 'w') as outfile:
#     json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
