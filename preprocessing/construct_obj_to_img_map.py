import numpy as np
import os
import h5py
import numpy as np
import jsonlines
import re
import json


for caption_dataset_name in ['GCC', 'SS']:
    cococap = json.load(open(f'data/coco_%s.json'%caption_dataset_name))

    coco_vocab = []
    for ix in cococap['ix_to_word'].keys():
        coco_vocab.append(cococap['ix_to_word'][ix])
    coco_word_to_ix = {}
    for ix, word in cococap['ix_to_word'].items():
        coco_word_to_ix[word] = ix

    vg_vocab = []
    vg_f = open('data/objects_vocab.txt')
    for line in vg_f.readlines():
        objs = line.strip().split(',')
        obj = objs[0]
        for i in range(len(objs)):
            if ' ' not in objs[i]:
                obj = objs[i]
                break
        vg_vocab.append(obj)

    vg_ix_to_coco_ix = {1600: -1} # background
    for i in range(1600):
        w = vg_vocab[i]
        vg_ix = i
        if w in coco_word_to_ix:
            coco_ix = coco_word_to_ix[w]
        else:
            coco_ix = -1
        vg_ix_to_coco_ix[vg_ix] = coco_ix
    print(vg_ix_to_coco_ix.keys())

    ########
    i = 0
    save_dir = f'data/vg_obj_to_%s_word_ix/'%caption_dataset_name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for f in os.listdir('data/features'):
        i += 1
        item = np.load('data/features/'+f)
        id = f.split('.jpg')[0]
        vg_class_coco_ix = np.argmax(item['obj_probs'], 1)
        for j in range(vg_class_coco_ix.shape[0]):
            vg_class_coco_ix[j] = vg_ix_to_coco_ix[vg_class_coco_ix[j]]
        np.save(f'data/vg_obj_to_%s_word_ix/'%caption_dataset_name + str(int(id)), vg_class_coco_ix) # str(int(id)) for COCO, #str(id) for others, since coco is like 0000123.npy
        if i % 1000 == 0:
            print('Processing #', i)
    print('Finish extracting visual tokens!')

    # construct object ix to image ix mapping
    MIN_VIS_TOKEN_PER_SENT = 3

    obj_ix_to_img_ix = {}
    objs_dir = f'data/vg_obj_to_%s_word_ix/'%caption_dataset_name
    json_file = json.load(open(f'data/coco_%s.json'%caption_dataset_name))
    images = json_file['images']
    for ix in range(len(images)):
        image = images[ix]
        if image['split'] != 'train':
            continue # only using training set
        img_id = int(image['id'])
        img_object_ixs = list(np.load(objs_dir+str(img_id)+'.npy'))
        for obj_ix in img_object_ixs:
            if obj_ix != -1:
                if obj_ix not in obj_ix_to_img_ix:
                    obj_ix_to_img_ix[obj_ix] = set()
                obj_ix_to_img_ix[obj_ix].add(ix)
        if ix % 1000 == 0:
            print('Processing ', ix, '/', len(images))

    np.save(f'data/coco_%s_obj_ix_to_img_ix'%(caption_dataset_name),
            np.array(dict(obj_ix_to_img_ix)))

    # filter useful sentences
    useful_sentence_ixs = set()
    count = 0

    labels = h5py.File(f'data/%s_label.h5' % caption_dataset_name, 'r')['labels']
    for ix in range(len(labels)):
        sentence = labels[ix]
        if sum([word_ix in obj_ix_to_img_ix for word_ix in sentence]) >= MIN_VIS_TOKEN_PER_SENT:
            useful_sentence_ixs.add(ix)
        count += 1
        if count % 1000 == 0:
            print('Processing ', count, '/', len(labels))

    np.save(f'data/coco_%s_useful_sentence_ixs'%caption_dataset_name, np.array(list(useful_sentence_ixs)))
    print('Finish constructing the obj-to-img-ix mapping!')