# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note the original is here: https://github.com/facebookresearch/dino/blob/main/visualize_attention.py

import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from dinov2.models.vision_transformer import vit_small, vit_large


if __name__ == '__main__':
    # image_size = (952, 952)
    image_size = (224, 224)
    output_dir = '.'
    patch_size = 14

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model = vit_large(
    #         patch_size=14,
    #         img_size=526,
    #         init_values=1.0,
    #         #ffn_layer="mlp",
    #         block_chunks=0
    # )

    # model.load_state_dict(torch.load('/media/cpii/Drive 03/datasets/dinov2_vitl14_reg4_pretrain.pth'))
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', force_reload=True)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()

    # img = Image.open('cow-on-the-beach2.jpg')
    # img = Image.open('barn_00.jpg')
    # img = Image.open('479390531_a237b6b5f4_o.jpg')
    # img = Image.open('3395188399_c4be0555c7_o.jpg')
    img = Image.open('pad_img0.png')
    # img = Image.open('resize_keep_ar.png')
    img = img.convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img) # [3,952,952]
    print(img.shape)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0) # [1,3,952,952]

    w_featmap = img.shape[-2] // patch_size # 68
    h_featmap = img.shape[-1] // patch_size

    print(img.shape)

    feats, attentions = model.get_last_self_attention(img.to(device)) 
    # feats, [1,4625,1024]
    # attentions, [1,16,4625,4625], correlation volume of each patch

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    # for every patch, patch_id
    patch_id = 0
    attentions = attentions[0, :, patch_id, 1:].reshape(nh, -1) # [16,4624]
    # weird: one pixel gets high attention over all heads?
    print(torch.max(attentions, dim=1)) 
    # attentions[:, 4342] = 0 
    # print(torch.max(attentions, dim=1)) 
    # attentions[:, 4341] = 0 
    # print(torch.max(attentions, dim=1)) 
    # attentions[:, 4274] = 0 
    # print(torch.max(attentions, dim=1)) 

    attentions = attentions.reshape(nh, w_featmap, h_featmap) # 【16，68，68】
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy() # [16,952,952]
    
    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)

    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    
    fname = os.path.join(output_dir, "attn-head_average" + ".png")
    plt.imsave(fname=fname, arr=np.mean(attentions,axis=0), format='png')

    fname = os.path.join(output_dir, "attn-head_max" + ".png")
    plt.imsave(fname=fname, arr=np.max(attentions,axis=0), format='png')