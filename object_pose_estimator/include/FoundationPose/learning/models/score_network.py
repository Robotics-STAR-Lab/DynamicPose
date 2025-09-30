# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../../../../')
from Utils import *
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
from network_modules import *
from Utils import *




class ScoreNetMultiPair(nn.Module):
  def __init__(self, cfg=None, c_in=4):
    super().__init__()
    self.cfg = cfg
    if self.cfg.use_BN:
      norm_layer = nn.BatchNorm2d
    else:
      norm_layer = None

    self.encoderA = nn.Sequential(
      ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
    )

    self.encoderAB = nn.Sequential(
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
    )

    embed_dim = 512
    num_heads = 4
    self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True)
    self.att_cross = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True)

    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)
    self.linear = nn.Linear(embed_dim, 1)


  def extract_feat(self, A, B):
    """
    @A: (B*L,C,H,W) L is num of pairs
    """
    bs = A.shape[0]  # B*L

    x = torch.cat([A,B], dim=0)
    x = self.encoderA(x)
    a = x[:bs]
    b = x[bs:]
    ab = torch.cat((a,b), dim=1)
    ab = self.encoderAB(ab)
    ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))
    ab, _ = self.att(ab, ab, ab)
    return ab.mean(dim=1).reshape(bs,-1)


  def forward(self, A, B, L):
    """
    @A: (B*L,C,H,W) L is num of pairs
    @L: num of pairs
    """
    output = {}
    bs = A.shape[0]//L
    feats = self.extract_feat(A, B)   #(B*L, C)
    x = feats.reshape(bs,L,-1)
    x, _ = self.att_cross(x, x, x)

    output['score_logit'] = self.linear(x).reshape(bs,L)  # (B,L)

    return output
  
# import tensorrt as trt
# logger = trt.Logger(trt.Logger.INFO)

# class ScoreNetMultiPair_trt(nn.Module):
#   def __init__(self, model_path):
#     super(ScoreNetMultiPair_trt, self).__init__()
#     # engine_path = "/home/linbei/workspace/d6d/src/Dynamic_6D/TEST/dy_model.trt"
#     engine_path = model_path
#     with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
#         engine = runtime.deserialize_cuda_engine(f.read())
#     self.engine = engine
#     if self.engine is not None:
#       self.context = self.engine.create_execution_context()
    
#     input_names = ["A", "B", "L"]
#     output_names = ["output"]
#     self.input_names = input_names
#     self.output_names = output_names
    
#     # self.bs = 8
  
#   def forward(self, *inputs):
#     t = time.time()
#     batch_size = inputs[0].shape[0]
#     # print(f'batch size: {batch_size}')
    
#     bindings = [None] * (len(self.input_names) + len(self.output_names))
#     outputs = [None] * len(self.output_names)
#     for i, input_name in enumerate(self.input_names):
#       idx = self.engine.get_binding_index(input_name)
#       bindings[idx] = inputs[i].contiguous().data_ptr()
      
#       ### set binding shape
#       shape = inputs[i].shape
#       self.context.set_binding_shape(idx, shape)
      
#     for i, output_name in enumerate(self.output_names):
#       idx = self.engine.get_binding_index(output_name)
#       dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
      
#       # print(f'tuple(self.engine.get_binding_shape(idx) = {tuple(self.engine.get_binding_shape(idx))}')
#       # shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))
#       shape = (batch_size, self.engine.get_binding_shape(idx)[-1])
#       device = torch_device_from_trt(self.engine.get_location(idx))
#       # print(f'output shape: {shape}')
#       output = torch.empty(size=shape, dtype=dtype, device=device)
#       outputs[i] = output
#       bindings[idx] = output.data_ptr()
    
#     stream = torch.cuda.current_stream().cuda_stream
#     self.context.execute_async_v2(bindings, stream)  # Use execute_async_v2
    
#     outputs = tuple(outputs)
#     if len(outputs) == 1:
#         outputs = outputs[0]
#     # return outputs
#     # print(f'outputs: {outputs}')
#     res = {}
#     res['score_logit'] = outputs * -1.0
    
#     print(f'\033[31mforward time: {time.time()-t}\033[0m')
#     return res
