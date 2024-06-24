# Zipformer-cvimodel

本项目将zipformer的语音识别模型的onnx和cvimodel部署到算能搭载CV181x TPU芯片的开发板上 

## 准备工作

onnx模型来自 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2 , 将里面的文件放入该目录中

目录结构：

```
.
├── README.md
├── assets
│   ├── image-20240624182705071.png
│   └── image-20240624182835736.png
├── decoder-epoch-99-avg-1.int8.onnx
├── decoder-epoch-99-avg-1.onnx
├── encoder-epoch-99-avg-1.int8.onnx
├── encoder-epoch-99-avg-1.onnx
├── joiner-epoch-99-avg-1.int8.onnx
├── joiner-epoch-99-avg-1.onnx
├── onnx_pretrained.py
├── scripts
│   ├── get_ioinfo.py
│   └── input_info.txt
├── test.sh
├── test_wavs
│   ├── 0.wav
│   ├── 1.wav
│   ├── 2.wav
│   ├── 3.wav
│   └── 8k.wav
└── tokens.txt
```

将onnx格式模型转为cvimodel（此处待完善）

## 运行

onnx推理程序运行环境为普通x86环境，安装相应的python包即可，运行./test_onnx.sh

```
(mlc-prebuilt) root@LAPTOP-N9IJI1E8:~/WORKSPACE_LTZ/_MLIR_/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20# ./test_onnx.sh
2024-06-25 00:06:39,862 INFO [onnx_pretrained.py:448] {'encoder_model_filename': 'encoder-epoch-99-avg-1.onnx', 'decoder_model_filename': 'decoder-epoch-99-avg-1.onnx', 'joiner_model_filename': 'joiner-epoch-99-avg-1.onnx', 'tokens': 'tokens.txt', 'sound_file': './test_wavs/0.wav'}
2024-06-25 00:06:41,111 INFO [onnx_pretrained.py:161] decode_chunk_len: 32
2024-06-25 00:06:41,111 INFO [onnx_pretrained.py:162] T: 39
2024-06-25 00:06:41,111 INFO [onnx_pretrained.py:163] num_encoder_layers: [2, 4, 3, 2, 4]
2024-06-25 00:06:41,111 INFO [onnx_pretrained.py:164] encoder_dims: [384, 384, 384, 384, 384]
2024-06-25 00:06:41,111 INFO [onnx_pretrained.py:165] attention_dims: [192, 192, 192, 192, 192]
2024-06-25 00:06:41,111 INFO [onnx_pretrained.py:166] cnn_module_kernels: [31, 31, 31, 31, 31]
2024-06-25 00:06:41,111 INFO [onnx_pretrained.py:167] left_context_len: [64, 32, 16, 8, 32]
2024-06-25 00:06:41,165 INFO [onnx_pretrained.py:241] context_size: 2
2024-06-25 00:06:41,165 INFO [onnx_pretrained.py:242] vocab_size: 6254
2024-06-25 00:06:41,184 INFO [onnx_pretrained.py:255] joiner_dim: 512
2024-06-25 00:06:41,185 INFO [onnx_pretrained.py:458] Constructing Fbank computer
2024-06-25 00:06:41,188 INFO [onnx_pretrained.py:461] Reading sound files: ./test_wavs/0.wav
2024-06-25 00:06:42,431 INFO [onnx_pretrained.py:513] ./test_wavs/0.wav
2024-06-25 00:06:42,431 INFO [onnx_pretrained.py:514] 昨天是 MONDAY TODAY IS LIBR THE DAY AFTER TOMORROW是星期三
2024-06-25 00:06:42,431 INFO [onnx_pretrained.py:516] Decoding Done
```

cvimodel可以在docker(sophgo/tpuc_dev:v3.2)里使用仿真器运行，也可以在搭载cv181x TPU芯片的开发板上运行时环境下执行推理

```
root@1bcf97f6a846:/workspace/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20# ./test_cvimodel.sh
{'encoder_model_filename': 'sherpa_encoder_bf16.cvimodel', 'decoder_model_filename': 'sherpa_decoder_bf16.cvimodel', 'joiner_model_filename': 'sherpa_joiner_bf16.cvimodel', 'tokens': 'tokens.txt', 'sound_file': './test_wavs/0.wav'}
decode_chunk_len: 32
T: 39
num_encoder_layers: [2, 4, 3, 2, 4]
encoder_dims: [384, 384, 384, 384, 384]
attention_dims: [192, 192, 192, 192, 192]
cnn_module_kernels: [31, 31, 31, 31, 31]
left_context_len: [64, 32, 16, 8, 32]
context_size: 2
vocab_size: 6254
joiner_dim: 512
Constructing Fbank computer
Reading sound files: ./test_wavs/0.wav
setenv:cv181x
Start TPU Simulator for cv181x
device[0] opened, 4294967296
version: 1.4.0
sherpa_encoder Build at 2024-06-18 11:54:20 For platform cv181x
(此处省略一些打印信息)
./test_wavs/0.wav
昨天是吧
Decoding Done
```

