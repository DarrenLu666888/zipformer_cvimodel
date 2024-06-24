# Zipformer-cvimodel

本项目将zipformer的语音识别模型的onnx和cvimodel部署到算能搭载CV181x TPU芯片的开发板上 

其中onnx模型来自 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2 , 将里面的文件放入该目录中，运行test.sh即可



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

