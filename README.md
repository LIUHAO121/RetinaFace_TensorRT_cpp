# RetinaFace-TensorRT in C++

## Envirenment

Telsa = T4


CUDA = 11.1


TensorRT = 8.0.3.4

## Step 1: pth2onnx

1. get source code of yolox

```
git clone git@github.com:biubug6/Pytorch_Retinaface.git
cd Pytorch_Retinaface

```

2. modify the code

```
# line 17 in data/config.py
#'pretrain': True
'pretrain': False

# line 24 in models/retinaface.py
# return out.view(out.shape[0], -1, 2) is modified into 
return out.view(-1, int(out.size(1) * out.size(2) * 2), 2)

# line 35 in models/retinaface.py
# return out.view(out.shape[0], -1, 4) is modified into
return out.view(-1, int(out.size(1) * out.size(2) * 2), 4)

# line 46 in models/retinaface.py
# return out.view(out.shape[0], -1, 10) is modified into
return out.view(-1, int(out.size(1) * out.size(2) * 2), 10)

# The following modification ensures the output of resize node is based on scale rather than shape such that dynamic batch can be achieved.
# line 89 in models/net.py
# up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest") is modified into
up3 = F.interpolate(output3, scale_factor=2, mode="nearest")

# line 93 in models/net.py
# up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest") is modified into
up2 = F.interpolate(output2, scale_factor=2, mode="nearest")

# The following code removes softmax (bug sometimes happens). At the same time, concatenate the output to simplify the decoding.
# line 123 in models/retinaface.py
# if self.phase == 'train':
#     output = (bbox_regressions, classifications, ldm_regressions)
# else:
#     output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
# return output
# the above is modified into:
output = (bbox_regressions, classifications, ldm_regressions)
return torch.cat(output, dim=-1)

# set 'opset_version=11' to ensure a successful export
# torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
#     input_names=input_names, output_names=output_names)
torch_out = torch.onnx._export(
        net,
        inputs,
        output_onnx,
        export_params=True, 
        verbose=False,
        input_names=input_names, 
        output_names=output_names,
        dynamic_axes={"input0": {0: 'batch'},
                      "output0": {0: 'batch'}} ,
        opset_version = 11
    )
```

3. download pth model and convert to onnx

```

# download model
https://github.com/biubug6/Pytorch_Retinaface#training .Then unzip it to the /weights . Here, we use mobilenet0.25_Final.pth

# export
python convert_to_onnx.py -m /path/to/model.pth --network mobile0.25 

```

## Step 2: onnx2engine

```
cd /path/to/TensorRT-8.0.3.4/bin
./trtexec --onnx=/path/to/model.onnx --minShapes=input:1x3x640x640 --optShapes=input:4x3x640x640 --maxShapes=input:16x3x640x640  --verbose --avgRuns=10 --plugins --saveEngine=/path/to/model.engine
```

## Step 3: make this project

First you should set the TensorRT path and CUDA path in CMakeLists.txt.

compile

```
git clone git@github.com:LIUHAO121/RetinaFace_TensorRT_cpp.git
cd YOLOX_TensorRT_cpp
mkdir -p build
cd build
cmake ..
make
cd ..
```

run

```
./build/retina_face /path/to/model.engine -i input.jpg 
```
