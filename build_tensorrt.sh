/usr/src/tensorrt/bin/trtexec \
  --onnx=/opt/nvidia/deepstream/deepstream-7.1/sources/apps/sample_apps/vfocals-intrusion-detection/yolov8s.onnx \
  --saveEngine=/opt/nvidia/deepstream/deepstream-7.1/sources/apps/sample_apps/vfocals-intrusion-detection/model.engine \
  --fp16 \
  --minShapes=input:1x3x1280x1280 \
  --optShapes=input:1x3x1280x1280 \
  --maxShapes=input:8x3x1280x1280