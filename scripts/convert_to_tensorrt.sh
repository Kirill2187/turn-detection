#!/bin/bash
set -e

ONNX_MODEL=${1:-model.onnx}
OUTPUT_DIR=${2:-models/turn_detection/1}

mkdir -p $OUTPUT_DIR

docker run --rm --gpus all \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/tensorrt:24.12-py3 \
    trtexec \
    --verbose \
    --onnx=/workspace/$ONNX_MODEL \
    --saveEngine=/workspace/$OUTPUT_DIR/model.plan \
    --minShapes=input_ids:1x8,attention_mask:1x8 \
    --optShapes=input_ids:2x128,attention_mask:2x128 \
    --maxShapes=input_ids:8x512,attention_mask:8x512 \

echo "Saved to $OUTPUT_DIR/model.plan"
