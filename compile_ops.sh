#!/bin/bash
# Compile CUDA extensions for pudm_extension.
# Run from the pudm_extension/ root directory.
set -e

echo "=== Compiling Chamfer3D ==="
cd src/metrics
python -c "
import torch
from torch.utils.cpp_extension import load
import os
chamfer = load(
    name='chamfer_3D',
    sources=[os.path.join(os.path.dirname(__file__), 'chamfer_cuda.cpp'),
             os.path.join(os.path.dirname(__file__), 'chamfer3D.cu')],
    verbose=True
)
print('Chamfer3D compiled successfully.')
"
cd ../..
echo "---- Chamfer3D ---> Done ----"

echo ""
echo "=== Compiling pointops ==="
cd src/ops/pointops
pip install -e .
cd ../../..
echo "---- pointops ---> Done ----"

echo ""
echo "Note: pointnet2_ops and chamfer3d_legacy use JIT compilation."
echo "They will compile automatically on first import."
echo ""
echo "All CUDA extensions ready!"
