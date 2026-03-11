"""Quick quality check: verify all key imports resolve."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import importlib

modules = [
    # Utils
    "src.utils.misc",
    "src.utils.pc_utils",
    "src.utils.config",
    # Data
    "src.data.dataset",
    "src.data.dataset_utils",
    # Generative
    "src.generative.base",
    "src.generative.ddpm",
    "src.generative.flow_matching",
    "src.generative",
    # Models (may fail without CUDA)
    "src.models.model_utils",
    "src.models.pointnet2_ssg_sem",
    "src.models.pnet",
    "src.models.pointnet2_with_pcld_condition",
]

passed = 0
failed = 0
for m in modules:
    try:
        importlib.import_module(m)
        print(f"[OK]   {m}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] {m} -> {e}")
        failed += 1

print(f"\nResults: {passed}/{passed + failed} passed")
