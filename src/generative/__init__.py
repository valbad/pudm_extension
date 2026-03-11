"""Generative strategies for point cloud upsampling."""
from src.generative.base import GenerativeStrategy
from src.generative.ddpm import DDPMStrategy
from src.generative.flow_matching import FlowMatchingStrategy

STRATEGIES = {
    "ddpm": DDPMStrategy,
    "flow_matching": FlowMatchingStrategy,
}


def get_strategy(name: str) -> GenerativeStrategy:
    """Factory function to get a generative strategy by name."""
    name = name.lower()
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[name]()
