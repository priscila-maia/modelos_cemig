"""Profile registry for model-agnostic pipeline defaults."""

from src.pipelines.profiles import qwen_v2


PROFILE_REGISTRY = {
    "qwen_v2": qwen_v2,
}


def get_profile_module(profile_name: str):
    if profile_name not in PROFILE_REGISTRY:
        valid = ", ".join(sorted(PROFILE_REGISTRY.keys()))
        raise ValueError(f"Unknown profile '{profile_name}'. Valid profiles: {valid}")
    return PROFILE_REGISTRY[profile_name]
