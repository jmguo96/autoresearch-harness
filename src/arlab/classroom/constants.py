"""Dependency-free identifiers shared with the installation bootstrap."""

D4RL_SUITE9 = tuple(
    f"{env}-{quality}-v2"
    for env in ("halfcheetah", "hopper", "walker2d")
    for quality in ("medium", "medium-replay", "medium-expert")
)
# Fixed before any experiment: one task from each environment AND each dataset
# regime. This diagonal sample is not a factorial study and cannot disentangle
# environment effects from dataset-regime effects. The final nine-task panel
# covers all combinations; coverage alone does not imply causal attribution.
EXPLORATION_TASKS = (
    "halfcheetah-medium-v2",
    "hopper-medium-replay-v2",
    "walker2d-medium-expert-v2",
)
