"""High ceilings, small initial jobs. Limits are NOT desired expenditure."""
from arlab.core.schema import Budget, ExplorationPolicy


def research_budget() -> Budget:
    return Budget(
        max_decisions=20000, max_trials=20000, max_checks=2000,
        max_attempts=40000, max_attempts_per_trial=3,
        max_train_steps=1_000_000_000, max_optimizer_steps=16_000_000_000,
        max_eval_episodes=200000, max_model_calls=40000,
        max_output_tokens=160_000_000, max_wall_seconds=60 * 86400,
        worker_timeout_seconds=6 * 3600,
    )


def adaptive_exploration() -> ExplorationPolicy:
    return ExplorationPolicy(adaptive=True, minimum_steps=1000, initial_steps=5000,
                             initial_eval_episodes=3, confirmation_seeds=[0, 1, 2])
