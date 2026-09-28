"""Run these explicitly in the legacy environment. Skips are not benchmark passes."""
import importlib.util

import pytest


@pytest.mark.d4rl
@pytest.mark.skipif(importlib.util.find_spec("d4rl") is None or importlib.util.find_spec("gym") is None,
                    reason="Legacy D4RL/Gym/MuJoCo are not installed")
def test_real_legacy_environment_registration():
    from arlab.tasks.iql.environments import make_environment
    env = make_environment({"kind": "d4rl", "id": "hopper-medium-v2"})
    try:
        obs = env.reset()
        assert len(obs) == 11
        assert len(env.step(env.action_space.sample())) in (4, 5)
    finally:
        env.close()
