"""Real validation-data pipeline through shared pools, scores, and executed returns."""

from __future__ import annotations

import json

import numpy as np
import pytest

from crlo.agents._diffusion_ql.policy import DiffusionQLBehavior
from crlo.agents.critic_factory import build_critic
from crlo.envs.ogbench_loader import load_crl_env_and_dataset
from crlo.eval.query_manifests import (
    build_manifest_rows,
    manifest_payload,
    sha256_file,
    stratum_ranks,
)
from crlo.eval.rollout_shards import merge_shards
from crlo.training.checkpoints import save_critic, save_diffusion
from crlo.utils.config_loader import CONFIG_DIR
from experiments import (
    build_candidate_pools,
    check_value_reliability,
    evaluate_candidate_returns,
    evaluate_reference_ordering,
    score_candidate_pools,
)


def _args(**values):
    result = []
    for key, value in values.items():
        result.append("--" + key.replace("_", "-"))
        if isinstance(value, (tuple, list)):
            result.extend(map(str, value))
        else:
            result.append(str(value))
    return result


def test_pointmaze_shared_pool_to_executed_returns(tmp_path):
    task = "pointmaze-large-navigate-v0"
    dataset_dir = CONFIG_DIR.parent / "data/ogbench"
    env, dataset = load_crl_env_and_dataset(task, 0.99, dataset_dir=str(dataset_dir))
    np.random.seed(17)
    batch = dataset.sample(8)
    obs_dim, act_dim = batch["observations"].shape[-1], batch["actions"].shape[-1]
    policy = DiffusionQLBehavior.create(17, act_dim, obs_dim, obs_dim, 4, (16,))
    for _ in range(3):
        assert np.isfinite(policy.update(batch)["ddpm_loss"])
    continuation = tmp_path / "continuation"
    save_diffusion(policy, continuation)
    pools = tmp_path / "pools"
    build_candidate_pools.main(
        _args(
            task=task,
            diffusion_ckpt=continuation,
            output_dir=pools,
            dataset_dir=dataset_dir,
            n_per_stratum=1,
            n_replicates=1,
            k_max=4,
        )
    )
    pool = np.load(pools / "pools.npz")
    assert pool["A"].shape == (3, 1, 4, act_dim)
    assert json.loads((pools / "pools_meta.json").read_text())["query_split"] == "val"
    for seed in (7, 8, 9):
        agent = build_critic("td_q", seed, batch, 3.0, 0.99, hidden_dims=(16,))
        for _ in range(2):
            agent, info = agent.update(batch)
            assert all(np.isfinite(np.asarray(value)).all() for value in info.values())
        checkpoint = tmp_path / f"reference_{seed}"
        save_critic(
            agent,
            checkpoint,
            dict(
                critic="td_q",
                seed=seed,
                obs_dim=obs_dim,
                action_dim=act_dim,
                bc_alpha=3.0,
                discount=0.99,
            ),
        )
        score_candidate_pools.main(
            _args(
                pools_dir=pools,
                checkpoint_dir=checkpoint,
                critic="td_q",
                label=f"reference_{seed}",
            )
        )
    common = dict(task=task, pools_dir=pools, refs=("reference_7", "reference_8", "reference_9"))
    evaluate_reference_ordering.main(_args(**common))
    evaluate_reference_ordering.main(
        _args(
            **common,
            mode="metrics",
            critics="reference_7",
            k_grid=(1, 2, 4),
            n_boot=20,
        )
    )
    assert (pools / f"reference_ordering_{task}.csv").is_file()
    rows = build_manifest_rows(
        list(range(3)),
        pool["stratum"],
        stratum_ranks(pool["stratum"]),
        np.arange(3),
        1,
        7002,
        4,
    )
    digest = sha256_file(pools / "pools.npz")[:16]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            manifest_payload(
                rows,
                {
                    "task": task,
                    "manifest": "integration_test",
                    "pool_sha16": digest,
                    "k": 4,
                    "tie_seed": 7002,
                    "val_data_sha16": "test",
                    "exclusion_manifest_sha256": "test",
                },
            )
        )
    )
    output = tmp_path / "returns"
    arguments = _args(
        task=task,
        pools_dir=pools,
        manifest_path=manifest,
        diffusion_ckpt=continuation,
        continuation_sha256=sha256_file(continuation / "params.msgpack"),
        expected_pool_sha16=digest,
        output_dir=output / "rollouts",
        horizon=3,
        n_shards=1,
        shard=0,
        rollouts_per_half=1,
        continuation_mode="deterministic",
        dataset_dir=dataset_dir,
    )
    assert evaluate_candidate_returns.main(arguments) == 0
    before = sha256_file(output / "rollouts/shard_0000.npz")
    assert evaluate_candidate_returns.main(arguments) == 0
    assert before == sha256_file(output / "rollouts/shard_0000.npz")
    mismatch = arguments.copy()
    mismatch[mismatch.index("--horizon") + 1] = "4"
    with pytest.raises(ValueError, match="protocol mismatch"):
        evaluate_candidate_returns.main(mismatch)
    merged = merge_shards(output / "rollouts", rows, 4, 1)
    np.testing.assert_array_equal(merged["disc_return"][:, :, 0], merged["disc_return"][:, :, 1])
    check_value_reliability.main(
        _args(
            mode="deterministic_primary",
            manifest_path=manifest,
            merged_path=output / "merged/full_pool_rollouts.npz",
            output_dir=output / "precheck",
            k=4,
            rollouts_per_half=1,
            n_boot=20,
        )
    )
    assert (output / "precheck/reliability_pointmaze.json").is_file()
    env.close()
