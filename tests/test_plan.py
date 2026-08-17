"""Splitting a model across GPUs of different sizes."""

from __future__ import annotations

import pytest

from p2pgpu.cluster.plan import PlanError, describe, layer_memory_mb, plan_split

# The user's actual cluster.
CLUSTER = [("rtx4050", 5 * 1024), ("rtx4080", 15 * 1024)]


def test_bigger_gpu_gets_more_layers():
    """Equal layer counts would OOM the 4050 while the 4080 sat two-thirds empty."""
    specs = plan_split(CLUSTER, num_layers=32, total_params=7_000_000_000)
    small, big = specs
    assert big.layer_count > small.layer_count
    assert small.first_layer == 0
    assert small.last_layer == big.first_layer  # contiguous, no gap
    assert big.last_layer == 32


def test_every_layer_is_placed_exactly_once():
    specs = plan_split(CLUSTER, num_layers=32, total_params=7_000_000_000)
    covered = [i for s in specs for i in range(s.first_layer, s.last_layer)]
    assert covered == list(range(32))


def test_no_node_is_pushed_past_its_headroom():
    specs = plan_split(CLUSTER, num_layers=32, total_params=7_000_000_000)
    assert all(s.utilisation <= 1.0 for s in specs)


def test_model_too_big_for_the_whole_cluster_is_refused_with_numbers():
    with pytest.raises(PlanError, match="GB usable"):
        plan_split(CLUSTER, num_layers=80, total_params=70_000_000_000)


def test_a_model_that_fits_neither_card_alone_fits_across_both():
    """The entire point: 13B at fp16 is ~26 GB, too big for either card."""
    with pytest.raises(PlanError):
        plan_split([("rtx4050", 5 * 1024)], num_layers=40, total_params=13_000_000_000)
    with pytest.raises(PlanError):
        plan_split([("rtx4080", 15 * 1024)], num_layers=40, total_params=13_000_000_000)
    # But at int4 across both, it places.
    specs = plan_split(CLUSTER, num_layers=40, total_params=13_000_000_000, dtype="int4")
    assert sum(s.layer_count for s in specs) == 40


def test_more_nodes_than_layers_is_refused():
    with pytest.raises(PlanError, match="cannot be split"):
        plan_split(CLUSTER, num_layers=1, total_params=1_000_000)


def test_dtype_changes_the_footprint_proportionally():
    fp16 = layer_memory_mb(1_000_000_000, 10, "float16")
    fp32 = layer_memory_mb(1_000_000_000, 10, "float32")
    assert fp32 == pytest.approx(fp16 * 2)


def test_unknown_dtype_is_named_in_the_error():
    with pytest.raises(PlanError, match="float8"):
        layer_memory_mb(1_000_000, 4, "float8")


def test_describe_reports_the_combined_total():
    specs = plan_split(CLUSTER, num_layers=32, total_params=7_000_000_000)
    assert "combined:" in describe(specs)
