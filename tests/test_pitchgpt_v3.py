"""
Tests required by PITCHGPT_V2_SPEC.md §8.3 **before any training run**, plus
the data-path equivalence checks the v3 cache depends on.

§8.3 checklist -> test mapping
------------------------------
1. Dynamic-context assertion (guards the ``6111cd6`` fix, §4.1)
   -> ``test_dynamic_context_not_constant_across_pa_positions``
      ``test_stage_b_count_trajectory_matches_advance_count``
2. Chain-rule identity (sums to 1; argmax vs sequential-greedy)
   -> ``test_chain_rule_reconstruction_sums_to_one``
      ``test_chain_rule_argmax_matches_sequential_greedy``
3. Parameter-budget assertion fails the build past the ceiling
   -> ``test_output_stack_param_budget`` / ``test_param_budget_assertion_fires``
4. Mask legality (no type 16; support >= 10 or a recorded fallback)
   -> ``test_sampling_masks_are_legal``
5. Provenance guard: a v3 sidecar declaring 2025/2026 is refused on load
   -> ``test_calibration_provenance_guard_refuses_gate_cohorts``
6. Ledger guard: the lockbox entry point is ``@holdout_access``-decorated and
   refuses while sealed
   -> ``test_lockbox_entry_point_is_sealed``
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.analytics.pitchgpt import (
    CONTEXT_DIM,
    NUM_COUNT_STATES,
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PAD_TOKEN,
    VOCAB_SIZE,
    PitchTokenizer,
)
from src.analytics.pitchgpt_sim import _advance_count
from src.analytics.pitchgpt_v3 import (
    EXPECTED_OUTPUT_STACK_PARAMS,
    FLAT_HEAD_PARAMS,
    MIN_SUPPORT,
    OUTPUT_STACK_PARAM_CEILING,
    UNKNOWN_TYPE_IDX,
    CalibrationProvenanceError,
    FactorizedPitchGPT,
    HeadTemperatures,
    MaskStats,
    SupportCounts,
    fields_from_token,
    load_calibration,
    token_from_fields,
    validate_calibration_provenance,
)
from src.analytics.pitchgpt_v3_data import (
    SeasonPolicyError,
    assert_season_policy,
    context_indices_to_tensor,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rng_model() -> FactorizedPitchGPT:
    torch.manual_seed(0)
    m = FactorizedPitchGPT()
    m.eval()
    return m


@pytest.fixture(scope="module")
def toy_support() -> SupportCounts:
    """Support built from a synthetic token stream with known holes."""
    rs = np.random.RandomState(7)
    # Dense support over types 0..5, zones 0..9, velos 0..4.
    t = rs.randint(0, 6, size=40_000)
    z = rs.randint(0, 10, size=40_000)
    v = rs.randint(0, 5, size=40_000)
    toks = token_from_fields(t, z, v)
    return SupportCounts.from_tokens(toks)


# ── §8.3.3 parameter budget ──────────────────────────────────────────────────


def test_output_stack_param_budget(rng_model: FactorizedPitchGPT) -> None:
    n = rng_model.output_stack_parameters()
    assert n == EXPECTED_OUTPUT_STACK_PARAMS, (
        f"§3.3 fixes the output stack at {EXPECTED_OUTPUT_STACK_PARAMS:,}; got {n:,}"
    )
    assert n < OUTPUT_STACK_PARAM_CEILING
    assert n < 0.25 * FLAT_HEAD_PARAMS
    # The plan's stated expectation: 283K output matrix -> ~40K.
    assert FLAT_HEAD_PARAMS / n > 9.0


def test_param_budget_assertion_fires() -> None:
    """A head stack over the §3.3 ceiling must abort the build."""
    with pytest.raises(AssertionError, match="§3.3 build assertion failed"):
        FactorizedPitchGPT(field_emb_dim=32, head_hidden=512)


# ── §8.3.2 chain-rule identity ───────────────────────────────────────────────


def _random_hidden(model: FactorizedPitchGPT, n: int = 24) -> torch.Tensor:
    torch.manual_seed(3)
    tokens = torch.randint(0, VOCAB_SIZE, (n, 5))
    ctx = torch.rand(n, 5, CONTEXT_DIM)
    with torch.no_grad():
        h = model.forward_hidden(tokens, ctx)
    return h[:, -1, :]


def test_chain_rule_reconstruction_sums_to_one(rng_model: FactorizedPitchGPT) -> None:
    h = _random_hidden(rng_model)
    probs = rng_model.full_token_probs(h)
    assert probs.shape == (h.shape[0], VOCAB_SIZE)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), sums


def test_chain_rule_factorisation_matches_token_arithmetic(
    rng_model: FactorizedPitchGPT,
) -> None:
    """``p(token)`` from the joint equals ``p_t · p_z|t · p_v|t,z`` per field."""
    h = _random_hidden(rng_model)
    joint = rng_model.full_token_probs(h)
    with torch.no_grad():
        p_t = torch.softmax(rng_model.type_logits(h), dim=-1)
        for t in (0, 5, 16):
            t_idx = torch.full((h.shape[0],), t, dtype=torch.long)
            p_z = torch.softmax(rng_model.zone_logits(h, t_idx), dim=-1)
            for z in (0, 13, 25):
                z_idx = torch.full((h.shape[0],), z, dtype=torch.long)
                p_v = torch.softmax(rng_model.velo_logits(h, t_idx, z_idx), dim=-1)
                for v in (0, 4):
                    tok = int(token_from_fields(t, z, v))
                    expect = p_t[:, t] * p_z[:, z] * p_v[:, v]
                    assert torch.allclose(joint[:, tok], expect, atol=1e-6)


def test_chain_rule_argmax_matches_sequential_greedy(
    rng_model: FactorizedPitchGPT,
) -> None:
    """§8.3.2, second clause — as amended by deviations-log entry 2.

    The spec asks that "the argmax token matches sequential-greedy decode".
    That is not a mathematical identity: greedy maximises each conditional in
    turn, which can miss the joint mode when a slightly-less-likely type
    carries a much sharper zone/velo distribution.  Measured disagreement on a
    random-init model is ~1.6% of rows.  What IS an identity, and what this
    test enforces, is:

      (a) the greedy path is exactly ``token_from_fields`` of the three
          per-head argmaxes, so field ordering / token arithmetic cannot be
          silently miswired;
      (b) the greedy token's reconstructed joint probability equals the
          product of its three conditional probabilities;
      (c) the joint argmax is never *less* likely than the greedy token.

    The agreement rate is reported (not asserted at 1.0) so the divergence
    stays a measured number.
    """
    h = _random_hidden(rng_model, n=64)
    greedy = rng_model.greedy_tokens(h)
    joint = rng_model.full_token_probs(h)
    joint_argmax = joint.argmax(dim=-1)

    with torch.no_grad():
        t_g = rng_model.type_logits(h).argmax(dim=-1)
        z_g = rng_model.zone_logits(h, t_g).argmax(dim=-1)
        v_g = rng_model.velo_logits(h, t_g, z_g).argmax(dim=-1)
        p_t = torch.softmax(rng_model.type_logits(h), dim=-1)
        p_z = torch.softmax(rng_model.zone_logits(h, t_g), dim=-1)
        p_v = torch.softmax(rng_model.velo_logits(h, t_g, z_g), dim=-1)
    rows = torch.arange(h.shape[0])

    # (a)
    assert torch.equal(greedy, token_from_fields(t_g, z_g, v_g))
    # (b)
    prod = p_t[rows, t_g] * p_z[rows, z_g] * p_v[rows, v_g]
    assert torch.allclose(joint[rows, greedy], prod, atol=1e-6)
    # (c)
    assert bool((joint[rows, joint_argmax] >= joint[rows, greedy] - 1e-9).all())

    agree = float((greedy == joint_argmax).float().mean())
    assert 0.0 <= agree <= 1.0
    print(f"greedy-vs-joint-argmax agreement (random init): {agree:.4f}")


def test_token_field_roundtrip() -> None:
    toks = torch.arange(VOCAB_SIZE)
    t, z, v = fields_from_token(toks)
    assert torch.equal(token_from_fields(t, z, v), toks)
    # Matches the v2 tokenizer's own decode.
    for tok in (0, 1, 129, 130, 2209):
        d = PitchTokenizer.decode(tok)
        tt, zz, vv = fields_from_token(torch.tensor([tok]))
        assert int(zz) == d["zone"] and int(vv) == d["velo_bucket"]


# ── §8.3.4 mask legality ─────────────────────────────────────────────────────


def test_sampling_masks_are_legal(
    rng_model: FactorizedPitchGPT, toy_support: SupportCounts
) -> None:
    rng_model.attach_support(toy_support)
    h = _random_hidden(rng_model, n=256)
    gen = torch.Generator().manual_seed(42)
    stats = MaskStats()
    tok, t_idx, z_idx, v_idx = rng_model.sample_tokens(
        h, generator=gen, stats=stats,
    )

    # §3.6.1 — index 16 is never emitted.
    assert int((t_idx == UNKNOWN_TYPE_IDX).sum()) == 0

    # §3.6.2/3 — every sampled field has support >= 10, or a fallback was
    # recorded for that class of event.
    tz = toy_support.type_zone
    tzv = toy_support.type_zone_velo
    n_zone_degenerate = n_velo_degenerate = 0
    for t, z, v in zip(t_idx.tolist(), z_idx.tolist(), v_idx.tolist()):
        # zone: conditional support, else the unconditional fallback, else the
        # §3.6.4 degenerate branch (uniform) — which must have been counted.
        if tz[t, z] < MIN_SUPPORT and toy_support.zone[z] < MIN_SUPPORT:
            assert not (toy_support.zone >= MIN_SUPPORT).any(), (
                f"zone {z} sampled for type {t} while a legal fallback existed"
            )
            n_zone_degenerate += 1
        if (
            tzv[t, z, v] < MIN_SUPPORT
            and toy_support.type_velo[t, v] < MIN_SUPPORT
        ):
            assert not (toy_support.type_velo[t] >= MIN_SUPPORT).any(), (
                f"velo {v} sampled for ({t},{z}) while a legal fallback existed"
            )
            n_velo_degenerate += 1

    # Degenerate draws are only acceptable because they were *recorded*.
    if n_zone_degenerate:
        assert stats.zone_degenerate > 0
    if n_velo_degenerate:
        assert stats.velo_degenerate > 0

    # Token arithmetic identity survives sampling.
    assert torch.equal(tok, token_from_fields(t_idx, z_idx, v_idx))
    d = stats.as_dict()
    assert set(d) == {"mask_fallbacks", "mask_degenerate"}


def test_precomputed_mask_tables_match_reference_semantics(
    rng_model: FactorizedPitchGPT, toy_support: SupportCounts
) -> None:
    """The hot-path tables must equal the row-at-a-time §3.6.4 reference."""
    rng_model.attach_support(toy_support)
    for t in range(NUM_PITCH_TYPES):
        ref = rng_model._mask_from_support(
            toy_support.type_zone[t], toy_support.zone, None, None, None,
        )
        assert np.array_equal(rng_model._zone_mask[t], ref), f"zone mask type {t}"
        for z in range(NUM_ZONES):
            ref_v = rng_model._mask_from_support(
                toy_support.type_zone_velo[t, z], toy_support.type_velo[t],
                None, None, None,
            )
            assert np.array_equal(rng_model._velo_mask[t, z], ref_v), (t, z)


def test_mask_fallback_is_recorded_when_conditional_support_empty(
    rng_model: FactorizedPitchGPT,
) -> None:
    """A type with zero conditional zone support must fall back and be counted."""
    rs = np.random.RandomState(1)
    t = np.full(5000, 3)
    z = rs.randint(0, 4, size=5000)
    v = rs.randint(0, 5, size=5000)
    sup = SupportCounts.from_tokens(token_from_fields(t, z, v))
    # Type 3 is the only supported type; force sampling of an unsupported type
    # by asking the mask helper directly for a type with no support at all.
    stats = MaskStats()
    mask = rng_model._mask_from_support(
        sup.type_zone[7], sup.zone, "zone_fallback", "zone_degenerate", stats,
    )
    assert stats.zone_fallback == 1
    assert mask[:4].all() and not mask[10:].any()

    # And when even the unconditional support is empty -> uniform + counted.
    empty = np.zeros(NUM_ZONES, dtype=np.int64)
    stats2 = MaskStats()
    mask2 = rng_model._mask_from_support(
        empty, empty, "zone_fallback", "zone_degenerate", stats2,
    )
    assert stats2.zone_degenerate == 1 and mask2.all()


# ── §8.3.1 dynamic mid-PA context ────────────────────────────────────────────


def test_dynamic_context_not_constant_across_pa_positions() -> None:
    """The count_state one-hot must move as the count advances (§4.1)."""
    balls, strikes = 0, 0
    vectors = []
    # ball, called_strike, foul, ball, swinging_strike -> count advances
    for outcome in (0, 1, 3, 0, 2):
        ctx = PitchTokenizer.encode_context(
            balls=balls, strikes=strikes, outs=1,
            on_1b=False, on_2b=False, on_3b=False,
            stand="R", inning=4, score_diff=0,
        )
        vectors.append(PitchTokenizer.context_to_tensor(ctx, ump_scalar=0.36))
        balls, strikes = _advance_count(balls, strikes, outcome)

    stacked = torch.stack(vectors)
    count_block = stacked[:, :NUM_COUNT_STATES]
    assert not torch.allclose(count_block[0], count_block[-1]), (
        "count_state one-hot is CONSTANT across within-PA positions — this is "
        "the static-context bug commit 6111cd6 fixed (§4.1)."
    )
    assert len({tuple(r.tolist()) for r in count_block}) >= 4
    # Every non-count field is PA-invariant.
    assert torch.allclose(
        stacked[:, NUM_COUNT_STATES:], stacked[0, NUM_COUNT_STATES:].expand(5, -1),
    )


def test_stage_b_count_trajectory_matches_advance_count() -> None:
    """The context builder used by Stage B derives counts via ``_advance_count``."""
    from src.analytics.pitchgpt_v3_rollout import pa_count_trajectory

    outcomes = np.array([0, 0, 1, 3, 3, 0])  # ball ball CS foul foul ball -> walk
    traj = pa_count_trajectory(outcomes, start=(0, 0))
    expect = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 2)]
    assert [tuple(x) for x in traj] == expect


def test_context_expansion_matches_reference_tokenizer() -> None:
    idx = np.array([[5, 2, 3, 1, 2, 4], [0, 0, 0, 0, 0, 0]])
    ump = np.array([0.36, -0.1], dtype=np.float32)
    got = context_indices_to_tensor(idx, ump)
    for row in range(2):
        want = PitchTokenizer.context_to_tensor(
            list(int(v) for v in idx[row]), ump_scalar=float(ump[row]),
        )
        assert torch.allclose(got[row], want)


# ── §8.3.5 calibration provenance guard (K5 second clause) ───────────────────


def _sidecar(season: int) -> dict:
    return {
        "fit_cohort_season": season,
        "fit_seed": 42,
        "fit_n_pas": 19625,
        "produced_by": "scripts/pitchgpt_v3_calibrate.py",
        "temperatures": {
            "T_type": 1.0, "T_zone": 1.0, "T_velo": 1.0, "T_outcome": 1.0,
        },
    }


@pytest.mark.parametrize("season", [2025, 2026])
def test_calibration_provenance_guard_refuses_gate_cohorts(season: int) -> None:
    with pytest.raises(CalibrationProvenanceError, match="provenance guard"):
        validate_calibration_provenance(_sidecar(season))


def test_calibration_provenance_requires_2023() -> None:
    with pytest.raises(CalibrationProvenanceError, match="requires fit_cohort_season"):
        validate_calibration_provenance(_sidecar(2024))
    assert validate_calibration_provenance(_sidecar(2023))


def test_calibration_provenance_requires_schema_keys() -> None:
    bad = _sidecar(2023)
    del bad["fit_seed"]
    with pytest.raises(CalibrationProvenanceError, match="missing required provenance"):
        validate_calibration_provenance(bad)


def test_load_calibration_refuses_tainted_file(tmp_path) -> None:
    p = tmp_path / "calibration_pitchgpt_v3.json"
    p.write_text(json.dumps(_sidecar(2026)))
    with pytest.raises(CalibrationProvenanceError):
        load_calibration(p)

    good = tmp_path / "good.json"
    good.write_text(json.dumps(_sidecar(2023)))
    temps, payload = load_calibration(good)
    assert isinstance(temps, HeadTemperatures)
    assert payload["fit_cohort_season"] == 2023


# ── §8.3.6 ledger guard ──────────────────────────────────────────────────────


def test_lockbox_entry_point_is_sealed() -> None:
    from src.holdout import HoldoutSealedError, contact_count, dataset_policy
    from scripts import pitchgpt_v3_lockbox_2026 as lb

    policy = dataset_policy(lb.LOCKBOX_DATASET) or {}
    assert policy.get("tier") == "lockbox"
    assert policy.get("unsealed") is False, (
        "the 2026 lockbox is no longer sealed — this test must be revisited "
        "together with the §5.5 one-contact protocol."
    )
    assert lb.LOCKBOX_ENTRY_POINT_IS_LEDGER_GATED
    # functools.wraps leaves the original on __wrapped__ — proof of decoration.
    assert hasattr(lb.run_lockbox_2026_grading, "__wrapped__")

    before = contact_count(lb.LOCKBOX_DATASET)
    with pytest.raises(HoldoutSealedError):
        lb.run_lockbox_2026_grading()
    assert contact_count(lb.LOCKBOX_DATASET) == before == 0


# ── Data policy (spec §5) ────────────────────────────────────────────────────


@pytest.mark.parametrize("season", [2025, 2026])
def test_season_policy_refuses_forbidden_seasons(season: int) -> None:
    with pytest.raises(SeasonPolicyError):
        assert_season_policy([2023, season])


def test_season_policy_allows_program_seasons() -> None:
    assert assert_season_policy([2015, 2022, 2023, 2024]) == [
        2015, 2022, 2023, 2024,
    ]


def test_season_policy_never_allows_2025_even_with_lockbox_flag() -> None:
    with pytest.raises(SeasonPolicyError):
        assert_season_policy([2025], allow_lockbox=True)
    assert assert_season_policy([2026], allow_lockbox=True) == [2026]


# ── Cache fidelity vs the reference data path ────────────────────────────────


def _db_conn():
    from pathlib import Path

    from src.db.schema import DEFAULT_DB_PATH, get_connection

    if not Path(DEFAULT_DB_PATH).exists():
        pytest.skip("baseball.duckdb not present")
    return get_connection(read_only=True)


def test_vectorised_score_diff_matches_reference() -> None:
    """The v3 cache's score reconstruction must equal ``pitchgpt``'s row walk."""
    from src.analytics.pitchgpt import _compute_per_pitch_score_diff
    from src.analytics.pitchgpt_v3_data import _SEQ_SQL, vectorised_score_diff

    conn = _db_conn()
    sql = _SEQ_SQL.format(
        season_filter="AND EXTRACT(YEAR FROM game_date) IN (2023)",
        pitcher_exclude_filter="",
        game_filter=(
            "AND game_pk IN (SELECT game_pk FROM (SELECT DISTINCT game_pk "
            "FROM pitches WHERE pitch_type IS NOT NULL AND "
            "EXTRACT(YEAR FROM game_date) = 2023) USING SAMPLE 25 ROWS "
            "(reservoir, 7))"
        ),
    )
    df = conn.execute(sql).fetchdf()
    assert len(df) > 1000
    ref = np.asarray(_compute_per_pitch_score_diff(df), dtype=np.int32)
    assert np.array_equal(ref, vectorised_score_diff(df))


def test_cache_matches_reference_dataset() -> None:
    """Bit-equivalence of the cached tensors with ``PitchSequenceDataset``."""
    from src.analytics.pitchgpt import PitchSequenceDataset
    from src.analytics.pitchgpt_v3_data import (
        CachedSequenceDataset,
        build_sequence_cache,
    )

    conn = _db_conn()
    ref = PitchSequenceDataset(conn, seasons=[2023], max_games=8)
    assert len(ref.sequences) > 10
    cache = build_sequence_cache(
        conn, seasons=[2023], max_games=None, sample_seed=42,
        game_pks=sorted(ref.game_pks),
    )
    ds = CachedSequenceDataset(cache)
    assert len(ds) == len(ref.sequences)

    def key(item):
        return (item[0], item[2])

    ref_items = [
        (tuple(i.tolist()), c.numpy(), tuple(t.tolist())) for i, c, t in ref.sequences
    ]
    v3_items = []
    for i in range(len(ds)):
        tok, cidx, ump, tgt = ds[i]
        ctx = context_indices_to_tensor(cidx.numpy(), ump.numpy()).numpy()
        v3_items.append((tuple(tok.tolist()), ctx, tuple(tgt.tolist())))

    for a, b in zip(sorted(ref_items, key=key), sorted(v3_items, key=key)):
        assert a[0] == b[0]
        assert a[2] == b[2]
        assert np.allclose(a[1], b[1], atol=1e-6)


# ── Data policy, continued ───────────────────────────────────────────────────


# ── §4.3.2 batch invariance of the rollout ───────────────────────────────────


def _rollout_fixtures():
    """A support-attached model + outcome head + a small PA-start batch."""
    from src.analytics.pitchgpt_v3 import V3OutcomeHead

    torch.manual_seed(11)
    m = FactorizedPitchGPT()
    rs = np.random.RandomState(5)
    toks = token_from_fields(
        rs.randint(0, 8, 60_000), rs.randint(0, 12, 60_000), rs.randint(0, 5, 60_000)
    )
    m.attach_support(SupportCounts.from_tokens(toks))
    m.eval()
    oh = V3OutcomeHead()
    oh.eval()
    ctx = torch.zeros(7, CONTEXT_DIM)
    ctx[:, 0] = 1.0                      # count state 0-0
    ctx[:, NUM_COUNT_STATES + 1] = 1.0   # 1 out
    ctx[:, -1] = 0.36
    return m, oh, ctx


def test_batched_and_single_pa_rollouts_agree() -> None:
    """A PA's trajectory must not depend on the batch it is rolled in (§4.3.2)."""
    from src.analytics.pitchgpt_v3_rollout import pa_uniform_block, rollout_pa_batch

    m, oh, ctx = _rollout_fixtures()
    P, S, H = ctx.shape[0], 9, 6
    start_count = np.zeros((P, 2), dtype=np.int64)
    unis = np.stack([pa_uniform_block(i, S, H) for i in range(P)])

    whole = rollout_pa_batch(
        m, oh, start_context=ctx, start_count=start_count, uniforms=unis, horizon=H,
    )
    for i in range(P):
        one = rollout_pa_batch(
            m, oh,
            start_context=ctx[i : i + 1],
            start_count=start_count[i : i + 1],
            uniforms=unis[i : i + 1],
            horizon=H,
        )
        for k in ("pitch_tokens", "outcomes", "pa_terminated", "pa_outcome", "final_count"):
            assert np.array_equal(one[k][0], whole[k][i]), f"{k} diverged for PA {i}"


def test_rollout_respects_masks_and_the_production_count_machine() -> None:
    """Sampled tokens are §3.6-legal and the count follows ``_advance_count``."""
    from src.analytics.pitchgpt_sim import ROLLOUT_PAD_PITCH, _TERMINAL_INPLAY_OUTCOMES
    from src.analytics.pitchgpt_v3_rollout import pa_uniform_block, rollout_pa_batch

    m, oh, ctx = _rollout_fixtures()
    P, S, H = ctx.shape[0], 12, 6
    unis = np.stack([pa_uniform_block(i, S, H) for i in range(P)])
    res = rollout_pa_batch(
        m, oh,
        start_context=ctx,
        start_count=np.zeros((P, 2), dtype=np.int64),
        uniforms=unis,
        horizon=H,
    )
    toks = res["pitch_tokens"]
    live = toks != ROLLOUT_PAD_PITCH
    t, z, v = fields_from_token(torch.from_numpy(toks[live]))
    assert int((t == UNKNOWN_TYPE_IDX).sum()) == 0, "§3.6.1: type 16 was emitted"
    sup = m.support
    ok = (
        (sup.type_zone[t.numpy(), z.numpy()] >= MIN_SUPPORT)
        | (sup.zone[z.numpy()] >= MIN_SUPPORT)
    )
    assert ok.all(), "§3.6.2: a zone with no legal support was sampled"

    # Replay the count through the production state machine.
    for i in range(P):
        for s in range(S):
            b, k = 0, 0
            for pos in range(H):
                o = int(res["outcomes"][i, s, pos])
                if o < 0 or o >= 7:
                    break
                if o in _TERMINAL_INPLAY_OUTCOMES:
                    assert res["pa_terminated"][i, s, pos]
                    assert int(res["pa_outcome"][i, s]) == o
                    break
                b, k = _advance_count(b, k, o)
                if b >= 4 or k >= 3:
                    assert res["pa_terminated"][i, s, pos]
                    break
            assert int(res["final_count"][i, s, 0]) == min(b, 4)
            assert int(res["final_count"][i, s, 1]) == min(k, 3)


# ── Sim integration is opt-in; production is untouched (§8.1) ────────────────


def test_rollout_v3_optin_refuses_without_an_explicit_opt_in(monkeypatch) -> None:
    from src.analytics.pitchgpt_sim import PAContext
    from src.analytics.pitchgpt_v3_rollout import (
        V3_SIM_OPTIN_ENV,
        V3SimOptInError,
        rollout_v3_optin,
        v3_sim_optin_enabled,
    )

    m, oh, _ = _rollout_fixtures()
    monkeypatch.delenv(V3_SIM_OPTIN_ENV, raising=False)
    assert v3_sim_optin_enabled() is False
    ctx = PAContext(
        pitcher_id=1, batter_id=2, count=(0, 0), outs=1,
        runners=(False, False, False), batter_stand="R", pitcher_throws="R",
        inning=4, inning_topbot="Top", score_diff=0, umpire_scalar=0.36,
    )
    with pytest.raises(V3SimOptInError, match="dev-tier"):
        rollout_v3_optin(ctx, model=m, outcome_head=oh, n_samples=2)

    monkeypatch.setenv(V3_SIM_OPTIN_ENV, "1")
    assert v3_sim_optin_enabled() is True
    res = rollout_v3_optin(ctx, model=m, outcome_head=oh, n_samples=2)
    assert res.sampling_metadata["backbone_version"] == "v3_factorized"


def test_production_sim_module_has_no_v3_path() -> None:
    """`pitchgpt_sim.py` must not reference v3 at all (§8.1, no alias change)."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "src" / "analytics" / "pitchgpt_sim.py"
    ).read_text(encoding="utf-8")
    assert "pitchgpt_v3" not in src, (
        "production sim now imports v3 — the opt-in boundary is broken"
    )

    from src.analytics.pitchgpt_sim import OutcomePredictorRegistry

    names = " ".join(str(n) for n in OutcomePredictorRegistry.list_registered())
    assert "v3" not in names, f"a v3 predictor is registered in production: {names}"


# ── §6 thresholds are the spec's, mechanically ───────────────────────────────


def test_g1_threshold_ladder_matches_the_spec_derivation() -> None:
    """§6.1: threshold(C) = max(0.010, 0.010 * sqrt(C / 7)) rounded to 0.005."""
    import math

    from src.analytics.pitchgpt_v3_gates import G1_THRESHOLDS

    for head, row in G1_THRESHOLDS.items():
        C = row["C"]
        raw = max(0.010, 0.010 * math.sqrt(C / 7))
        want = round(raw / 0.005) * 0.005
        assert abs(row["cwECE"] - want) < 1e-9, (head, row["cwECE"], want)
        assert abs(row["TACE"] - want) < 1e-9, (head, row["TACE"], want)
    assert {h: r["C"] for h, r in G1_THRESHOLDS.items()} == {
        "type": 17, "zone": 26, "velo": 5, "outcome": 7,
    }


def test_fixed_gate_constants_match_the_frozen_spec() -> None:
    from src.analytics import pitchgpt_v3_gates as g

    assert (g.CWECE_BINS, g.TACE_BINS, g.TACE_THRESHOLD) == (15, 15, 1e-3)
    assert g.MIN_CLASS_OBS == 100
    assert (g.KCE_ALPHA, g.KCE_BOOTSTRAP, g.KCE_PROBE_LOGIT_SCALE) == (0.01, 1000, 1.10)
    assert (g.PIT_BINS, g.PIT_KS_MAX) == (20, 0.03)
    assert abs(g.PIT_BIN_LO - 0.0335) < 1e-12 and abs(g.PIT_BIN_HI - 0.0750) < 1e-12
    assert (g.MARGINAL_ABS_TOL, g.MARGINAL_REL_TOL) == (0.01, 0.10)
    assert (g.WOBA_ABS_TOL, g.PA_LEN_ABS_TOL) == (0.015, 0.5)
    assert g.CAL_VALID_COVERAGE_MIN == 0.95
    assert (g.COUNT_STATE_ECE_BINS, g.COUNT_STATE_ECE_MAX, g.COUNT_STATE_MIN_OBS) == (
        10, 0.02, 500,
    )
    assert g.POSITION_GAP_MAX == 0.01          # the 1.0pp 0.6.2 died on
    assert (g.DECISION_DECILES, g.DECISION_MIN_N) == (10, 500)
    assert (g.DECISION_DECILE_GAP_MAX, g.DECISION_WEIGHTED_GAP_MAX) == (0.01, 0.005)
    assert (g.ANTI_UNFAILABILITY_FACTOR, g.ANTI_UNFAILABILITY_FLOOR) == (0.6, 0.005)


def test_anti_unfailability_tightens_but_never_loosens() -> None:
    from src.analytics.pitchgpt_v3_gates import tighten_threshold

    # Frozen v2 comfortably inside the line -> tighten to 0.6x its value.
    r = tighten_threshold(0.020, 0.010)
    assert r["tightened"] and abs(r["effective_threshold"] - 0.006) < 1e-12
    # Floor at 0.005.
    r = tighten_threshold(0.020, 0.002)
    assert abs(r["effective_threshold"] - 0.005) < 1e-12
    # Frozen v2 fails -> the threshold stands as written, never loosened.
    r = tighten_threshold(0.010, 0.080)
    assert not r["tightened"] and r["effective_threshold"] == 0.010
    for spec, v2 in ((0.010, 0.004), (0.015, 0.2), (0.020, 0.0201), (0.010, 0.010)):
        assert tighten_threshold(spec, v2)["effective_threshold"] <= spec


def test_gate_statistics_recover_known_answers() -> None:
    """Sanity-check the estimators on constructed data."""
    from src.analytics.pitchgpt_v3_gates import (
        classwise_ece, decision_deciles, ks_uniform, position_marginal_gap, top1_ece,
    )

    rs = np.random.RandomState(0)
    n = 20_000
    # A perfectly calibrated 4-class model: labels drawn from its own probs.
    p = rs.dirichlet(np.ones(4) * 2.0, size=n)
    y = np.array([rs.choice(4, p=row) for row in p])
    assert classwise_ece(p, y)["classwise_ece"] < 0.01
    assert top1_ece(p, y) < 0.02
    # A deliberately overconfident model must score badly.
    sharp = p ** 4
    sharp /= sharp.sum(axis=1, keepdims=True)
    assert classwise_ece(sharp, y)["classwise_ece"] > classwise_ece(p, y)["classwise_ece"]

    assert ks_uniform(np.linspace(0, 1, 10_001)) < 0.001
    assert ks_uniform(rs.beta(2, 5, size=20_000)) > 0.1

    gap = position_marginal_gap(
        np.array([[0.5, 0.5], [0.4, 0.6]]), np.array([[0.5, 0.5], [0.42, 0.58]]),
    )
    assert abs(gap["max_abs_gap"] - 0.02) < 1e-12 and not gap["pass"]

    # Decision calibration: a truly calibrated predictor passes at a sample
    # size where the 1.0pp per-decile line is above estimation noise; a 5pp
    # biased one fails.
    m = 400_000
    pred = rs.uniform(0.05, 0.5, size=m)
    act = (rs.uniform(size=m) < pred).astype(float)
    good = decision_deciles(pred, act)
    assert good["pass"], good["weighted_mean_abs_gap"]
    biased = decision_deciles(pred, np.clip(act - 0.05, 0.0, 1.0))
    assert not biased["pass"]
    assert biased["weighted_mean_abs_gap"] > good["weighted_mean_abs_gap"]


def test_pa_terminal_classifiers_are_mirrors() -> None:
    """Empirical and rollout PA-terminal classification must agree by construction."""
    from src.analytics.pitchgpt_sim import ROLLOUT_PAD_OUTCOME
    from src.analytics.pitchgpt_v3_infer import (
        PA_BB, PA_HIT, PA_K, PA_OUT, PA_TRUNC,
        classify_rollout_terminals, empirical_pa_terminals,
    )

    # ball x4 -> BB; CS,CS,SS -> K; foul x6 -> truncated; in_play_hit -> hit.
    seqs = [[0, 0, 0, 0], [1, 1, 2], [3, 3, 3, 3, 3, 3], [3, 5], [4]]
    want = [PA_BB, PA_K, PA_TRUNC, PA_HIT, PA_OUT]
    n = len(seqs)
    pa = {
        "pa_outc": np.full((n, 6), -1, dtype=np.int8),
        "pa_len": np.array([len(s) for s in seqs], dtype=np.int8),
        "pa_ctx_idx": np.zeros((n, 6, 6), dtype=np.int8),
    }
    for i, s in enumerate(seqs):
        pa["pa_outc"][i, : len(s)] = s
    assert empirical_pa_terminals(pa).tolist() == want

    # Same sequences seen as rollout output must classify identically.
    outc = pa["pa_outc"]
    term = np.zeros((n, 6), dtype=bool)
    pa_outcome = np.full(n, ROLLOUT_PAD_OUTCOME, dtype=np.int64)
    final = np.zeros((n, 2), dtype=np.int64)
    for i, s in enumerate(seqs):
        b = k = 0
        for j, o in enumerate(s):
            if o in (4, 5, 6):
                term[i, j] = True
                pa_outcome[i] = o
                break
            b, k = _advance_count(b, k, o)
            if b >= 4 or k >= 3:
                term[i, j] = True
                break
        final[i] = (min(b, 4), min(k, 3))
    assert classify_rollout_terminals(pa_outcome, final, term).tolist() == want
    assert outc.shape == (n, 6)


def test_no_v3_code_path_hardcodes_a_forbidden_season() -> None:
    """No v3 *executable* statement may embed 2025/2026 as a literal.

    AST-based so docstrings and comments are ignored and a real hazard — a
    season literal reaching a SQL string or a default argument — cannot hide.
    The only sanctioned literals are the policy constant itself and the
    ledger-gated lockbox door.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    allowed_files = {"pitchgpt_v3_lockbox_2026.py"}
    allowed_targets = {
        "FORBIDDEN_SEASONS", "FORBIDDEN_FIT_COHORTS",
        "BUDGETED_SEASON", "LOCKBOX_SEASON",
    }
    offenders: list[str] = []

    for p in sorted(root.glob("scripts/pitchgpt_v3_*.py")) + sorted(
        root.glob("src/analytics/pitchgpt_v3*.py")
    ):
        if p.name in allowed_files:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        sanctioned: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                names = {t.id for t in targets if isinstance(t, ast.Name)}
                if names & allowed_targets:
                    for sub in ast.walk(node):
                        sanctioned.add(id(sub))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, int)
                and node.value in (2025, 2026)
                and id(node) not in sanctioned
            ):
                offenders.append(f"{p.name}:{node.lineno}: literal {node.value}")

    assert not offenders, (
        "forbidden-season literals in v3 code:\n" + "\n".join(offenders)
    )


# ── §6.3 protocol: subsample disclosure and bandwidth pinning (§9 13/14) ─────


def test_skce_reports_its_subsample_provenance() -> None:
    """§9 entry 13: an SKCE number may never be published without ``n_used``.

    §6.3 authorizes no subsample; the O(n^2) U-statistic forces one, so the
    estimator has to hand every caller the row count it actually used.
    """
    from src.analytics.pitchgpt_v3_gates import KCE_SUBSAMPLE, skce_test

    rng = np.random.default_rng(0)
    n = KCE_SUBSAMPLE + 500
    p = rng.dirichlet(np.ones(4), size=n)
    y = np.array([rng.choice(4, p=row) for row in p])

    r = skce_test(p, y, n_boot=5)
    assert r["n_available"] == n
    assert r["n_used"] == KCE_SUBSAMPLE
    assert r["subsampled"] is True
    assert r["subsample_cap"] == KCE_SUBSAMPLE
    for k in ("n_used", "n_available", "subsample_cap", "subsample_seed"):
        assert k in r, f"{k} must be reported so a bare SKCE cannot be published"

    small = skce_test(p[:50], y[:50], n_boot=5)
    assert small["subsampled"] is False and small["n_used"] == 50


def test_skce_same_bandwidth_is_the_only_like_for_like_comparison() -> None:
    """§9 entry 14: two SKCEs are comparable only under one shared kernel.

    Guards the §6.2 side-by-side: the frozen-v2 reference must be handed v3's
    bandwidth, because the median heuristic lands somewhere different on each
    model's probabilities.
    """
    from src.analytics.pitchgpt_v3_gates import (
        median_heuristic_bandwidth,
        skce_test,
    )

    rng = np.random.default_rng(7)
    p_a = rng.dirichlet(np.ones(5), size=400)
    p_b = rng.dirichlet(np.ones(5) * 6.0, size=400)     # sharper model
    y = np.array([rng.choice(5, p=row) for row in p_a])

    bw_a = median_heuristic_bandwidth(p_a, seed=42)
    bw_b = median_heuristic_bandwidth(p_b, seed=42)
    assert abs(bw_a - bw_b) > 1e-6, "the two models must disagree on bandwidth"

    shared = skce_test(p_b, y, bandwidth=bw_a, n_boot=5)
    own = skce_test(p_b, y, n_boot=5)
    assert shared["bandwidth"] == pytest.approx(bw_a)
    assert own["bandwidth"] == pytest.approx(bw_b)
    assert shared["bandwidth_source"] == "pinned"
    assert own["bandwidth_source"] == "median-heuristic-fitted-on-this-cohort"
    assert shared["skce"] != own["skce"], (
        "different kernels give different SKCEs — which is exactly why the "
        "frozen-v2 reference may not fit its own"
    )


def test_graded_skce_refuses_to_fit_a_bandwidth_on_the_graded_cohort() -> None:
    """§6.3 fixes the bandwidth BEFORE the contact — the graded path enforces it."""
    from src.analytics.pitchgpt_v3_gates import (
        BandwidthNotPinnedError,
        graded_skce_test,
        skce_test,
    )

    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(4), size=200)
    y = np.array([rng.choice(4, p=row) for row in p])

    with pytest.raises(BandwidthNotPinnedError):
        skce_test(p, y, n_boot=5, allow_bandwidth_fit=False)

    with pytest.raises(BandwidthNotPinnedError):
        graded_skce_test("zone", p, y, {"type": 0.5}, n_boot=5)

    r = graded_skce_test("type", p, y, {"type": 0.5}, n_boot=5)
    assert r["bandwidth"] == pytest.approx(0.5)
    assert r["bandwidth_source"] == "pinned"


def test_pinned_bandwidth_artifact_is_write_once_and_provenance_guarded(
    tmp_path,
) -> None:
    """The pinned kernel may only come from the 2024 dev cohort, and only once."""
    from src.analytics.pitchgpt_v3_gates import (
        KCE_BANDWIDTH_FIT_SEASON,
        load_pinned_bandwidths,
        save_pinned_bandwidths,
    )

    assert KCE_BANDWIDTH_FIT_SEASON == 2024
    path = tmp_path / "bw.json"
    save_pinned_bandwidths(
        path, {"type": 0.5, "zone": 0.6}, {"fit_cohort_season": 2024},
    )
    bw, payload = load_pinned_bandwidths(path)
    assert bw == {"type": 0.5, "zone": 0.6}
    assert payload["fit_cohort_season"] == 2024

    with pytest.raises(FileExistsError):
        save_pinned_bandwidths(path, {"type": 0.9}, {"fit_cohort_season": 2024})

    # A bandwidth fitted on a budgeted or sealed cohort is never legal.
    for season in (2025, 2026):
        with pytest.raises(ValueError):
            save_pinned_bandwidths(
                tmp_path / f"{season}.json", {"type": 0.5},
                {"fit_cohort_season": season},
            )
        tainted = tmp_path / f"load_{season}.json"
        tainted.write_text(
            json.dumps({"bandwidths": {"type": 0.5}, "fit_cohort_season": season})
        )
        with pytest.raises(ValueError):
            load_pinned_bandwidths(tainted)


def test_dev_gates_frozen_v2_reference_shares_the_v3_bandwidth() -> None:
    """§6.2's side-by-side must not refit the kernel on frozen v2's probs.

    Source-level, because the regression it guards is a *missing argument*:
    ``skce_test(p2, y2)`` silently refits, and the resulting numbers look
    perfectly plausible.
    """
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "pitchgpt_v3_dev_gates.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "skce_test"
    ]
    assert calls, "expected skce_test call sites in the dev-gate script"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "bandwidth" in kwargs, (
            f"skce_test at line {call.lineno} omits bandwidth=, so it silently "
            "refits the median heuristic on whatever it was handed. Every call "
            "site must state its kernel explicitly (§9 entry 14)."
        )

    # Exactly one call may fit the kernel — the one that establishes the pin,
    # on v3's dev probabilities. Every other call must replay a value.
    fitting = [
        c for c in calls
        for kw in c.keywords
        if kw.arg == "bandwidth"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is None
    ]
    assert len(fitting) == 1, (
        "expected exactly one bandwidth-establishing skce_test call in the "
        f"dev-gate script, found {len(fitting)} at lines "
        f"{[c.lineno for c in fitting]}"
    )


def test_lockbox_entry_point_pins_kce_bandwidth() -> None:
    """The §5.5 door must replay the dev-pinned kernel, never fit a 2026 one."""
    from pathlib import Path

    from scripts import pitchgpt_v3_lockbox_2026 as lb
    from src.analytics.pitchgpt_v3_gates import (
        KCE_PINNED_BANDWIDTH_FILENAME,
        load_pinned_bandwidths,
    )

    assert lb.LOCKBOX_G2_REQUIRES_PINNED_BANDWIDTH is True
    assert hasattr(lb, "load_graded_kce_bandwidths")

    # Safe while sealed: reads only the recorded dev artifact, no 2026 row.
    path = (
        Path(__file__).resolve().parents[1] / "models" / KCE_PINNED_BANDWIDTH_FILENAME
    )
    assert path.exists(), (
        "§6.3 requires the bandwidth to be recorded BEFORE the contact; the "
        "pinned artifact is missing, so the lockbox run must not proceed."
    )
    bw, payload = load_pinned_bandwidths(path)
    assert set(bw) == {"type", "zone", "velo", "outcome"}
    assert payload["fit_cohort_season"] == 2024
    assert lb.load_graded_kce_bandwidths() == bw


# ── K6 claims integrity ──────────────────────────────────────────────────────


def test_claim_values_match_their_cited_audit_artifacts() -> None:
    """Every v3 claim's headline numbers must agree with the audit it cites.

    The regression this guards actually happened: claim ``pitchgpt_v3_killB``
    published the 2024 *dev* Spearman rho (0.2571) as its 2023 *fit-cohort*
    figure, contradicting the very audit JSON named in its ``dataset`` block
    (rho = 0.6).  §9 entry 15.
    """
    import hashlib
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    claims = yaml.safe_load(
        (root / "docs" / "claims" / "claims.yaml").read_text(encoding="utf-8")
    )["claims"]
    by_id = {c["id"]: c for c in claims}

    expected = {
        "pitchgpt_v3_killB": (
            "spearman_rho_position_kl",
            lambda a: a["secondary_position_kl"]["spearman_rho"],
        ),
        "pitchgpt_v3_dev_gate_fail": (
            "G4_spearman_rho_position_kl",
            lambda a: a["gates"]["G4"]["secondary_position_kl_spearman"][
                "spearman_rho"
            ],
        ),
    }

    for claim_id, (field, extract) in expected.items():
        claim = by_id[claim_id]
        path = root / claim["dataset"]["path"]
        assert path.exists(), f"{claim_id} cites a missing audit: {path}"

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == claim["dataset"]["sha256"], (
            f"{claim_id}: cited sha256 does not match the artifact on disk"
        )

        audit = json.loads(path.read_text(encoding="utf-8"))
        assert claim["value"][field] == pytest.approx(extract(audit), abs=5e-5), (
            f"{claim_id}.{field} disagrees with its OWN cited audit artifact"
        )

    # The two rho values belong to different cohorts and must stay distinct.
    fit_rho = by_id["pitchgpt_v3_killB"]["value"]["spearman_rho_position_kl"]
    dev_rho = by_id["pitchgpt_v3_dev_gate_fail"]["value"][
        "G4_spearman_rho_position_kl"
    ]
    assert fit_rho != dev_rho, (
        "the 2023 fit-cohort and 2024 dev rho are different measurements; "
        "equal values mean one surface was copied from the other again"
    )
