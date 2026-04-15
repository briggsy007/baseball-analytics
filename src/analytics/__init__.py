"""Analytics modules for the baseball analytics platform."""

__all__: list[str] = []

# ── Shared infrastructure ──
from src.analytics.base import BaseAnalyticsModel
from src.analytics.registry import ModelRegistry

__all__ += [
    "BaseAnalyticsModel",
    "ModelRegistry",
]

# ── Matchup analysis ──
try:
    from src.analytics.matchups import (
        get_matchup_stats,
        get_pitcher_profile,
        get_batter_profile,
        estimate_matchup_woba,
        generate_matchup_card,
        get_lineup_matchups,
    )
    __all__ += [
        "get_matchup_stats",
        "get_pitcher_profile",
        "get_batter_profile",
        "estimate_matchup_woba",
        "generate_matchup_card",
        "get_lineup_matchups",
    ]
except ImportError:
    pass

# ── Bullpen strategy ──
try:
    from src.analytics.bullpen import (
        calculate_fatigue_score,
        get_bullpen_state,
        calculate_leverage_index,
        recommend_reliever,
        get_upcoming_batters,
    )
    __all__ += [
        "calculate_fatigue_score",
        "get_bullpen_state",
        "calculate_leverage_index",
        "recommend_reliever",
        "get_upcoming_batters",
    ]
except ImportError:
    pass

# ── Anomaly detection (built by another engineer) ──
try:
    from src.analytics.anomaly import (
        calculate_pitcher_baselines,
        detect_velocity_anomaly,
        detect_spin_anomaly,
        detect_release_point_drift,
        detect_pitch_mix_anomaly,
        detect_batter_anomalies,
        GameAnomalyMonitor,
    )
    __all__ += [
        "calculate_pitcher_baselines",
        "detect_velocity_anomaly",
        "detect_spin_anomaly",
        "detect_release_point_drift",
        "detect_pitch_mix_anomaly",
        "detect_batter_anomalies",
        "GameAnomalyMonitor",
    ]
except ImportError:
    pass

# ── Stuff+ pitch quality model ──
try:
    from src.analytics.stuff_model import (
        train_stuff_model,
        calculate_stuff_plus,
        get_pitch_grade,
        batch_calculate_stuff_plus,
    )
    __all__ += [
        "train_stuff_model",
        "calculate_stuff_plus",
        "get_pitch_grade",
        "batch_calculate_stuff_plus",
    ]
except ImportError:
    pass

# ── Pitch sequencing analysis ──
try:
    from src.analytics.pitch_sequencing import (
        get_pitch_sequence,
        analyze_sequencing_patterns,
        detect_sequencing_anomaly,
        get_batter_pitch_type_vulnerability,
        recommend_pitch_plan,
    )
    __all__ += [
        "get_pitch_sequence",
        "analyze_sequencing_patterns",
        "detect_sequencing_anomaly",
        "get_batter_pitch_type_vulnerability",
        "recommend_pitch_plan",
    ]
except ImportError:
    pass

# ── Pitch Sequence Expected Threat (PSET) ──
try:
    from src.analytics.pset import (
        PSETModel,
        build_transition_matrix,
        calculate_pset,
        batch_calculate as pset_batch_calculate,
        get_best_sequences,
        compute_predictability,
        compute_tunnel_scores,
    )
    __all__ += [
        "PSETModel",
        "build_transition_matrix",
        "calculate_pset",
        "pset_batch_calculate",
        "get_best_sequences",
        "compute_predictability",
        "compute_tunnel_scores",
    ]
except ImportError:
    pass

# ── Player Sharpe Ratio & Efficient Lineup Frontier ──
try:
    from src.analytics.sharpe_lineup import (
        SharpeLineupModel,
        calculate_player_sharpe,
        batch_player_sharpe,
        compute_correlation_matrix,
        optimize_lineup,
        efficient_frontier,
        get_sharpe_leaderboard,
    )
    __all__ += [
        "SharpeLineupModel",
        "calculate_player_sharpe",
        "batch_player_sharpe",
        "compute_correlation_matrix",
        "optimize_lineup",
        "efficient_frontier",
        "get_sharpe_leaderboard",
    ]
except ImportError:
    pass

# ── Pitch Implied Volatility Surface (PIVS) ──
try:
    from src.analytics.volatility_surface import (
        PitchVolatilitySurfaceModel,
        calculate_volatility_surface,
        batch_calculate as pivs_batch_calculate,
        compare_surfaces,
        classify_outcome,
        compute_cell_entropy,
        smooth_surface,
    )
    __all__ += [
        "PitchVolatilitySurfaceModel",
        "calculate_volatility_surface",
        "pivs_batch_calculate",
        "compare_surfaces",
        "classify_outcome",
        "compute_cell_entropy",
        "smooth_surface",
    ]
except ImportError:
    pass

# ── CausalWAR -- Causal Inference Player Valuation ──
try:
    from src.analytics.causal_war import (
        CausalWARModel,
        CausalWARConfig,
        train as causal_war_train,
        calculate_causal_war,
        batch_calculate as causal_war_batch_calculate,
        get_leaderboard as causal_war_get_leaderboard,
    )
    __all__ += [
        "CausalWARModel",
        "CausalWARConfig",
        "causal_war_train",
        "calculate_causal_war",
        "causal_war_batch_calculate",
        "causal_war_get_leaderboard",
    ]
except ImportError:
    pass

# ── Pitch Decay Rate (PDR) -- per-pitch-type fatigue cliff detection ──
try:
    from src.analytics.pitch_decay import (
        PitchDecayRateModel,
        calculate_pdr,
        batch_calculate as pdr_batch_calculate,
        get_first_to_die,
        compute_pitch_type_quality,
        detect_cliff_point,
        get_game_cliff_data,
    )
    __all__ += [
        "PitchDecayRateModel",
        "calculate_pdr",
        "pdr_batch_calculate",
        "get_first_to_die",
        "compute_pitch_type_quality",
        "detect_cliff_point",
        "get_game_cliff_data",
    ]
except ImportError:
    pass

# ── Lineup Order Flow Toxicity (LOFT) ──
try:
    from src.analytics.loft import (
        LOFTModel,
        classify_pitch_flow,
        compute_game_loft,
        compute_pitcher_baseline,
        detect_toxicity_events,
        batch_game_analysis,
        calculate_loft,
    )
    __all__ += [
        "LOFTModel",
        "classify_pitch_flow",
        "compute_game_loft",
        "compute_pitcher_baseline",
        "detect_toxicity_events",
        "batch_game_analysis",
        "calculate_loft",
    ]
except ImportError:
    pass

# ── Viscoelastic Workload Response (VWR) ──
try:
    from src.analytics.viscoelastic_workload import (
        ViscoelasticWorkloadModel,
        compute_pitch_stress,
        compute_strain_state,
        creep_compliance,
        calculate_vwr,
        batch_calculate as vwr_batch_calculate,
        fit_pitcher_parameters as vwr_fit_pitcher_parameters,
        predict_recovery as vwr_predict_recovery,
    )
    __all__ += [
        "ViscoelasticWorkloadModel",
        "compute_pitch_stress",
        "compute_strain_state",
        "creep_compliance",
        "calculate_vwr",
        "vwr_batch_calculate",
        "vwr_fit_pitcher_parameters",
        "vwr_predict_recovery",
    ]
except ImportError:
    pass

# ── Baserunner Gravity Index (BGI) ──
try:
    from src.analytics.baserunner_gravity import (
        BaserunnerGravityModel,
        BGIConfig,
        train as bgi_train,
        calculate_bgi,
        batch_calculate as bgi_batch_calculate,
        get_gravity_leaderboard,
        compute_runner_threat_rate,
        compute_gravity_effect,
    )
    __all__ += [
        "BaserunnerGravityModel",
        "BGIConfig",
        "bgi_train",
        "calculate_bgi",
        "bgi_batch_calculate",
        "get_gravity_leaderboard",
        "compute_runner_threat_rate",
        "compute_gravity_effect",
    ]
except ImportError:
    pass

# ── Allostatic Batting Load (ABL) ──
try:
    from src.analytics.allostatic_load import (
        AllostaticLoadModel,
        ABLConfig,
        calculate_abl,
        batch_calculate as abl_batch_calculate,
        compute_game_stressors,
        compute_leaky_load,
        validate_against_outcomes as abl_validate_against_outcomes,
        predict_recovery_days,
    )
    __all__ += [
        "AllostaticLoadModel",
        "ABLConfig",
        "calculate_abl",
        "abl_batch_calculate",
        "compute_game_stressors",
        "compute_leaky_load",
        "abl_validate_against_outcomes",
        "predict_recovery_days",
    ]
except ImportError:
    pass

# ── Defensive Pressing Intensity (DPI) ──
try:
    from src.analytics.defensive_pressing import (
        DefensivePressingModel,
        DPIConfig,
        train_expected_out_model,
        compute_expected_outs,
        calculate_game_dpi,
        calculate_team_dpi,
        batch_calculate as dpi_batch_calculate,
        get_player_dpi,
        compute_spray_angle,
        build_bip_features,
        get_team_game_dpi_timeline,
    )
    __all__ += [
        "DefensivePressingModel",
        "DPIConfig",
        "train_expected_out_model",
        "compute_expected_outs",
        "calculate_game_dpi",
        "calculate_team_dpi",
        "dpi_batch_calculate",
        "get_player_dpi",
        "compute_spray_angle",
        "build_bip_features",
        "get_team_game_dpi_timeline",
    ]
except ImportError:
    pass

# ── PitchGPT -- Transformer-based pitch sequence model ──
try:
    from src.analytics.pitchgpt import (
        PitchGPT,
        PitchGPTModel,
        PitchTokenizer,
        PitchSequenceDataset,
        train_pitchgpt,
        calculate_predictability as pitchgpt_calculate_predictability,
        calculate_disruption_index,
        batch_calculate as pitchgpt_batch_calculate,
    )
    __all__ += [
        "PitchGPT",
        "PitchGPTModel",
        "PitchTokenizer",
        "PitchSequenceDataset",
        "train_pitchgpt",
        "pitchgpt_calculate_predictability",
        "calculate_disruption_index",
        "pitchgpt_batch_calculate",
    ]
except ImportError:
    pass

# ── Win probability (built by another engineer) ──
try:
    from src.analytics.win_probability import (
        build_run_expectancy_matrix,
        build_win_expectancy_table,
        calculate_win_probability,
        calculate_win_prob_delta,
        build_win_prob_curve,
        format_base_out_state,
    )
    __all__ += [
        "build_run_expectancy_matrix",
        "build_win_expectancy_table",
        "calculate_win_probability",
        "calculate_win_prob_delta",
        "build_win_prob_curve",
        "format_base_out_state",
    ]
except ImportError:
    pass
