#!/usr/bin/env python
"""CRC Overall Survival Risk Prediction Pipeline — Modular Orchestrator.

This file is the single entry point for the pipeline.  All implementation
logic lives in the following modules (each ≤ 2 000 lines):

    _pipeline_core   – environment checks, config, constants, RunContext,
                       data I/O, survival utilities, IPCW, evaluation, plotting
    _causal_infer    – causal RMST doubly-robust estimation, causal screening,
                       Cox fitting, Uno C-index, prediction figures
    _gan_augment     – synthetic-data QC, GAN class hierarchy
                       (ConditionalTabularGAN / SurvivalConditionalTabularGAN /
                       FeatureSpaceGAN), GAN training & selection
    _step_common     – clinical feature encoding, model-fitting helpers,
                       external-cohort loading, legacy monolithic run_pipeline,
                       main(), main_step03_5()
    _step01          – Step 01: multi-omics preprocessing
    _step02          – Step 02: causal feature screening
    _step03_04       – Step 03: multi-omics pretraining
                       Step 04: survival prediction modeling
    _step04_5_kalman_risk_smoothing
                     – Step 04.5: Kalman risk smoothing (Stage 1 + Stage 2)
                       Multi-model fusion, multi-timepoint trajectory
                       POSITIONING: post-hoc calibration, NOT primary model selection
    _step04_6_lstm_kalman_risk_trajectory
                     – Step 04.6: LSTM + Kalman risk trajectory (Stage 3)
                       POSITIONING: exploratory/supplementary ONLY
    _step05_rest     – Step 05: validation & visualization
                       Step 06: interpretability & biology
                       Multi-timepoint analysis
                       run_full_pipeline()
"""

from __future__ import annotations

# ── Re-export every public symbol for backward compatibility ────────────
from _pipeline_core import *   # noqa: F401,F403
from _causal_infer import *    # noqa: F401,F403
from _gan_augment import *     # noqa: F401,F403
from _step_common import *     # noqa: F401,F403
from _step01 import *          # noqa: F401,F403
from _step02 import *          # noqa: F401,F403
from _step03_04 import *       # noqa: F401,F403
from _step05_rest import *     # noqa: F401,F403

# ── Explicit imports for clarity (entry points & orchestrator) ──────────
from _pipeline_core import (
    enforce_project_environment,
    initialize_run,
    default_config,
    load_config,
    resolve_path,
    RunContext,
)

from _step_common import (
    main as legacy_main,
    main_step03_5,
    run_pipeline,
    build_arg_parser,
    build_parser_step03_5,
)

from _step01 import main_step01, build_parser_step01
from _step02 import main_step02, build_parser_step02
from _step03_04 import main_step03, main_step04, build_parser_step03, build_parser_step04
from _step04_5_kalman_risk_smoothing import main_step04_5, build_parser_step04_5
from _step04_6_lstm_kalman_risk_trajectory import main_step04_6, build_parser_step04_6
from _step05_rest import (
    main_step05,
    main_step06,
    main_multi_timepoint,
    run_full_pipeline,
    build_parser_step05,
    build_parser_step06,
    build_parser_multi_timepoint,
)


if __name__ == "__main__":
    import sys
    sys.exit(run_full_pipeline())
