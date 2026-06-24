from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _pipeline_core import *  # noqa: F401,F403
from _causal_infer import *   # noqa: F401,F403

# ──────────────────────────────────────────────────────────────────────────────
# Kalman Filter 1D Implementation
# ──────────────────────────────────────────────────────────────────────────────

class KalmanFilter1D:
    """Simple 1D Kalman filter for risk score smoothing.
    
    State model: x_t = x_{t-1} + w_t  (random walk)
    Observation model: z_t = x_t + v_t
    
    Parameters estimated from training data:
    - process_variance (w_t variance)
    - observation_variance (v_t variance)
    """
    
    def __init__(self, process_variance: float = 0.01, observation_variance: float = 0.1):
        self.process_variance = process_variance
        self.observation_variance = observation_variance
    
    def fit(self, observations: np.ndarray) -> None:
        """Estimate variances from training observations."""
        obs = observations[np.isfinite(observations)]
        if len(obs) < 2:
            return
        
        # Estimate observation variance from data variance
        self.observation_variance = float(np.var(obs))
        
        # Process variance as fraction of observation variance
        # (smaller = smoother trajectory)
        self.process_variance = self.observation_variance * 0.01
    
    def smooth(self, observations: np.ndarray) -> np.ndarray:
        """Apply Kalman smoothing to observation sequence.
        
        Args:
            observations: Array of noisy observations (can contain NaN)
        
        Returns:
            Smoothed estimates
        """
        n = len(observations)
        if n == 0:
            return np.array([])
        
        # Initialize
        x_pred = np.zeros(n)
        x_filtered = np.zeros(n)
        P_pred = np.zeros(n)
        P_filtered = np.zeros(n)
        
        # Forward pass (filtering)
        x_filtered[0] = np.nanmean(observations) if np.isfinite(observations[0]) else 0.0
        P_filtered[0] = self.observation_variance
        
        for t in range(1, n):
            # Prediction
            x_pred[t] = x_filtered[t-1]
            P_pred[t] = P_filtered[t-1] + self.process_variance
            
            # Update (if observation available)
            if np.isfinite(observations[t]):
                K = P_pred[t] / (P_pred[t] + self.observation_variance)
                x_filtered[t] = x_pred[t] + K * (observations[t] - x_pred[t])
                P_filtered[t] = (1 - K) * P_pred[t]
            else:
                x_filtered[t] = x_pred[t]
                P_filtered[t] = P_pred[t]
        
        # Backward pass (smoothing)
        x_smoothed = np.zeros(n)
        x_smoothed[-1] = x_filtered[-1]
        
        for t in range(n-2, -1, -1):
            if P_pred[t+1] > 0:
                A = P_filtered[t] / P_pred[t+1]
                x_smoothed[t] = x_filtered[t] + A * (x_smoothed[t+1] - x_pred[t+1])
            else:
                x_smoothed[t] = x_filtered[t]
        
        return x_smoothed
    
    def fuse_multiple_observations(self, obs_matrix: np.ndarray) -> np.ndarray:
        """Fuse multiple noisy observations at each time point.
        
        Args:
            obs_matrix: (n_samples, n_observations) array
        
        Returns:
            Fused estimates (n_samples,)
        """
        n_samples, n_obs = obs_matrix.shape
        fused = np.zeros(n_samples)
        
        for i in range(n_samples):
            valid = obs_matrix[i, np.isfinite(obs_matrix[i, :])]
            if len(valid) == 0:
                fused[i] = np.nan
            elif len(valid) == 1:
                fused[i] = valid[0]
            else:
                # Weighted average with inverse variance weights
                # Assuming equal observation quality for simplicity
                fused[i] = np.mean(valid)
        
        return fused


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Multi-model Risk Smoothing
# ──────────────────────────────────────────────────────────────────────────────

def smooth_multimodel_risks(
    risk_table: pd.DataFrame,
    split: pd.DataFrame,
    risk_columns: list[str],
    ctx: RunContext,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fuse multiple model risk scores with a Kalman-inspired latent-risk smoother.
    
    Each model provides a noisy observation of the same latent risk. We use a
    train-split-derived variance heuristic to compute a fusion score. We do not
    treat model order as a time axis.
    """
    ctx.logger.info("Stage 1: Kalman fusion of multi-model risk scores")
    
    risk_matrix = risk_table[risk_columns].to_numpy(dtype=float)
    train_mask = split["split"].to_numpy() == "train"
    train_risks = risk_matrix[train_mask, :]
    
    kf = KalmanFilter1D()
    all_train_obs = train_risks[np.isfinite(train_risks)]
    if len(all_train_obs) > 0:
        kf.fit(all_train_obs)
    
    ctx.logger.info(
        f"Kalman parameters (fitted on train): process_var={kf.process_variance:.6f}, "
        f"obs_var={kf.observation_variance:.6f}"
    )
    
    fused_risks = kf.fuse_multiple_observations(risk_matrix)
    result_table = risk_table.copy()
    result_table["kalman_fused_risk_score"] = fused_risks
    result_table["kalman_smoothed_risk_score"] = fused_risks
    
    utility_records = []
    try:
        endpoint_cols = ["PATIENT_ID", "event", "time_months"]
        endpoint = split[endpoint_cols].copy()
        valid_mask = split["split"].to_numpy() == "internal_validation"
        valid_ep = endpoint[valid_mask].copy()
        valid_risks = fused_risks[valid_mask]
        try:
            from lifelines.utils import concordance_index as _lifelines_ci
            t = valid_ep["time_months"].to_numpy(dtype=float)
            e = valid_ep["event"].to_numpy(dtype=int)
            finite = np.isfinite(valid_risks)
            if finite.sum() >= 5:
                c_index = float(_lifelines_ci(t[finite], -valid_risks[finite], e[finite]))
            else:
                c_index = float("nan")
        except Exception as exc:
            ctx.logger.warning(f"C-index evaluation failed: {exc}")
            c_index = float("nan")
        utility_records.append(
            {
                "score_type": "kalman_fused",
                "split": "internal_validation",
                "n": int(valid_mask.sum()),
                "events": int(valid_ep["event"].sum()),
                "harrell_cindex": c_index,
                "note": "fusion-only; model order is not treated as time",
            }
        )
        ctx.logger.info(f"kalman_fused C-index (internal validation): {c_index:.4f}")
    except Exception as exc:
        ctx.logger.warning(f"Stage 1 evaluation failed: {exc}")
    
    return result_table, utility_records


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Multi-timepoint Risk Trajectory
# ──────────────────────────────────────────────────────────────────────────────

def generate_multi_timepoint_trajectories(
    clinical: pd.DataFrame,
    split: pd.DataFrame,
    base_risk_table: pd.DataFrame,
    primary_risk_col: str,
    tau_list: list[float],
    ctx: RunContext,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Generate risk trajectories across multiple time horizons using pseudo-observations.
    
    For each tau, we:
    1. Create fixed endpoint with add_fixed_endpoint
    2. Compute IPCW weights with compute_ipcw_weights_tagged
    3. Compute pseudo-observations with compute_pseudo_observations_tagged
    4. Train a simple risk model (or reuse base risk)
    5. Apply Kalman smoothing across time
    """
    ctx.logger.info(f"Stage 2: Multi-timepoint trajectory analysis (tau = {tau_list})")
    
    # Prepare clinical endpoint
    if "event" not in clinical.columns or "time_months" not in clinical.columns:
        time_col, event_col = detect_event_time_columns(clinical)
        clinical["time_months"] = safe_numeric(clinical[time_col])
        clinical["event"] = coerce_event(clinical[event_col])
    
    # Initialize trajectory tables
    trajectory_raw = pd.DataFrame({
        "PATIENT_ID": clinical["PATIENT_ID"].to_numpy(),
    })
    
    # For each tau, compute pseudo-risk and risk score
    for tau in tau_list:
        ctx.logger.info(f"Processing tau = {tau} months")
        
        # Add fixed endpoint
        ep_tau = add_fixed_endpoint(clinical.copy(), tau)
        
        # Compute IPCW weights
        ep_tau = compute_ipcw_weights_tagged(ep_tau, tau)
        
        # Compute pseudo-observations
        ep_tau = compute_pseudo_observations_tagged(ep_tau, tau)
        
        tag = f"_{int(tau)}m"
        pseudo_col = f"pseudo_risk{tag}_clipped"
        
        # Merge back to trajectory table
        trajectory_raw = trajectory_raw.merge(
            ep_tau[["PATIENT_ID", pseudo_col]],
            on="PATIENT_ID",
            how="left"
        )
        
        # Use base risk as approximation (scaled to [0,1] range)
        # In a full implementation, we would train tau-specific models
        if primary_risk_col in base_risk_table.columns:
            base_risks = base_risk_table[primary_risk_col].to_numpy()
            # Simple scaling to probability-like range
            risk_finite = base_risks[np.isfinite(base_risks)]
            if len(risk_finite) > 0:
                risk_min, risk_max = np.percentile(risk_finite, [5, 95])
                scaled_risk = (base_risks - risk_min) / (risk_max - risk_min + 1e-8)
                scaled_risk = np.clip(scaled_risk, 0, 1)
            else:
                scaled_risk = np.full(len(base_risks), 0.5)
            
            trajectory_raw[f"risk_score{tag}"] = scaled_risk
    
    # Apply Kalman smoothing across time for each patient
    ctx.logger.info("Applying Kalman smoothing across time horizons")
    
    kf = KalmanFilter1D()
    
    # Fit on train split
    train_mask = split["split"].to_numpy() == "train"
    risk_cols = [f"risk_score_{int(tau)}m" for tau in tau_list]
    train_trajectory = trajectory_raw.loc[train_mask, risk_cols].to_numpy(dtype=float)
    all_train_obs = train_trajectory[np.isfinite(train_trajectory)]
    if len(all_train_obs) > 0:
        kf.fit(all_train_obs)
    
    ctx.logger.info(f"Multi-timepoint Kalman parameters: "
                   f"process_var={kf.process_variance:.6f}, "
                   f"obs_var={kf.observation_variance:.6f}")
    
    # Smooth each patient's trajectory
    trajectory_kalman = trajectory_raw.copy()
    
    for i, row in trajectory_raw.iterrows():
        obs = np.array([row[col] for col in risk_cols], dtype=float)
        if np.isfinite(obs).sum() >= 2:
            smoothed = kf.smooth(obs)
            for j, tau in enumerate(tau_list):
                trajectory_kalman.loc[i, f"risk_score_{int(tau)}m"] = smoothed[j]
    
    # Compute summary statistics
    summary_records = []
    
    for tau in tau_list:
        tag = f"_{int(tau)}m"
        risk_col_raw = f"risk_score{tag}"
        pseudo_col = f"pseudo_risk{tag}_clipped"
        
        if risk_col_raw in trajectory_raw.columns and pseudo_col in trajectory_raw.columns:
            raw_risks = trajectory_raw[risk_col_raw].to_numpy()
            kalman_risks = trajectory_kalman[risk_col_raw].to_numpy()
            pseudo = trajectory_raw[pseudo_col].to_numpy()
            
            # Correlation with pseudo-observations (internal validation split)
            valid_mask = split["split"].to_numpy() == "internal_validation"
            
            raw_valid = raw_risks[valid_mask]
            kalman_valid = kalman_risks[valid_mask]
            pseudo_valid = pseudo[valid_mask]
            
            # Compute correlation
            finite_mask = np.isfinite(raw_valid) & np.isfinite(pseudo_valid)
            if finite_mask.sum() >= 5:
                try:
                    corr_raw = float(np.corrcoef(raw_valid[finite_mask], pseudo_valid[finite_mask])[0, 1])
                    corr_kalman = float(np.corrcoef(kalman_valid[finite_mask], pseudo_valid[finite_mask])[0, 1])
                except Exception:
                    corr_raw = corr_kalman = float("nan")
            else:
                corr_raw = corr_kalman = float("nan")
            
            summary_records.append({
                "tau_months": int(tau),
                "split": "internal_validation",
                "n": int(valid_mask.sum()),
                "correlation_raw_vs_pseudo": corr_raw,
                "correlation_kalman_vs_pseudo": corr_kalman,
            })
            
            ctx.logger.info(f"tau={int(tau)}m: corr(raw,pseudo)={corr_raw:.3f}, "
                          f"corr(kalman,pseudo)={corr_kalman:.3f}")
    
    summary_df = pd.DataFrame(summary_records) if summary_records else pd.DataFrame()
    
    return trajectory_raw, trajectory_kalman, [summary_df.to_dict(orient="records") if not summary_df.empty else []]


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def build_parser_step04_5() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kalman risk smoothing: Stage 1 (multi-model fusion) + Stage 2 (multi-timepoint trajectory)."
    )
    add_common_args(parser)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Internal validation fraction (must match Step 04)."
    )
    parser.add_argument(
        "--tau-grid",
        type=str,
        default="12,24,36,48,60",
        help="Comma-separated time horizons (months) for multi-timepoint analysis."
    )
    return parser


def main_step04_5() -> int:
    parser = build_parser_step04_5()
    args = parser.parse_args()
    ctx = initialize_run(__file__, args)
    cfg = ctx.cfg
    endpoint = args.endpoint
    
    # Parse tau grid
    tau_list = [float(t.strip()) for t in args.tau_grid.split(",")]
    ctx.logger.info(f"Multi-timepoint tau grid: {tau_list}")
    
    # Input directories
    survival_models_dir = ctx.data_dir("survival_models")
    preprocessed_dir = ctx.data_dir("preprocessed")
    
    # Check required inputs
    risk_table_path = survival_models_dir / "internal_validation_risk_scores.tsv"
    split_path = survival_models_dir / "tcga_train_internal_validation_split.tsv"
    
    if not risk_table_path.exists():
        ctx.add_warning(f"Risk table not found: {risk_table_path}")
        ctx.add_warning("Step 04 must be run first to generate risk scores.")
        ctx.finalize([])
        return 1
    
    if not split_path.exists():
        ctx.add_warning(f"Split file not found: {split_path}")
        ctx.finalize([])
        return 1
    
    # Load data
    ctx.logger.info(f"Loading risk table from {risk_table_path}")
    risk_table = pd.read_csv(risk_table_path, sep="\t")
    
    ctx.logger.info(f"Loading split from {split_path}")
    split = pd.read_csv(split_path, sep="\t")
    
    # Identify risk score columns
    risk_columns = [
        col for col in risk_table.columns
        if col.endswith("_risk_score") and col != "kalman_fused_risk_score" and col != "kalman_smoothed_risk_score"
    ]
    
    if len(risk_columns) == 0:
        ctx.add_warning("No risk score columns found in risk table.")
        ctx.finalize([risk_table_path])
        return 1
    
    ctx.logger.info(f"Found {len(risk_columns)} risk score columns: {risk_columns}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1: Multi-model Kalman smoothing
    # ──────────────────────────────────────────────────────────────────────────
    
    smoothed_table, utility_records = smooth_multimodel_risks(
        risk_table=risk_table,
        split=split,
        risk_columns=risk_columns,
        ctx=ctx,
    )
    
    # Write smoothed risk scores
    smoothed_path = survival_models_dir / "kalman_smoothed_risk_scores.tsv"
    ctx.write_table(
        smoothed_path,
        smoothed_table,
        "analysis_data",
        "Kalman-smoothed risk scores (multi-model fusion)."
    )
    
    # Write utility report
    utility_path = survival_models_dir / "kalman_smoothing_utility.tsv"
    ctx.write_table(
        utility_path,
        pd.DataFrame(utility_records),
        "analysis_data",
        "Kalman smoothing utility metrics."
    )
    
    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2: Multi-timepoint trajectory
    # ──────────────────────────────────────────────────────────────────────────
    
    # Load clinical endpoint
    clinical_path = preprocessed_dir / f"tcga_{endpoint.lower()}_clinical_endpoint_qc.tsv"
    if not clinical_path.exists():
        # Try loading from raw source
        clinical_path = Path(cfg["cohorts"]["tcga"]["clinical_patient"])
    
    ctx.logger.info(f"Loading clinical endpoint from {clinical_path}")
    clinical = read_cbio_table(str(clinical_path))
    
    # Ensure endpoint columns
    if endpoint == "OS":
        if "OS_STATUS" in clinical.columns and "OS_MONTHS" in clinical.columns:
            clinical["event"] = parse_survival_status(clinical["OS_STATUS"])
            clinical["time_months"] = numeric_series(clinical["OS_MONTHS"])
    
    # Use primary risk column (clinical_ajcc_cox by convention)
    primary_risk_col = "clinical_ajcc_cox_risk_score"
    if primary_risk_col not in smoothed_table.columns and len(risk_columns) > 0:
        primary_risk_col = risk_columns[0]
    
    trajectory_raw, trajectory_kalman, summary_list = generate_multi_timepoint_trajectories(
        clinical=clinical,
        split=split,
        base_risk_table=smoothed_table,
        primary_risk_col=primary_risk_col,
        tau_list=tau_list,
        ctx=ctx,
    )
    
    # Write trajectories
    traj_raw_path = survival_models_dir / "multi_timepoint_risk_trajectory_raw.tsv"
    ctx.write_table(
        traj_raw_path,
        trajectory_raw,
        "analysis_data",
        "Multi-timepoint risk trajectory (raw, before Kalman smoothing)."
    )
    
    traj_kalman_path = survival_models_dir / "multi_timepoint_risk_trajectory_kalman.tsv"
    ctx.write_table(
        traj_kalman_path,
        trajectory_kalman,
        "analysis_data",
        "Multi-timepoint risk trajectory (Kalman-smoothed across time)."
    )
    
    if summary_list and len(summary_list) > 0 and len(summary_list[0]) > 0:
        summary_path = survival_models_dir / "multi_timepoint_kalman_summary.tsv"
        ctx.write_table(
            summary_path,
            pd.DataFrame(summary_list[0]),
            "analysis_data",
            "Multi-timepoint Kalman smoothing summary."
        )
    
    # ──────────────────────────────────────────────────────────────────────────
    # Write manifest
    # ──────────────────────────────────────────────────────────────────────────
    
    manifest = {
        "module": "_step04_5_kalman_risk_smoothing",
        "endpoint": endpoint,
        "tau_grid": tau_list,
        "risk_columns_used": risk_columns,
        "primary_risk_col": primary_risk_col,
        "stage1_outputs": {
            "smoothed_risk_scores": str(smoothed_path.name),
            "utility_report": str(utility_path.name),
        },
        "stage2_outputs": {
            "trajectory_raw": str(traj_raw_path.name),
            "trajectory_kalman": str(traj_kalman_path.name),
        },
        "positioning": "post-hoc calibration and auxiliary fusion; not eligible for primary model selection",
    }
    
    manifest_path = survival_models_dir / "kalman_smoothing_manifest.json"
    ctx.write_json(
        manifest_path,
        manifest,
        "analysis_data",
        "Kalman smoothing manifest (Stage 1 + Stage 2)."
    )
    
    ctx.logger.info("Kalman risk smoothing (Stage 1 + Stage 2) completed successfully.")
    ctx.finalize([risk_table_path, split_path, clinical_path])
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main_step04_5())
