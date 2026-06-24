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
# LSTM Risk Trajectory Model
# ──────────────────────────────────────────────────────────────────────────────

class LSTMRiskTrajectory:
    """LSTM model for multi-horizon risk trajectory prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model = None
        self.device = None
        self.feature_scaler = None
        self.training_history: list[dict[str, float]] = []

        if torch is None or nn is None:
            raise RuntimeError("PyTorch not available. Cannot use LSTM model.")

    def _build_model(self, n_timepoints: int):
        import torch.nn as nn_torch

        class LSTMNet(nn_torch.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout, n_timepoints):
                super().__init__()
                self.lstm = nn_torch.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn_torch.Linear(hidden_dim, n_timepoints)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_out = lstm_out[:, -1, :]
                return self.fc(last_out)

        return LSTMNet(self.input_dim, self.hidden_dim, self.num_layers, self.dropout, n_timepoints)

    def fit(
        self,
        X_train: np.ndarray,
        pseudo_targets: np.ndarray,
        ipcw_weights: np.ndarray,
        X_val: np.ndarray | None = None,
        pseudo_val: np.ndarray | None = None,
        ipcw_val: np.ndarray | None = None,
    ):
        import torch as torch_lib
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        self.device = torch_lib.device("cuda" if torch_lib.cuda.is_available() else "cpu")
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        n_timepoints = pseudo_targets.shape[1]
        self.model = self._build_model(n_timepoints)
        self.model.to(self.device)

        X_train_seq = X_train_scaled[:, np.newaxis, :]
        X_tensor = torch_lib.FloatTensor(X_train_seq)
        y_tensor = torch_lib.FloatTensor(pseudo_targets)
        w_tensor = torch_lib.FloatTensor(ipcw_weights)
        train_dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        val_loader = None
        if X_val is not None and pseudo_val is not None:
            X_val_scaled = self.feature_scaler.transform(X_val)
            X_val_seq = X_val_scaled[:, np.newaxis, :]
            val_dataset = TensorDataset(
                torch_lib.FloatTensor(X_val_seq),
                torch_lib.FloatTensor(pseudo_val),
                torch_lib.FloatTensor(ipcw_val) if ipcw_val is not None else torch_lib.ones_like(torch_lib.FloatTensor(pseudo_val)),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch_lib.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        def ipcw_masked_mse(pred, target, weight):
            valid_mask = (weight > 0) & torch_lib.isfinite(target) & torch_lib.isfinite(weight)
            if not bool(valid_mask.any().item()):
                return torch_lib.tensor(0.0, device=pred.device, requires_grad=True)
            safe_target = torch_lib.where(valid_mask, target, pred.detach())
            safe_weight = torch_lib.where(valid_mask, weight, torch_lib.zeros_like(weight))
            mse = ((pred - safe_target) ** 2) * safe_weight
            per_sample_sum = mse.sum(dim=1)
            per_sample_count = valid_mask.sum(dim=1).clamp_min(1)
            per_sample_loss = per_sample_sum / per_sample_count
            active_sample_mask = valid_mask.any(dim=1)
            return per_sample_loss[active_sample_mask].mean()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y, batch_w in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_w = batch_w.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = ipcw_masked_mse(pred, batch_y, batch_w)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            val_loss = float("nan")
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch_lib.no_grad():
                    for batch_X, batch_y, batch_w in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        batch_w = batch_w.to(self.device)
                        pred = self.model(batch_X)
                        loss = ipcw_masked_mse(pred, batch_y, batch_w)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

            self.training_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch as torch_lib

        if self.model is None or self.feature_scaler is None:
            raise RuntimeError("Model not fitted yet.")
        X_scaled = self.feature_scaler.transform(X)
        X_seq = X_scaled[:, np.newaxis, :]
        self.model.eval()
        with torch_lib.no_grad():
            X_tensor = torch_lib.FloatTensor(X_seq).to(self.device)
            pred = self.model(X_tensor)
            return pred.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: LSTM + Kalman Risk Trajectory
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_feature_block(clinical: pd.DataFrame, embedding: pd.DataFrame | None, feature_source: str) -> tuple[pd.DataFrame, list[str]]:
    clinical_work = clinical.copy()
    if "PATIENT_ID" not in clinical_work.columns:
        raise ValueError("clinical must contain PATIENT_ID.")

    clinical_cols = []
    for col in ["AGE", "SEX"]:
        if col in clinical_work.columns:
            clinical_cols.append(col)
    if not clinical_cols:
        clinical_cols = [c for c in clinical_work.columns if c != "PATIENT_ID"]

    clinical_block = clinical_work[["PATIENT_ID"] + clinical_cols].copy()
    for col in clinical_cols:
        if clinical_block[col].dtype == object or str(clinical_block[col].dtype).startswith("category"):
            clinical_block[col] = clinical_block[col].map({"Male": 1, "Female": 0, "M": 1, "F": 0}).fillna(0)
        clinical_block[col] = pd.to_numeric(clinical_block[col], errors="coerce")
    for col in clinical_cols:
        med = pd.to_numeric(clinical_block[col], errors="coerce").median()
        clinical_block[col] = pd.to_numeric(clinical_block[col], errors="coerce").fillna(med)

    if feature_source == "clinical":
        return clinical_block, clinical_cols

    if embedding is None:
        raise ValueError("embedding is required for embedding-based feature sources.")
    if "PATIENT_ID" not in embedding.columns:
        raise ValueError("embedding must contain PATIENT_ID.")

    embedding_cols = [c for c in embedding.columns if c != "PATIENT_ID"]
    embedding_block = embedding[["PATIENT_ID"] + embedding_cols].copy()
    for col in embedding_cols:
        embedding_block[col] = pd.to_numeric(embedding_block[col], errors="coerce")
        embedding_block[col] = embedding_block[col].fillna(embedding_block[col].median())

    if feature_source == "embedding":
        return embedding_block, embedding_cols

    merged = clinical_block.merge(embedding_block, on="PATIENT_ID", how="inner")
    merged_cols = [c for c in merged.columns if c != "PATIENT_ID"]
    return merged, merged_cols


def _build_tau_sequence_features(
    base_features: pd.DataFrame,
    tau_list: list[float],
) -> tuple[np.ndarray, list[str]]:
    feature_cols = [c for c in base_features.columns if c != "PATIENT_ID"]
    x_base = base_features[feature_cols].to_numpy(dtype=float)
    if x_base.size == 0:
        raise ValueError("No usable feature columns for Stage 3.")

    tau_arr = np.asarray(tau_list, dtype=float)
    if tau_arr.size == 0:
        raise ValueError("tau_list cannot be empty.")
    tau_norm = (tau_arr - tau_arr.min()) / max(float(tau_arr.max() - tau_arr.min()), 1e-8)
    tau_log = np.log1p(tau_arr)
    tau_log = (tau_log - tau_log.min()) / max(float(tau_log.max() - tau_log.min()), 1e-8)
    tau_features = np.stack([tau_norm, tau_log], axis=1)

    seq_list = []
    for tvec in tau_features:
        repeated = np.repeat(x_base[:, np.newaxis, :], repeats=1, axis=1)
        tblock = np.tile(tvec.reshape(1, 1, -1), (len(base_features), 1, 1))
        seq_list.append(np.concatenate([repeated, tblock], axis=2))
    seq = np.concatenate(seq_list, axis=1)
    seq_cols = feature_cols + ["tau_norm", "tau_log1p"]
    return seq, seq_cols


def train_lstm_kalman_trajectory(
    features: pd.DataFrame,
    clinical: pd.DataFrame,
    split: pd.DataFrame,
    tau_list: list[float],
    feature_cols: list[str],
    ctx: RunContext,
    lstm_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ctx.logger.info("Stage 3: True tau-list LSTM + Kalman risk trajectory")

    if "PATIENT_ID" not in features.columns:
        raise ValueError("features must contain PATIENT_ID for alignment.")
    if "PATIENT_ID" not in split.columns:
        raise ValueError("split must contain PATIENT_ID for alignment.")
    if "split" not in split.columns:
        raise ValueError("split must contain split labels.")

    if torch is None or nn is None:
        ctx.add_warning("PyTorch not available. Skipping LSTM module.")
        return pd.DataFrame(), pd.DataFrame(), {}

    clinical_work = clinical.copy()
    if not {"OS_STATUS", "OS_MONTHS"}.issubset(clinical_work.columns):
        ctx.add_warning("clinical is missing OS_STATUS/OS_MONTHS; pseudo-observation construction may fail.")

    feature_frame = features[["PATIENT_ID"] + feature_cols].copy().drop_duplicates("PATIENT_ID").reset_index(drop=True)
    feature_frame = feature_frame.merge(split[["PATIENT_ID", "split"]], on="PATIENT_ID", how="inner")
    if feature_frame.empty:
        ctx.add_warning("No overlapping PATIENT_ID values between features and split.")
        return pd.DataFrame(), pd.DataFrame(), {}

    patient_order = feature_frame["PATIENT_ID"].tolist()
    aligned_base = pd.DataFrame({"PATIENT_ID": patient_order})
    split_aligned = aligned_base.merge(split[["PATIENT_ID", "split"]], on="PATIENT_ID", how="left")
    if split_aligned["split"].isna().any():
        missing = int(split_aligned["split"].isna().sum())
        ctx.add_warning(f"{missing} patients in features are missing from split and will be dropped.")
        split_aligned = split_aligned.dropna(subset=["split"]).reset_index(drop=True)
        patient_order = split_aligned["PATIENT_ID"].tolist()
        feature_frame = feature_frame[feature_frame["PATIENT_ID"].isin(patient_order)].reset_index(drop=True)
        aligned_base = pd.DataFrame({"PATIENT_ID": patient_order})

    pseudo_tables: list[pd.DataFrame] = []
    ipcw_tables: list[pd.DataFrame] = []
    for tau in tau_list:
        ep_tau = add_fixed_endpoint(clinical_work.copy(), tau)
        ep_tau = compute_ipcw_weights_tagged(ep_tau, tau)
        ep_tau = compute_pseudo_observations_tagged(ep_tau, tau)
        tag = f"_{int(tau)}m"
        pseudo_col = f"pseudo_risk{tag}_clipped"
        ipcw_col = f"ipcw_weight{tag}"
        tau_table = ep_tau[["PATIENT_ID", pseudo_col, ipcw_col]].copy()
        tau_table = aligned_base.merge(tau_table, on="PATIENT_ID", how="left")
        pseudo_tables.append(tau_table[["PATIENT_ID", pseudo_col]].rename(columns={pseudo_col: f"pseudo_{int(tau)}m"}))
        ipcw_tables.append(tau_table[["PATIENT_ID", ipcw_col]].rename(columns={ipcw_col: f"ipcw_{int(tau)}m"}))

    label_base = aligned_base.copy()
    for tau in tau_list:
        label_base = label_base.merge(pseudo_tables[tau_list.index(tau)], on="PATIENT_ID", how="left")
        label_base = label_base.merge(ipcw_tables[tau_list.index(tau)], on="PATIENT_ID", how="left")

    train_ids = split_aligned.loc[split_aligned["split"] == "train", "PATIENT_ID"].tolist()
    val_ids = split_aligned.loc[split_aligned["split"] == "internal_validation", "PATIENT_ID"].tolist()

    pseudo_train_matrix = label_base.loc[label_base["PATIENT_ID"].isin(train_ids), [f"pseudo_{int(t)}m" for t in tau_list]].to_numpy(dtype=float)
    pseudo_val_matrix = label_base.loc[label_base["PATIENT_ID"].isin(val_ids), [f"pseudo_{int(t)}m" for t in tau_list]].to_numpy(dtype=float)
    ipcw_train_matrix = label_base.loc[label_base["PATIENT_ID"].isin(train_ids), [f"ipcw_{int(t)}m" for t in tau_list]].to_numpy(dtype=float)
    ipcw_val_matrix = label_base.loc[label_base["PATIENT_ID"].isin(val_ids), [f"ipcw_{int(t)}m" for t in tau_list]].to_numpy(dtype=float)

    X_all = feature_frame[feature_cols].to_numpy(dtype=float)
    X_train = feature_frame.loc[feature_frame["PATIENT_ID"].isin(train_ids), feature_cols].to_numpy(dtype=float)
    X_val = feature_frame.loc[feature_frame["PATIENT_ID"].isin(val_ids), feature_cols].to_numpy(dtype=float)

    ctx.logger.info(f"LSTM training: X_train={X_train.shape}, pseudo_train={pseudo_train_matrix.shape}")
    ctx.logger.info(f"LSTM validation: X_val={X_val.shape}, pseudo_val={pseudo_val_matrix.shape}")

    try:
        lstm = LSTMRiskTrajectory(
            input_dim=X_train.shape[1],
            hidden_dim=lstm_config.get("hidden_dim", 64),
            num_layers=lstm_config.get("num_layers", 1),
            dropout=lstm_config.get("dropout", 0.3),
            learning_rate=lstm_config.get("learning_rate", 0.001),
            epochs=lstm_config.get("epochs", 100),
            batch_size=lstm_config.get("batch_size", 32),
            early_stopping_patience=lstm_config.get("early_stopping_patience", 10),
        )
        ctx.logger.info("Training LSTM model...")
        lstm.fit(
            X_train=X_train,
            pseudo_targets=pseudo_train_matrix,
            ipcw_weights=ipcw_train_matrix,
            X_val=X_val,
            pseudo_val=pseudo_val_matrix,
            ipcw_val=ipcw_val_matrix,
        )
        lstm_pred_all = lstm.predict(X_all)
        ctx.logger.info("LSTM training completed successfully")
    except Exception as exc:
        ctx.add_warning(f"LSTM training failed: {exc}")
        import traceback
        ctx.logger.error(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame(), {}

    from _step04_5_kalman_risk_smoothing import KalmanFilter1D

    kf = KalmanFilter1D()
    train_mask = feature_frame["PATIENT_ID"].isin(train_ids).to_numpy()
    train_preds = lstm_pred_all[train_mask]
    all_train_obs = train_preds[np.isfinite(train_preds)]
    if len(all_train_obs) > 0:
        kf.fit(all_train_obs)

    ctx.logger.info(
        f"Kalman smoothing LSTM predictions: process_var={kf.process_variance:.6f}, obs_var={kf.observation_variance:.6f}"
    )

    lstm_kalman_all = np.zeros_like(lstm_pred_all)
    for i in range(len(lstm_pred_all)):
        traj = lstm_pred_all[i, :]
        if np.isfinite(traj).sum() >= 2:
            lstm_kalman_all[i, :] = kf.smooth(traj)
        else:
            lstm_kalman_all[i, :] = traj

    trajectory_lstm = pd.DataFrame({"PATIENT_ID": feature_frame["PATIENT_ID"].to_numpy()})
    trajectory_lstm_kalman = trajectory_lstm.copy()
    for j, tau in enumerate(tau_list):
        tag = f"_{int(tau)}m"
        trajectory_lstm[f"lstm_risk{tag}"] = lstm_pred_all[:, j]
        trajectory_lstm_kalman[f"lstm_kalman_risk{tag}"] = lstm_kalman_all[:, j]

    metadata = {
        "model_type": "lstm_kalman_tau_sequence",
        "positioning": "exploratory/supplementary - NOT eligible for primary model selection",
        "tau_grid": tau_list,
        "feature_cols": feature_cols,
        "lstm_config": lstm_config,
        "training_history": lstm.training_history if lstm else [],
        "n_train": int(len(train_ids)),
        "n_val": int(len(val_ids)),
        "sequence_mode": "true_tau_sequence",
    }
    return trajectory_lstm, trajectory_lstm_kalman, metadata


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def build_parser_step04_6() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exploratory LSTM + Kalman risk trajectory (Stage 3) - SUPPLEMENTARY ONLY."
    )
    add_common_args(parser)
    parser.add_argument("--tau-grid", type=str, default="12,24,36,48,60", help="Comma-separated time horizons (months).")
    parser.add_argument("--lstm-hidden-dim", type=int, default=64, help="LSTM hidden dimension.")
    parser.add_argument("--lstm-num-layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--lstm-dropout", type=float, default=0.3, help="LSTM dropout rate.")
    parser.add_argument("--lstm-epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--lstm-batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument(
        "--feature-source",
        type=str,
        default="clinical_plus_embedding",
        choices=["clinical", "embedding", "clinical_plus_embedding"],
        help="Feature set for LSTM input.",
    )
    return parser


def main_step04_6() -> int:
    parser = build_parser_step04_6()
    args = parser.parse_args()
    ctx = initialize_run(__file__, args)
    cfg = ctx.cfg
    endpoint = args.endpoint

    tau_list = [float(t.strip()) for t in args.tau_grid.split(",") if t.strip()]
    lstm_config = {
        "hidden_dim": args.lstm_hidden_dim,
        "num_layers": args.lstm_num_layers,
        "dropout": args.lstm_dropout,
        "learning_rate": 0.001,
        "epochs": args.lstm_epochs,
        "batch_size": args.lstm_batch_size,
        "early_stopping_patience": 10,
    }

    ctx.logger.warning("=" * 80)
    ctx.logger.warning("IMPORTANT POSITIONING:")
    ctx.logger.warning("This module implements EXPLORATORY LSTM + Kalman analysis.")
    ctx.logger.warning("It is NOT a primary survival model.")
    ctx.logger.warning("It is NOT eligible for primary model selection.")
    ctx.logger.warning("Results are for supplementary/sensitivity analysis only.")
    ctx.logger.warning("=" * 80)

    survival_models_dir = ctx.data_dir("survival_models")
    preprocessed_dir = ctx.data_dir("preprocessed")

    split_path = survival_models_dir / "tcga_train_internal_validation_split.tsv"
    if not split_path.exists():
        ctx.add_warning(f"Split file not found: {split_path}")
        ctx.finalize([])
        return 1

    split = pd.read_csv(split_path, sep="\t")

    clinical_path = preprocessed_dir / f"tcga_{endpoint.lower()}_clinical_endpoint_qc.tsv"
    if not clinical_path.exists():
        clinical_path = Path(cfg["cohorts"]["tcga"]["clinical_patient"])
    clinical = read_cbio_table(str(clinical_path))

    if endpoint == "OS" and {"OS_STATUS", "OS_MONTHS"}.issubset(clinical.columns):
        clinical["event"] = parse_survival_status(clinical["OS_STATUS"])
        clinical["time_months"] = numeric_series(clinical["OS_MONTHS"])

    embedding = None
    embedding_path = ctx.data_dir("embeddings") / "combined_omics_embedding.tsv"
    if args.feature_source in ("embedding", "clinical_plus_embedding"):
        if not embedding_path.exists():
            ctx.add_warning(f"Embedding not found: {embedding_path}")
            ctx.finalize([split_path, clinical_path])
            return 1
        embedding = pd.read_csv(embedding_path, sep="\t")

    try:
        features, feature_cols = _prepare_feature_block(clinical, embedding, args.feature_source)
        features = features.merge(split[["PATIENT_ID", "split"]], on="PATIENT_ID", how="inner")
    except Exception as exc:
        ctx.add_warning(f"Failed to prepare features: {exc}")
        ctx.finalize([split_path, clinical_path])
        return 1

    ctx.logger.info(f"Features: {len(features)} patients, {len(feature_cols)} features")
    ctx.logger.info(f"Tau grid: {tau_list}")

    trajectory_lstm, trajectory_lstm_kalman, metadata = train_lstm_kalman_trajectory(
        features=features,
        clinical=clinical,
        split=split,
        tau_list=tau_list,
        feature_cols=feature_cols,
        ctx=ctx,
        lstm_config=lstm_config,
    )

    if trajectory_lstm.empty:
        ctx.add_warning("LSTM training failed or skipped.")
        ctx.finalize([split_path, clinical_path])
        return 1

    lstm_path = survival_models_dir / "lstm_risk_trajectory_raw.tsv"
    ctx.write_table(lstm_path, trajectory_lstm, "analysis_data", "LSTM risk trajectory (raw predictions) - EXPLORATORY ONLY.")

    lstm_kalman_path = survival_models_dir / "lstm_risk_trajectory_kalman_smoothed.tsv"
    ctx.write_table(
        lstm_kalman_path,
        trajectory_lstm_kalman,
        "analysis_data",
        "LSTM + Kalman risk trajectory (smoothed) - EXPLORATORY ONLY.",
    )

    metadata_path = survival_models_dir / "lstm_kalman_trajectory_manifest.json"
    ctx.write_json(metadata_path, metadata, "analysis_data", "LSTM + Kalman trajectory metadata - EXPLORATORY ONLY.")

    ctx.logger.info("LSTM + Kalman trajectory analysis (Stage 3) completed.")
    ctx.logger.warning("Remember: These results are exploratory/supplementary only.")

    ctx.finalize([split_path, clinical_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main_step04_6())
