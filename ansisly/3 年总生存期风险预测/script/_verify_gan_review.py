"""Focused verification harness for the GAN-module fixes.

Loads the REAL TCGA clinical endpoint, builds the clinical feature matrix with
the pipeline's own functions, and exercises:
  1. synthetic_data_qc new fields + behaviour (raw GAN vs calibrated vs SMOTE)
  2. ConditionalTabularGAN / SurvivalConditionalTabularGAN fit+sample (new arch)
  3. SmoteLikeAugmenter
  4. select_train_only_gan_augmentation end-to-end (QC pass + AUC/AP gain)

Runs with reduced epochs/grid for speed; the goal is to validate logic and the
direction/magnitude of the QC and utility changes, not to reproduce the full run.
"""
import importlib.util, dataclasses, warnings, json, sys
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

spec = importlib.util.spec_from_file_location("os3", "3YearsOS.py")
m = importlib.util.module_from_spec(spec)
sys.modules["os3"] = m  # needed so @dataclasses.dataclass can resolve the module
spec.loader.exec_module(m)

ROOT = Path("../../../").resolve()
clin_path = ROOT / "rawData/preprocessed/tcga_os_clinical_endpoint_qc.tsv"

tcga = m.add_fixed_36m_endpoint(m.read_tsv(clin_path), tau=36.0).set_index("PATIENT_ID", drop=False)
tcga = m.compute_ipcw_weights(tcga, 36.0)
tcga = m.compute_pseudo_observations(tcga, 36.0)

# train/val split (stratified-ish) — use 80% as "train_real"
rng = np.random.default_rng(20260604)
ids = list(tcga.index)
rng.shuffle(ids)
cut = int(len(ids) * 0.8)
train_ids, val_ids = ids[:cut], ids[cut:]
train_endpoint = tcga.loc[train_ids].copy()

clinical_pipe, clinical_names = m.build_clinical_preprocessor(train_endpoint)
X_all = pd.DataFrame(clinical_pipe.transform(tcga), index=tcga.index, columns=clinical_names)
X_train_real = X_all.loc[train_ids]

y_obs, _ = m.censor_safe_binary_training(train_endpoint)
print(f"=== Data regime ===")
print(f"clinical features: {len(clinical_names)}  | train rows: {len(X_train_real)}  | labeled(train): {len(y_obs)}  | event rate: {y_obs.mean():.3f}")

# ---------------------------------------------------------------------------
print("\n=== 1. QC new fields: GAN(raw) vs GAN(calibrated) vs SMOTE ===")
gan_fit_ids = list(y_obs.index)
Xg = X_train_real.loc[gan_fit_ids]
cond = m.build_survival_gan_conditions(train_endpoint, gan_fit_ids)

sgan = m.SurvivalConditionalTabularGAN(latent_dim=32, epochs=80, patience=20, batch_size=32, random_seed=42)
sgan.fit(Xg, cond)
print("survival GAN status:", sgan.status)
n_syn = len(gan_fit_ids)
cond_arr = cond[m.SURVIVAL_GAN_CONDITION_COLUMNS].to_numpy(float)
syn_raw = sgan.sample(n_syn, cond_arr)
ev_mask = cond["death_by_36m"].to_numpy(int) == 1
real_lab = y_obs.to_numpy(int)
syn_cal = m._moment_match_calibrate(syn_raw, Xg, ev_mask, real_lab)

qc_raw = m.synthetic_data_qc(Xg, syn_raw, random_seed=42, min_good=4)
qc_cal = m.synthetic_data_qc(Xg, syn_cal, random_seed=42, min_good=4)
print(f"GAN raw       : good={qc_raw['good_count']}/5  ks_stat={qc_raw['ks_stat_mean']} (tol {qc_raw['ks_tol']}) ks_good={qc_raw['ks_good']}  wd={qc_raw['wd_mean']:.3f}(tol {qc_raw['wd_tol']:.3f}) wd_good={qc_raw['wd_good']}  pca={qc_raw['pca_good']} dcr={qc_raw['dcr_good']} mia={qc_raw['mia_good']}")
print(f"GAN calibrated: good={qc_cal['good_count']}/5  ks_stat={qc_cal['ks_stat_mean']} (tol {qc_cal['ks_tol']}) ks_good={qc_cal['ks_good']}  wd={qc_cal['wd_mean']:.3f}(tol {qc_cal['wd_tol']:.3f}) wd_good={qc_cal['wd_good']}  pca={qc_cal['pca_good']} dcr={qc_cal['dcr_good']} mia={qc_cal['mia_good']}")

sm = m.SmoteLikeAugmenter(k_neighbors=5, random_seed=42)
sm.fit(Xg, cond)
print("smote status:", sm.status)
syn_sm = sm.sample(n_syn, cond_arr)
syn_sm_cal = m._moment_match_calibrate(syn_sm, Xg, ev_mask, real_lab)
qc_sm = m.synthetic_data_qc(Xg, syn_sm_cal, random_seed=42, min_good=4)
print(f"SMOTE calib   : good={qc_sm['good_count']}/5  ks_stat={qc_sm['ks_stat_mean']} (tol {qc_sm['ks_tol']}) ks_good={qc_sm['ks_good']}  wd={qc_sm['wd_mean']:.3f}(tol {qc_sm['wd_tol']:.3f}) wd_good={qc_sm['wd_good']}  pca={qc_sm['pca_good']} dcr={qc_sm['dcr_good']} mia={qc_sm['mia_good']}")

# ---------------------------------------------------------------------------
print("\n=== 2. End-to-end selection (reduced grid/epochs) ===")
cfg = m.PipelineConfig(
    project_root=ROOT, raw_data_root=ROOT / "rawData", plan_path=ROOT / "nonexistent",
    output_root=ROOT / "_tmp_out",
    gan_epochs=80, gan_patience=20, gan_latent_dim=32,
    gan_aug_ratio_candidates=(0.5, 1.0),
    gan_sampling_strategies=("balanced_event", "overall"),
    gan_use_feature_space=False,
)
audit = m.QuietAuditLog()
X_aug, ep_aug, report = m.select_train_only_gan_augmentation(X_train_real, train_endpoint, cfg, audit)

base = report.get("baseline_metrics", {})
print(f"baseline      : AUC={base.get('auc_36m_observed')}, AP={base.get('average_precision_36m')}, Brier={base.get('brier_36m_ipcw')}")
print(f"selection status: {report.get('status')}  n_synthetic={report.get('n_synthetic')}")
cands = report.get("candidates", [])
print(f"\n{'strategy':16s} {'ratio':5s} {'method':22s} {'QC':4s} {'AUCgain':8s} {'APgain':8s} {'util'}")
for c in sorted(cands, key=lambda c: -(float(c.get('auc_gain_vs_baseline') or -9) + float(c.get('ap_delta_vs_baseline') or -9))):
    print(f"{c['sampling_strategy']:16s} {c['ratio']:<5.2f} {str(c.get('method','')):22s} {c['qc_good_count']}/5  "
          f"{c.get('auc_gain_vs_baseline',float('nan')):+.5f} {c.get('ap_delta_vs_baseline',float('nan')):+.5f}  "
          f"qc={c['qc_passed']} util={c['utility_passed']}")
if report.get("status") == "fitted":
    print(f"\nSELECTED: method={report.get('selected_method')} strategy={report.get('selected_sampling_strategy')} "
          f"ratio={report.get('selected_aug_ratio')}")
    print(f"  train-only CV AUC={report.get('train_only_cv_auc_36m_observed'):.5f} "
          f"(baseline {base.get('auc_36m_observed'):.5f}, gain {report.get('train_only_cv_auc_36m_observed')-base.get('auc_36m_observed'):+.5f})")
    print(f"  train-only CV AP ={report.get('train_only_cv_average_precision_36m'):.5f} "
          f"(baseline {base.get('average_precision_36m'):.5f}, gain {report.get('train_only_cv_average_precision_36m')-base.get('average_precision_36m'):+.5f})")
print("\n=== verification complete ===")
