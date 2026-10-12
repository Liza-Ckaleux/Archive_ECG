import os
import yaml
import pandas as pd
import joblib
import pyedflib
import numpy as np

from process_ecg import process_patient


# =========================
# CONFIG
# =========================

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =========================
# EDF LOADER (ROBUST)
# =========================

def load_edf(path):
    f = pyedflib.EdfReader(path)

    signals = [f.readSignal(i) for i in range(f.signals_in_file)]
    labels = f.getSignalLabels()

    f.close()

    def norm(x):
        return x.strip().upper().replace(" ", "").replace("_", "")

    return {norm(l): s for l, s in zip(labels, signals)}


# =========================
# CHANNEL SAFE GETTER
# =========================

def get_channel(ecg, names):
    for n in names:
        key = n.strip().upper().replace(" ", "").replace("_", "")
        if key in ecg:
            return ecg[key]
    return None


def safe(x, name):
    if x is None:
        raise ValueError(f"Missing channel: {name}")
    return np.asarray(x)


# =========================
# INFERENCE PIPELINE
# =========================

def run_inference(config):
    paths = config["inference_paths"]
    ecg_cfg = config["ecg"]

    input_dir = paths["input_dir"]
    output_dir = paths["output_dir"]

    model = joblib.load(paths["model_path"])
    scaler = joblib.load(paths["scaler_path"])

    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".edf")]
    results = []

    print(f"Found {len(files)} EDF files")

    for file in files:
        path = os.path.join(input_dir, file)

        try:
            ecg = load_edf(path)

            # =========================
            # MAP CHANNELS
            # =========================

            V1 = safe(get_channel(ecg, ["V1", "ECGV1", "ECG_V1"]), "V1")
            V2 = safe(get_channel(ecg, ["V2", "ECGV2", "ECG_V2"]), "V2")
            V3 = safe(get_channel(ecg, ["V3", "ECGV3", "ECG_V3"]), "V3")
            V4 = safe(get_channel(ecg, ["V4", "ECGV4", "ECG_V4"]), "V4")
            V5 = safe(get_channel(ecg, ["V5", "ECGV5", "ECG_V5"]), "V5")
            V6 = safe(get_channel(ecg, ["V6", "ECGV6", "ECG_V6"]), "V6")

            I  = safe(get_channel(ecg, ["I", "ECGI", "ECG_I"]), "I")
            II = safe(get_channel(ecg, ["II", "ECGII", "ECG_II"]), "II")

            ecg_data = {
                "V1": V1, "V2": V2, "V3": V3, "V4": V4,
                "V5": V5, "V6": V6, "I": I, "II": II
            }

            # =========================
            # FEATURE EXTRACTION
            # =========================

            features = process_patient(
                ecg_data,
                sfreq=ecg_cfg["sfreq"],
                num_periods=ecg_cfg["num_periods"],
                num_regions=ecg_cfg["num_regions"]
            )

            X = pd.DataFrame([features])
            X_scaled = scaler.transform(X)

            proba = model.predict_proba(X_scaled)[0, 1]
            pred = int(proba > 0.5)

            results.append({
                "file": file,
                "probability": float(proba),
                "prediction": pred
            })

            print(f"[OK] {file} -> {proba:.4f}")

        except Exception as e:
            print(f"[SKIP] {file}: {e}")

            results.append({
                "file": file,
                "probability": None,
                "prediction": None
            })

    # =========================
    # SAVE RESULTS
    # =========================

    out_path = os.path.join(output_dir, "inference_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)

    print(f"\nSaved results to: {out_path}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    run_inference(config)