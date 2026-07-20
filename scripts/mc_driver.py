#!/usr/bin/env python3
"""Driver riprendibile per il BLOCCO B (10 feature) + C (KAN) del multiclass
su dati reali completi, un modello per invocazione con checkpoint."""
import sys, os, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "."); sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")
import compare_models_multiclass as mc
import section_310_unified_feature_engineering as fe
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from utils import get_models

OUT = "results/multiclass_full_real.csv"
CK = "/tmp/mc_state.pkl"

def main():
    t0 = time.time()
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34.0
    if os.path.exists(CK):
        with open(CK, "rb") as f:
            state = pickle.load(f)
    else:
        state = {"done": {}}

    df_raw = pd.read_csv("train_test_network.csv")
    df_clean = mc.prepare_base_dataframe(df_raw).reset_index(drop=True)
    le = LabelEncoder().fit(df_clean[mc.TARGET_M])
    Xfe = fe.build_unified_features_ton(df_clean).reset_index(drop=True)
    num, cat = fe.UNIFIED_NUMERIC_FEATURES, fe.UNIFIED_CATEGORICAL_FEATURES
    X10 = Xfe[num + cat].copy()
    y10 = le.transform(df_clean[mc.TARGET_M])
    Xtr, Xte, ytr, yte = train_test_split(X10, y10, test_size=mc.TEST_SIZE,
                                          random_state=mc.RANDOM_STATE, stratify=y10)
    # pipeline IDENTICA a run_user_models_mc (build_preprocessor + sanitizer)
    import utils
    from sklearn.pipeline import Pipeline
    from compare_models import FeatureNameSanitizer
    models = utils.get_models(task="multiclass")
    order = ["Decision Tree", "LightGBM", "XGBoost", "Logistic Regression", "Random Forest"]
    for name in order:
        if name in state["done"] or name not in models:
            continue
        if time.time() - t0 > budget:
            print(f"CHECKPOINT: {len(state['done'])} modelli fatti"); return
        pre, _, _ = utils.build_preprocessor(Xtr)
        est = models[name]
        if hasattr(est, "n_jobs"): est.n_jobs = -1
        pipe = Pipeline([("preprocessor", pre),
                         ("sanitize", FeatureNameSanitizer()),
                         ("model", est)])
        pipe.fit(Xtr, ytr)
        res, _, _, _ = utils.evaluate_multiclass_pipeline(pipe, Xte, yte, name)
        state["done"][name] = {"macro_f1": res.get("macro_f1"),
                               "weighted_f1": res.get("weighted_f1")}
        with open(CK, "wb") as f: pickle.dump(state, f)
        print(f"ok {name}: macro_f1={state['done'][name]['macro_f1']:.4f} t={time.time()-t0:.0f}s", flush=True)

    rows = [{"model": k, **v} for k, v in state["done"].items()]
    pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).sort_values("macro_f1", ascending=False).round(4).to_string(index=False))
    print("DONE")

if __name__ == "__main__":
    main()
