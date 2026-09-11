import sys
import os

# --- AGGRESSIVE MONKEY PATCH FOR UMAP/SKLEARN COMPATIBILITY ---
import warnings
warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")

try:
    import sklearn.utils.validation
    import sklearn.utils
    import inspect
    if not hasattr(sklearn.utils.validation.check_array, '_patched'):
        _original_check_array = sklearn.utils.validation.check_array
        _sig = inspect.signature(_original_check_array)
        _supports_ensure = 'ensure_all_finite' in _sig.parameters
        _supports_force = 'force_all_finite' in _sig.parameters
        def _patched_check_array(*args, **kwargs):
            if 'force_all_finite' in kwargs and _supports_ensure and not _supports_force:
                kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
            if 'ensure_all_finite' in kwargs and _supports_force and not _supports_ensure:
                kwargs['force_all_finite'] = kwargs.pop('ensure_all_finite')
            return _original_check_array(*args, **kwargs)
        _patched_check_array._patched = True
        sklearn.utils.validation.check_array = _patched_check_array
        if hasattr(sklearn.utils, 'check_array'):
            sklearn.utils.check_array = _patched_check_array
        print("🔧 Applied adaptive monkey-patch for UMAP/Sklearn compatibility.")
    else:
        _patched_check_array = sklearn.utils.validation.check_array

except ImportError:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, LSTM, Lambda, Multiply
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.initializers import he_normal
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import joblib
import json
import math
import random
import scipy.stats as stats
import pickle

# --- DEPENDENCY CHECK ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP library not found. Run '!pip install shap' if in notebook.")
    SHAP_AVAILABLE = False

try:
    import umap
    if hasattr(umap, 'utils') and hasattr(umap.utils, 'check_array'):
        umap.utils.check_array = sklearn.utils.validation.check_array
    if hasattr(umap, 'check_array'):
        umap.check_array = sklearn.utils.validation.check_array
    for name, mod in list(sys.modules.items()):
        if 'umap' in name and hasattr(mod, 'check_array'):
            mod.check_array = sklearn.utils.validation.check_array
            
    UMAP_AVAILABLE = True
    print("✅ UMAP library found and patched.")
except ImportError:
    print("⚠️ UMAP library NOT found. Visualizations will be skipped.")
    UMAP_AVAILABLE = False

# --- REPRODUCIBILITY SETUP ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"✅ Random Seeds set to {SEED} for Reproducibility")

# --- HYPERPARAMETERS ---
OPTIMIZER_NAME = 'sgd'
LEARNING_RATE = 0.01
KERNEL_SIZE = 5
POOL_SIZE = 3
K_VALUE = 0.5
BATCH_SIZE = 64
EPOCHS = 50
ACTIVATION_NAME = 'relu'

# --- DATA PATH CONFIGURATION ---
DATA_PATH_TRAIN_X = '/kaggle/input/datasets/engrsiyab/phd-dataset/emb_train_protbert.npy'
DATA_PATH_TEST_X = '/kaggle/input/datasets/engrsiyab/phd-dataset/emb_test_protbert.npy'
DATA_PATH_TRAIN_Y = '/kaggle/input/datasets/engrsiyab/phd-dataset/train.xlsx'
DATA_PATH_TEST_Y = '/kaggle/input/datasets/engrsiyab/phd-dataset/test.xlsx'

IN_KAGGLE = os.path.exists('/kaggle')

if IN_KAGGLE:
    print("🌍 Running on Kaggle Environment")

DASHBOARD_DATA = {
    "metrics": {},
    "visualizations": {}
}

# --- HELPER FUNCTIONS ---

def load_data():
    print("--- Loading Data ---")
    try:
        df_train = pd.read_excel(DATA_PATH_TRAIN_Y, na_filter=False)
        df_test = pd.read_excel(DATA_PATH_TEST_Y, na_filter=False)
        y_train = np.array(df_train['label'])
        y_test = np.array(df_test['label'])
        
        X_train = np.load(DATA_PATH_TRAIN_X, allow_pickle=True)
        X_test = np.load(DATA_PATH_TEST_X, allow_pickle=True)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        
        scaler = MinMaxScaler()
        if len(X_train.shape) == 3:
            N, L, C = X_train.shape
            X_train = X_train.reshape(N, L*C)
            X_test = X_test.reshape(X_test.shape[0], -1)
            
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"✅ Data Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None

def calculate_metrics(y_true, y_pred_prob, model_name):
    y_pred_class = np.argmax(y_pred_prob, axis=1) if y_pred_prob.shape[1] > 1 else (y_pred_prob > 0.5).astype(int)
    prob_pos = y_pred_prob[:, 1] if y_pred_prob.shape[1] == 2 else y_pred_prob
        
    acc = accuracy_score(y_true, y_pred_class)
    try:
        auc = roc_auc_score(y_true, prob_pos)
    except:
        auc = 0.5
    f1 = f1_score(y_true, y_pred_class, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_class)
    prec = precision_score(y_true, y_pred_class, zero_division=0)
    rec = recall_score(y_true, y_pred_class, zero_division=0)
    
    print(f"[{model_name}] Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    return {
        "acc": float(acc), "auc": float(auc), "f1": float(f1),
        "mcc": float(mcc), "prec": float(prec), "rec": float(rec)
    }

def glu_std(inputs, units):
    x = Dense(units, activation='sigmoid')(inputs)
    x1 = Dense(units)(inputs)
    return Multiply()([x, x1])

def swiglu(inputs, units):
    x = Dense(units)(inputs)
    x = Lambda(lambda z: z * tf.sigmoid(float(K_VALUE) * z))(x)
    x1 = Dense(units)(inputs)
    return Multiply()([x, x1])

def reglu(inputs, units):
    x = Dense(units, activation='relu')(inputs)
    x1 = Dense(units)(inputs)
    return Multiply()([x, x1])

def geglu(inputs, units):
    x = Dense(units, activation='gelu')(inputs)
    x1 = Dense(units)(inputs)
    return Multiply()([x, x1])

def hsigglu(inputs, units):
    x = Dense(units, activation='hard_sigmoid')(inputs)
    x1 = Dense(units)(inputs)
    return Multiply()([x, x1])

def tanhglu(inputs, units):
    x = Dense(units, activation='tanh')(inputs)
    x1 = Dense(units)(inputs)
    return Multiply()([x, x1])

def step_decay(epoch):
    initial_lrate = LEARNING_RATE
    drop = 0.6
    epochs_drop = 3.0
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

def get_optimizer():
    if OPTIMIZER_NAME.lower() == 'adam':
        return Adam(learning_rate=LEARNING_RATE)
    else:
        return SGD(learning_rate=LEARNING_RATE, momentum=0.5, nesterov=False)

# --- MODEL BUILDERS ---

def build_mlp(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    features = Dense(128, activation='relu', name='features')(x)
    outputs = Dense(2, activation='softmax')(features)
    model = Model(inputs, outputs, name="MLP")
    model.compile(optimizer=get_optimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_dim):
    inputs = Input(shape=(1, input_dim)) 
    x = LSTM(128, return_sequences=False)(inputs)
    x = Dropout(0.3)(x)
    features = Dense(64, activation='relu', name='features')(x)
    outputs = Dense(2, activation='softmax')(features)
    model = Model(inputs, outputs, name="LSTM")
    model.compile(optimizer=get_optimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(input_dim):
    inputs = Input(shape=(input_dim, 1))
    x = Conv1D(128, KERNEL_SIZE, strides=2, padding='same', kernel_initializer=he_normal(seed=42))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(POOL_SIZE, padding='same')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    features = Dense(64, activation='relu', name='features')(x)
    outputs = Dense(2, activation='softmax')(features)
    model = Model(inputs, outputs, name="CNN")
    model.compile(optimizer=get_optimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_glu(input_dim, glu_func, glu_name="GLU"):
    inputs = Input(shape=(input_dim, 1))
    x = Conv1D(128, KERNEL_SIZE, strides=2, padding='same', kernel_initializer=he_normal(seed=42))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(POOL_SIZE, padding='same')(x)
    x = Dropout(0.15)(x)
    
    x = glu_func(x, 64)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.15)(x)
    
    features = Dense(32, activation='relu', name='features')(x)
    x = Dropout(0.15)(features)
    x = BatchNormalization()(x)
    
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name=f"CNN-{glu_name}")
    model.compile(optimizer=get_optimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def perform_data_validation(X_train, y_train, X_test, y_test):
    print("   🧪 Running Data Validation Tests...")
    validation_report = {}
    train_balance = np.mean(y_train)
    test_balance = np.mean(y_test)
    validation_report["Class Balance"] = {
        "Train Positive Rate": f"{train_balance:.2%}",
        "Test Positive Rate": f"{test_balance:.2%}",
        "Status": "Pass" if abs(train_balance - test_balance) < 0.1 else "Warning: Distribution Shift"
    }
    validation_report["Feature Scaling"] = {
        "Train Max": float(np.max(X_train)),
        "Train Min": float(np.min(X_train)),
        "Test Max": float(np.max(X_test)),
        "Status": "Pass" if np.max(X_train) <= 1.001 and np.min(X_train) >= -0.001 else "Fail"
    }
    validation_report["Dimensions"] = {
        "Train Features": X_train.shape[1],
        "Test Features": X_test.shape[1],
        "Status": "Pass" if X_train.shape[1] == X_test.shape[1] else "Fail"
    }
    return validation_report

def get_model_builder(model_name, glu_variants):
    if model_name == "MLP":
        return lambda dim: build_mlp(dim)
    elif model_name == "LSTM":
        return lambda dim: build_lstm(dim)
    elif model_name == "CNN":
        return lambda dim: build_cnn(dim)
    elif model_name.startswith("CNN-"):
        glu_type = model_name.replace("CNN-", "")
        if glu_type in glu_variants:
            return lambda dim: build_cnn_glu(dim, glu_variants[glu_type], glu_type)
    return None

def perform_kfold_cv(model_name, builder_func, X, y, input_dim, k=5):
    print(f"\n   🔄 Performing {k}-Fold Cross-Validation for Best Model: {model_name}...")
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    
    cv_metrics = {"acc": [], "prec": [], "rec": [], "f1": [], "mcc": [], "auc": []}
    cv_roc_data = []
    
    fold = 1
    for train_ix, val_ix in kfold.split(X, y):
        # FIX: Reset graph session before each fold to avoid gradient registry leaks
        tf.keras.backend.clear_session()
        
        X_t, y_t = X[train_ix], y[train_ix]
        X_v, y_v = X[val_ix], y[val_ix]
        
        if "CNN" in model_name:
            X_t = X_t.reshape(-1, input_dim, 1)
            X_v = X_v.reshape(-1, input_dim, 1)
        elif "LSTM" in model_name:
            X_t = X_t.reshape(-1, 1, input_dim)
            X_v = X_v.reshape(-1, 1, input_dim)
            
        model = builder_func(input_dim)
        model.fit(X_t, y_t, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[EarlyStopping(patience=5)])
        
        y_probs = model.predict(X_v, verbose=0)
        fold_metrics = calculate_metrics(y_v, y_probs, f"CV-Fold-{fold}")
        
        for key in cv_metrics:
            cv_metrics[key].append(fold_metrics[key])
            
        pos_probs = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs.flatten()
        fpr, tpr, _ = roc_curve(y_v, pos_probs)
        roc_auc = roc_auc_score(y_v, pos_probs)
        
        cv_roc_data.append({
            "fold": fold,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": roc_auc
        })

        print(f"      Fold {fold}/{k}: Acc = {fold_metrics['acc']:.4f}, AUC = {roc_auc:.4f}")
        fold += 1
        
    return cv_metrics, cv_roc_data

def compute_visualizations(features, labels, name):
    print(f"   🎨 Computing t-SNE & UMAP for {name} ({features.shape})...")
    if features.shape[0] > 2000:
        indices = np.random.choice(features.shape[0], 2000, replace=False)
        feat_subset = features[indices]
        lbl_subset = labels[indices]
    else:
        feat_subset = features
        lbl_subset = labels
        
    viz_data = {"labels": lbl_subset.tolist()}
    
    try:
        tsne = TSNE(n_components=2, perplexity=30)
        tsne_res = tsne.fit_transform(feat_subset)
        viz_data["tsne"] = {"x": tsne_res[:, 0].tolist(), "y": tsne_res[:, 1].tolist()}
    except Exception as e:
        viz_data["tsne"] = {"x": [], "y": []}

    if UMAP_AVAILABLE:
        try:
            import umap.utils
            if hasattr(umap.utils, 'check_array') and umap.utils.check_array != sklearn.utils.validation.check_array:
                 umap.utils.check_array = sklearn.utils.validation.check_array

            reducer = umap.UMAP(n_components=2)
            umap_res = reducer.fit_transform(feat_subset)
            viz_data["umap"] = {"x": umap_res[:, 0].tolist(), "y": umap_res[:, 1].tolist()}
        except Exception as e:
             viz_data["umap"] = {"x": [], "y": []}
    else:
        viz_data["umap"] = {"x": [], "y": []}
        
    DASHBOARD_DATA["visualizations"][name] = viz_data

def compute_shap_importance(model, X_train, X_test, model_name):
    if not SHAP_AVAILABLE: return

    print(f"   ✨ Computing SHAP Feature Importance for {model_name}...")
    import matplotlib.pyplot as plt
    import io
    import base64

    try:
        n_bg = 50
        n_eval = 20
        
        bg_idx = np.random.choice(X_train.shape[0], min(n_bg, X_train.shape[0]), replace=False)
        eval_tr_idx = np.random.choice(X_train.shape[0], min(n_eval, X_train.shape[0]), replace=False)
        eval_te_idx = np.random.choice(X_test.shape[0], min(n_eval, X_test.shape[0]), replace=False)
        
        bg_data = X_train[bg_idx]
        eval_train = X_train[eval_tr_idx]
        eval_test = X_test[eval_te_idx]
        
        used_explainer = "Kernel"

        try:
            explainer = shap.DeepExplainer(model, bg_data)
            shap_values_tr = explainer.shap_values(eval_train)
            shap_values_te = explainer.shap_values(eval_test)
            used_explainer = "Deep"
        except Exception:
            is_3d = len(X_train.shape) == 3
            original_shape = X_train.shape[1:]
            
            if is_3d:
                bg_data = bg_data.reshape(bg_data.shape[0], -1)
                eval_train = eval_train.reshape(eval_train.shape[0], -1)
                eval_test = eval_test.reshape(eval_test.shape[0], -1)
            
            def predict_wrapper(X_batch):
                if is_3d:
                    X_reshaped = X_batch.reshape((X_batch.shape[0],) + original_shape)
                    return model.predict(X_reshaped, verbose=0)
                return model.predict(X_batch, verbose=0)

            bg_summary = shap.kmeans(bg_data, 10) if bg_data.shape[0] > 10 else bg_data
            explainer = shap.KernelExplainer(predict_wrapper, bg_summary)
            shap_values_tr = explainer.shap_values(eval_train, nsamples='auto', silent=True)
            shap_values_te = explainer.shap_values(eval_test, nsamples='auto', silent=True)

        def process_vals(sv):
            if isinstance(sv, list): 
                v = sv[1] if len(sv) == 2 else sv[0]
            elif len(sv.shape) == 3 and sv.shape[-1] == 2: 
                v = sv[:, :, 1]
            else: 
                v = sv
            
            v = np.array(v)
            if len(v.shape) > 2: 
                v = v.reshape(v.shape[0], -1)
            return v

        raw_sv_te = process_vals(shap_values_te)
        imp_te = np.abs(raw_sv_te).mean(axis=0)
        
        raw_sv_tr = process_vals(shap_values_tr)
        imp_tr = np.abs(raw_sv_tr).mean(axis=0)
        
        top_indices = np.argsort(imp_te)[::-1][:20]
        
        try:
            plt.figure(figsize=(10, 6))
            X_disp = eval_test
            if len(X_disp.shape) > 2: X_disp = X_disp.reshape(X_disp.shape[0], -1)
            
            shap.summary_plot(raw_sv_te, X_disp, show=False, max_display=20, plot_type="dot")
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            if "shap_plots" not in DASHBOARD_DATA: DASHBOARD_DATA["shap_plots"] = {}
            DASHBOARD_DATA["shap_plots"][model_name] = f"data:image/png;base64,{img_str}"
            
        except Exception as plot_e:
            print(f"      ⚠️ Failed to generate SHAP beeswarm plot: {plot_e}")

        if "shap_importance" not in DASHBOARD_DATA: DASHBOARD_DATA["shap_importance"] = {}
        DASHBOARD_DATA["shap_importance"][model_name] = {
            "indices": top_indices.tolist(),
            "train": imp_tr[top_indices].tolist(),
            "test": imp_te[top_indices].tolist()
        }
        print(f"   ✅ SHAP Computed for {model_name} using {used_explainer}Explainer")

    except Exception as e:
        print(f"   ⚠️ SHAP Computation Failed: {e}")

# --- MAIN EXECUTION ---

def run_experiment():
    X_train, y_train, X_test, y_test = load_data()
    if X_train is None: return

    results_table = []
    
    def log_performance(model, X_tr, y_tr, X_te, y_te, name):
        y_p_tr = model.predict(X_tr, verbose=0)
        y_p_te = model.predict(X_te, verbose=0)
             
        m_tr = calculate_metrics(y_tr, y_p_tr, f"{name} (Train)")
        m_tr["Model"] = name
        m_tr["Dataset"] = "Train"
        results_table.append(m_tr)
        
        m_te = calculate_metrics(y_te, y_p_te, name)
        DASHBOARD_DATA["metrics"][name] = {
            "train": m_tr.copy(),
            "test": m_te.copy()
        }
        m_te["Model"] = name
        m_te["Dataset"] = "Test"
        results_table.append(m_te)

        y_pred_class_tr = np.argmax(y_p_tr, axis=1) if y_p_tr.shape[1] > 1 else (y_p_tr > 0.5).astype(int)
        y_pred_class_te = np.argmax(y_p_te, axis=1) if y_p_te.shape[1] > 1 else (y_p_te > 0.5).astype(int)
        
        if "confusion_matrices" not in DASHBOARD_DATA: DASHBOARD_DATA["confusion_matrices"] = {}
        DASHBOARD_DATA["confusion_matrices"][name] = {
            "train": confusion_matrix(y_tr, y_pred_class_tr).tolist(),
            "test": confusion_matrix(y_te, y_pred_class_te).tolist()
        }
        
        prob_pos_tr = y_p_tr[:, 1] if y_p_tr.shape[1] > 1 else y_p_tr
        prob_pos_te = y_p_te[:, 1] if y_p_te.shape[1] > 1 else y_p_te.flatten()
        
        if "model_predictions" not in DASHBOARD_DATA: DASHBOARD_DATA["model_predictions"] = {}
        DASHBOARD_DATA["model_predictions"][name] = {
            "pos": prob_pos_te[y_te == 1].tolist(),
            "neg": prob_pos_te[y_te == 0].tolist()
        }
        
        fpr_tr, tpr_tr, _ = roc_curve(y_tr, prob_pos_tr)
        fpr_te, tpr_te, _ = roc_curve(y_te, prob_pos_te)
        
        if "roc_curves" not in DASHBOARD_DATA: DASHBOARD_DATA["roc_curves"] = {}
        DASHBOARD_DATA["roc_curves"][name] = {
            "train": {"fpr": fpr_tr.tolist(), "tpr": tpr_tr.tolist(), "auc": m_tr["auc"]},
            "test": {"fpr": fpr_te.tolist(), "tpr": tpr_te.tolist(), "auc": m_te["auc"]}
        }
        
        prec_tr, rec_tr, _ = precision_recall_curve(y_tr, prob_pos_tr)
        prec_te, rec_te, _ = precision_recall_curve(y_te, prob_pos_te)
        
        if "pr_curves" not in DASHBOARD_DATA: DASHBOARD_DATA["pr_curves"] = {}
        DASHBOARD_DATA["pr_curves"][name] = {
            "train": {"prec": prec_tr.tolist(), "rec": rec_tr.tolist()},
            "test": {"prec": prec_te.tolist(), "rec": rec_te.tolist()}
        }
        
        return y_p_te, m_te

    input_dim = X_train.shape[1]
    
    best_model_obj = None
    best_model_name = ""
    max_test_acc = -1.0
    
    def check_best(model, name, metrics):
        nonlocal best_model_obj, best_model_name, max_test_acc
        if metrics["acc"] > max_test_acc:
            max_test_acc = metrics["acc"]
            best_model_obj = model
            best_model_name = name
            print(f"   🏆 New Best Model: {name} (Acc: {max_test_acc:.4f})")

    compute_visualizations(X_test, y_test, "Raw Embeddings")

    # --- 1. MLP ---
    print("\nTraining MLP...")
    tf.keras.backend.clear_session()
    mlp = build_mlp(input_dim)
    mlp.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[EarlyStopping(patience=5)])
    
    y_prob, m_te = log_performance(mlp, X_train, y_train, X_test, y_test, "MLP")
    check_best(mlp, "MLP", m_te)
    
    feat_model = Model(inputs=mlp.input, outputs=mlp.get_layer('features').output)
    features = feat_model.predict(X_test, verbose=0)
    compute_visualizations(features, y_test, "MLP")
    
    # --- 2. LSTM ---
    print("\nTraining LSTM...")
    tf.keras.backend.clear_session()
    X_train_seq = X_train.reshape(-1, 1, input_dim)
    X_test_seq = X_test.reshape(-1, 1, input_dim)
    
    lstm = build_lstm(input_dim)
    lstm.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[EarlyStopping(patience=5)])
    
    y_prob, m_te = log_performance(lstm, X_train_seq, y_train, X_test_seq, y_test, "LSTM")
    check_best(lstm, "LSTM", m_te)
    
    feat_model = Model(inputs=lstm.input, outputs=lstm.get_layer('features').output)
    features = feat_model.predict(X_test_seq, verbose=0)
    compute_visualizations(features, y_test, "LSTM")

    # --- 3. CNN ---
    print("\nTraining CNN...")
    tf.keras.backend.clear_session()
    X_train_cnn = X_train.reshape(-1, input_dim, 1)
    X_test_cnn = X_test.reshape(-1, input_dim, 1)
    
    cnn = build_cnn(input_dim)
    cnn.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[EarlyStopping(patience=5)])
    
    y_prob, m_te = log_performance(cnn, X_train_cnn, y_train, X_test_cnn, y_test, "CNN")
    check_best(cnn, "CNN", m_te)
    
    feat_model = Model(inputs=cnn.input, outputs=cnn.get_layer('features').output)
    features = feat_model.predict(X_test_cnn, verbose=0)
    compute_visualizations(features, y_test, "CNN")

    # --- 4. CNN-GLU Variants ---
    glu_variants = {
        "GLU": glu_std,
        "SwiGLU": swiglu,
        "ReGLU": reglu,
        "GeGLU": geglu,
        "HsigGLU": hsigglu,
        "TanhGLU": tanhglu
    }

    for glu_name, glu_func in glu_variants.items():
        model_name = f"CNN-{glu_name}"
        print(f"\nTraining {model_name}...")
        
        try:
            tf.keras.backend.clear_session()
            cnn_glu = build_cnn_glu(input_dim, glu_func, glu_name)
            lrate = LearningRateScheduler(step_decay)
            cnn_glu.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[lrate, EarlyStopping(patience=10)])
            
            y_prob, m_te = log_performance(cnn_glu, X_train_cnn, y_train, X_test_cnn, y_test, model_name)
            check_best(cnn_glu, model_name, m_te)
            
            feat_model = Model(inputs=cnn_glu.input, outputs=cnn_glu.get_layer('features').output)
            features = feat_model.predict(X_test_cnn, verbose=0)
            compute_visualizations(features, y_test, model_name)

        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}")

    # --- DATA VALIDATION ---
    val_report = perform_data_validation(X_train, y_train, X_test, y_test)
    DASHBOARD_DATA["validation_report"] = val_report

    # --- CROSS VALIDATION ---
    if best_model_name:
        builder = get_model_builder(best_model_name, glu_variants)
        if builder:
            cv_metrics_dict, cv_roc_data = perform_kfold_cv(best_model_name, builder, X_train, y_train, input_dim, k=5)
            
            cv_stats = {}
            for metric, values in cv_metrics_dict.items():
                cv_stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values
                }

            DASHBOARD_DATA["cv_results"] = {
                "model": best_model_name,
                "metrics": cv_stats,
                "roc_data": cv_roc_data
            }
        
        # Probability Distribution
        try:
            if "CNN" in best_model_name: X_in = X_test_cnn
            elif "LSTM" in best_model_name: X_in = X_test_seq
            else: X_in = X_test
             
            y_probs = best_model_obj.predict(X_in, verbose=0)
            pos_probs = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs.flatten()

            DASHBOARD_DATA["prob_dist"] = {
                "pos": pos_probs[y_test == 1].tolist(),
                "neg": pos_probs[y_test == 0].tolist()
            }
        except Exception as e:
            print(f"⚠️ Could not compute probability distribution: {e}")

    # --- SHAP FOR BEST MODEL (RUN AT VERY END TO AVOID TF GRAPH POLLUTION) ---
    if best_model_obj is not None:
        print(f"\n✨ Calculating SHAP for Best Model: {best_model_name}")
        X_tr_shap = X_train
        X_te_shap = X_test
        
        if "CNN" in best_model_name:
            X_tr_shap = X_train_cnn
            X_te_shap = X_test_cnn
        elif "LSTM" in best_model_name:
            X_tr_shap = X_train_seq
            X_te_shap = X_test_seq
            
        compute_shap_importance(best_model_obj, X_tr_shap, X_te_shap, best_model_name)

    # --- SAVE RESULTS ---
    print("\n💾 Saving Results...")
    base_dir = '/kaggle/working' if IN_KAGGLE else os.path.dirname(os.path.abspath(__file__))
    
    if best_model_obj is not None and best_model_name:
        safe_name = best_model_name.replace(" ", "_").replace("/", "_")
        full_model_path = os.path.join(base_dir, f"{safe_name}_best_model.keras")
        weights_path = os.path.join(base_dir, f"{safe_name}.weights.h5")
        best_model_obj.save(full_model_path)
        best_model_obj.save_weights(weights_path)
        
        preprocessing_data = {
            "scaler": "MinMaxScaler",
            "input_dim": int(input_dim),
            "train_shape": tuple(X_train.shape),
            "test_shape": tuple(X_test.shape)
        }
        preprocessing_path = os.path.join(base_dir, f"{safe_name}_preprocessing.pkl")
        with open(preprocessing_path, "wb") as f:
            pickle.dump(preprocessing_data, f)
            
        training_info = {
            "best_model": best_model_name,
            "metrics": DASHBOARD_DATA.get("metrics", {}),
            "cv_results": DASHBOARD_DATA.get("cv_results", {}),
            "validation_report": DASHBOARD_DATA.get("validation_report", {})
        }
        history_path = os.path.join(base_dir, f"{safe_name}_training_info.json")
        with open(history_path, "w") as f:
            json.dump(training_info, f)
    
    try:
        df_results = pd.DataFrame(results_table)
        cols = ["Model", "Dataset"] + [c for c in df_results.columns if c not in ["Model", "Dataset"]]
        df_results = df_results[cols]
        excel_path = os.path.join(base_dir, 'prott5_results.xlsx')
        df_results.to_excel(excel_path, index=False)
        print(f"✅ Tabular Results saved to '{excel_path}'")
    except Exception as e:
        print(f"❌ Error saving tabular results: {e}")

    # Save Dashboard HTML
    json_str = json.dumps(DASHBOARD_DATA)
    final_html = html_template.replace('const dashboardData = {};', f'const dashboardData = {json_str};')
    
    html_path = os.path.join(base_dir, 'prott5_dashboard.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"✅ Self-contained Dashboard saved to '{html_path}'")
    print(f"✅ Experiment Complete.")

# Define HTML Template placeholder for complete generation
html_template = r"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Model Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdn.sheetjs.com/xlsx-latest/package/dist/xlsx.full.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; text-align: center; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h2 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
        .plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .metric-table { width: 100%; border-collapse: collapse; }
        .metric-table th, .metric-table td { padding: 12px; border: 1px solid #ddd; text-align: center; }
        .metric-table th { background-color: #f8f9fa; }
        .model-section { border-left: 5px solid #3498db; padding-left: 15px; margin-top: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Multi-Model Performance Comparison</h1>
            <p>MLP vs LSTM vs CNN vs CNN-GLU Variants</p>
        </div>
        <div class="card">
            <h2>🧪 Data Validation & Integrity Checks</h2>
            <div style="overflow-x: auto;">
                <table class="metric-table" id="validation_table">
                    <thead>
                        <tr><th>Test</th><th>Details</th><th>Status</th></tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
        <div class="card">
            <h2>🏆 Main Model Comparison (with Best GLU)</h2>
            <div class="plot-grid">
                <div id="comparison_plot_train" style="height: 400px;"></div>
                <div id="comparison_plot_test" style="height: 400px;"></div>
            </div>
            <div class="plot-grid" style="margin-top: 20px;">
                <div id="radar_plot_train" style="height: 400px;"></div>
                <div id="radar_plot_test" style="height: 400px;"></div>
            </div>
            <div class="plot-grid" style="margin-top: 20px;">
                <div id="roc_comparison_plot_train" style="height: 500px;"></div>
                <div id="roc_comparison_plot_test" style="height: 500px;"></div>
            </div>
            <div class="plot-grid" style="margin-top: 20px;">
                <div id="pr_comparison_plot_train" style="height: 500px;"></div>
                <div id="pr_comparison_plot_test" style="height: 500px;"></div>
            </div>
        </div>
        <div class="card">
            <h2>🔄 Cross-Validation Analysis (Best Model)</h2>
            <div class="plot-grid">
                <div id="cv_scores_plot" style="height: 400px;"></div>
                <div id="cv_comparison_plot" style="height: 400px;"></div>
            </div>
            <h3 style="margin-top: 30px;">Detailed CV Metrics</h3>
            <div style="overflow-x: auto;">
                <table class="metric-table" id="cv_table">
                    <thead>
                        <tr><th>Fold</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>MCC</th><th>AUC</th></tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
        <div class="card">
            <h2>📉 Prediction Probability Distribution</h2>
            <div id="prob_dist_plot" style="height: 500px;"></div>
        </div>
        <div class="card">
            <h2>🔬 GLU Variants Comparison</h2>
            <div class="plot-grid">
                <div id="glu_comparison_plot_train" style="height: 400px;"></div>
                <div id="glu_comparison_plot_test" style="height: 400px;"></div>
            </div>
            <div class="plot-grid" style="margin-top: 20px;">
                <div id="glu_radar_plot_train" style="height: 400px;"></div>
                <div id="glu_radar_plot_test" style="height: 400px;"></div>
            </div>
             <div class="plot-grid" style="margin-top: 20px;">
                <div id="glu_roc_comparison_plot_train" style="height: 500px;"></div>
                <div id="glu_roc_comparison_plot_test" style="height: 500px;"></div>
            </div>
             <div class="plot-grid" style="margin-top: 20px;">
                <div id="glu_pr_comparison_plot_train" style="height: 500px;"></div>
                <div id="glu_pr_comparison_plot_test" style="height: 500px;"></div>
            </div>
        </div>
        <div class="card">
             <h2>📊 Detailed Metrics</h2>
             <div style="overflow-x: auto; margin-top: 20px;">
                <table class="metric-table" id="comparison_table">
                    <thead>
                        <tr><th>Model</th><th>Dataset</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>MCC</th><th>AUC</th></tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
        <div class="card">
            <h2>🧩 Confusion Matrices</h2>
            <div id="cm_container" class="plot-grid"></div>
        </div>
        <div class="card" id="shap_section" style="display: none;">
            <h2>✨ Feature Importance (SHAP - Best Model)</h2>
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1; min-width: 450px;">
                    <h3>Global Feature Impact (Beeswarm)</h3>
                    <div id="shap_image_container" style="text-align: center;"></div>
                </div>
                <div style="flex: 1; min-width: 450px;">
                     <h3>Feature Importance Magnitude</h3>
                     <div id="shap_plot" style="height: 500px;"></div>
                </div>
            </div>
        </div>
        <div class="card" id="density_map_section">
            <h2>🌊 Predicted Probability Density Maps</h2>
            <div id="density_grid" class="plot-grid"></div>
        </div>
        <div class="card" id="cv_roc_section" style="display: none;">
             <h2>🔄 Cross-Validation vs. Independent Test ROC</h2>
             <div id="cv_roc_plot" style="height: 600px;"></div>
        </div>
        <div class="card">
            <h2>🎨 Feature Space Visualization (t-SNE & UMAP)</h2>
            <div id="viz_container"></div>
        </div>
    </div>
    <script>
        const dashboardData = {}; 

        function exportToExcel(divId, filename) {
            const div = document.getElementById(divId);
            if (!div || !div.data) return;
            const data = div.data;
            let ws_data = [];
            const type = data[0].type;
            
            if (type === 'bar' || type === 'scatterpolar' || type === 'scatter') {
                ws_data.push(["Series", "X-Axis", "Y-Axis"]);
                data.forEach(trace => {
                    const x = trace.x || trace.theta; 
                    const y = trace.y || trace.r;    
                    const name = trace.name || "Data";
                    if (x && y) {
                        for (let i = 0; i < x.length; i++) {
                            ws_data.push([name, x[i], y[i]]);
                        }
                    }
                });
            } else if (type === 'heatmap') {
                const z = data[0].z;
                const x = data[0].x || Array.from({length: z[0].length}, (_, i) => i);
                const y = data[0].y || Array.from({length: z.length}, (_, i) => i);
                ws_data.push(["", ...x]); 
                for (let i = 0; i < z.length; i++) {
                    ws_data.push([y[i], ...z[i]]);
                }
            }
            const wb = XLSX.utils.book_new();
            const ws = XLSX.utils.aoa_to_sheet(ws_data);
            XLSX.utils.book_append_sheet(wb, ws, "Data");
            XLSX.writeFile(wb, `${filename}.xlsx`);
        }

        function getDownloadConfig(filename) {
            return {
                responsive: true,
                displaylogo: false,
                toImageButtonOptions: { format: 'png', filename: filename, height: 1200, width: 1600, scale: 4 },
                modeBarButtonsToAdd: [
                    {
                        name: 'Download PDF',
                        icon: Plotly.Icons.camera, 
                        click: function(gd) {
                            Plotly.toImage(gd, {format: 'png', height: 1200, width: 1600, scale: 4})
                                .then(function(dataUrl) {
                                    const { jsPDF } = window.jspdf;
                                    const pdf = new jsPDF({ orientation: 'landscape', unit: 'px', format: [1600, 1200] });
                                    pdf.addImage(dataUrl, 'PNG', 0, 0, 1600, 1200);
                                    pdf.save(`${filename}.pdf`);
                                });
                        }
                    },
                    {
                        name: 'Download Excel Data',
                        icon: { width: 512, height: 512, path: "M64 0C28.7 0 0 28.7 0 64V448c0 35.3 28.7 64 64 64H384c35.3 0 64-28.7 64-64V160H320c-17.7 0-32-14.3-32-32V0H64zM256 0V128H384L256 0zM112 256H208v48H112V256zm144 0H336v48H256V256zM112 352H208v48H112V352zm144 0H336v48H256V352z" },
                        click: function(gd) { exportToExcel(gd.id, filename); }
                    }
                ]
            };
        }

        function updateDashboard(data) {
            if (data.validation_report) {
                const valTable = document.querySelector('#validation_table tbody');
                valTable.innerHTML = '';
                for (const [testName, result] of Object.entries(data.validation_report)) {
                    const row = document.createElement('tr');
                    let details = '';
                    for (const [k, v] of Object.entries(result)) {
                        if (k !== 'Status') details += `<b>${k}:</b> ${v} <br>`;
                    }
                    const statusColor = result.Status.includes('Pass') ? 'green' : 'red';
                    row.innerHTML = `<td>${testName}</td><td style="text-align: left;">${details}</td><td style="color:${statusColor}; font-weight:bold;">${result.Status}</td>`;
                    valTable.appendChild(row);
                }
            }

            if (data.cv_results && data.cv_results.metrics) {
                const cv = data.cv_results;
                const metrics = cv.metrics;
                const metricKeys = ['acc', 'prec', 'rec', 'f1', 'mcc', 'auc'];
                const metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUC'];
                const metricColors = ['#3498db', '#9b59b6', '#2ecc71', '#e67e22', '#f1c40f', '#e74c3c'];
                const traces = [];
                const folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'];
                
                metricKeys.forEach((key, i) => {
                    traces.push({ x: folds, y: metrics[key].values, type: 'bar', name: metricLabels[i], marker: { color: metricColors[i] } });
                });

                Plotly.newPlot('cv_scores_plot', traces, { title: `5-Fold CV Metrics Breakdown (${cv.model})`, barmode: 'group', yaxis: { range: [0, 1.1], title: 'Score' } });

                const bestModelMetrics = data.metrics[cv.model];
                const testAcc = bestModelMetrics ? bestModelMetrics.test.acc : 0;
                const cvMeanAcc = metrics.acc.mean;

                const traceComp = {
                    x: ['CV Mean Accuracy', 'Test Set Accuracy'],
                    y: [cvMeanAcc, testAcc],
                    type: 'bar',
                    marker: { color: ['#e74c3c', '#2ecc71'] },
                    text: [cvMeanAcc.toFixed(4), testAcc.toFixed(4)],
                    textposition: 'auto'
                };
                Plotly.newPlot('cv_comparison_plot', [traceComp], { title: 'Robustness Check: CV vs Test (Accuracy)', yaxis: { range: [0, 1.1], title: 'Accuracy' } });

                const tbody = document.querySelector('#cv_table tbody');
                tbody.innerHTML = '';
                for (let i = 0; i < 5; i++) {
                    const row = document.createElement('tr');
                    row.innerHTML = `<td>Fold ${i+1}</td><td>${metrics.acc.values[i].toFixed(4)}</td><td>${metrics.prec.values[i].toFixed(4)}</td><td>${metrics.rec.values[i].toFixed(4)}</td><td>${metrics.f1.values[i].toFixed(4)}</td><td>${metrics.mcc.values[i].toFixed(4)}</td><td>${metrics.auc.values[i].toFixed(4)}</td>`;
                    tbody.appendChild(row);
                }
                const meanRow = document.createElement('tr');
                meanRow.style.fontWeight = 'bold';
                meanRow.style.backgroundColor = '#f0f0f0';
                meanRow.innerHTML = `<td>Mean ± Std</td><td>${metrics.acc.mean.toFixed(4)} ± ${metrics.acc.std.toFixed(4)}</td><td>${metrics.prec.mean.toFixed(4)} ± ${metrics.prec.std.toFixed(4)}</td><td>${metrics.rec.mean.toFixed(4)} ± ${metrics.rec.std.toFixed(4)}</td><td>${metrics.f1.mean.toFixed(4)} ± ${metrics.f1.std.toFixed(4)}</td><td>${metrics.mcc.mean.toFixed(4)} ± ${metrics.mcc.std.toFixed(4)}</td><td>${metrics.auc.mean.toFixed(4)} ± ${metrics.auc.std.toFixed(4)}</td>`;
                tbody.appendChild(meanRow);
            }

            if (data.prob_dist) {
                const tracePos = { x: data.prob_dist.pos, type: 'histogram', opacity: 0.6, name: 'Positive Class', marker: { color: 'green' }, xbins: { start: 0, end: 1, size: 0.05 } };
                const traceNeg = { x: data.prob_dist.neg, type: 'histogram', opacity: 0.6, name: 'Negative Class', marker: { color: 'red' }, xbins: { start: 0, end: 1, size: 0.05 } };
                Plotly.newPlot('prob_dist_plot', [tracePos, traceNeg], { title: 'Prediction Confidence Distribution (Best Model)', barmode: 'overlay', xaxis: { title: 'Predicted Probability', range: [0, 1] }, yaxis: { title: 'Count' } });
            }

            if (data.model_predictions) {
                const grid = document.getElementById('density_grid');
                Object.keys(data.model_predictions).forEach(modelName => {
                    const divId = `density_${modelName}`;
                    const div = document.createElement('div');
                    div.id = divId;
                    grid.appendChild(div);

                    const posData = data.model_predictions[modelName].pos;
                    const negData = data.model_predictions[modelName].neg;

                    const tracePos = { x: posData, type: 'histogram', histnorm: 'probability density', opacity: 0.5, name: 'Positive', marker: { color: '#3498db' }, xbins: { start: 0, end: 1, size: 0.02 } };
                    const traceNeg = { x: negData, type: 'histogram', histnorm: 'probability density', opacity: 0.5, name: 'Negative', marker: { color: '#e74c3c' }, xbins: { start: 0, end: 1, size: 0.02 } };

                    Plotly.newPlot(divId, [traceNeg, tracePos], { title: `${modelName} Density`, barmode: 'overlay', xaxis: { title: 'Predicted Probability', range: [0, 1] }, yaxis: { title: 'Density' }, showlegend: true, legend: { x: 0.8, y: 1 } }, getDownloadConfig(`Density_${modelName}`));
                });
            }

            if (data.cv_results && data.cv_results.roc_data) {
                 document.getElementById('cv_roc_section').style.display = 'block';
                 const rocTraces = [];
                 data.cv_results.roc_data.forEach(fold => {
                     rocTraces.push({ x: fold.fpr, y: fold.tpr, mode: 'lines', name: `Fold ${fold.fold} (AUC: ${fold.auc.toFixed(3)})`, line: { color: 'rgba(52, 152, 219, 0.4)', width: 1.5 }, showlegend: true });
                 });
                 const bestModelName = data.cv_results.model;
                 if (data.roc_curves[bestModelName]) {
                     const testRoc = data.roc_curves[bestModelName].test;
                     rocTraces.push({ x: testRoc.fpr, y: testRoc.tpr, mode: 'lines', name: `Indep. Test (AUC: ${testRoc.auc.toFixed(3)})`, line: { color: '#e74c3c', width: 4 }, showlegend: true });
                 }
                 rocTraces.push({ x: [0, 1], y: [0, 1], mode: 'lines', name: 'Random', line: { dash: 'dash', color: 'gray' }, showlegend: false });

                 Plotly.newPlot('cv_roc_plot', rocTraces, { title: `5-Fold CV vs. Independent Test ROC (${data.cv_results.model})`, xaxis: { title: 'False Positive Rate' }, yaxis: { title: 'True Positive Rate' }, hovermode: 'closest' });
            }

            if (data.shap_plots || data.shap_importance) {
                 document.getElementById('shap_section').style.display = 'block';
            }

            if (data.shap_plots) {
                const shapModel = Object.keys(data.shap_plots)[0];
                if (shapModel) {
                     const imgContainer = document.getElementById('shap_image_container');
                     if (imgContainer) {
                        imgContainer.innerHTML = `<img src="${data.shap_plots[shapModel]}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 10px;" alt="SHAP Beeswarm Plot" />`;
                     }
                }
            }

            if (data.shap_importance) {
                 const shapDiv = document.getElementById('shap_plot');
                 if (shapDiv) {
                     Object.keys(data.shap_importance).forEach(modelName => {
                         const shapData = data.shap_importance[modelName];
                         Plotly.newPlot(shapDiv, [
                            { x: shapData.indices.map(i => `Dim ${i}`), y: shapData.train, type: 'bar', name: 'Train Importance', marker: { color: '#3498db' } },
                            { x: shapData.indices.map(i => `Dim ${i}`), y: shapData.test, type: 'bar', name: 'Test Importance', marker: { color: '#e74c3c' } }
                          ], { title: `Top 20 Important Features - ${modelName}`, xaxis: { title: 'Feature Dimension', tickangle: -45 }, yaxis: { title: 'Mean |SHAP Value|' }, margin: { b: 100 }, barmode: 'group' }, getDownloadConfig('SHAP_Importance'));
                     });
                 }
            }

            const allModels = Object.keys(data.metrics);
            const gluModels = allModels.filter(m => m.startsWith('CNN-') && m !== 'CNN');
            const baseModels = ['MLP', 'LSTM', 'CNN'];
            
            let bestGluModel = null;
            let maxAcc = -1;
            gluModels.forEach(m => {
                if (data.metrics[m] && data.metrics[m].test.acc > maxAcc) {
                    maxAcc = data.metrics[m].test.acc;
                    bestGluModel = m;
                }
            });
            
            const mainModels = [...baseModels];
            if (bestGluModel && !mainModels.includes(bestGluModel)) mainModels.push(bestGluModel);
            const gluModelsToUse = gluModels.length > 0 ? gluModels : [];

            const metrics = ['acc', 'prec', 'rec', 'f1'];
            const metricNames = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
            
            const createBarChart = (divId, datasetType, models, title, colorway) => {
                const traces = metrics.map((metric, i) => {
                    return {
                        x: models,
                        y: models.map(m => data.metrics[m] ? data.metrics[m][datasetType][metric] : 0),
                        name: metricNames[i],
                        type: 'bar',
                        text: models.map(m => data.metrics[m] ? data.metrics[m][datasetType][metric].toFixed(4) : "N/A"),
                        textposition: 'auto'
                    };
                });
                Plotly.newPlot(divId, traces, { barmode: 'group', title: title, yaxis: { range: [0, 1.1] }, colorway: colorway }, getDownloadConfig(title.replace(/\s/g, '_')));
            };

            const createRadarChart = (divId, datasetType, models, title, colorway) => {
                const radarTraces = models.map(model => {
                    if (!data.metrics[model]) return null;
                    const m = data.metrics[model][datasetType];
                    return { type: 'scatterpolar', r: [m.acc, m.prec, m.rec, m.f1], theta: ['Accuracy', 'Precision', 'Recall', 'F1 Score'], fill: 'toself', name: model };
                }).filter(t => t !== null);
                
                Plotly.newPlot(divId, radarTraces, { polar: { radialaxis: { visible: true, range: [0, 1] } }, title: title, colorway: colorway }, getDownloadConfig(title.replace(/\s/g, '_')));
            };

            const createRocChart = (divId, datasetType, models, title, colorway) => {
                if (!data.roc_curves) return;
                const rocTraces = [{ x: [0, 1], y: [0, 1], mode: 'lines', line: { dash: 'dash', color: 'gray' }, name: 'Random' }];
                
                models.forEach(model => {
                    if (data.roc_curves[model] && data.roc_curves[model][datasetType]) {
                        const roc = data.roc_curves[model][datasetType];
                        rocTraces.push({ x: roc.fpr, y: roc.tpr, mode: 'lines', name: `${model} (AUC: ${roc.auc.toFixed(3)})` });
                    }
                });
                Plotly.newPlot(divId, rocTraces, { title: title, xaxis: { title: 'False Positive Rate' }, yaxis: { title: 'True Positive Rate' }, colorway: colorway }, getDownloadConfig(title.replace(/\s/g, '_')));
            };

            const createPrChart = (divId, datasetType, models, title, colorway) => {
                if (!data.pr_curves) return;
                const prTraces = [];
                models.forEach(model => {
                    if (data.pr_curves[model] && data.pr_curves[model][datasetType]) {
                        const pr = data.pr_curves[model][datasetType];
                        prTraces.push({ x: pr.rec, y: pr.prec, mode: 'lines', name: `${model}` });
                    }
                });
                Plotly.newPlot(divId, prTraces, { title: title, xaxis: { title: 'Recall' }, yaxis: { title: 'Precision', range: [0, 1.05] }, colorway: colorway }, getDownloadConfig(title.replace(/\s/g, '_')));
            };

            const createConfusionMatrix = (containerId, modelName) => {
                if (!data.confusion_matrices || !data.confusion_matrices[modelName]) return;
                const cmData = data.confusion_matrices[modelName].test;
                const xLabels = ['Pred: Non-Onco', 'Pred: Onco'];
                const yLabels = ['True: Non-Onco', 'True: Onco'];
                
                const div = document.createElement('div');
                div.id = `cm_${modelName}`;
                div.style.height = '400px';
                document.getElementById(containerId).appendChild(div);
                
                const annotations = [];
                for (let i = 0; i < yLabels.length; i++) {
                    for (let j = 0; j < xLabels.length; j++) {
                        annotations.push({ x: xLabels[j], y: yLabels[i], text: String(cmData[i][j]), font: { color: 'white', size: 16 }, showarrow: false });
                    }
                }
                Plotly.newPlot(div.id, [{ z: cmData, x: xLabels, y: yLabels, type: 'heatmap', colorscale: 'Blues', showscale: false }], { title: `Confusion Matrix - ${modelName} (Test)`, xaxis: { side: 'bottom' }, yaxis: { autorange: 'reversed' }, annotations: annotations }, getDownloadConfig(`CM_${modelName}`));
            };

            const mainColors = ['#008080', '#FF7F50', '#800080', '#00BFFF', '#FFD700']; 
            createBarChart('comparison_plot_train', 'train', mainModels, 'Main Models Performance (Train)', mainColors);
            createBarChart('comparison_plot_test', 'test', mainModels, 'Main Models Performance (Test)', mainColors);
            createRadarChart('radar_plot_train', 'train', mainModels, 'Main Models Profile (Train)', mainColors);
            createRadarChart('radar_plot_test', 'test', mainModels, 'Main Models Profile (Test)', mainColors);
            createRocChart('roc_comparison_plot_train', 'train', mainModels, 'Main Models ROC (Train)', mainColors);
            createRocChart('roc_comparison_plot_test', 'test', mainModels, 'Main Models ROC (Test)', mainColors);
            createPrChart('pr_comparison_plot_train', 'train', mainModels, 'Main Models PR Curve (Train)', mainColors);
            createPrChart('pr_comparison_plot_test', 'test', mainModels, 'Main Models PR Curve (Test)', mainColors);

            const allVizModels = [...mainModels, ...gluModelsToUse];
            const uniqueModels = [...new Set(allVizModels)];
            uniqueModels.forEach(m => createConfusionMatrix('cm_container', m));

            const gluColors = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3'];
            if (gluModelsToUse.length > 0) {
                createBarChart('glu_comparison_plot_train', 'train', gluModelsToUse, 'GLU Variants Performance (Train)', gluColors);
                createBarChart('glu_comparison_plot_test', 'test', gluModelsToUse, 'GLU Variants Performance (Test)', gluColors);
                createRadarChart('glu_radar_plot_train', 'train', gluModelsToUse, 'GLU Variants Profile (Train)', gluColors);
                createRadarChart('glu_radar_plot_test', 'test', gluModelsToUse, 'GLU Variants Profile (Test)', gluColors);
                createRocChart('glu_roc_comparison_plot_train', 'train', gluModelsToUse, 'GLU Variants ROC (Train)', gluColors);
                createRocChart('glu_roc_comparison_plot_test', 'test', gluModelsToUse, 'GLU Variants ROC (Test)', gluColors);
                createPrChart('glu_pr_comparison_plot_train', 'train', gluModelsToUse, 'GLU Variants PR Curve (Train)', gluColors);
                createPrChart('glu_pr_comparison_plot_test', 'test', gluModelsToUse, 'GLU Variants PR Curve (Test)', gluColors);
            }

            const tbody = document.querySelector('#comparison_table tbody');
            const tableModels = [...new Set([...baseModels, ...gluModelsToUse])];
            
            tableModels.forEach(model => {
                if (!data.metrics[model]) return;
                const rowTr = document.createElement('tr');
                const mTr = data.metrics[model].train;
                rowTr.innerHTML = `<td style="font-weight: bold;">${model}</td><td><span style="color: #e67e22; font-weight: bold;">Train</span></td><td>${mTr.acc.toFixed(4)}</td><td>${mTr.prec.toFixed(4)}</td><td>${mTr.rec.toFixed(4)}</td><td>${mTr.f1.toFixed(4)}</td><td>${mTr.mcc.toFixed(4)}</td><td>${mTr.auc.toFixed(4)}</td>`;
                tbody.appendChild(rowTr);

                const rowTe = document.createElement('tr');
                const mTe = data.metrics[model].test;
                rowTe.innerHTML = `<td style="font-weight: bold;">${model}</td><td><span style="color: #27ae60; font-weight: bold;">Test</span></td><td>${mTe.acc.toFixed(4)}</td><td>${mTe.prec.toFixed(4)}</td><td>${mTe.rec.toFixed(4)}</td><td>${mTe.f1.toFixed(4)}</td><td>${mTe.mcc.toFixed(4)}</td><td>${mTe.auc.toFixed(4)}</td>`;
                tbody.appendChild(rowTe);
            });

            const container = document.getElementById('viz_container');
            const vizOrder = ['Raw Embeddings', ...tableModels];
            
            vizOrder.forEach(name => {
                if (!data.visualizations[name]) return; 
                const section = document.createElement('div');
                section.className = 'model-section';
                section.innerHTML = `<h3>${name} Feature Space</h3><div class="plot-grid" id="grid_${name}"></div>`;
                container.appendChild(section);

                const grid = document.getElementById(`grid_${name}`);
                const tsneDiv = document.createElement('div');
                tsneDiv.id = `tsne_${name}`;
                grid.appendChild(tsneDiv);
                
                Plotly.newPlot(tsneDiv.id, [{ x: data.visualizations[name].tsne.x, y: data.visualizations[name].tsne.y, mode: 'markers', type: 'scatter', marker: { color: data.visualizations[name].labels, colorscale: 'Viridis', size: 6, opacity: 0.7 }, text: data.visualizations[name].labels.map(l => l == 1 ? 'Oncogenic' : 'Non-Oncogenic') }], { title: `${name} - t-SNE`, xaxis: { title: 'Dim 1' }, yaxis: { title: 'Dim 2' } }, getDownloadConfig(`${name}_tSNE`));

                if (data.visualizations[name].umap && data.visualizations[name].umap.x.length > 0) {
                     const umapDiv = document.createElement('div');
                     umapDiv.id = `umap_${name}`;
                     grid.appendChild(umapDiv);
                     
                     Plotly.newPlot(umapDiv.id, [{ x: data.visualizations[name].umap.x, y: data.visualizations[name].umap.y, mode: 'markers', type: 'scatter', marker: { color: data.visualizations[name].labels, colorscale: 'Viridis', size: 6, opacity: 0.7 }, text: data.visualizations[name].labels.map(l => l == 1 ? 'Oncogenic' : 'Non-Oncogenic') }], { title: `${name} - UMAP`, xaxis: { title: 'Dim 1' }, yaxis: { title: 'Dim 2' } }, getDownloadConfig(`${name}_UMAP`));
                }
            });
        }
        updateDashboard(dashboardData);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    run_experiment()