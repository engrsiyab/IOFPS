import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization, Activation, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import itertools

# --- CONFIGURATION ---
EXPERIMENT_MODE = True
OPTIMIZER_NAME = 'adam'
EPOCHS_PER_RUN = 5
BATCH_SIZE = 32

os.makedirs('/kaggle/working', exist_ok=True)

# --- DASHBOARD DATA COLLECTION ---
dashboard_data = {
    "metrics": {},
    "dist": {},
    "balance": [],
    "loss": {},
    "pca": {},
    "cm": [],
    "roc": {},
    "experiments": []
}

def get_activation(act_name, k=1):
    act_name = act_name.lower()
    if act_name == 'swish':
        return lambda x: x * tf.sigmoid(k * x)
    elif act_name == 'sigmoid':
        return 'sigmoid'
    elif act_name == 'relu':
        return 'relu'
    elif act_name == 'tanh':
        return 'tanh'
    elif act_name == 'gelu':
        return tf.keras.activations.gelu
    else:
        return 'relu'

def glu_unit(inputs, units, activation_name, k_value):
    x = Dense(units)(inputs)
    x1 = Dense(units)(inputs)

    act_fn = get_activation(activation_name, k_value)
    x = Activation(act_fn)(x) if callable(act_fn) else Activation(act_fn)(x)
    return x * x1

def build_and_train_model(X_train, y_train, X_test, y_test, params, input_dim):
    K.clear_session()

    inputs = Input((input_dim, 1))
    x = Conv1D(128, kernel_size=params['filter_size'], strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)

    act_fn = get_activation(params['activation'], params['k'])
    x = Activation(act_fn)(x) if callable(act_fn) else Activation(act_fn)(x)

    x = MaxPooling1D(pool_size=params['pool_size'], padding='same')(x)
    x = Dropout(0.2)(x)

    x = glu_unit(x, 64, params['activation'], params['k'])
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.15)(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs)

    opt = SGD(learning_rate=params['lr'], momentum=0.9) if OPTIMIZER_NAME.lower() == 'sgd' else Adam(learning_rate=params['lr'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS_PER_RUN,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop]
    )
    return model, history

# --- LOAD DATA ---
dataset = pd.read_excel('/kaggle/input/datasets/engrsiyab/phd-dataset/train.xlsx', na_filter=False)
dataset_test = pd.read_excel('/kaggle/input/datasets/engrsiyab/phd-dataset/test.xlsx', na_filter=False)

y_train = dataset['label'].values
y_test = dataset_test['label'].values

X_train = np.load('/kaggle/input/datasets/engrsiyab/phd-dataset/emb_train_prott5.npy')
X_test = np.load('/kaggle/input/datasets/engrsiyab/phd-dataset/emb_test_prott5.npy')

if X_train.ndim == 2:
    X_train = X_train[..., None]
if X_test.ndim == 2:
    X_test = X_test[..., None]

input_dim = X_train.shape[1]

param_grid = {
    'filter_size': [1, 3, 5, 7],
    'pool_size': [2, 3],
    'activation': ['sigmoid', 'relu', 'tanh', 'gelu', 'swish'],
    'k': [0.5, 1, 10, 100],
    'lr': [0.1, 0.01]
}

combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

results, cache = [], {}

for i, params in enumerate(combinations):
    print(f"[{i+1}/{len(combinations)}] {params}")

    cache_key = (params['filter_size'], params['pool_size'], params['activation'], params['lr'])

    if params['activation'] != 'swish' and cache_key in cache:
        res = cache[cache_key].copy()
        res['k'] = params['k']
        results.append(res)
        dashboard_data['experiments'].append(res)
        continue

    model, _ = build_and_train_model(X_train, y_train, X_test, y_test, params, input_dim)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    res = {
        'filter_size': params['filter_size'],
        'pool_size': params['pool_size'],
        'activation': params['activation'],
        'k': params['k'],
        'lr': params['lr'],
        'acc': float(accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'prec': float(precision_score(y_test, y_pred, zero_division=0)),
        'rec': float(recall_score(y_test, y_pred, zero_division=0))
    }

    results.append(res)
    dashboard_data['experiments'].append(res)

    if params['activation'] != 'swish':
        cache[cache_key] = res

# --- SAVE JSON ---
best_run = max(results, key=lambda x: x['f1'])
dashboard_data["metrics"] = {
    "acc": best_run['acc'],
    "auc": 0.0,
    "f1": best_run['f1'],
    "mcc": 0.0,
    "prec": best_run['prec'],
    "rec": best_run['rec']
}

json_path = '/kaggle/working/dashboard_data_MLTweakGS.json'
with open(json_path, 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print(f"✅ JSON saved to: {json_path}")

# --- GENERATE HTML ---
print("📁 Available input folders:", os.listdir('/kaggle/input'))

TEMPLATE_PATH = '/kaggle/input/gradient-s/dashboard_template_MLTweakGS.html'  # change if needed

if not os.path.exists(TEMPLATE_PATH):
    raise FileNotFoundError(f"❌ Template not found: {TEMPLATE_PATH}")

with open(TEMPLATE_PATH, 'r') as f:
    html_template = f.read()

json_str = json.dumps(dashboard_data)
final_html = html_template.replace(
    '// Data placeholders - these will be replaced by Python script',
    f'const dashboardData = {json_str}; updateDashboard(dashboardData);'
)

html_out_path = '/kaggle/working/dashboard_MLTweakGS.html'
with open(html_out_path, 'w') as f:
    f.write(final_html)

print(f"✅ HTML saved to: {html_out_path}")
print("📂 Working dir files:", os.listdir('/kaggle/working'))
