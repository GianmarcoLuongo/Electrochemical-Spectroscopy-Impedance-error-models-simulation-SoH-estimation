from skopt import BayesSearchCV
import pandas as pd
import numpy as np
import sys
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error, max_error
from sklearn.pipeline import Pipeline
from CompositeKernel import CompositeKernel
from skopt.space import Real

np.random.seed(42)

# ========================================
# PARAMETRI DA TERMINALE
# ========================================
DATA_PATH = sys.argv[1]
LABEL = sys.argv[2]

print(f"Dataset: {DATA_PATH}")
print(f"Label: {LABEL}")

# ========================================
# PARTIAL SCALER
# ========================================
class PartialScaler:
    def __init__(self, n_features_to_scale):
        self.scaler = StandardScaler()
        self.n_features_to_scale = n_features_to_scale

    def fit(self, X, y=None):
        self.scaler.fit(X[:, :self.n_features_to_scale])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[:, :self.n_features_to_scale] = self.scaler.transform(X[:, :self.n_features_to_scale])
        return X_copy

    def fit_transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[:, :self.n_features_to_scale] = self.scaler.fit_transform(X[:, :self.n_features_to_scale])
        return X_copy

# ========================================
# CV GENERATOR
# ========================================
def CVGenerator(dataset):
    SoH_levels = np.sort(dataset["SoH"].unique())
    batteries_per_SoH = {SoH: dataset.query(f"SoH == {SoH}")["Cell_Name"].unique() for SoH in SoH_levels}
    folds = [
        [28, 29, 2, 12, 17, 15],  # Fold 1: 2 from 100%, 1 from 95%, 1 from 90%, 1 from 85%, 1 from 80%
        [3, 4, 13, 20, 18, 30],   # Fold 2: 1 from 100%, 2 from 95%, 1 from 90%, 1 from 85%, 1 from 80%
        [17, 21, 14, 19, 31, 5],  # Fold 3: 1 from 100%, 1 from 95%, 1 from 90%, 2 from 85%, 1 from 80%
        [15, 23, 24, 32, 6, 22]   # Fold 4: 1 from 100%, 1 from 95%, 1 from 90%, 1 from 85%, 2 from 80%
    ]
    for batt_test in folds:
        trainingSet = dataset[~dataset['Cell_Name'].isin(batt_test)]
        testSet = dataset[dataset['Cell_Name'].isin(batt_test)]
        yield (trainingSet.index.values.astype(int), testSet.index.values.astype(int))


# ========================================
# CARICA DATASET
# ========================================
df = pd.read_csv(DATA_PATH)
features = [col for col in df.columns if '-Re' in col or '-Im' in col] + ['Temp', 'SoC']
X = df[features].values
X_eis_len = len(features) - 2
y = df['SoH_Actual'].values / 100.0

# ========================================
# TROVA RANGE PARAMETRI
# ========================================
print("\n" + "="*50)
print("STEP 1: Ricerca range parametri")
print("="*50)

pipeline_explore = Pipeline([
    ('scaler', PartialScaler(n_features_to_scale=X_eis_len)),
    ('gpr', GaussianProcessRegressor(
        kernel=CompositeKernel(),
        normalize_y=True
    ))
])

kernel_param_list_explore = []
for fold, (train_idx, test_idx) in enumerate(CVGenerator(df)):
    X_train = X[train_idx]
    y_train = y[train_idx]
    pipeline_explore.fit(X_train, y_train)
    K = pipeline_explore.named_steps["gpr"].kernel_
    params_dict = {
        "variance_eis": K.variance_eis,
        "lengthscale_eis": K.lengthscale_eis,
        "variance_temp": K.variance_temp,
        "lengthscale_temp": K.lengthscale_temp,
        "variance_soc": K.variance_soc
    }
    kernel_param_list_explore.append(params_dict)
    print(f"  Fold {fold+1}: {params_dict}")

df_kp = pd.DataFrame(kernel_param_list_explore)
ranges = {}
for col in df_kp.columns:
    ranges[col] = (0.2 * df_kp[col].min(), 5 * df_kp[col].max())

print("\nRange trovati:")
for name, (lo, hi) in ranges.items():
    print(f"  {name}: [{lo}, {hi}]")

# ========================================
# BAYES SEARCH CV
# ========================================
print("\n" + "="*50)
print("STEP 2: BayesSearchCV")
print("="*50)

pipeline = Pipeline([
    ('scaler', PartialScaler(n_features_to_scale=X_eis_len)),
    ('gpr', GaussianProcessRegressor(
        kernel=CompositeKernel(),
        normalize_y=True,
        optimizer=None
    ))
])

param_grid = {
    'gpr__alpha': Real(1e-10, 1e-5, prior='log-uniform'),
    'gpr__kernel__variance_eis': Real(ranges['variance_eis'][0], ranges['variance_eis'][1], prior='log-uniform'),
    'gpr__kernel__lengthscale_eis': Real(ranges['lengthscale_eis'][0], ranges['lengthscale_eis'][1], prior='log-uniform'),
    'gpr__kernel__variance_temp': Real(ranges['variance_temp'][0], ranges['variance_temp'][1], prior='log-uniform'),
    'gpr__kernel__lengthscale_temp': Real(ranges['lengthscale_temp'][0], ranges['lengthscale_temp'][1], prior='log-uniform'),
    'gpr__kernel__variance_soc': Real(ranges['variance_soc'][0], ranges['variance_soc'][1], prior='log-uniform'),
}

opt = BayesSearchCV(pipeline, param_grid, cv=CVGenerator(df), n_iter=5, verbose=1)
opt.fit(X, y)

print("Best parameters:", opt.best_params_)
pipeline.set_params(**opt.best_params_)

# ========================================
# CV 
# ========================================
print("\n" + "="*50)
print("STEP 3: Cross-validation finale")
print("="*50)

MAEs, R2s, RMSEs, Max_Errors = [], [], [], []
kernel_param_list = []

for fold, (train_idx, test_idx) in enumerate(CVGenerator(df)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    MAEs.append(mae)
    R2s.append(r2)
    RMSEs.append(rmse)
    Max_Errors.append(max_err)

    print(f"\nFold {fold+1}:")
    print(f"  MAE = {mae}")
    print(f"  RMSE = {rmse}")
    print(f"  MAX_ERR = {max_err}")
    print(f"  R2 = {r2}")
    print(f"  Kernel: {pipeline.named_steps['gpr'].kernel_}")
    print(f"  Alpha: {pipeline.named_steps['gpr'].alpha}")

    K = pipeline.named_steps["gpr"].kernel_
    params_dict = {
        "variance_eis": K.variance_eis,
        "lengthscale_eis": K.lengthscale_eis,
        "variance_temp": K.variance_temp,
        "lengthscale_temp": K.lengthscale_temp,
        "variance_soc": K.variance_soc
    }
    kernel_param_list.append(params_dict)

df_kernel_params = pd.DataFrame(kernel_param_list, index=[f"Fold {i+1}" for i in range(len(kernel_param_list))])
print("\nParametri kernel per fold:")
print(df_kernel_params)

# ========================================
# RISULTATI E SALVATAGGIO
# ========================================
print("\n" + "="*50)
print("RISULTATI FINALI")
print("="*50)
print(f"Average RMSE: {np.mean(RMSEs):.4f} ± {np.std(RMSEs):.4f}")
print(f"Average MAE:  {np.mean(MAEs):.4f} ± {np.std(MAEs):.4f}")
print(f"Average MAX:  {np.mean(Max_Errors):.4f} ± {np.std(Max_Errors):.4f}")
print(f"Average R2:   {np.mean(R2s):.4f} ± {np.std(R2s):.4f}")

metriche_finali = {
    'RMSE': f"{np.mean(RMSEs):.4f} ± {np.std(RMSEs):.4f}",
    'MAE': f"{np.mean(MAEs):.4f} ± {np.std(MAEs):.4f}",
    'MAX_ERR': f"{np.mean(Max_Errors):.4f} ± {np.std(Max_Errors):.4f}",
    'R2': f"{np.mean(R2s):.4f} ± {np.std(R2s):.4f}",
}

file_name = "metriche_modelli_optimized.csv"
row_name = f"GPR_{LABEL}"

if os.path.exists(file_name):
    df_existing = pd.read_csv(file_name, index_col=0)
    df_existing.loc[row_name] = metriche_finali
    df_existing.to_csv(file_name)
else:
    pd.DataFrame([metriche_finali], index=[row_name]).to_csv(file_name, index=True)

print(f"\nMetriche salvate come '{row_name}' in {file_name}")
