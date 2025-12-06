import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. Load the dataset (cleaned in R)
# ------------------------------------------------------------

def load_phoneme_data(path="../data/phoneme_python.csv"):
    """
    Loads the phoneme dataset cleaned in R:
    - id removed
    - Class mapped to nasal/oral
    - no missing values
    """

    df = pd.read_csv(path)
    df["Class"] = df["Class"].astype("category")

    print("Dataset loaded.")
    print("Shape:", df.shape)
    print("Class distribution:\n", df["Class"].value_counts(normalize=True))

    return df


# ------------------------------------------------------------
# 2. Outlier treatment (IQR-based winsorization)
# ------------------------------------------------------------

def winsorize_iqr(df, factor=1.5):
    """
    Caps outliers using the standard IQR rule:
      lower = Q1 - 1.5*IQR
      upper = Q3 + 1.5*IQR

    Applies winsorization column-wise.

    Returns a transformed copy of df.
    """

    df_wins = df.copy()

    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        # cap values
        df_wins[col] = np.clip(df[col], lower, upper)

    return df_wins


# ------------------------------------------------------------
# 3. Scaling (fit on train, apply on test)
# ------------------------------------------------------------

def scale_train_test(X_train, X_test):
    """
    Standardize features using parameters fitted ONLY on train.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    return X_train_scaled, X_test_scaled, scaler


# ------------------------------------------------------------
# 4. Preprocessing for a single fold of nested CV
# ------------------------------------------------------------

def prepare_fold_data(df, train_idx, test_idx):
    """
    Preprocessing steps inside nested CV:
      1) Split into train/test
      2) Apply IQR-based winsorization (fit on train → apply to both)
      3) Fit StandardScaler on train, apply to both
    """

    # ---- split ----
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    X_train = train.drop(columns=["Class"])
    y_train = train["Class"]

    X_test = test.drop(columns=["Class"])
    y_test = test["Class"]

    # ---- Winsorization (fit on train only) ----
    winsor_limits = {}
    X_train_wins = X_train.copy()
    X_test_wins = X_test.copy()

    for col in X_train.columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        winsor_limits[col] = (lower, upper)

        X_train_wins[col] = np.clip(X_train[col], lower, upper)
        X_test_wins[col] = np.clip(X_test[col], lower, upper)

    # ---- Scaling (fit only on train winsorized) ----
    X_train_scaled, X_test_scaled, scaler = scale_train_test(
        X_train_wins, X_test_wins
    )

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "winsor_limits": winsor_limits,
        "scaler": scaler
    }
