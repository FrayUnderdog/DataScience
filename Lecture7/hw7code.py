# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------
# Configuration
# ------------------------------
CSV_PATH = "amr_ds.csv"  # change to "/mnt/data/amr_ds.csv" if needed

STATE_NAMES = ['Ampicillin', 'Penicillin', 'Not_MDR']
OBS_NAMES = ['No Infection', 'Infection']  # order matters for emission lookup

# Emission probabilities: rows=states (Amp, Pen, NMDR), cols=observations (No Infection, Infection)
EMISSION_TABLE = np.array([
    [0.4, 0.6],  # Ampicillin
    [0.3, 0.7],  # Penicillin
    [0.8, 0.2],  # Not_MDR
], dtype=float)

# Observation sequence for HMM/Viterbi (exact strings mapped below)
OBS_SEQ_STR = [
    'Infection after surgery',
    'No infection after surgery',
    'Infection after surgery'
]


# ------------------------------
# Utilities
# ------------------------------
def safe_div(a: float, b: float) -> float:
    """Safely divide a/b with zero denominator protection."""
    return (a / b) if b != 0 else 0.0


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column names and ensure required columns exist."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    required = set(STATE_NAMES)
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Expected columns {STATE_NAMES}, but got {list(df.columns)}")
    return df


# ------------------------------
# Part 1: Naive Bayes (Bernoulli) with 75/25 split
# ------------------------------
def bernoulli_nb_fit_predict_accuracy(df: pd.DataFrame, random_state: int = 6105) -> float:
    """
    Train a Bernoulli Naive Bayes classifier from scratch on two binary features
    (Ampicillin, Penicillin) to predict Not_MDR, and return test accuracy.
    Laplace smoothing is used for stability.
    """
    X = df[['Ampicillin', 'Penicillin']].astype(int)
    y = df['Not_MDR'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    classes = np.sort(y.unique())
    alpha = 1.0  # Laplace smoothing

    # Class priors
    class_counts = np.array([(y_train == c).sum() for c in classes], dtype=float)
    class_priors = class_counts / class_counts.sum()

    # Conditional probs P(x_j=1 | class=c)
    cond_prob = np.zeros((len(classes), X_train.shape[1]))
    for i, c in enumerate(classes):
        X_c = X_train[y_train == c]
        cond_prob[i] = (X_c.sum(axis=0) + alpha) / (X_c.shape[0] + 2 * alpha)

    # Predict in log-space
    Xb = X_test.values
    log_proba = np.zeros((Xb.shape[0], len(classes)))
    for i in range(len(classes)):
        lp = np.log(class_priors[i] + 1e-12)
        p = np.clip(cond_prob[i], 1e-9, 1 - 1e-9)
        log_like = (Xb * np.log(p) + (1 - Xb) * np.log(1 - p)).sum(axis=1)
        log_proba[:, i] = lp + log_like

    y_pred = classes[np.argmax(log_proba, axis=1)]
    acc = (y_pred == y_test.values).mean()
    return float(acc)


# ------------------------------
# Part 2: Markov Chain from counts and stationary distribution
# ------------------------------
def build_transition_matrix_and_stationary(df: pd.DataFrame):
    """
    Build the transition matrix for states [Ampicillin, Penicillin, Not_MDR]
    based on the counts specified in the assignment, then compute the stationary distribution.
    Returns (transition_matrix, stationary_probs).
    """
    data_np = df[STATE_NAMES].to_numpy(dtype=int)

    # Counts per spec
    amp_pen = int(np.sum((data_np[:, 0] == 1) & (data_np[:, 1] == 1)))
    amp_nmdr = int(np.sum((data_np[:, 0] == 1) & (data_np[:, 2] == 1)))
    pen_nmdr = int(np.sum((data_np[:, 1] == 1) & (data_np[:, 2] == 1)))

    tm = np.array([
        [0.0, safe_div(amp_pen, amp_nmdr + amp_pen), safe_div(amp_nmdr, amp_nmdr + amp_pen)],
        [safe_div(amp_pen, pen_nmdr + amp_pen), 0.0, safe_div(pen_nmdr, pen_nmdr + amp_pen)],
        [safe_div(amp_nmdr, amp_nmdr + pen_nmdr), safe_div(pen_nmdr, amp_nmdr + pen_nmdr), 0.0],
    ], dtype=float)

    # Stationary distribution: solve pi P = pi, sum(pi)=1
    eigvals, eigvecs = np.linalg.eig(tm.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stationary = np.real(eigvecs[:, idx])
    stationary = np.where(stationary < 0, 0, stationary)  # guard against negative zeros
    stationary = stationary / stationary.sum()

    counts = {'amp_pen': amp_pen, 'amp_nmdr': amp_nmdr, 'pen_nmdr': pen_nmdr}
    return tm, stationary, counts


# ------------------------------
# Part 3: HMM / Viterbi
# ------------------------------
def map_observation_to_index(obs_str: str) -> int:
    """
    Map observation strings in the prompt to indices in OBS_NAMES.
    Accepts both exact table labels and the '... after surgery' variants.
    """
    s = obs_str.strip().lower()
    if 'no infection' in s:
        return OBS_NAMES.index('No Infection')
    if 'infection' in s:
        return OBS_NAMES.index('Infection')
    raise ValueError(f"Unknown observation string: {obs_str}")


def viterbi_decode(A: np.ndarray, B: np.ndarray, pi: np.ndarray, obs_idx: list) -> list:
    """
    Standard Viterbi in log-space.
    A: (N,N) transition matrix
    B: (N,M) emission matrix
    pi: (N,) initial distribution
    obs_idx: list of observation indices
    Returns: list of state indices (most probable path)
    """
    N = A.shape[0]
    T = len(obs_idx)

    A_log = np.log(np.clip(A, 1e-12, 1.0))
    B_log = np.log(np.clip(B, 1e-12, 1.0))
    pi_log = np.log(np.clip(pi, 1e-12, 1.0))

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # Initialization
    delta[0] = pi_log + B_log[:, obs_idx[0]]
    psi[0] = 0

    # Recursion
    for t in range(1, T):
        for j in range(N):
            scores = delta[t - 1] + A_log[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = scores[psi[t, j]] + B_log[j, obs_idx[t]]

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path.tolist()


# ------------------------------
# Main
# ------------------------------
def main():
    # Load and normalize columns
    df = pd.read_csv(CSV_PATH)
    df = normalize_columns(df)

    # Part 1: Naive Bayes accuracy
    nb_acc = bernoulli_nb_fit_predict_accuracy(df)
    print("\n=== Part 1: Naive Bayes (Bernoulli) ===")
    print(f"Test accuracy: {nb_acc:.4f}")

    # Part 2: Transition matrix and stationary distribution
    tm, stationary, counts = build_transition_matrix_and_stationary(df)
    print("\n=== Part 2: Markov Chain from counts ===")
    print("Counts:", counts)
    print("\nTransition Matrix (rows->columns):")
    tm_df = pd.DataFrame(tm, index=STATE_NAMES, columns=STATE_NAMES)
    print(tm_df.round(6))
    print("\nStationary distribution:")
    stat_series = pd.Series(stationary, index=STATE_NAMES)
    print(stat_series.round(6))

    # Part 3: HMM/Viterbi with given emissions and observations
    obs_idx = [map_observation_to_index(s) for s in OBS_SEQ_STR]
    pi = stationary.copy()  # use stationary distribution as initial
    path_idx = viterbi_decode(A=tm, B=EMISSION_TABLE, pi=pi, obs_idx=obs_idx)
    path_states = [STATE_NAMES[i] for i in path_idx]

    print("\n=== Part 3: HMM / Viterbi Decoding ===")
    print("Observations:", OBS_SEQ_STR)
    print("Most probable hidden-state sequence:", path_states)


if __name__ == "__main__":
    main()
