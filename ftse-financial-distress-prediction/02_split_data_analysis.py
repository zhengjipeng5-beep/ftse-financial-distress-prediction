"""Financial-distress modelling.

The script uses a chronological train/validation/test split. All preprocessing
parameters are estimated on the training period, hyperparameters and decision
thresholds are selected on the validation period, and the test period is used
only for final evaluation and post-estimation interpretation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler


# Configuration
RANDOM_SEED = 42
N_TREES = 500
N_BOOTSTRAP_REPLICATIONS = 2_000
WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99

FEATURES = [
    "Current_Ratio",
    "Liabilities_to_Assets",
    "ROA",
    "EBIT_to_Assets",
    "CFO_to_Liabilities",
    "RE_to_Assets",
]
TARGET = "FD_t_plus_1"

TARGET_YEAR_RANGES = {
    "Training": (2008, 2018),
    "Validation": (2019, 2020),
    "Test": (2021, 2024),
}

LOGISTIC_C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
RF_PARAMETER_GRID = {
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_features": ["sqrt", 0.5],
}

EXPECTED_SIGNS = {
    "Current_Ratio": "Negative",
    "Liabilities_to_Assets": "Positive",
    "ROA": "Negative",
    "EBIT_to_Assets": "Negative",
    "CFO_to_Liabilities": "Negative",
    "RE_to_Assets": "Negative",
}


# Data preparation
def load_model_data(file_path):
    """Load the panel, validate its structure, and retain valid one-year labels."""
    data = pd.read_excel(
        file_path,
        sheet_name="Clean_Panel",
        na_values=["NaN", "#N/A N/A", "#N/A", "N/A"],
    )

    required_columns = ["Ticker", "Year", "Target_Year", *FEATURES, TARGET]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"These columns are missing: {missing_columns}")

    model_data = data.loc[data[TARGET].notna()].copy()
    model_data[TARGET] = model_data[TARGET].astype(int)
    model_data["Year"] = model_data["Year"].astype(int)
    model_data["Target_Year"] = model_data["Target_Year"].astype(int)

    mismatch_count = (
        model_data["Target_Year"].ne(model_data["Year"] + 1).sum()
    )
    if mismatch_count:
        raise ValueError(
            f"Found {mismatch_count} observations where Target_Year != Year + 1."
        )

    # Preserve the source-row order. With a fixed random seed, changing row
    # order changes which observations are selected by Random Forest bootstrap
    # samples and therefore prevents exact reproduction of the original run.
    return model_data


def split_by_target_year(model_data):
    """Create non-overlapping chronological samples using outcome years."""
    samples = {
        name: model_data.loc[
            model_data["Target_Year"].between(start_year, end_year)
        ].copy()
        for name, (start_year, end_year) in TARGET_YEAR_RANGES.items()
    }

    for name, sample in samples.items():
        if sample.empty:
            raise ValueError(f"The {name.lower()} sample is empty.")

    if not (
        samples["Training"]["Target_Year"].max()
        < samples["Validation"]["Target_Year"].min()
        <= samples["Validation"]["Target_Year"].max()
        < samples["Test"]["Target_Year"].min()
    ):
        raise ValueError("The chronological samples overlap or are out of order.")

    return samples


def print_sample_summary(name, sample):
    feature_data = sample[FEATURES]
    labels = sample[TARGET]
    print(f"\n{name.upper()} SET")
    print("-" * 50)
    print(
        "Target years:",
        int(sample["Target_Year"].min()),
        "to",
        int(sample["Target_Year"].max()),
    )
    print("Observations:", len(sample))
    print("Unique firms:", sample["Ticker"].nunique())
    print("Target distribution:\n", labels.value_counts().sort_index())
    print("Distress rate:", labels.mean())
    print("Missing values by feature:\n", feature_data.isna().sum())


def preprocess_samples(samples):
    """Winsorise, impute, and scale without using post-training information."""
    raw_features = {name: sample[FEATURES].copy() for name, sample in samples.items()}

    lower_limits = raw_features["Training"].quantile(WINSOR_LOWER_QUANTILE)
    upper_limits = raw_features["Training"].quantile(WINSOR_UPPER_QUANTILE)
    winsorised = {
        name: values.clip(lower=lower_limits, upper=upper_limits, axis=1)
        for name, values in raw_features.items()
    }

    imputer = SimpleImputer(strategy="median")
    imputed = {}
    for name, values in winsorised.items():
        transformed = (
            imputer.fit_transform(values)
            if name == "Training"
            else imputer.transform(values)
        )
        imputed[name] = pd.DataFrame(
            transformed, columns=FEATURES, index=values.index
        )

    scaler = StandardScaler()
    scaled = {}
    for name, values in imputed.items():
        transformed = (
            scaler.fit_transform(values)
            if name == "Training"
            else scaler.transform(values)
        )
        scaled[name] = pd.DataFrame(
            transformed, columns=FEATURES, index=values.index
        )

    for name in samples:
        if imputed[name].isna().any().any():
            raise ValueError(f"Missing values remain in the {name.lower()} sample.")
        if not np.isfinite(scaled[name].to_numpy()).all():
            raise ValueError(f"Non-finite scaled values found in {name.lower()} sample.")

    parameters = pd.DataFrame(
        {
            "Lower_1_Percent": lower_limits,
            "Upper_99_Percent": upper_limits,
            "Training_Median": imputer.statistics_,
            "Scaler_Mean": scaler.mean_,
            "Scaler_Standard_Deviation": scaler.scale_,
        },
        index=FEATURES,
    )
    parameters.index.name = "Feature"

    return imputed, scaled, parameters


def build_descriptive_statistics(imputed, samples):
    """Summarise model inputs after training-based winsorisation and imputation."""
    full_features = pd.concat(imputed.values()).sort_index()
    full_labels = pd.concat(
        [samples[name][TARGET] for name in samples]
    ).sort_index()

    statistics = (
        full_features.agg(["count", "mean", "median", "std", "min", "max"])
        .T.reset_index()
        .rename(
            columns={
                "index": "Variable",
                "count": "N",
                "mean": "Mean",
                "median": "Median",
                "std": "Standard_Deviation",
                "min": "Minimum",
                "max": "Maximum",
            }
        )
    )
    statistics["N"] = statistics["N"].astype(int)

    group_means = (
        full_features.assign(Distress_Status=full_labels)
        .groupby("Distress_Status")[FEATURES]
        .mean()
        .T.rename(columns={0: "Non_Distressed_Mean", 1: "Distressed_Mean"})
        .reset_index(names="Variable")
    )
    statistics = statistics.merge(
        group_means, on="Variable", how="left", validate="one_to_one"
    )
    statistics["Distressed_Minus_Non_Distressed"] = (
        statistics["Distressed_Mean"] - statistics["Non_Distressed_Mean"]
    )
    return statistics


# Model selection and evaluation
def tune_logistic_regression(X_train, y_train, X_validation, y_validation):
    results = []
    for c_value in LOGISTIC_C_VALUES:
        model = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=1_000,
        )
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        results.append(
            {
                "C": c_value,
                "Validation_AP": average_precision_score(
                    y_validation, probabilities
                ),
                "Validation_ROC_AUC": roc_auc_score(
                    y_validation, probabilities
                ),
            }
        )

    results_frame = pd.DataFrame(results).sort_values(
        "Validation_AP", ascending=False, kind="stable"
    ).reset_index(drop=True)
    return results_frame, float(results_frame.loc[0, "C"])


def tune_random_forest(X_train, y_train, X_validation, y_validation):
    """Return both the reporting table and the original best parameter types."""
    results = []
    for parameters in ParameterGrid(RF_PARAMETER_GRID):
        model = RandomForestClassifier(
            n_estimators=N_TREES,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **parameters,
        )
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        results.append(
            {
                **parameters,
                "Validation_AP": average_precision_score(
                    y_validation, probabilities
                ),
                "Validation_ROC_AUC": roc_auc_score(
                    y_validation, probabilities
                ),
            }
        )

    best_result = max(results, key=lambda row: row["Validation_AP"])
    best_parameters = {
        name: best_result[name]
        for name in ["max_depth", "min_samples_leaf", "max_features"]
    }
    results_frame = pd.DataFrame(results).sort_values(
        "Validation_AP", ascending=False, kind="stable"
    ).reset_index(drop=True)
    return results_frame, best_parameters


def threshold_table(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )
    precision = precision[:-1]
    recall = recall[:-1]
    f1_values = 2 * precision * recall / (precision + recall + 1e-10)
    table = pd.DataFrame(
        {
            "Threshold": thresholds,
            "Precision": precision,
            "Recall": recall,
            "F1": f1_values,
        }
    )
    best_index = table["F1"].idxmax()
    return table, float(table.loc[best_index, "Threshold"])


def trapezoidal_pr_auc(y_true, probabilities):
    """Compute PR-AUC by integration; this is distinct from average precision."""
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    return auc(recall, precision)


def evaluate_predictions(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    metrics = {
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": trapezoidal_pr_auc(y_true, probabilities),
        "Average_Precision": average_precision_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, predictions, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }
    return metrics, predictions


def validation_result(model_name, y_true, probabilities, selected_threshold):
    metrics, _ = evaluate_predictions(
        y_true, probabilities, selected_threshold
    )
    return {
        "Model": model_name,
        "PR_AUC": metrics["PR_AUC"],
        "Average_Precision": metrics["Average_Precision"],
        "ROC_AUC": metrics["ROC_AUC"],
        "Selected_Threshold": selected_threshold,
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
    }


def threshold_comparison_row(
    model_name, y_true, probabilities, threshold, threshold_type
):
    metrics, _ = evaluate_predictions(y_true, probabilities, threshold)
    return {
        "Model": model_name,
        "Threshold_Type": threshold_type,
        "Threshold": threshold,
        **{
            name: metrics[name]
            for name in ["Precision", "Recall", "F1", "TN", "FP", "FN", "TP"]
        },
    }


def print_evaluation(model_name, sample_name, y_true, predictions, metrics):
    print(f"\n{model_name} — {sample_name.upper()} RESULTS")
    print("-" * 60)
    print("Confusion matrix:\n", confusion_matrix(y_true, predictions, labels=[0, 1]))
    print(
        "Classification report:\n",
        classification_report(
            y_true, predictions, digits=4, zero_division=0
        ),
    )
    for metric in [
        "ROC_AUC",
        "PR_AUC",
        "Average_Precision",
        "Balanced_Accuracy",
        "Specificity",
    ]:
        print(f"{metric}: {metrics[metric]:.6f}")


# Interpretation and diagnostics
def logistic_coefficient_table(model):
    table = pd.DataFrame(
        {"Feature": FEATURES, "Coefficient": model.coef_[0]}
    )
    table["Absolute_Coefficient"] = table["Coefficient"].abs()
    table["Odds_Ratio"] = np.exp(table["Coefficient"])
    table["Expected_Sign"] = table["Feature"].map(EXPECTED_SIGNS)
    table["Observed_Sign"] = np.select(
        [table["Coefficient"] > 0, table["Coefficient"] < 0],
        ["Positive", "Negative"],
        default="Zero",
    )
    table["Sign_Agreement"] = table["Expected_Sign"].eq(
        table["Observed_Sign"]
    )
    table["Direction"] = np.where(
        table["Coefficient"] > 0,
        "Increases predicted distress",
        "Decreases predicted distress",
    )
    return table.sort_values("Absolute_Coefficient", ascending=False).reset_index(
        drop=True
    )


def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "."
    return ""


def clustered_logistic_inference(X_train, y_train, training_tickers):
    """Estimate supplementary unweighted, unpenalised firm-clustered inference.

    This auxiliary model supports coefficient interpretation only. It does not
    replace the class-weighted, L2-regularised predictive model.
    """
    model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        class_weight=None,
        random_state=RANDOM_SEED,
        max_iter=10_000,
    )
    model.fit(X_train, y_train)

    design = np.column_stack([np.ones(len(X_train)), X_train.to_numpy()])
    parameter_names = ["Intercept", *FEATURES]
    coefficients = np.concatenate([model.intercept_, model.coef_[0]])
    probabilities = model.predict_proba(X_train)[:, 1]
    y_array = y_train.to_numpy(dtype=float)

    weights = np.clip(probabilities * (1 - probabilities), 1e-12, None)
    information = design.T @ (design * weights[:, None])
    bread = np.linalg.pinv(information)
    conventional_se = np.sqrt(np.clip(np.diag(bread), 0, None))
    conventional_z = coefficients / conventional_se
    conventional_p = 2 * norm.sf(np.abs(conventional_z))

    groups = training_tickers.astype(str).to_numpy()
    unique_groups = np.unique(groups)
    score_residuals = y_array - probabilities
    cluster_meat = np.zeros_like(information)
    for group_name in unique_groups:
        group_score = design[groups == group_name].T @ score_residuals[
            groups == group_name
        ]
        cluster_meat += np.outer(group_score, group_score)

    n_observations, n_parameters = design.shape
    n_clusters = len(unique_groups)
    if n_clusters <= 1 or n_observations <= n_parameters:
        raise ValueError("Insufficient observations or firms for clustered inference.")

    correction = (
        n_clusters
        / (n_clusters - 1)
        * (n_observations - 1)
        / (n_observations - n_parameters)
    )
    cluster_covariance = correction * bread @ cluster_meat @ bread
    cluster_se = np.sqrt(np.clip(np.diag(cluster_covariance), 0, None))
    cluster_z = coefficients / cluster_se
    cluster_p = 2 * norm.sf(np.abs(cluster_z))
    critical_value = norm.ppf(0.975)
    cluster_ci_lower = coefficients - critical_value * cluster_se
    cluster_ci_upper = coefficients + critical_value * cluster_se

    coefficient_results = pd.DataFrame(
        {
            "Term": parameter_names,
            "Coefficient": coefficients,
            "Odds_Ratio": np.exp(coefficients),
            "Conventional_SE": conventional_se,
            "Conventional_Z": conventional_z,
            "Conventional_P_Value": conventional_p,
            "Cluster_Robust_SE": cluster_se,
            "Cluster_Robust_Z": cluster_z,
            "Cluster_Robust_P_Value": cluster_p,
            "Cluster_Robust_CI_Lower": cluster_ci_lower,
            "Cluster_Robust_CI_Upper": cluster_ci_upper,
            "Odds_Ratio_CI_Lower": np.exp(cluster_ci_lower),
            "Odds_Ratio_CI_Upper": np.exp(cluster_ci_upper),
            "Significance": [significance_stars(value) for value in cluster_p],
        }
    )

    epsilon = 1e-15
    fitted = np.clip(probabilities, epsilon, 1 - epsilon)
    full_log_likelihood = np.sum(
        y_array * np.log(fitted) + (1 - y_array) * np.log(1 - fitted)
    )
    null_probability = np.clip(y_array.mean(), epsilon, 1 - epsilon)
    null_log_likelihood = np.sum(
        y_array * np.log(null_probability)
        + (1 - y_array) * np.log(1 - null_probability)
    )
    lr_statistic = 2 * (full_log_likelihood - null_log_likelihood)
    lr_degrees_of_freedom = len(FEATURES)
    distressed_observations = int(y_train.sum())

    model_results = pd.DataFrame(
        {
            "Number_of_Observations": [n_observations],
            "Number_of_Firm_Clusters": [n_clusters],
            "Distressed_Observations": [distressed_observations],
            "Events_Per_Predictor": [distressed_observations / len(FEATURES)],
            "Full_Log_Likelihood": [full_log_likelihood],
            "Null_Log_Likelihood": [null_log_likelihood],
            "Likelihood_Ratio_Statistic": [lr_statistic],
            "LR_Degrees_of_Freedom": [lr_degrees_of_freedom],
            "Likelihood_Ratio_P_Value": [
                chi2.sf(lr_statistic, lr_degrees_of_freedom)
            ],
            "McFadden_Pseudo_R_Squared": [
                1 - full_log_likelihood / null_log_likelihood
            ],
        }
    )
    return coefficient_results, model_results


def calculate_vif(X_train_scaled):
    results = []
    for feature_name in FEATURES:
        other_features = [name for name in FEATURES if name != feature_name]
        regression = LinearRegression().fit(
            X_train_scaled[other_features], X_train_scaled[feature_name]
        )
        r_squared = regression.score(
            X_train_scaled[other_features], X_train_scaled[feature_name]
        )
        results.append(
            {
                "Feature": feature_name,
                "R_Squared": r_squared,
                "VIF": np.inf if r_squared >= 1 else 1 / (1 - r_squared),
            }
        )
    return pd.DataFrame(results).sort_values("VIF", ascending=False)


def univariate_logistic_analysis(
    X_train, y_train, X_validation, y_validation, selected_c
):
    results = []
    for feature_name in FEATURES:
        model = LogisticRegression(
            C=selected_c,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=1_000,
        )
        model.fit(X_train[[feature_name]], y_train)
        probability = model.predict_proba(X_validation[[feature_name]])[:, 1]
        coefficient = model.coef_[0, 0]
        results.append(
            {
                "Feature": feature_name,
                "Univariate_Coefficient": coefficient,
                "Univariate_Odds_Ratio": np.exp(coefficient),
                "Validation_AP": average_precision_score(
                    y_validation, probability
                ),
            }
        )
    return pd.DataFrame(results).sort_values("Validation_AP", ascending=False)


# Clustered bootstrap
BOOTSTRAP_METRICS = [
    "ROC_AUC",
    "AP",
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
]


def calculate_bootstrap_metrics(y_true, probabilities, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "AP": average_precision_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1": f1_score(y_true, predictions, zero_division=0),
    }


def ticker_clustered_bootstrap(
    y_test,
    test_tickers,
    logistic_probabilities,
    logistic_predictions,
    rf_probabilities,
    rf_predictions,
):
    """Resample firms with replacement while retaining all their firm-years."""
    y_array = np.asarray(y_test)
    ticker_array = test_tickers.astype(str).to_numpy()
    logistic_probability_array = np.asarray(logistic_probabilities)
    logistic_prediction_array = np.asarray(logistic_predictions)
    rf_probability_array = np.asarray(rf_probabilities)
    rf_prediction_array = np.asarray(rf_predictions)

    if test_tickers.isna().any() or len(ticker_array) != len(y_array):
        raise ValueError("Test tickers are missing or misaligned with predictions.")

    unique_tickers = np.unique(ticker_array)
    n_firms = len(unique_tickers)
    random_generator = np.random.default_rng(RANDOM_SEED)
    maximum_attempts = N_BOOTSTRAP_REPLICATIONS * 100

    logistic_results = []
    rf_results = []
    difference_results = []
    attempts = 0

    while len(logistic_results) < N_BOOTSTRAP_REPLICATIONS:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError("Unable to generate enough valid bootstrap samples.")

        sampled_tickers = random_generator.choice(
            unique_tickers, size=n_firms, replace=True
        )
        indices = np.concatenate(
            [np.flatnonzero(ticker_array == ticker) for ticker in sampled_tickers]
        )
        # Retain the original random-number sequence for exact reproducibility
        # of the reported bootstrap intervals.
        random_generator.shuffle(indices)
        bootstrap_y = y_array[indices]
        if np.unique(bootstrap_y).size < 2:
            continue

        logistic_metrics = calculate_bootstrap_metrics(
            bootstrap_y,
            logistic_probability_array[indices],
            logistic_prediction_array[indices],
        )
        rf_metrics = calculate_bootstrap_metrics(
            bootstrap_y,
            rf_probability_array[indices],
            rf_prediction_array[indices],
        )
        logistic_results.append(logistic_metrics)
        rf_results.append(rf_metrics)
        difference_results.append(
            {
                metric: rf_metrics[metric] - logistic_metrics[metric]
                for metric in BOOTSTRAP_METRICS
            }
        )

    return (
        pd.DataFrame(logistic_results),
        pd.DataFrame(rf_results),
        pd.DataFrame(difference_results),
        n_firms,
        attempts,
    )


def summarise_bootstrap(bootstrap_frame, model_name, n_firms):
    rows = []
    for metric_name in BOOTSTRAP_METRICS:
        values = bootstrap_frame[metric_name].dropna()
        rows.append(
            {
                "Bootstrap_Method": "Ticker-clustered percentile bootstrap",
                "Number_of_Test_Firms": n_firms,
                "Number_of_Replications": N_BOOTSTRAP_REPLICATIONS,
                "Model": model_name,
                "Metric": metric_name,
                "Bootstrap_Mean": values.mean(),
                "CI_Lower_2.5%": values.quantile(0.025),
                "CI_Upper_97.5%": values.quantile(0.975),
            }
        )
    return pd.DataFrame(rows)


def summarise_paired_differences(difference_frame, n_firms):
    rows = []
    for metric_name in BOOTSTRAP_METRICS:
        values = difference_frame[metric_name].dropna()
        rows.append(
            {
                "Bootstrap_Method": "Ticker-clustered percentile bootstrap",
                "Number_of_Test_Firms": n_firms,
                "Number_of_Replications": N_BOOTSTRAP_REPLICATIONS,
                "Metric": metric_name,
                "RF_minus_Logistic_Mean": values.mean(),
                "CI_Lower_2.5%": values.quantile(0.025),
                "CI_Upper_97.5%": values.quantile(0.975),
            }
        )
    return pd.DataFrame(rows)


def publication_bootstrap_table(
    logistic_test_metrics,
    rf_test_metrics,
    logistic_intervals,
    rf_intervals,
    difference_intervals,
    n_firms,
):
    result_keys = {
        "ROC_AUC": "ROC_AUC",
        "AP": "Average_Precision",
        "Accuracy": "Accuracy",
        "Balanced_Accuracy": "Balanced_Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "Specificity": "Specificity",
        "F1": "F1",
    }
    logistic_lookup = logistic_intervals.set_index("Metric")
    rf_lookup = rf_intervals.set_index("Metric")
    difference_lookup = difference_intervals.set_index("Metric")

    rows = []
    for bootstrap_metric, result_key in result_keys.items():
        logistic_estimate = logistic_test_metrics[result_key]
        rf_estimate = rf_test_metrics[result_key]
        rows.append(
            {
                "Bootstrap_Method": "Ticker-clustered percentile bootstrap",
                "Number_of_Test_Firms": n_firms,
                "Number_of_Replications": N_BOOTSTRAP_REPLICATIONS,
                "Metric": bootstrap_metric,
                "LR_Test_Estimate": logistic_estimate,
                "LR_CI_Lower_2.5%": logistic_lookup.loc[
                    bootstrap_metric, "CI_Lower_2.5%"
                ],
                "LR_CI_Upper_97.5%": logistic_lookup.loc[
                    bootstrap_metric, "CI_Upper_97.5%"
                ],
                "RF_Test_Estimate": rf_estimate,
                "RF_CI_Lower_2.5%": rf_lookup.loc[
                    bootstrap_metric, "CI_Lower_2.5%"
                ],
                "RF_CI_Upper_97.5%": rf_lookup.loc[
                    bootstrap_metric, "CI_Upper_97.5%"
                ],
                "RF_Minus_LR_Test_Difference": rf_estimate - logistic_estimate,
                "Difference_CI_Lower_2.5%": difference_lookup.loc[
                    bootstrap_metric, "CI_Lower_2.5%"
                ],
                "Difference_CI_Upper_97.5%": difference_lookup.loc[
                    bootstrap_metric, "CI_Upper_97.5%"
                ],
            }
        )
    return pd.DataFrame(rows)


# Figures
def plot_test_curves(
    output_folder,
    y_test,
    logistic_probabilities,
    rf_probabilities,
    logistic_metrics,
    rf_metrics,
):
    logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_probabilities)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(
        logistic_fpr,
        logistic_tpr,
        linewidth=2,
        label=f"Logistic Regression (AUC = {logistic_metrics['ROC_AUC']:.3f})",
    )
    plt.plot(
        rf_fpr,
        rf_tpr,
        linewidth=2,
        label=f"Random Forest (AUC = {rf_metrics['ROC_AUC']:.3f})",
    )
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on the Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "test_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    logistic_precision, logistic_recall, _ = precision_recall_curve(
        y_test, logistic_probabilities
    )
    rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probabilities)
    distress_rate = y_test.mean()

    plt.figure(figsize=(8, 6))
    plt.plot(
        logistic_recall,
        logistic_precision,
        linewidth=2,
        label=(
            "Logistic Regression "
            f"(AP = {logistic_metrics['Average_Precision']:.3f})"
        ),
    )
    plt.plot(
        rf_recall,
        rf_precision,
        linewidth=2,
        label=f"Random Forest (AP = {rf_metrics['Average_Precision']:.3f})",
    )
    plt.axhline(
        distress_rate,
        linestyle="--",
        color="grey",
        label=f"Random baseline ({distress_rate:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves on the Test Set")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_folder / "test_precision_recall_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_confusion_matrices(
    output_folder,
    y_test,
    logistic_predictions,
    rf_predictions,
    logistic_threshold,
    rf_threshold,
):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        logistic_predictions,
        labels=[0, 1],
        display_labels=["Non-distress", "Distress"],
        cmap="Blues",
        colorbar=False,
        ax=axes[0],
    )
    axes[0].set_title(
        f"Logistic Regression\nThreshold = {logistic_threshold:.3f}"
    )
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        rf_predictions,
        labels=[0, 1],
        display_labels=["Non-distress", "Distress"],
        cmap="Oranges",
        colorbar=False,
        ax=axes[1],
    )
    axes[1].set_title(f"Random Forest\nThreshold = {rf_threshold:.3f}")
    figure.suptitle("Confusion Matrices on the Test Set", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        output_folder / "test_confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_model_interpretation(
    output_folder, logistic_coefficients, rf_permutation_importance
):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    logistic_plot = logistic_coefficients.sort_values("Coefficient")
    colours = np.where(
        logistic_plot["Coefficient"] > 0, "firebrick", "steelblue"
    )
    axes[0].barh(
        logistic_plot["Feature"], logistic_plot["Coefficient"], color=colours
    )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Logistic Regression Coefficients")
    axes[0].set_xlabel("Standardised coefficient")
    axes[0].set_ylabel("")

    rf_plot = rf_permutation_importance.sort_values("Mean_AP_Decrease")
    axes[1].barh(
        rf_plot["Feature"],
        rf_plot["Mean_AP_Decrease"],
        xerr=rf_plot["Standard_Deviation"],
        color="darkorange",
        alpha=0.85,
        capsize=3,
    )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Random Forest Permutation Importance")
    axes[1].set_xlabel("Mean decrease in average precision")
    axes[1].set_ylabel("")

    figure.suptitle("Model Interpretation", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        output_folder / "model_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def calibration_analysis(
    output_folder, y_test, logistic_probabilities, rf_probabilities, y_train
):
    logistic_brier = brier_score_loss(y_test, logistic_probabilities)
    rf_brier = brier_score_loss(y_test, rf_probabilities)
    baseline_probability = y_train.mean()
    baseline_brier = brier_score_loss(
        y_test, np.full(len(y_test), baseline_probability)
    )

    brier_results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "Training-prevalence baseline",
            ],
            "Test_Set_Brier_Score": [
                logistic_brier,
                rf_brier,
                baseline_brier,
            ],
            "Fixed_Predicted_Probability": [
                np.nan,
                np.nan,
                baseline_probability,
            ],
        }
    )

    logistic_observed, logistic_predicted = calibration_curve(
        y_test, logistic_probabilities, n_bins=10, strategy="quantile"
    )
    rf_observed, rf_predicted = calibration_curve(
        y_test, rf_probabilities, n_bins=10, strategy="quantile"
    )
    logistic_calibration = pd.DataFrame(
        {
            "Mean_Predicted_Probability": logistic_predicted,
            "Observed_Distress_Rate": logistic_observed,
        }
    )
    rf_calibration = pd.DataFrame(
        {
            "Mean_Predicted_Probability": rf_predicted,
            "Observed_Distress_Rate": rf_observed,
        }
    )

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    plt.plot(
        logistic_predicted,
        logistic_observed,
        marker="o",
        linewidth=2,
        label=f"Logistic Regression (Brier = {logistic_brier:.3f})",
    )
    plt.plot(
        rf_predicted,
        rf_observed,
        marker="s",
        linewidth=2,
        label=f"Random Forest (Brier = {rf_brier:.3f})",
    )
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Distress Rate")
    plt.title("Calibration Curves on the Test Set")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_folder / "test_calibration_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return brier_results, logistic_calibration, rf_calibration


# Main analysis
def main():
    project_folder = Path(__file__).resolve().parent
    input_path = project_folder / "clear_panel_with_one_year_fd_labels.xlsx"
    output_folder = project_folder / "outputs"
    output_folder.mkdir(exist_ok=True)

    model_data = load_model_data(input_path)
    samples = split_by_target_year(model_data)
    for name, sample in samples.items():
        print_sample_summary(name, sample)

    labels = {name: sample[TARGET].copy() for name, sample in samples.items()}
    imputed, scaled, preprocessing_parameters = preprocess_samples(samples)
    preprocessing_parameters.to_csv(
        output_folder / "preprocessing_parameters.csv"
    )

    descriptive_statistics = build_descriptive_statistics(imputed, samples)
    descriptive_statistics.to_csv(
        output_folder / "chapter4_descriptive_statistics.csv", index=False
    )

    logistic_tuning, best_logistic_c = tune_logistic_regression(
        scaled["Training"],
        labels["Training"],
        scaled["Validation"],
        labels["Validation"],
    )
    rf_tuning, best_rf_parameters = tune_random_forest(
        imputed["Training"],
        labels["Training"],
        imputed["Validation"],
        labels["Validation"],
    )
    logistic_tuning.to_csv(
        output_folder / "logistic_tuning_results.csv", index=False
    )
    rf_tuning.to_csv(
        output_folder / "random_forest_tuning_results.csv", index=False
    )

    print("\nSelected Logistic C:", best_logistic_c)
    print("Selected Random Forest parameters:", best_rf_parameters)

    # The selected validation results now flow directly into the final models.
    logistic_model = LogisticRegression(
        C=best_logistic_c,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=1_000,
    )
    rf_model = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **best_rf_parameters,
    )
    logistic_model.fit(scaled["Training"], labels["Training"])
    rf_model.fit(imputed["Training"], labels["Training"])

    logistic_validation_probabilities = logistic_model.predict_proba(
        scaled["Validation"]
    )[:, 1]
    rf_validation_probabilities = rf_model.predict_proba(
        imputed["Validation"]
    )[:, 1]
    logistic_thresholds, logistic_threshold = threshold_table(
        labels["Validation"], logistic_validation_probabilities
    )
    rf_thresholds, rf_threshold = threshold_table(
        labels["Validation"], rf_validation_probabilities
    )
    logistic_thresholds.to_csv(
        output_folder / "logistic_validation_thresholds.csv", index=False
    )
    rf_thresholds.to_csv(
        output_folder / "random_forest_validation_thresholds.csv", index=False
    )

    validation_comparison = pd.DataFrame(
        [
            validation_result(
                "Logistic Regression",
                labels["Validation"],
                logistic_validation_probabilities,
                logistic_threshold,
            ),
            validation_result(
                "Random Forest",
                labels["Validation"],
                rf_validation_probabilities,
                rf_threshold,
            ),
        ]
    )
    validation_comparison.to_csv(
        output_folder / "final_validation_model_comparison.csv", index=False
    )

    validation_threshold_comparison = pd.DataFrame(
        [
            threshold_comparison_row(
                "Logistic Regression",
                labels["Validation"],
                logistic_validation_probabilities,
                0.50,
                "Conventional 0.50",
            ),
            threshold_comparison_row(
                "Logistic Regression",
                labels["Validation"],
                logistic_validation_probabilities,
                logistic_threshold,
                "Selected on validation F1",
            ),
            threshold_comparison_row(
                "Random Forest",
                labels["Validation"],
                rf_validation_probabilities,
                0.50,
                "Conventional 0.50",
            ),
            threshold_comparison_row(
                "Random Forest",
                labels["Validation"],
                rf_validation_probabilities,
                rf_threshold,
                "Selected on validation F1",
            ),
        ]
    )
    validation_threshold_comparison.to_csv(
        output_folder / "final_tuned_validation_threshold_comparison.csv",
        index=False,
    )

    logistic_test_probabilities = logistic_model.predict_proba(
        scaled["Test"]
    )[:, 1]
    rf_test_probabilities = rf_model.predict_proba(imputed["Test"])[:, 1]
    logistic_test_metrics, logistic_test_predictions = evaluate_predictions(
        labels["Test"], logistic_test_probabilities, logistic_threshold
    )
    rf_test_metrics, rf_test_predictions = evaluate_predictions(
        labels["Test"], rf_test_probabilities, rf_threshold
    )
    print_evaluation(
        "Logistic Regression",
        "final test",
        labels["Test"],
        logistic_test_predictions,
        logistic_test_metrics,
    )
    print_evaluation(
        "Random Forest",
        "final test",
        labels["Test"],
        rf_test_predictions,
        rf_test_metrics,
    )

    test_comparison = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "Fixed_Threshold": logistic_threshold,
                **logistic_test_metrics,
            },
            {
                "Model": "Random Forest",
                "Fixed_Threshold": rf_threshold,
                **rf_test_metrics,
            },
        ]
    )
    test_comparison.to_csv(
        output_folder / "final_test_model_comparison.csv", index=False
    )

    test_predictions = samples["Test"][
        ["Ticker", "Year", "Target_Year", TARGET]
    ].copy()
    test_predictions["Logistic_Probability"] = logistic_test_probabilities
    test_predictions["Logistic_Prediction"] = logistic_test_predictions
    test_predictions["Random_Forest_Probability"] = rf_test_probabilities
    test_predictions["Random_Forest_Prediction"] = rf_test_predictions
    test_predictions.to_csv(
        output_folder / "final_test_predictions.csv", index=False
    )

    plot_test_curves(
        output_folder,
        labels["Test"],
        logistic_test_probabilities,
        rf_test_probabilities,
        logistic_test_metrics,
        rf_test_metrics,
    )
    plot_confusion_matrices(
        output_folder,
        labels["Test"],
        logistic_test_predictions,
        rf_test_predictions,
        logistic_threshold,
        rf_threshold,
    )

    logistic_coefficients = logistic_coefficient_table(logistic_model)
    inference_coefficients, inference_model_test = clustered_logistic_inference(
        scaled["Training"],
        labels["Training"],
        samples["Training"].loc[scaled["Training"].index, "Ticker"],
    )
    logistic_coefficients.to_csv(
        output_folder / "logistic_coefficients.csv", index=False
    )
    inference_coefficients.to_csv(
        output_folder / "logistic_inference_cluster_robust.csv", index=False
    )
    inference_model_test.to_csv(
        output_folder / "logistic_inference_model_test.csv", index=False
    )

    rf_impurity_importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Impurity_Importance": rf_model.feature_importances_,
        }
    ).sort_values("Impurity_Importance", ascending=False)

    # Test-set permutation importance is descriptive only and is not used to
    # alter model parameters, thresholds, or predictor selection.
    rf_permutation_result = permutation_importance(
        rf_model,
        imputed["Test"],
        labels["Test"],
        scoring="average_precision",
        n_repeats=50,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_permutation_importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Mean_AP_Decrease": rf_permutation_result.importances_mean,
            "Standard_Deviation": rf_permutation_result.importances_std,
        }
    ).sort_values("Mean_AP_Decrease", ascending=False)
    rf_impurity_importance.to_csv(
        output_folder / "rf_impurity_importance.csv", index=False
    )
    rf_permutation_importance.to_csv(
        output_folder / "rf_permutation_importance.csv", index=False
    )
    plot_model_interpretation(
        output_folder, logistic_coefficients, rf_permutation_importance
    )

    training_correlations = imputed["Training"].corr()
    training_correlations.to_csv(
        output_folder / "training_feature_correlations.csv"
    )
    calculate_vif(scaled["Training"]).to_csv(
        output_folder / "vif_results.csv", index=False
    )
    univariate_logistic_analysis(
        scaled["Training"],
        labels["Training"],
        scaled["Validation"],
        labels["Validation"],
        best_logistic_c,
    ).to_csv(output_folder / "univariate_logistic_results.csv", index=False)

    brier_results, logistic_calibration, rf_calibration = calibration_analysis(
        output_folder,
        labels["Test"],
        logistic_test_probabilities,
        rf_test_probabilities,
        labels["Training"],
    )
    brier_results.to_csv(
        output_folder / "test_brier_score_comparison.csv", index=False
    )
    logistic_calibration.to_csv(
        output_folder / "logistic_calibration.csv", index=False
    )
    rf_calibration.to_csv(output_folder / "rf_calibration.csv", index=False)

    (
        logistic_bootstrap,
        rf_bootstrap,
        paired_differences,
        n_test_firms,
        bootstrap_attempts,
    ) = ticker_clustered_bootstrap(
        labels["Test"],
        samples["Test"].loc[labels["Test"].index, "Ticker"],
        logistic_test_probabilities,
        logistic_test_predictions,
        rf_test_probabilities,
        rf_test_predictions,
    )
    logistic_intervals = summarise_bootstrap(
        logistic_bootstrap, "Logistic Regression", n_test_firms
    )
    rf_intervals = summarise_bootstrap(
        rf_bootstrap, "Random Forest", n_test_firms
    )
    test_intervals = pd.concat(
        [logistic_intervals, rf_intervals], ignore_index=True
    )
    difference_intervals = summarise_paired_differences(
        paired_differences, n_test_firms
    )
    publication_summary = publication_bootstrap_table(
        logistic_test_metrics,
        rf_test_metrics,
        logistic_intervals,
        rf_intervals,
        difference_intervals,
        n_test_firms,
    )
    test_intervals.to_csv(
        output_folder / "test_metric_confidence_intervals.csv", index=False
    )
    difference_intervals.to_csv(
        output_folder / "paired_model_differences.csv", index=False
    )
    publication_summary.to_csv(
        output_folder / "chapter4_bootstrap_summary.csv", index=False
    )

    print("\nAnalysis completed successfully.")
    print("Output folder:", output_folder)
    print("Valid clustered-bootstrap replications:", len(logistic_bootstrap))
    print("Clustered-bootstrap sampling attempts:", bootstrap_attempts)


if __name__ == "__main__":
    main()
