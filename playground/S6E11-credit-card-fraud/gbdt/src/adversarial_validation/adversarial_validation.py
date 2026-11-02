import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


# TODO: add auto_merge if threshold looks good
def adversarial_validation(
        datasets: dict,
        target: str,
        folds: int = 5,
        seed: int = 42,
        print_details: bool = False,
        ) -> float:
    plt.figure(figsize=(8, 6))

    av_train = datasets["train"].copy()
    av_extra = datasets["extra"].copy()

    av_train.drop(target, axis=1, inplace=True)
    av_extra.drop(target, axis=1, inplace=True)

    av_train["av_label"] = 1
    av_extra["av_label"] = 0

    adversarial_df = pd.concat([
      av_train,
      av_extra
    ], axis=0, ignore_index=True)

    adversarial_df = adversarial_df.sample(frac=1, random_state=seed)

    estimator = LGBMClassifier(
        objective="binary",
        verbose=-1
        )

    X = adversarial_df.copy()
    y = X.pop("av_label")

    enc = OrdinalEncoder()
    cat_cols = X.copy().select_dtypes("object").columns
    X[cat_cols] = pd.DataFrame(enc.fit_transform(X[cat_cols].copy()), columns=cat_cols)

    bool_cols = X.copy().select_dtypes("bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)

    fold = []
    fprs = []
    tprs = []
    aucs = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        estimator.fit(X_train, y_train)
        y_hat = estimator.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_hat)
        roc_auc = auc(fpr, tpr)

        fold.append(i)
        tprs.append(tpr)
        fprs.append(fpr)
        aucs.append(roc_auc)

        plt.plot(fpr, tpr, lw=1, alpha=0.7, label=f'Fold {i+1} ROC curve (AUC = {roc_auc:.2f})')

    av_results = pd.DataFrame({
        'Fold': fold,
        'AUC': aucs
    })

    if print_details:
        plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, label='Chance')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC AUC Curves from Adversarial Validation')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

        print("\n-- Feature Importance --")
        fig, ax = plt.subplots(figsize=(12, 6))
        from lightgbm import plot_importance
        plot_importance(estimator, ax=ax)
        plt.show()

        print("\n-- Overall Results --")
        display(av_results)

    print(f"\n Mean AUC score: {av_results['AUC'].mean():.2f}")
    return av_results['AUC'].mean()
