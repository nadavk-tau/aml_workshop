import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay

from sklearn.model_selection import StratifiedKFold


def _draw_roc_curve_for_mutation(mutation_name, classifier_runner, fig_path, cv=None):
    cv = cv if cv else StratifiedKFold(n_splits=6)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    y = classifier_runner.y[mutation_name]
    X = classifier_runner.X

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier_runner.pipeline.fit(X.iloc[train], y[train])
        viz = RocCurveDisplay.from_estimator(classifier_runner.pipeline, X.iloc[test], y[test],
                                             name="ROC fold {}".format(i),alpha=0.3, lw=1, ax=ax,)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), lw=2, alpha=0.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey",
                    alpha=0.2, label=r"$\pm$ 1 std. dev.")

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"Mutation {mutation_name}",
    )
    ax.legend(loc="lower right")

    plt.savefig(fig_path)


# Based on: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
def _draw_pr_curve_for_mutation(mutation_name, classifier_runner, fig_path, cv=None):
    cv = cv if cv else StratifiedKFold(n_splits=5)
    y = classifier_runner.y[mutation_name]
    X = classifier_runner.X

    f, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(X[y==0,0], X[y==0,1], color='blue', s=2, label='y=0')
    axes[0].scatter(X[y!=0,0], X[y!=0,1], color='red', s=2, label='y=1')
    axes[0].set_xlabel('X[:,0]')
    axes[0].set_ylabel('X[:,1]')
    axes[0].legend(loc='lower left', fontsize='small')

    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(cv.split(X)):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        classifier_runner.fit(Xtrain, ytrain)
        pred_proba = classifier_runner.predict_proba(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, pred_proba[:,1])
        lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
        axes[1].step(recall, precision, label=lab)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'Overall AUC=%.4f' % (auc(recall, precision))
    axes[1].step(recall, precision, label=lab, lw=2, color='black')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='lower left', fontsize='small')

    f.tight_layout()

    # fig, ax = plt.subplots()
    # for i, (train, test) in enumerate(cv.split(X, y)):
    #     classifier_runner.pipeline.fit(X.iloc[train], y[train])
    #     viz = PrecisionRecallDisplay.from_estimator(classifier_runner.pipeline, X.iloc[test], y[test],
    #                                                 name="PR fold {}".format(i), alpha=0.3, lw=1, ax=ax,)
    # ax.set(
    #     # xlim=[-0.05, 1.05],
    #     # ylim=[-0.05, 1.05],
    #     title=f"Mutation {mutation_name}",
    # )
    # ax.legend(loc="lower right")

    plt.savefig(fig_path)


def analyze_classifier_roc(classifier_runner, fig_path, cv=None):
    for mutation in classifier_runner.y.columns:
        _draw_roc_curve_for_mutation(mutation, classifier_runner, fig_path.replace("mutation_name", mutation), cv)


def analyze_classifier_pr(classifier_runner, fig_path, cv=None):
    for mutation in classifier_runner.y.columns:
        _draw_pr_curve_for_mutation(mutation, classifier_runner, fig_path.replace("mutation_name", mutation), cv)
