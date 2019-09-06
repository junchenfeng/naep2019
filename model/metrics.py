from sklearn import metrics


def scorer(estimator, X, y):
    prob_pred = estimator.predict_proba(X)[:, 1]
    y_pred = estimator.predict(X)
    fpr, tpr, thresholds = metrics.roc_curve(y, prob_pred)
    auc = metrics.auc(fpr, tpr)
    kappa = metrics.cohen_kappa_score(y, y_pred)
    adj_auc = 2 * (auc - 0.5) if auc > 0.5 else 0
    adj_kappa = max(kappa, 0)
    return adj_auc + adj_kappa
