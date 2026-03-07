# Federated Multi-City Transport Mode Detection

In federated multi-city transport mode detection, each city has a different local data distribution and often different class imbalance. 
Because of this, optimizing standard cross-entropy may not align well with the metric of interest, especially for minority transport modes.
Since AUPRC/AP is more appropriate for imbalanced classification, we propose to replace the usual cross-entropy objective with a PR-oriented surrogate. 
However, the SOAP method is formulated for binary classification, so we adapt it to the multiclass setting through one-vs-rest binarization: for each transport class, we treat that class as positive and all other classes as negative, optimize a SOAP-like loss locally on each city, and aggregate the model updates in a federated learning framework. We then evaluate the global model using per-class AUPRC and macro-AUPRC, rather than relying only on accuracy.
