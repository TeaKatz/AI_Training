class CrossValidation:
    def __init__(self, metrics, k_folds=10, scaler=None):
        # Initial properties
        self.metrics = metrics
        self.k_folds = k_folds
        self.scaler = scaler
        self.scores = []

    def eval(self, model, x, y, **kwargs):
        # Divide training set into k folds
        kf = KFold(n_splits=self.k_folds)
        self.scores = []
        for i, (train_index, val_index) in enumerate(kf.split(x)):
            # Get validation fold
            val_x, val_y = x[val_index], y[val_index]

            # Get training fold
            train_x, train_y = x[train_index], y[train_index]

            # Normalization
            if scaler is not None:
                train_x = scaler.fit_transform(train_x)
                val_x = scaler.transform(val_x)

            # Train model on training set
            model.fit(train_x, train_y, **kwargs)

            # Evaluate model on validation set
            pred_y = model.predict(val_x)
            score = self.metrics(val_y.reshape(-1, 1), pred_y.reshape(-1, 1))

            # Save evaluation result
            self.scores.append(score)
        # Average all evaluation results
        mean_score = np.mean(self.scores)
        return mean_score
