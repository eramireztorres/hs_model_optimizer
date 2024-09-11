class ModelSelector:
    def __init__(self):
        """
        Initialize the ModelSelector class.
        """
        self.best_model = None
        self.best_metrics = None

    def select_best_model(self, models, metrics_history):
        """
        Select the best model based on the evaluation metrics (F1 score as the primary metric).

        Args:
            models (list): List of trained models.
            metrics_history (list): List of dictionaries containing metrics for each model.

        Returns:
            object: The best model based on F1 score.
            dict: The metrics of the best model.
        """
        best_f1_score = -1
        best_index = -1

        for index, metrics in enumerate(metrics_history):
            if metrics['f1_score'] > best_f1_score:
                best_f1_score = metrics['f1_score']
                best_index = index

        self.best_model = models[best_index]
        self.best_metrics = metrics_history[best_index]

        return self.best_model, self.best_metrics
