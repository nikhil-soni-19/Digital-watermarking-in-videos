# Define the EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        """
        Early stopping to stop training when validation loss doesn't improve.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in monitored value to qualify as improvement.
            verbose (bool): If True, prints a message when early stopping is triggered.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, encoder, decoder):
        """
        Check if training should be stopped.

        Args:
            val_loss (float): Current validation loss
            encoder (nn.Module): Encoder model
            decoder (nn.Module): Decoder model

        Returns:
            bool: True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            # Save best model state
            self.best_model_state = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict()
            }
            if self.verbose:
                print(f"Validation loss decreased to {val_loss:.4f}. Saving model...")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")

        return self.early_stop

    def load_best_model(self, encoder, decoder):
        """
        Load the best model states saved during training.

        Args:
            encoder (nn.Module): Encoder model
            decoder (nn.Module): Decoder model
        """
        if self.best_model_state is not None:
            encoder.load_state_dict(self.best_model_state['encoder'])
            decoder.load_state_dict(self.best_model_state['decoder'])
            if self.verbose:
                print("Loaded best model states")
