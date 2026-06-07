from .train_cnn import train as train_cnn, evaluate as evaluate_cnn
from .train_gnn import train as train_gnn, evaluate as evaluate_gnn

__all__ = ["train_cnn", "evaluate_cnn", "train_gnn", "evaluate_gnn"]
