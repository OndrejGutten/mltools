import pandas as pd

class Relabeller:
    def __init__(self, relabel_map_path):
        relabel_map = pd.read_csv(relabel_map_path)
        if relabel_map.shape[1] != 2:
            raise ValueError("Relabeller map must have exactly two columns. First column is the original label, second column is the new label.")
        self.relabel_map = dict(zip(relabel_map.iloc[:, 0], relabel_map.iloc[:, 1]))

    def predict(self, original_labels: list):
        return [self.relabel_map.get(label,-1) for label in original_labels]
