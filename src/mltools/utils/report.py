import pickle

class Report():
    """
    Class for storing objects and their attributes and saving it to a file."""
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.data = {}
        self.verbose = verbose

    def add(self, key_chain: str | list[str], value):
        """Add a key-value pair to the report."""
        if self.verbose:
            print(f"Adding {key_chain} : {value} to report {self.name}")
        if isinstance(key_chain, list):
            subdict = self.data
            for k in key_chain[:-1]:
                if subdict.get(k) is None:
                    subdict[k] = {}
                subdict = subdict[k]
            subdict[key_chain[-1]] = value
        else:
            self.data[key_chain] = value

    def save(self, file_path: str):
        """Save the report to a file."""
        if self.verbose:
            print(f"Saving report {self.name} to {file_path}")
        with open(file_path, 'wb') as f:
            pickle.dump(self.data, f)