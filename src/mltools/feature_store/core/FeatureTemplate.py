from mltools.feature_store.utils import utils
from abc import abstractmethod
from mltools.feature_store.core import interface

'''
###
All classes derived from this class pick arguments listed in 'compute_args'. The list passed includes:
- dlznik_ids: List of debtor IDs for which the feature should be computed.
- pohladavky_dlznikov: DataFrame containing the debts of the debtors.
- solo_pohladavky_dlznikov: DataFrame containing the solo debts of the debtors (if applicable).
- reference_times: Timestamps indicating the time of reference for the computation.
- db_primary_engine: sqlalchemy.engine for the source database.
- feature_store_client: Database connector (client) for the target database.
- deriver: An instance of the Deriver class used for deriving specific values from the debts
###
'''

# NO FEATURE SHOULD RETURN NAN/NONE/NULL VALUES!

# FeatureTemplate is an ABSTRACT CLASS!
class FeatureTemplate(interface.FeatureDefinition):
    # compute_args is a list of arguments that are expected by the compute method
    compute_args = []  # subclasses must define this list - these are the arguments passed to the _compute method by parent (FeatureTemplate) compute method
    source_prerequisite_features = []  # features this feature depends on from the source database - will be used to check if any underlying data has changed
    prerequisite_features = []  # features this feature depends on from the target database - currently not used
    stale_after_n_days = None  # if set, the feature will be considered stale after this many days and recalculation will be triggered
    
    def compute(self, **kwargs):
        # Check required arguments
        missing = [arg for arg in self.compute_args if arg not in kwargs]
        if missing:
            raise ValueError(f"Missing required arguments: {missing}")

        # Call the subclass-specific logic
        kwargs_to_pass = {k: kwargs[k] for k in self.compute_args}

        if self._decide_parallelization(**kwargs):
            return utils.parallelize_call(self._compute, kwargs_to_pass, num_processes=8)
        else:
            return self._compute(**{k: kwargs[k] for k in self.compute_args})

    def _decide_parallelization(self, **kwargs):
        return False
        if 'dlznik_ids' in kwargs and len(kwargs['dlznik_ids']) > 1000 and hasattr(self, 'parallelize') and self.parallelize:
            # If there are more than 1000 debtor IDs, parallelize the computation
            print('Parallelizing computation for feature:', self.__class__.__name__)
            return True
        print('Not parallelizing computation for feature:', self.__class__.__name__)
        return False

    @abstractmethod
    def _compute(self, **kwargs):
        pass

    def get_feature_names(self):
        if not hasattr(self, 'features'):
            raise ValueError(f"Feature {self} does not have attribute_names defined.")
        else:
            if isinstance(self.features, list):
                return [a.name for a in self.features]
            return [self.features.name]
    
