import pandas as pd
from abc import abstractmethod, ABC
from typing import final

from mltools.feature_store.utils import utils
from mltools.feature_store.core import Metadata, Register

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

class FeatureCalculator(ABC):
    # ===== THIS MUST BE IMPLEMENTED IN SUBCLASSES =====
    compute_args = []  # compute_args is a list of arguments that are expected by the compute method. Implementations must specify which arguments they need.
    features : list[Metadata] # list of metadata for features calculated by this calculator. Implementations must specify this.
    prerequisite_features = []  # features this feature depends on from the FeatureStore database. If any are requested "prerequisite_features" argument must be requested in compute_args. Implementations may use this optionally.
    
    @abstractmethod
    def _compute(self, **kwargs) -> dict[Metadata, pd.DataFrame]:
        '''Compute the feature based using any arguments necessary.'''
        pass

    # ===== THESE ARE FINAL METHODS, DO NOT OVERRIDE =====
    @final
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

    @final
    def _decide_parallelization(self, **kwargs):
        return False
        if 'dlznik_ids' in kwargs and len(kwargs['dlznik_ids']) > 1000 and hasattr(self, 'parallelize') and self.parallelize:
            # If there are more than 1000 debtor IDs, parallelize the computation
            print('Parallelizing computation for feature:', self.__class__.__name__)
            return True
        print('Not parallelizing computation for feature:', self.__class__.__name__)
        return False

    @final
    def __init_subclass__(cls):
        Register._FEATURE_CALCULATOR_REGISTER[cls.__name__] = cls
        return super().__init_subclass__()