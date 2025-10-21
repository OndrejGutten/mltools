class FeatureNotFoundError(Exception):
    """Exception raised when a feature is not found in the database."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class SchemaMismatchError(Exception):
    """Exception raised when there is a schema mismatch."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class FeatureAlreadyExistsError(Exception):
    """Exception raised when trying to create a feature that already exists."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message      

class FeatureDefinitionError(Exception):
    """Exception raised when there is an error in the feature definition."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class DatabaseConnectionError(Exception):
    """Exception raised when there is a database connection error."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message