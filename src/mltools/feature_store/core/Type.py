import enum

class FeatureType(enum.Enum):
    # EVENT-type feature represents an event, e.g. a payment, phone call, etc. Each requires an event_id_column attribute with the name of the column that will serve as the unique identifier for the event.
    # Output shall include the event_id column. event_id_column must not be included in attribute_names - this is not a separate attribute, it will be part of the output DataFrame together with the actual value, reference_time, entity_id, calculation_time.
    # Each new event is expected to be added to the database and all events in a date range are expected to be read from the database.
    EVENT = "EVENT" 

    # STATE-type feature represents a state of the entity, e.g. current debt, current number of calls, etc. No further attributes are required.
    # Each new state is expected to be added to the database. Only the most recent state (with respect to a given reference time) is expected to be read from the database.
    STATE = "STATE"

    # HELPER-type feature is used to derive other features, e.g. fetching a list of entities or kwargs to be passed to other FeatureDefinition functions
    # It is not expected to be written to the database, hence its structure is less strict.
    HELPER = "HELPER"

    # TIMESTAMP-type feature is used to store a date instead of timediff. Timediff will be calcuated upon retrieval => will save a lot of storage space.
    # Similar to state - each new state is expected to be added to the database and only the most recent state (with respect to a given reference time) is expected to be read from the database.
    # This type means the value will be a timestamp and should trigger calculation of a timediff (between the timestamp and the reference time) upon retrieval.
    TIMESTAMP = "TIMESTAMP"

