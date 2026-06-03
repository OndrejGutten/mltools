import decimal
from io import StringIO
import pandas as pd
import numpy as np
import datetime
import sqlalchemy

from typing import Literal
from sqlalchemy.orm import Session

from mltools.feature_store.core import Metadata, Register, Type
from mltools.utils import report, utils as general_utils
from mltools.utils.errors import FeatureNotFoundError, SchemaMismatchError

from ..internal import FeatureStoreModel
from sqlalchemy import Boolean, Float, MetaData, PrimaryKeyConstraint, String, Table, Column, Integer, Numeric, DateTime

# TODO: general_utils.sanitize_timestamps should be done at the top, avoid multiple calls

# Source - https://stackoverflow.com/a
# Posted by jaumebonet
# Retrieved 2026-01-21, License - CC BY-SA 4.0

import numpy as np
from psycopg2.extensions import register_adapter, AsIs

def adapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))

def adapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def adapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

def adapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)

def adapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

register_adapter(np.float64, adapt_numpy_float64)
register_adapter(np.int64, adapt_numpy_int64)
register_adapter(np.float32, adapt_numpy_float32)
register_adapter(np.int32, adapt_numpy_int32)
register_adapter(np.ndarray, adapt_numpy_array)

class FeatureStoreClient():
    def __init__(self, db_flavor: str, username : str, password : str, address : str):
        self.username = username
        self.password = password
        self.db_flavor = db_flavor.lower()
        self.db_address = address
        self.connection_string = f"{self.db_flavor}://{self.username}:{self.password}@{self.db_address}"
        self.connection_string_printable = f"{self.db_flavor}://{self.username}:***@{self.db_address}"
        # Map PostgreSQL types to pandas types
        self.typedict_postgres_to_pandas = {
            'integer': ['int64', 'bool'],
            'bigint': ['int64', 'bool'],
            'smallint': ['int64', 'bool'],
            'real': ['float64', 'bool'],
            'double precision': ['float64', 'bool'],
            'float': ['float64', 'bool'],
            'text': 'object',
            'varchar': 'object',
            'character varying': 'object',
            'char': 'object',
            'boolean': 'bool',
            'bool': 'bool',
            'numeric': 'float64',
            'decimal': 'float64',
            'date': ['datetime64[ns]', 'datetime64[us]'],
            'timestamp': ['datetime64[ns]', 'datetime64[us]'],
            'timestamp without time zone': ['datetime64[ns]', 'datetime64[us]'],
            'timestamp with time zone': ['datetime64[ns]', 'datetime64[us]'],
            'bytea': 'object'
        }

    def load_feature_metadata(self, feature_name : str, version : int = None) -> Metadata.Metadata:
        # load feature metadata from the feature registry table
        # if version is None, load latest version
        with Session(self.engine) as session:
            existing = session.query(FeatureStoreModel.FeatureRegistry).filter_by(feature_name=feature_name).first()
            if not existing:
                raise FeatureNotFoundError(f"Feature '{feature_name}' not found in the feature registry.")
            if version is not None:
                version_metadata = next((v for v in existing.versions if v.version == version), None)
                if not version_metadata:
                    raise FeatureNotFoundError(f"Feature '{feature_name}' with version '{version}' not found in the feature registry.")
            else:
                version_metadata = max(existing.versions, key=lambda v: v.created_at)

            metadata = Metadata.Metadata(
                name = existing.feature_name,
                entity_id_name = existing.entity_id_name,
                feature_type = Type.FeatureType(version_metadata.feature_type),
                data_type = version_metadata.data_type,
                stale_after_n_days = version_metadata.stale_after_n_days,
                description = existing.description,
                version_description = version_metadata.version_description,
                version = version_metadata.version,
                event_id_name = version_metadata.event_id_name,
                value_column = version_metadata.value_column,
                reference_time_column = version_metadata.reference_time_column,
            )
            return metadata

    def load_feature(self, feature_name: str, version: int = None):
        metadata = self._check_if_feature_exists(feature_name, version)

        table_name = metadata.table_name
        schema_name = self._metadata_type_to_schema(metadata.metadata_type)

        try:
            with self.engine.connect() as conn:
                # Load the feature table into a DataFrame
                feature_df = pd.read_sql_table(table_name, conn, schema=schema_name) # NOTE: DELETED parse_dates argument because this should be handled by pd.read_sql_table automatically when reading from postgres. Does it?

                if feature_df.empty:
                    feature_df = self._read_empty_table_with_dtypes(self.engine, metadata.table_name, schema=schema_name)

            feature_df = general_utils.sanitize_timestamps(feature_df)
            print(f"Feature '{feature_name}' loaded successfully.")
            return feature_df
        except Exception as e:
            print(f"Error loading feature '{feature_name}': {e}")
            return None

    def _read_empty_table_with_dtypes(self, engine, table_name, schema=None):
        metadata = MetaData()
        table = Table(table_name, metadata, schema=schema, autoload_with=engine)

        # Step 1: load empty frame with correct columns
        df = pd.read_sql(table.select().limit(0), engine)

        # Step 2: construct dtype map from SQLAlchemy column definitions
        dtype_mapping_dict = self._map_postgres_types_to_pandas(table)
        
        return df.astype(dtype_mapping_dict)

    def _map_postgres_types_to_pandas(self, table: Table):

        dtype_map = {}
        for col in table.columns:
            try:
                py = col.type.python_type   # <— This is the actual correct python type
            except NotImplementedError:
                # Some types (e.g. JSON, ARRAY) may not have python_type
                continue

            name = col.name
            if py is int:
                dtype_map[name] = "Int64"   # pandas nullable int (capital I)
            elif py is decimal.Decimal:
                dtype_map[name] = "float64"  # nullable by default
            elif py is bool:
                dtype_map[name] = "boolean"  # pandas nullable bool
            elif py is float:
                dtype_map[name] = "float64"  # nullable by default
            elif py is str:
                dtype_map[name] = "string"   # nullable by default
            elif py.__name__ == "datetime":
                dtype_map[name] = "datetime64[ns]"
            elif py.__name__ == "date":
                dtype_map[name] = "datetime64[ns]"
            # extend as needed
        
        return dtype_map

    # TODO: load_feature within date range (reference time/calculation time) / entity ID
    def load_most_recent_feature_value_wrt_reference_time(self, feature_name: str, reference_time: pd.Timestamp = pd.Timestamp.now(),  groupby_key = 'dlznik_id', reference_time_column = 'reference_time'):
        previously_computed_values_df = self.load_feature(feature_name = feature_name)
        previously_computed_values_df = previously_computed_values_df[previously_computed_values_df[reference_time_column] <= reference_time]
        previously_computed_values_df.sort_values(by = reference_time_column, ascending=False, inplace=True)
        latest_values_df = previously_computed_values_df.groupby(groupby_key, as_index=False).head(1)
        return latest_values_df

    '''
    def delete_data(self, feature_name: str, module_name: str, period_start : datetime.datetime, period_end : datetime.datetime, reference_column = 'reference_time'):
        # TODO: TEST
        # cleanup the feature table for a given period
        DB_address = self._feature_and_module_name_to_DB_address(feature_name, module_name)
        if not self._check_if_object_exists(DB_address):
            raise FeatureNotFoundError(f"Feature '{DB_address}' does not exist in the database.")

        if period_start is None:
            period_start = datetime.datetime.min
        if period_end is None:
            period_end = datetime.datetime.max

        try:
            with self.engine.connect() as conn:
                count_result = conn.execute(
                    sqlalchemy.text(f"SELECT COUNT(*) FROM {DB_address} WHERE {reference_column} >= :start AND {reference_column} <= :end"),
                    {"start": period_start, "end": period_end}
                )
                count = count_result.scalar()
                conn.execute(
                    sqlalchemy.text(f"DELETE FROM {DB_address} WHERE {reference_column} >= :start AND {reference_column} <= :end"),
                    {"start": period_start, "end": period_end}
                )
            print(f"Cleanup for feature '{feature_name}' completed for period {period_start} to {period_end}. Deleted {count} records.")
        except Exception as e:
            print(f"Error during cleanup for feature '{feature_name}': {e}")
    '''

    def submit_features(self, calculated_features_dict: dict):
        current_report = report.Report(f'Feature_submission_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
        written_data_list = []
        for feature_metadata, df in calculated_features_dict.items():
            try:
                # and write each DataFrame to the target database
                current_report.add([feature_metadata.feature_name, 'submitted_rows'], calculated_features_dict.get(feature_metadata).shape[0])
                if feature_metadata.feature_type == Type.FeatureType.EVENT:
                    written_data = self.write_feature(data = df, metadata = feature_metadata)
                    written_data_list.append(written_data)
                elif feature_metadata.feature_type == Type.FeatureType.STATE or feature_metadata.feature_type == Type.FeatureType.TIMESTAMP:
                    written_data = self.update_feature(data = df, metadata = feature_metadata)
                    written_data_list.append(written_data)
                current_report.add([feature_metadata.feature_name, 'written_rows'], written_data.shape[0])
            except Exception as e:
                print(f"ERROR WRITING FEATURE '{feature_metadata.feature_name}' to database: {e}. CONTINUING WITH NEXT FEATURE.")
        return written_data_list, current_report


    def write_feature(self, data: pd.DataFrame, metadata: Metadata):
        metadata_object, schema_name, table_name = self._validate_feature_metadata(metadata)
        versioned_data = self._wrap_feature_data(data, metadata)

        if metadata.feature_type == Type.FeatureType.EVENT:
            existing_entries_df = self.load_feature(metadata.feature_name)
            existing_entries = existing_entries_df.event_id.to_numpy()
            data_to_write = versioned_data[~versioned_data.event_id.isin(existing_entries)]
            print(f"Removing existing entries from the data to be written. {len(data_to_write)} records remaining.")
        else:
            print(f"No event ID for this feature. Writing all data to the database.")
            data_to_write = versioned_data

        try:
            self._to_sql(data = data_to_write,
                         table_name = table_name,
                         schema_name = schema_name)
            print(f"{data.shape[0]} records for feature '{metadata.feature_name}' written to database.")
        except Exception as e:
            print(f"Error writing feature '{metadata.feature_name}' to database: {e}")    
        
        self._log_feature_submission(feature_metadata = metadata_object,
                                     submitted_data= data,
                                     written_data = data_to_write)
        return data_to_write


    def update_feature(self, data : pd.DataFrame, metadata: Metadata, model_id = None):

        # TODO: already loaded features could be optionally passed to avoid loading them again
        data = general_utils.sanitize_timestamps(data)
        data.sort_values(by=metadata.reference_time_column, inplace=True)
    
        if not metadata.value_column in data.columns:
            raise ValueError("value_column must refer to a column in the feature_df")
    
        metadata_object, schema_name, table_name = self._validate_feature_metadata(metadata)

        if data.empty:
            self._log_feature_submission(feature_metadata = metadata_object,
                    submitted_data= data,
                    written_data = data)
            return data

        versioned_data = self._wrap_feature_data(data, metadata)

        if schema_name == 'predictions' and model_id is not None:
            all_values_df = pd.read_sql(f"""SELECT * FROM {schema_name}.{table_name} as pt WHERE pt.model_id = {str(model_id)}""", self.engine)
        else:
            all_values_df = self.load_feature(metadata.feature_name)


        all_values_df.sort_values(by=metadata.reference_time_column, inplace=True)

        if all_values_df.empty:
            update_df = data
            changed_rows_entities = update_df.entity_id
        else:
            feature_plus_latest_df = pd.merge_asof(
                    versioned_data,
                    all_values_df,
                    on=metadata.reference_time_column,
                    by='entity_id',
                    direction='backward',
                    suffixes=('_new', '_current')
            )

            # changed rows:
            # new value has to be different for the update to happen (~equal_values)
            # if any value is null/NA the != operator returns True.
            # We actually want to write all these cases except when both values were calculated to be null.
            # This is equivalent to (both_values_null & old_value_calculated) because new_value_calculate is always True (otherwise it would not be part of the result in merge_asof)

            equal_values = (feature_plus_latest_df[metadata.value_column + '_new'] == feature_plus_latest_df[metadata.value_column + '_current']).to_numpy()
            equal_values[pd.isna(equal_values)] = False # make sure that pd.NA values are treated as unequal and not as NA
            old_value_calculated = feature_plus_latest_df['calculation_time_current'].notna()
            both_values_null = feature_plus_latest_df[metadata.value_column + '_new'].isnull().to_numpy() & feature_plus_latest_df[metadata.value_column + '_current'].isnull().to_numpy()
            update_mask = ~equal_values & ~(both_values_null & old_value_calculated)
            changed_rows_entities = feature_plus_latest_df[update_mask].loc[:,'entity_id'].to_numpy()
            update_df = data[data['entity_id'].isin(changed_rows_entities)]

        if len(changed_rows_entities) == 0:
            print(f"No changes detected for feature '{metadata.feature_name}'. No data written.")

        else:
            self._to_sql(data=update_df, table_name=table_name, schema_name=schema_name)
            print(f"Changes detected for feature '{metadata.feature_name}'. Writing {update_df.shape[0]} rows to database.")

        self._log_feature_submission(feature_metadata = metadata_object,
                            submitted_data= data,
                            written_data = update_df)
        return update_df

    def _check_if_feature_exists(self, feature_name: str, version : int = None):
        # check if feature exists in the feature registry. If it does, return its metadata
        with Session(self.engine) as session:
            existing = session.query(FeatureStoreModel.FeatureRegistry).filter_by(feature_name=feature_name).first()
            if not existing:
                raise ValueError(f"Feature '{feature_name}' not found in the feature registry.")
            if version is not None:
                version_metadata = next((v for v in existing.versions if v.version == version), None)
                if not version_metadata:
                    raise ValueError(f"Feature '{feature_name}' version {version} not found in the feature registry.")
            return existing

    # Accepted dict shapes for a PREDICTION entry in features_to_collect.
    # Anything that doesn't match one of these exactly is rejected with a
    # single uniform error.
    _PRED_DICT_KEYS_INT = frozenset({"table", "version", "output_name"})
    _PRED_DICT_KEYS_STR = frozenset({"table", "version", "model_name", "output_name"})
    _VERSION_SENTINELS = ("PIT", "production")

    @staticmethod
    def _parse_features_to_collect(features_to_collect) -> list[dict]:
        """Normalize features_to_collect into a list of structured entries.

        Each input element is one of:
          - A bare feature_name string (FEATURE-type feature only). The output
            column is implicitly named after the feature.
          - A dict in exactly one of these two shapes (PREDICTION-type only):
              {"table": str, "version": int,                "output_name": str}
              {"table": str, "version": "PIT" | "production",
                              "model_name": str, "output_name": str}
            `table` is the registered feature_name. `version` int (not bool)
            pins to a literal ModelRegister.id. `version="production"`
            resolves once at call time against ProductionModel.name; "PIT"
            resolves per-row against ProductionHistory.

        Returns a list of normalized dicts:
            {
              "feature_name":  str,                   # FeatureRegistry key
              "output_name":   str,                   # result column name
              "model_id_spec": None
                            | ("int",        int)
                            | ("production", str)    # model_name
                            | ("PIT",        str),   # model_name
            }

        Raises ValueError with a single uniform message on any malformed
        dict (missing required key, unknown extra key, wrong type, wrong
        literal). Raises separately on duplicate output_name (including the
        implicit output_names from bare-string FEATURE entries) and on
        non-str/non-dict entries.
        """
        if features_to_collect is None:
            return []

        # Single str/dict convenience (mirrors to_array's scalar behavior).
        if isinstance(features_to_collect, (str, dict)):
            features_to_collect = [features_to_collect]

        def _bad_dict(entry):
            return ValueError(
                f"PREDICTION entry {entry!r} does not match the expected format. "
                f"Use either "
                f"{{'table': str, 'version': int, 'output_name': str}} or "
                f"{{'table': str, 'version': 'PIT'|'production', "
                f"'model_name': str, 'output_name': str}}."
            )

        entries: list[dict] = []
        seen_output_names: set[str] = set()

        for entry in features_to_collect:
            if isinstance(entry, str):
                if not entry:
                    raise ValueError("Empty string in features_to_collect.")
                feature_name = entry
                output_name = entry
                spec = None
            elif isinstance(entry, dict):
                keys = frozenset(entry.keys())
                version = entry.get("version")
                # bool is a subclass of int; reject before the int branch.
                if isinstance(version, bool):
                    raise _bad_dict(entry)
                if isinstance(version, int):
                    if keys != FeatureStoreClient._PRED_DICT_KEYS_INT:
                        raise _bad_dict(entry)
                    spec = ("int", version)
                elif isinstance(version, str):
                    if (
                        version not in FeatureStoreClient._VERSION_SENTINELS
                        or keys != FeatureStoreClient._PRED_DICT_KEYS_STR
                    ):
                        raise _bad_dict(entry)
                    model_name = entry["model_name"]
                    if not isinstance(model_name, str) or not model_name:
                        raise _bad_dict(entry)
                    spec = (version, model_name)
                else:
                    raise _bad_dict(entry)

                feature_name = entry["table"]
                output_name = entry["output_name"]
                if not isinstance(feature_name, str) or not feature_name:
                    raise _bad_dict(entry)
                if not isinstance(output_name, str) or not output_name:
                    raise _bad_dict(entry)
            else:
                raise ValueError(
                    f"features_to_collect entries must be a feature_name string "
                    f"(FEATURE-type) or a dict (PREDICTION-type). Got: {entry!r}"
                )

            if output_name in seen_output_names:
                raise ValueError(
                    f"Duplicate output_name '{output_name}' in features_to_collect."
                )
            seen_output_names.add(output_name)
            entries.append({
                "feature_name": feature_name,
                "output_name": output_name,
                "model_id_spec": spec,
            })

        return entries

    def _resolve_production_model_name(self, model_name: str) -> int:
        """Return the model_id currently pointed to by ProductionModel.name == model_name.

        Single-row lookup, called once per unique production-model-name at the
        start of a collect_features call. Raises ValueError if no such row
        exists.
        """
        with Session(self.engine) as session:
            row = (
                session.query(FeatureStoreModel.ProductionModel.model_id)
                .filter(FeatureStoreModel.ProductionModel.name == model_name)
                .first()
            )
        if row is None:
            raise ValueError(
                f"ProductionModel name '{model_name}' not found. "
                f"Use set_production_model(name='{model_name}', model_id=...) "
                f"to register a production slot before referencing it."
            )
        return row[0]

    def _populate_temp_model_id_from_production_history(
        self, session: Session, temp_table_name: str, name: str,
    ) -> None:
        """For each row in the temp table, fill `model_id` with whichever
        model_id was the production pointer for `name` at that row's
        reference_time.

        Lookup: for each row r, the picked model_id is the one from the
        ProductionHistory entry with model_name=name and the largest
        promoted_at <= r.reference_time.

        Raises ValueError if any row's reference_time predates the first
        promotion of `name` (i.e. no entry exists with promoted_at <= that
        reference_time).
        """
        metadata_schema = FeatureStoreModel.SCHEMAS.METADATA.value

        # Clear first so the post-update NULL check is unambiguous, regardless
        # of any value left behind by a previous feature in the same call.
        session.execute(sqlalchemy.text(
            f"UPDATE {temp_table_name} SET model_id = NULL"
        ))

        session.execute(sqlalchemy.text(f"""
            UPDATE {temp_table_name} t
            SET model_id = src.model_id
            FROM (
                SELECT l.row_id, ph.model_id
                FROM {temp_table_name} l
                LEFT JOIN LATERAL (
                    SELECT model_id
                    FROM "{metadata_schema}"."ProductionHistory" ph
                    WHERE ph.model_name = :name
                      AND ph.promoted_at <= l.reference_time
                    ORDER BY ph.promoted_at DESC
                    LIMIT 1
                ) ph ON TRUE
            ) src
            WHERE t.row_id = src.row_id
        """), {"name": name})

        earliest_unresolved = session.execute(sqlalchemy.text(
            f"SELECT MIN(reference_time) FROM {temp_table_name} WHERE model_id IS NULL"
        )).scalar()
        if earliest_unresolved is not None:
            raise ValueError(
                f"ProductionModel name '{name}' has no ProductionHistory entry "
                f"promoted at or before reference_time {earliest_unresolved}. "
                f"Either promote a model under that name before that point "
                f"(set_production_model) or supply a literal int model_id."
            )

    def _validate_model_ids_exist(self, model_ids: set[int]) -> None:
        """Raise if any of the supplied model_ids is absent from ModelRegister."""
        if not model_ids:
            return
        with Session(self.engine) as session:
            found = {
                row[0]
                for row in session.query(FeatureStoreModel.ModelRegister.id)
                .filter(FeatureStoreModel.ModelRegister.id.in_(model_ids))
                .all()
            }
        missing = model_ids - found
        if missing:
            raise ValueError(
                f"model_id(s) not found in ModelRegister: {sorted(missing)}"
            )

    def collect_features(self,
                         entities_to_collect : np.ndarray,
                         reference_times : datetime.datetime | np.ndarray[datetime.datetime],
                         features_to_collect : list,
                         output_reference_time_column : str = None,
                         reference_time_comparison: Literal['<=', '<'] = '<' # If the events are logged with a daily resolution (i.e. they happened at any point during the day) then we presume they are not available at the start of the day, so we use '<' comparison. If reading prerequisites we assume these were already calculated with this limitation in mind. Hence, we use '<=' comparison. Example: Event happens on day X and is logged into DB with date X. Features calculated for day X do not know about the event yet. Thus, calculators should see this event only for reference times > X (i.e. we use '<'). The feature will reflect this event on days X+1 and beyond. However, if we are using this feature as a prerequisite we want to use '<=' when retrieving it - otherwise the dependent feature would reflect the event on X+2.
                         ) -> pd.DataFrame:
        # connect to the feature store

        entities_to_collect = general_utils.to_array(entities_to_collect).astype(float) # TODO: entity ID type? - hardcoded to be float here; move elsewhere
        reference_times = general_utils.to_datetime_array(reference_times)

        # Parse features_to_collect:
        #   - FEATURE-type features: bare feature_name string. Output column is
        #     the feature_name.
        #   - PREDICTION-type features: dict in one of the two shapes documented
        #     on _parse_features_to_collect; output column is `output_name`.
        # Bare-string vs dict enforcement against metadata_type happens after
        # the registry lookup below.
        entries = self._parse_features_to_collect(features_to_collect)
        output_names = [e["output_name"] for e in entries]

        all_features_df = pd.DataFrame(columns = output_names, index = entities_to_collect)
        matched_df = pd.DataFrame(columns = output_names, index = entities_to_collect)
        stale_df = pd.DataFrame(columns = output_names, index = entities_to_collect)

        if len(entries) == 0:
            return all_features_df, matched_df, stale_df

        if output_reference_time_column is not None and output_reference_time_column in output_names:
            raise ValueError(f"The output_reference_time_column '{output_reference_time_column}' cannot be in features_to_collect output_names. Rename it to avoid conflicts.")

        # check that all referenced feature_names are registered
        try:
            metadata_list = [self._check_if_feature_exists(e["feature_name"]) for e in entries]
        except Exception as e:
            raise ValueError(f"Failed to load feature metadata for features_to_collect: {e}")

        # Guardrails on the FEATURE / PREDICTION contract:
        #   - bare-string entry (spec is None) requires metadata_type == FEATURE.
        #   - dict entry (spec is not None) requires metadata_type == PREDICTION.
        for entry, metadata in zip(entries, metadata_list):
            is_prediction = metadata.metadata_type == Type.MetadataType.PREDICTION.value
            has_spec = entry["model_id_spec"] is not None
            if is_prediction and not has_spec:
                raise ValueError(
                    f"Feature '{metadata.feature_name}' has metadata_type=PREDICTION "
                    f"and must be passed as a dict with explicit version and output_name."
                )
            if not is_prediction and has_spec:
                raise ValueError(
                    f"Feature '{metadata.feature_name}' has metadata_type="
                    f"'{metadata.metadata_type}', not PREDICTION; pass it as a bare "
                    f"feature_name string."
                )

        # Guardrail: every caller-supplied int model_id must exist in ModelRegister.
        # "PIT" and "production" names are resolved against ProductionHistory /
        # ProductionModel whose model_id columns already FK to ModelRegister, so
        # those don't need extra validation here.
        int_model_ids = {
            spec[1] for spec in (e["model_id_spec"] for e in entries)
            if spec is not None and spec[0] == "int"
        }
        self._validate_model_ids_exist(int_model_ids)

        # Pre-resolve every distinct "production" model_name to a concrete int.
        # Done once per call; the int is then applied as a constant across all
        # reference_times of that entry (snapshot semantics — by design for
        # "production"; use "PIT" if you need historically-correct per-row).
        production_resolved: dict[str, int] = {}
        for entry in entries:
            spec = entry["model_id_spec"]
            if spec is not None and spec[0] == "production":
                model_name = spec[1]
                if model_name not in production_resolved:
                    production_resolved[model_name] = self._resolve_production_model_name(model_name)

        # check that all features share the same entity type
        entity_id_names = set([metadata.entity_id_name for metadata in metadata_list])
        if len(entity_id_names) > 1:
            raise ValueError(f"All features to collect must share the same entity type. Found entity types: {entity_id_names}")
        shared_entity_id_name = entity_id_names.pop()

        feature_staleness_dict = {metadata.feature_name: metadata.stale_after_n_days for metadata in metadata_list}

        # CREATE TEMP TABLE with unique name to avoid conflicts across calls
        session = Session(self.engine)
        unique_suffix = int(datetime.datetime.now().timestamp() * 1000000)  # microsecond precision
        temp_table_name = f"entities_to_collect_{unique_suffix}"
        
        session.execute(sqlalchemy.text(f"""
            CREATE TEMP TABLE {temp_table_name} (
                row_id serial PRIMARY KEY,
                entity_id numeric,
                reference_time timestamp,
                model_id integer
            ) ON COMMIT PRESERVE ROWS;
            """))

        buf = StringIO()
        for i, (e, t) in enumerate(zip(entities_to_collect, reference_times), start=1):
            buf.write(f"{i}\t{e}\t{t.isoformat()}\n")
        buf.seek(0)

        # raw psycopg2 cursor
        conn = session.connection().connection
        cur = conn.cursor()
        cur.copy_from(buf, temp_table_name, sep="\t", columns=("row_id", "entity_id", "reference_time"))

        try:
            for entry, feature_metadata in zip(entries, metadata_list):
                output_name = entry["output_name"]
                spec = entry["model_id_spec"]
                print(f"Collecting historical data for feature '{feature_metadata.feature_name}' -> column '{output_name}'...")
                if feature_metadata is None:
                    raise ValueError(f"Feature metadata '{feature_metadata.feature_name}' not found in the feature register.")

                # Populate the temp table's model_id column so the lateral join
                # can filter on it. Four cases:
                #   - spec is None (FEATURE-type): SQL omits the filter.
                #   - ("int", n):           constant n across all rows.
                #   - ("production", name): constant — resolved upfront to int.
                #   - ("PIT", name):        per-row from ProductionHistory.
                if spec is None:
                    model_id_applicable = False
                elif spec[0] == "int":
                    session.execute(
                        sqlalchemy.text(f"UPDATE {temp_table_name} SET model_id = :mid"),
                        {"mid": spec[1]},
                    )
                    model_id_applicable = True
                elif spec[0] == "production":
                    session.execute(
                        sqlalchemy.text(f"UPDATE {temp_table_name} SET model_id = :mid"),
                        {"mid": production_resolved[spec[1]]},
                    )
                    model_id_applicable = True
                elif spec[0] == "PIT":
                    self._populate_temp_model_id_from_production_history(
                        session=session,
                        temp_table_name=temp_table_name,
                        name=spec[1],
                    )
                    model_id_applicable = True
                else:  # pragma: no cover — parser would have raised already
                    raise ValueError(f"Unknown model_id_spec kind: {spec!r}")

                historical_data, matched_flag, stale_flag = self.get_historical_data_sql(
                    session,
                    feature_metadata=feature_metadata,
                    entities=entities_to_collect,
                    reference_times=reference_times,
                    reference_time_comparison=reference_time_comparison,
                    reference_time_column='reference_time',
                    expiration_days=feature_staleness_dict.get(feature_metadata.feature_name, None),
                    temp_table_name=temp_table_name,
                    model_id_applicable=model_id_applicable,
                )

                # assign to all_features_df using the caller-supplied output_name
                all_features_df[output_name] = historical_data
                matched_df[output_name] = matched_flag
                stale_df[output_name] = stale_flag
        except Exception as e:
            session.close()
            raise ValueError(f"Failed to collect historical data: {e}")

        session.close()

        # return reference_time as a column?
        if output_reference_time_column is not None:
            all_features_df[output_reference_time_column] = reference_times
            
        # return the collected features
        return all_features_df, matched_df, stale_df

    def get_historical_data_sql(self, session: Session, feature_metadata: Metadata.Metadata, entities: list, reference_times: list, reference_time_comparison : Literal['<=', '<'], temp_table_name : str, reference_time_column = 'reference_time', expiration_days: int = None, model_id_applicable: bool = False,) -> np.ndarray:

        if feature_metadata.metadata_type == Type.MetadataType.FEATURE.value:
            schema_name = "features"
            value_name = "value"
        else:
            schema_name = "predictions"
            value_name = "prediction"

        # Optional model_id filter inside the lateral join. The per-row model_id
        # is pre-populated into the temp table by the caller (either as a
        # constant for an int request, or per-row from ProductionHistory for a
        # string ProductionModel-name request). Only meaningful for PREDICTION
        # tables; FEATURE tables have no model_id column.
        model_id_filter = "AND r.model_id = l.model_id" if model_id_applicable else ""

        SQL_QUERY = f"""
        SELECT
            l.entity_id,
            l.reference_time AS left_time,
            r.{reference_time_column} AS matched_time,
            r.{value_name} AS value
        FROM {temp_table_name} l
        LEFT JOIN LATERAL (
            SELECT r.*
            FROM {schema_name}.{feature_metadata.table_name} r
            WHERE r.entity_id = l.entity_id
            AND r.reference_time {reference_time_comparison} l.reference_time
            {model_id_filter}
            ORDER BY r.reference_time DESC
            LIMIT 1
        ) r ON TRUE
        ORDER BY l.row_id;
        """

        # named parameters dict (pass this to session.execute)
        SQL_PARAMS = {"entities": entities, "reference_times": reference_times}

        # Execute the main query directly and get results as DataFrame
        result = session.execute(sqlalchemy.text(SQL_QUERY), SQL_PARAMS)
        result_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        # Get type mapping from the actual feature table instead of creating temp table
        if not result_df.empty:
            # Reflect the actual feature table to get proper type mappings
            feature_meta = MetaData()
            feature_table = Table(feature_metadata.table_name, feature_meta, 
                                schema=schema_name, autoload_with=session.get_bind())
            
            # Map types for the columns we actually have
            type_mapping = {}
            value_col = next(col for col in feature_table.columns if col.name == 'value' or col.name == 'prediction')
            value_col_copy = Column('value', value_col.type)

            feature_meta_temp = MetaData()
            type_mapping.update(self._map_postgres_types_to_pandas(Table('temp', feature_meta_temp, value_col_copy)))

            result_df = result_df.astype(type_mapping)
        
        matched_flag = result_df['matched_time'].notnull().to_numpy().astype(bool)

        if expiration_days is not None:
            # Remove values that are too old
            # null_value = self.dtype_null_value(result_df.dtypes['value'])
            # compute age in days; if matched_time is None use +inf so it will be treated as stale
            left = pd.to_datetime(result_df['left_time'])
            matched = pd.to_datetime(result_df['matched_time'])
            age = (left - matched).dt.days.fillna(np.inf).to_numpy()
            stale_flag = (age > expiration_days).astype(bool)
            #result_df.loc[:, 'value'] = result_df.loc[:, 'value'].where(~stale_flag, other=null_value) # this should not be done here - stale values may be returned and it is the user's decision to decide if those with a stale flag should be used or not
        else:
            stale_flag = np.full(result_df.shape[0], False, dtype=bool)
        
        return result_df['value'].to_numpy(), matched_flag, stale_flag


    def get_historical_data(self, all_values_df: pd.DataFrame, entities: list, reference_times: list, reference_time_column: str, expiration_days: int = None) -> np.ndarray:
        null_value = self.dtype_null_value(all_values_df.dtypes['value'])
        matched_flag_name = '__matched_flag' if '__matched_flag' not in all_values_df.columns else '___matched_flag'
        all_values_df[matched_flag_name] = True
        
        original_order_index_name = '__original_order_index' if '__original_order_index' not in all_values_df.columns else '__old_index'
        entity_df = pd.DataFrame({'entity_id': entities, reference_time_column: reference_times, original_order_index_name: list(range(len(entities)))})
        
        all_values_df = all_values_df.sort_values(by=[reference_time_column])
        all_values_df[f'{reference_time_column}_loaded_feature'] = all_values_df[reference_time_column]

        sorted_entity_df = entity_df.sort_values(by=[reference_time_column])

        # Perform point-in-time join
        merged = pd.merge_asof(
            sorted_entity_df,
            all_values_df,
            by='entity_id',
            on=reference_time_column,
            direction="backward",
            suffixes=("", "_loaded_feature"),
        )

        merged.sort_values(by=original_order_index_name, inplace=True)

        matched_flag = merged[matched_flag_name].notnull().to_numpy().astype(bool)

        if expiration_days is not None:
            # Remove values that are too old
            age = (merged[reference_time_column] - merged[f"{reference_time_column}_loaded_feature"]).dt.days.to_numpy()
            stale_flag = (age > expiration_days).astype(bool)
            merged.loc[:, 'value'] = merged.loc[:, 'value'].where(~stale_flag, other=null_value)
        else:
            stale_flag = np.full(merged.shape[0], False, dtype=bool)

        if null_value is not None:
            merged.loc[:, 'value'] = merged.loc[:, 'value'].fillna(null_value)
        return merged['value'].to_numpy(), matched_flag, stale_flag

    def dtype_null_value(self, dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return np.nan
        elif pd.api.types.is_float_dtype(dtype):
            return np.nan
        elif pd.api.types.is_string_dtype(dtype):
            return None
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.NaT
        else:
            return np.nan

    def collect_events_in_date_range(self,
                                        feature_name: str,
                                        date_start: datetime.datetime,
                                        date_end: datetime.datetime,
                                        reference_time_column: str = 'reference_time'
                                        ) -> pd.DataFrame:
        # connect to the feature store
        try:
            self.connect()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to the feature store: {e}")

        feature_metadata = self._check_if_feature_exists(feature_name=feature_name)

        if feature_metadata is None:
            raise ValueError(f"Feature metadata '{feature_name}' not found in the feature register.")

        # load feature values in the date range
        loaded_feature = self.load_feature(feature_metadata.feature_name)

        loaded_feature = loaded_feature[
            (loaded_feature[reference_time_column] >= date_start) &
            (loaded_feature[reference_time_column] <= date_end)
        ]

        return loaded_feature

    def register_entity_set(self, name : str, description : str, entity_id_name : str):
        new_entity_set = FeatureStoreModel.EntitySetRegister(name = name, description = description, entity_id_name = entity_id_name)
        with Session(self.engine) as session:
            existing = session.query(FeatureStoreModel.EntitySetRegister).filter_by(name=name).first()
            if not existing:
                session.add(new_entity_set)
                session.commit()
            else:
                print(f"Entity set '{name}' already exists. Skipping registration.")

    def delete_entity_set(self, name : str):
        with Session(self.engine) as session:
            entity_set = session.query(FeatureStoreModel.EntitySetRegister).filter_by(name=name).first()
            if entity_set:
                session.delete(entity_set)
                session.commit()

    def retrieve_entity_set(self, name : str) -> FeatureStoreModel.EntitySetInfo:
        with Session(self.engine) as session:
            entity_set = session.query(FeatureStoreModel.EntitySetRegister).filter_by(name=name).first()
            if entity_set:
                members = [member.member_id for member in entity_set.members]
                entity_set_info = FeatureStoreModel.EntitySetInfo(
                    name = entity_set.name,
                    description = entity_set.description,
                    entity_id_name = entity_set.entity_id_name,
                    members = members
                )
                return entity_set_info
            else:
                raise ValueError(f"Entity set with name '{name}' not found.")

    def remove_members_from_entity_set(self, entity_set_name : str, member_ids : list[int] | Literal['all']):
        with Session(self.engine) as session:
            entity_set = session.query(FeatureStoreModel.EntitySetRegister).filter_by(name=entity_set_name).first()
            if not entity_set:
                raise ValueError(f"Entity set with name '{entity_set_name}' not found.")
        
            if member_ids == 'all':
                entity_set.members.clear()
            else:
                for member_id in member_ids:
                    member = session.query(FeatureStoreModel.EntitySetMember).filter_by(member_id=member_id).first()
                    if member in entity_set.members:
                        entity_set.members.remove(member)
            session.commit()

    def change_members_of_entity_set(self, entity_set_name : str, member_ids : list[int], mode : Literal['add','set']):
        if mode not in ['add', 'set']:
            raise ValueError("mode must be either 'add' or 'set'")
        with Session(self.engine) as session:
            entity_set = session.query(FeatureStoreModel.EntitySetRegister).filter_by(name=entity_set_name).first()
            if not entity_set:
                raise ValueError(f"Entity set with name '{entity_set_name}' not found.")
            if mode == 'set':
                # Clear existing members
                entity_set.members.clear()
            # Add new members
            for member_id in member_ids:
                member = session.query(FeatureStoreModel.EntitySetMember).filter_by(member_id=member_id).first()
                if member is None:
                    member = FeatureStoreModel.EntitySetMember(member_id=member_id)
                entity_set.members.append(member)
            session.commit()

    def assign_model_id(self, model_uri: str):
        with Session(self.engine) as session:
            model = session.query(FeatureStoreModel.ModelRegister).filter_by(model_uri=model_uri).first()
            if not model:
                print(f"Model with URI '{model_uri}' not found. Registering new model.")
                model = FeatureStoreModel.ModelRegister(model_uri=model_uri)
                session.add(model)
                session.commit()
            return model.id

    def set_production_model(self, name: str, model_id: int):
        with Session(self.engine) as session:
            model_with_model_id = session.query(FeatureStoreModel.ModelRegister).filter_by(id=model_id).first()
            if not model_with_model_id:
                raise ValueError(f"Model with ID '{model_id}' not found.")

            # Upsert ProductionModel with name=name and model_id=model_with_model_id.id
            production_model = session.query(FeatureStoreModel.ProductionModel).filter_by(name=name).first()
            if production_model:
                production_model.model_id = model_with_model_id.id
            else:
                production_model = FeatureStoreModel.ProductionModel(name=name, model_id=model_with_model_id.id)
                session.add(production_model)
            session.commit()
            log = FeatureStoreModel.ProductionHistory(model_name=name, model_id=model_with_model_id.id, promoted_at=datetime.datetime.now())
            session.add(log)
            session.commit()

    def connect(self):
        """Establish a connection to the SQLite database."""
        try:
            if not hasattr(self, 'engine'):
                self.engine = sqlalchemy.create_engine(f'{self.connection_string}',
                                pool_size=10,            # or 1 if you're not parallelizing
                                max_overflow=0,         # don't allow going over pool_size
                                pool_timeout=10,        # optional: wait 10s before failing
                                pool_recycle=1800       # optional: close idle connections after 30 min
                            )

            with self.engine.begin() as conn:
                # Create schemas if they do not exist
                for schema in FeatureStoreModel.SCHEMAS:
                    conn.execute(sqlalchemy.text(f'CREATE SCHEMA IF NOT EXISTS "{schema.value}"'))
                # Create tables if they do not exist
            FeatureStoreModel.Base.metadata.create_all(self.engine)
            print(f"✅ Connected to database at {self.connection_string_printable}")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")

    def disconnect(self):
        """Disconnect from the database."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            print(f"✅ Disconnected from database at {self.connection_string_printable}")
        else:
            print("❌ No active database connection to disconnect.")

    def _check_if_object_exists(self, schema: str, table_name: str) -> bool:
        query = sqlalchemy.text("""
            SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema AND table_name = :table
            )
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"schema": schema, "table": table_name})
            return result.scalar()  # returns True or False
        
    def query(self, sql_query: str):
        # Query the PostgreSQL DB using SQLAlchemy engine
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(sql_query))
                rows = result.fetchall()
                return rows
        except Exception as e:
            print(f"Error querying database: {e}")
            return None

    def _to_sql(self, data : pd.DataFrame, table_name: str, schema_name : str = None):
        data = general_utils.sanitize_timestamps(data)
        # cast float to numeric
        cast_to_numeric_dict = {col: sqlalchemy.types.Numeric() for col in data.select_dtypes(include=['float64']).columns}
        with self.engine.begin() as conn:
            data.to_sql(name = table_name,
                        schema=schema_name,
                        con=conn,
                        if_exists='append',
                        index=False,
                        dtype = cast_to_numeric_dict,
                        )

    def _log_feature_submission(self, feature_metadata: FeatureStoreModel.FeatureRegistry, submitted_data : pd.DataFrame, written_data : pd.DataFrame):
        
        # Reference time is written only if it is equal for all rows; otherwise NULL is stored
        submitted_rows = submitted_data.shape[0]
        written_rows = written_data.shape[0]
        unique_entity_ids_submitted = submitted_data['entity_id'].nunique()
        unique_reference_times_submitted = submitted_data['reference_time'].nunique()
        unique_entity_ids_written = written_data['entity_id'].nunique()
        unique_reference_times_written = written_data['reference_time'].nunique()
        reference_time = None if unique_reference_times_written != 1 else written_data['reference_time'].iloc[0]
        submission_time = datetime.datetime.now()

        with Session(self.engine) as session:
            submission_log = FeatureStoreModel.FeatureSubmissionsLog(
                feature_id=feature_metadata.id,                
                submitted_rows=submitted_rows,
                written_rows=written_rows,
                unique_entity_ids_submitted=unique_entity_ids_submitted,
                unique_reference_times_submitted=unique_reference_times_submitted,
                unique_entity_ids_written=unique_entity_ids_written,
                unique_reference_times_written=unique_reference_times_written,
                reference_time_written=reference_time,
                submission_time=submission_time
            )
            session.add(submission_log)
            session.commit()

    def _validate_feature_metadata(self, feature_metadata: Metadata):
        '''
        Check if the given feature metadata match any registry entry.
        If not, create a new registry entry + corresponding table and return the schema+table name.
        If yes but any values do no match, raise SchemaMismatchError.
        If yes and all values match, return the schema+table name.
        '''

        with Session(self.engine) as session:
            existing = session.query(FeatureStoreModel.FeatureRegistry).filter_by(feature_name=feature_metadata.feature_name).first()
            matching_version = [v for v in existing.versions if v.version == feature_metadata.version] if existing else None
            if matching_version is not None and len(matching_version) > 1:
                raise SchemaMismatchError(f"Multiple versions found for feature '{feature_metadata.feature_name}' version {feature_metadata.version}. Database integrity issue.")

            if existing and matching_version:
                mismatched_values = {}
                # direct comparison of FeatureRegistry attributes
                for attr in ['entity_id_name', 'data_type', 'stale_after_n_days']:
                    if getattr(existing, attr) != getattr(feature_metadata, attr):
                        mismatched_values[attr] = (getattr(existing, attr), getattr(feature_metadata, attr))
                # translated comparison of FeatureRegistry attributes - feature_type and metadata_type
                for attr in ['feature_type', 'metadata_type']:
                    if getattr(existing, attr) != getattr(feature_metadata, attr).value:
                        mismatched_values[attr] = (getattr(existing, attr), getattr(feature_metadata, attr))
                # direct comparison of FeatureLog attributes
                for attr in ['description', 'version_description']:
                    if getattr(matching_version[0], attr) != getattr(feature_metadata, attr):
                        mismatched_values[attr] = (getattr(matching_version[0], attr), getattr(feature_metadata, attr))
                
                # any mismatches found?
                if mismatched_values:
                        raise SchemaMismatchError(f"Metadata mismatch for feature '{feature_metadata.feature_name}' version {feature_metadata.version}. Offending entries: {mismatched_values}")
                return existing, self._metadata_type_to_schema(existing.metadata_type), existing.table_name
            else:
                if not existing:
                    # assign new table name
                    table_name = self._create_table(feature_metadata)
                    # create new feature entry
                    new_feature = FeatureStoreModel.FeatureRegistry(
                        table_name = table_name,
                        feature_name = feature_metadata.feature_name,
                        entity_id_name = feature_metadata.entity_id_name,
                        feature_type = feature_metadata.feature_type.value,
                        data_type = feature_metadata.data_type,
                        stale_after_n_days = feature_metadata.stale_after_n_days,
                        metadata_type = feature_metadata.metadata_type.value
                    )
                    session.add(new_feature)
                    session.commit()
                    existing = new_feature

                if not matching_version:
                    # create new version log
                    new_version_log = FeatureStoreModel.FeatureLog(
                        feature_id = existing.id,
                        version = feature_metadata.version,
                        created_at = datetime.datetime.now(),
                        description = feature_metadata.description,
                    version_description = feature_metadata.version_description
                    )
                    session.add(new_version_log)
                    session.commit()

                return existing, self._metadata_type_to_schema(existing.metadata_type), existing.table_name

    def _metadata_type_to_schema(self, metadata_type_str: str) -> str:
        # metadata_type is either loaded from DB or created as sqlalchemy object -> it is a string, not enum
        if metadata_type_str == Type.MetadataType.FEATURE.value:
            return FeatureStoreModel.SCHEMAS.FEATURES.value
        elif metadata_type_str == Type.MetadataType.PREDICTION.value:
            return FeatureStoreModel.SCHEMAS.PREDICTIONS.value
        elif metadata_type_str == Type.MetadataType.METRIC.value:
            return FeatureStoreModel.SCHEMAS.METRICS.value
        else:
            raise ValueError(f"Unsupported metadata type: {metadata_type_str}")

    def _wrap_feature_data(self, data : pd.DataFrame, metadata : Metadata ): # version number should be FK
        if metadata.metadata_type == Type.MetadataType.FEATURE:
            data['version'] = metadata.version
        return data

    def _create_table(self, feature_metadata: Metadata.Metadata) -> str:
        schema_name = self._metadata_type_to_schema(feature_metadata.metadata_type.value)
        candidate_name = f"{feature_metadata.feature_name.lower()}"

        if self._check_if_object_exists(schema = schema_name, table_name = candidate_name):
            return candidate_name
        else:
            if feature_metadata.feature_type == Type.FeatureType.STATE or feature_metadata.feature_type == Type.FeatureType.TIMESTAMP:
                # Create table with name candidate_name and columns from feature_metadata
                metadata_obj = MetaData(schema=schema_name)
                table = Table(
                    candidate_name,
                    metadata_obj,
                    Column("entity_id", Numeric),
                    Column(feature_metadata.value_column,  self._translate_data_type(feature_metadata.data_type)),
                    Column(feature_metadata.reference_time_column, DateTime),
                    Column('calculation_time', DateTime),
                    Column('version', Integer),
                    PrimaryKeyConstraint('entity_id', feature_metadata.reference_time_column, name=f'pk_{candidate_name}')
                )
                metadata_obj.create_all(self.engine)
                return candidate_name
            elif feature_metadata.feature_type == Type.FeatureType.EVENT:
                # Create table with name candidate_name and columns from feature_metadata
                metadata_obj = MetaData(schema=schema_name)
                table = Table(
                    candidate_name,
                    metadata_obj,
                    Column("event_id", Numeric, primary_key=True),
                    Column("entity_id", Numeric),
                    Column(feature_metadata.value_column, self._translate_data_type(feature_metadata.data_type)),
                    Column(feature_metadata.reference_time_column, DateTime),
                    Column('calculation_time', DateTime),
                    Column('version', Integer),
                )
                metadata_obj.create_all(self.engine)
                return candidate_name
            else:
                raise ValueError(f"Creating feature tables for {feature_metadata.feature_name} of type {feature_metadata.feature_type} is not implemented yet.")
            
    def _translate_data_type(self, data_type: str):
        if data_type == 'float' or data_type == 'numeric':
            return Numeric
        elif data_type == 'int':
            return Integer
        elif data_type == 'string' or data_type == 'str':
            return String
        elif 'datetime' in data_type or 'timestamp' in data_type or data_type == 'date':
            return DateTime
        elif data_type == 'bool' or data_type == 'boolean':
            return Boolean
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

