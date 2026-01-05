import decimal
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
                version_metadata = existing.versions.created_at.desc().first()

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
        dtype_map = {}

        for col in table.columns:
            try:
                py = col.type.python_type   # <— This is the actual correct python type
            except NotImplementedError:
                # Some types (e.g. JSON, ARRAY) may not have python_type
                continue

            name = col.name
            if py is int:
                dtype_map[name] = "int"     # pandas nullable int
            elif py is decimal.Decimal:
                dtype_map[name] = "float"
            elif py is bool:
                dtype_map[name] = "bool"   # pandas nullable bool
            elif py is float:
                dtype_map[name] = "float"
            elif py is str:
                dtype_map[name] = "string"
            elif py.__name__ == "datetime":
                dtype_map[name] = "datetime64[ns]"
            elif py.__name__ == "date":
                dtype_map[name] = "datetime64[ns]"
            # extend as needed

        # Step 3: apply the dtype map
        return df.astype(dtype_map)

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
            # and write each DataFrame to the target database
            current_report.add([feature_metadata.feature_name, 'submitted_rows'], calculated_features_dict.get(feature_metadata).shape[0])
            if feature_metadata.feature_type == Type.FeatureType.EVENT:
                written_data = self.write_feature(data = df, metadata = feature_metadata)
                written_data_list.append(written_data)
            elif feature_metadata.feature_type == Type.FeatureType.STATE or feature_metadata.feature_type == Type.FeatureType.TIMESTAMP:
                written_data = self.update_feature(data = df, metadata = feature_metadata)
                written_data_list.append(written_data)
            current_report.add([feature_metadata.feature_name, 'written_rows'], written_data.shape[0])
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


    def update_feature(self, data : pd.DataFrame, metadata: Metadata):

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

        all_values_df = self.load_feature(metadata.feature_name)
        all_values_df.sort_values(by=metadata.reference_time_column, inplace=True)
        
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

    def collect_features(self,
                         entities_to_collect : np.ndarray,
                         reference_times : datetime.datetime | np.ndarray[datetime.datetime],
                         features_to_collect : np.ndarray[str],
                         output_reference_time_column : str = None,
                         ) -> pd.DataFrame:
        # connect to the feature store

        entities_to_collect = general_utils.to_array(entities_to_collect).astype(float) # TODO: entity ID type? - hardcoded to be float here; move elsewhere
        reference_times = general_utils.to_datetime_array(reference_times)
        features_to_collect = general_utils.to_array(features_to_collect)

        all_features_df = pd.DataFrame(columns = features_to_collect, index = entities_to_collect)
        matched_df = pd.DataFrame(columns = features_to_collect, index = entities_to_collect)
        stale_df = pd.DataFrame(columns = features_to_collect, index = entities_to_collect)

        if len(features_to_collect) == 0:
            return all_features_df, matched_df, stale_df
        
        if output_reference_time_column is not None and output_reference_time_column in features_to_collect:
            raise ValueError(f"The output_reference_time_column '{output_reference_time_column}' cannot be in features_to_collect. Rename it to avoid conflicts.")

        # check that all features_to_collect are 1) registered 
        try: 
            metadata_list = [self._check_if_feature_exists(feature_name) for feature_name in features_to_collect]
        except Exception as e:
            raise ValueError(f"Failed to load feature metadata for features_to_collect: {e}")
        
        # check that all features share the same entity type
        entity_id_names = set([metadata.entity_id_name for metadata in metadata_list])
        if len(entity_id_names) > 1:
            raise ValueError(f"All features to collect must share the same entity type. Found entity types: {entity_id_names}")
        shared_entity_id_name = entity_id_names.pop()

        feature_staleness_dict = {metadata.feature_name: metadata.stale_after_n_days for metadata in metadata_list}

        for feature_metadata in metadata_list:
            if feature_metadata is None:
                raise ValueError(f"Feature metadata '{feature_metadata.feature_name}' not found in the feature register.")

            loaded_feature = self.load_feature(feature_name=feature_metadata.feature_name)
            historical_data, matched_flag, stale_flag = self.get_historical_data(all_values_df=loaded_feature,
                                                          entities=entities_to_collect,
                                                          reference_times=reference_times,
                                                          reference_time_column='reference_time',
                                                          expiration_days=feature_staleness_dict.get(feature_metadata.feature_name, None))

            # assign to all_features_df
            all_features_df[feature_metadata.feature_name] = historical_data
            matched_df[feature_metadata.feature_name] = matched_flag
            stale_df[feature_metadata.feature_name] = stale_flag

        # return reference_time as a column?
        if output_reference_time_column is not None:
            all_features_df[output_reference_time_column] = reference_times
            
        # return the collected features
        return all_features_df, matched_df, stale_df

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
                reference_time_submitted=reference_time,
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