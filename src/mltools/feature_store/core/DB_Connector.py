import sqlite3
import pandas as pd
import numpy as np
import datetime
import sqlalchemy

from abc import abstractmethod

from mltools.utils.utils import sanitize_timestamps
from mltools.feature_store.utils import schema_from_dataframe
from mltools.utils.errors import FeatureAlreadyExistsError, FeatureNotFoundError, SchemaMismatchError
from mltools.feature_store.core import interface

# TODO: sanitize_timestamps should be done at the top, avoid multiple calls

class DB_Connector_Template(interface.DB_Connector):
    # connect is responsible for creating sqlalchemy engine
    def __init__(self, db_path: str):
        self.db_path = db_path
        # for SQLite timestamp_format is expected to be passed as a parameter
    
    def load_feature(self, feature_name: str, module_name : str, timestamp_columns: list[str] = None):
        # translate feature_address to table we are looking for:
        DB_address = self._feature_and_module_name_to_DB_address(feature_name, module_name)

        # check if feature exists
        if not self._check_if_object_exists(DB_address):
            raise FeatureNotFoundError(f"Feature '{DB_address}' does not exist in the database.")

        # attempt loading
        # TODO: handle schema in DB_address
        dot_split = DB_address.split('.')
        kwargs = {}
        if len(dot_split) == 2:
            schema, table_name = dot_split
            kwargs['schema'] = schema
        elif len(dot_split) == 1:
            table_name = dot_split[0]
        else:
            raise ValueError(f"Invalid DB_address format: {DB_address}. Expected 'schema.table_name' or 'table_name'.")
        
        try:
            if hasattr(self, 'timestamp_format') and self.timestamp_format is not None and timestamp_columns is not None:
                kwargs['parse_dates'] = {col: self.timestamp_format for col in timestamp_columns}

            with self.engine.connect() as conn:
                # Load the feature table into a DataFrame
                feature_df = pd.read_sql_table(table_name, conn, **kwargs)

            feature_df = sanitize_timestamps(feature_df)
            print(f"Feature '{feature_name}' loaded successfully.")
            return feature_df
        except Exception as e:
            print(f"Error loading feature '{feature_name}': {e}")
            return None

    # TODO: load_feature within date range (reference time/calculation time) / entity ID
    def load_most_recent_feature_value_wrt_reference_time(self, feature_name: str, module_name: str, reference_time: pd.Timestamp = pd.Timestamp.now(),  groupby_key = 'dlznik_id', reference_time_column = 'reference_time'):
        previously_computed_values_df = self.load_feature(feature_name = feature_name, module_name=module_name, timestamp_columns = ['calculation_time', 'reference_time'])
        previously_computed_values_df = previously_computed_values_df[previously_computed_values_df[reference_time_column] <= reference_time]
        previously_computed_values_df.sort_values(by = reference_time_column, ascending=False, inplace=True)
        latest_values_df = previously_computed_values_df.groupby(groupby_key, as_index=False).head(1)
        return latest_values_df

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

    def write_feature(self, feature_name: str, module_name : str, feature_df: pd.DataFrame, unique_ID_column: str = None):
        feature_df = sanitize_timestamps(feature_df)
        DB_address = self._feature_and_module_name_to_DB_address(feature_name, module_name)
        # check if feature exists
        if not self._check_if_object_exists(DB_address):
            self._create_table_from_feature(DB_address=DB_address, feature_df=feature_df)
            return feature_df

        # check if feature schema matches the data
        self._compare_schemas(DB_address = DB_address, df_to_upload=feature_df)

        feature_df = sanitize_timestamps(feature_df)

        #if unique_ID_column is provided, check if the feature_df contains it + ensure only non-duplicate entries (as given by unique_ID_column) are written
        if unique_ID_column:
            if unique_ID_column not in feature_df.columns:
                raise ValueError(f"unique_ID_column '{unique_ID_column}' is not present in the provided data.")
            try:
                existing_entries_df = self.load_feature(
                        feature_name = feature_name,
                        module_name = module_name,
                )
                existing_entries = existing_entries_df[unique_ID_column].to_numpy()
            except FeatureNotFoundError:
                print(f"Feature '{feature_name}' not found in the database. No existing entries to compare against. Writing all data.")
            except Exception as e:
                raise RuntimeError(f"Error loading existing entries for feature '{feature_name}': {e}")

            feature_df = feature_df[~feature_df[unique_ID_column].isin(existing_entries)]
            print(f"Removing existing entries from the data to be written. {len(feature_df)} records remaining.")
        else:
            print(f"No unique_ID_column provided. Writing all data to the database.")

        # write the data
        try:
            self._to_sql(feature_df, DB_address, self.engine)
            print(f"{feature_df.shape[0]} records for feature '{feature_name}' written to database.")
            return feature_df
        except Exception as e:
            print(f"Error writing feature '{feature_name}' to database: {e}")

    def update_feature(self, feature_name: str, module_name: str, feature_df: pd.DataFrame, value_column : str, reference_time_column: str = 'reference_time',  groupby_key : str = 'dlznik_id'):
        if feature_df.empty:
            return feature_df

        # TODO: already loaded features could be optionally passed to avoid loading them again
        feature_df = sanitize_timestamps(feature_df)
        feature_df.sort_values(by=reference_time_column, inplace=True)
    
        if not value_column in feature_df.columns:
            raise ValueError("value_column must refer to a column in the feature_df")
    
        DB_address = self._feature_and_module_name_to_DB_address(feature_name, module_name)
        self._compare_schemas(DB_address = DB_address, df_to_upload=feature_df)

        feature_df_schema = schema_from_dataframe(feature_df)
        timestamp_columns = [key for key, value in feature_df_schema.items() if value in ['datetime64[ns]']] 
        try:
            all_values_df = self.load_feature(feature_name, module_name, timestamp_columns = timestamp_columns)
            all_values_df.sort_values(by=reference_time_column, inplace=True)
        except FeatureNotFoundError:
            if feature_df.empty:
                print(f"Feature '{feature_name}' not found in the database and no data provided to create it. Skipping update.")
                return
            
            print(f"Feature '{feature_name}' not found in the database. Creating new feature with {feature_df.shape[0]} records.")
            self._create_table_from_feature(DB_address=DB_address, feature_df=feature_df)
            return feature_df
        
        feature_plus_latest_df = pd.merge_asof(
                feature_df,
                all_values_df,
                on=reference_time_column,
                by= groupby_key,
                direction='backward',
                suffixes=('_new', '_current')
        )

        # changed rows:
        # new value has to be different for the update to happen (~equal_values)
        # if any value is null/NA the != operator returns True.
        # We actually want to write all these cases except when both values were calculated to be null.
        # This is equivalent to (both_values_null & old_value_calculated) because new_value_calculate is always True (otherwise it would not be part of the result in merge_asof)
        equal_values = (feature_plus_latest_df[value_column + '_new'] == feature_plus_latest_df[value_column + '_current']).to_numpy()
        old_value_calculated = feature_plus_latest_df['calculation_time_current'].notna()
        both_values_null = feature_plus_latest_df[value_column + '_new'].isnull().to_numpy() & feature_plus_latest_df[value_column + '_current'].isnull().to_numpy()
        update_mask = ~equal_values & ~(both_values_null & old_value_calculated)
        changed_rows_entities = feature_plus_latest_df[update_mask].loc[:,groupby_key].to_numpy()
        update_df = feature_df[feature_df[groupby_key].isin(changed_rows_entities)]
        if len(changed_rows_entities) == 0:
            print(f"No changes detected for feature '{feature_name}'. No data written.")
            return update_df
        else:
            self._to_sql(update_df, DB_address, self.engine)
            print(f"Changes detected for feature '{feature_name}'. Writing {update_df.shape[0]} rows to database.")
            return update_df
    '''
    def _split_multifeature(feature_df : pd.DataFrame, data_columns : list[str]):
        #TODO: would be more consistent to use feature.attribute_names instead of data_columns argument
        """
        Splits a DataFrame with multiple features into separate DataFrames for each feature.
        Assumes that the Dataframe has
            - data columns - for each of these a separate DataFrame will be created
            - 'non_data_columns' - these columns will be copied in all resulting DataFrames
        """
        # Identify data columns and non-data columns
        if len (data_columns) == 0:
            raise ValueError("No data columns found in the DataFrame. Are non_data_columns specified correctly?")
        if not np.all(data_column in feature_df.columns for data_column in data_columns):
            raise ValueError(f"Some data columns are not present in the DataFrame: {set(data_columns) - set(feature_df.columns)}")
        if len (data_columns) == 1:
            # If there is only one data column, return the original DataFrame
            return [feature_df]
        split_dfs = []
        for data_column in data_columns:
            other_data_columns = [col for col in data_columns if col != data_column]
            split_df = feature_df.drop(columns=other_data_columns).copy()
            split_dfs.append(split_df)

        return split_dfs
    '''
    def _create_table_from_feature(self, DB_address : str, feature_df : pd.DataFrame):
        feature_df = sanitize_timestamps(feature_df)
        # given a dataframe, create a table in the DB
        if not self._check_if_object_exists(DB_address):
            feature_df = sanitize_timestamps(feature_df)
            self._to_sql(feature_df, DB_address, self.engine)
            print(f"Feature '{DB_address}' created in database.")
            print(f"{feature_df.shape[0]} records for feature '{DB_address}' written to database.")
        else:
            raise FeatureAlreadyExistsError(f"Feature '{DB_address}' already exists in the database.")

    @abstractmethod
    def _to_sql(self, df : pd.DataFrame, DB_address: str, engine : sqlalchemy.engine.base.Engine):
        pass

    @abstractmethod
    def _check_if_object_exists(self, DB_address: str):
        pass

    @abstractmethod
    def _get_feature_schema(self, DB_address: str):
        pass

    @abstractmethod
    def _compare_schemas(self, DB_address : str, df_to_upload: pd.DataFrame):
        pass

    @abstractmethod
    def query(self, sql_query: str):
        pass
    
    @abstractmethod
    def _feature_and_module_name_to_DB_address(self, feature_name : str, module_name : str) -> str:
        pass

    @abstractmethod
    def connect(self):
        pass

class PostgreSQL_DB_Connector(DB_Connector_Template):
    def __init__(self, db_path: str):
        self.db_path = db_path
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

    def connect(self):
        """Establish a connection to the SQLite database."""
        try:
            # Assuming db_path in the form of username:password@host:port/dbname
            if not hasattr(self, 'engine'):
                self.engine = sqlalchemy.create_engine(f'postgresql+psycopg2://{self.db_path}',
                                pool_size=10,            # or 1 if you're not parallelizing
                                max_overflow=0,         # don't allow going over pool_size
                                pool_timeout=10,        # optional: wait 10s before failing
                                pool_recycle=1800       # optional: close idle connections after 30 min
                            )

            with self.engine.connect() as conn:
                pass
            print(f"✅ Connected to database at {self.db_path}")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")

    def disconnect(self):
        """Disconnect from the database."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            print(f"✅ Disconnected from database at {self.db_path}")
        else:
            print("❌ No active database connection to disconnect.")

    def _check_if_object_exists(self, DB_address: str):
        
        # Assume DB_address is a string in the format "schema.table_name"
        schema, table = DB_address.split('.', 1)

        query = sqlalchemy.text("""
            SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema AND table_name = :table
            )
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"schema": schema, "table": table})
            return result.scalar()  # returns True or False
        
    def _get_feature_schema(self, DB_address: dict):
        # Query PostgreSQL for the schema of the feature (table)
        schema, table_name = DB_address.split('.', 1)

        query = sqlalchemy.text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table
            ORDER BY ordinal_position
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"schema": schema, "table": table_name})
            schema = result.fetchall()
            return schema

    def _compare_schemas(self, DB_address: dict, df_to_upload: pd.DataFrame):
        # check if the incoming dataframe has non-zero length
        if df_to_upload.empty:
            return
        
        # Check if table exists
        if not self._check_if_object_exists(DB_address):
            print(f"Table '{DB_address}' does not exist. No schema comparison needed.")
            return

        feature_db_schema = self._get_feature_schema(DB_address)
        feature_df_schema = schema_from_dataframe(df_to_upload)

        db_schema_dict = {col[0]: self.typedict_postgres_to_pandas.get(col[1].lower(), 'object') for col in feature_db_schema}
        if db_schema_dict.keys() != feature_df_schema.keys():
            raise SchemaMismatchError(f"Schema mismatch for feature '{DB_address}'. Expected {feature_db_schema}, got {feature_df_schema}.")

        for key, val in feature_df_schema.items():
            db_type = db_schema_dict.get(key)
            if isinstance(db_type, list):
                if val not in db_type:
                    raise SchemaMismatchError(f"Schema mismatch for feature '{DB_address}'. Expected {feature_db_schema}, got {feature_df_schema}.")
            else:
                if val != db_type:
                    raise SchemaMismatchError(f"Schema mismatch for feature '{DB_address}'. Expected {feature_db_schema}, got {feature_df_schema}. Offending column: {key} with type {val} does not match expected type {db_type}.")

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

    def _feature_and_module_name_to_DB_address(self, feature_name : str, module_name : str) -> str:
        # convert feature and module name to table name
        return f"{module_name}.{feature_name}"


    def _to_sql(self, df : pd.DataFrame, DB_address: str, engine : sqlalchemy.engine.base.Engine):
        dot_split = DB_address.split('.')
        kwargs = {}
        if len(dot_split) == 2:
            schema, table_name = dot_split
            kwargs['schema'] = schema
            # Check if schema exists, if not - create it
            with engine.begin() as conn:
                schema_exists_query = sqlalchemy.text("""
                    SELECT schema_name FROM information_schema.schemata WHERE schema_name = :schema
                """)
                result = conn.execute(schema_exists_query, {"schema": schema})
                if not result.fetchone():
                    conn.execute(sqlalchemy.text(f"CREATE SCHEMA IF NOT EXISTS \"{schema}\""))
        elif len(dot_split) == 1:
            table_name = dot_split[0]
        else:
            raise ValueError(f"Invalid DB_address format: {DB_address}. Expected 'schema.table_name' or 'table_name'.")
        
        # cast float to numeric
        cast_to_numeric_dict = {col: sqlalchemy.types.Numeric() for col in df.select_dtypes(include=['float64']).columns}
        with engine.begin() as conn:
            df.to_sql(table_name, conn, if_exists='append', index=False, dtype = cast_to_numeric_dict, **kwargs)

class SQLite_DB_Connector(DB_Connector_Template):
    def __init__(self, db_path: str, timestamp_format: str = "%Y-%m-%d %H:%M:%S"):
        self.db_path = db_path
        self.timestamp_format = timestamp_format
        self.typedict_sqlite_to_pandas = {
            'integer': ['int64','bool'],
            'int': ['int64','bool'],
            'bigint': ['int64', 'bool'],
            'smallint': ['int64', 'bool'],
        
            'real': ['float64','bool'],
            'double': ['float64','bool'],
            'double precision': ['float64','bool'],
            'float': ['float64','bool'],

            'text': 'object',
            'varchar': 'object',
            'char': 'object',
            'nvarchar': 'object',

            'boolean': 'bool',
            'bool': 'bool',

            'numeric': 'float64',       
            'decimal': 'float64',
            'date': ['datetime64[ns]','datetime64[us]'],
            'datetime': ['datetime64[ns]','datetime64[us]'],
            'timestamp': ['datetime64[ns]','datetime64[us]'],

            'blob': 'object'
        }
    def connect(self):
        """Establish a connection to the SQLite database."""
        try:
            self.engine = sqlalchemy.create_engine(f'sqlite:///{self.db_path}')
            # Test connection
            with self.engine.connect() as conn:
                pass
            print(f"✅ Connected to database at {self.db_path}")
        except sqlite3.Error as e:
            print(f"❌ Error connecting to database: {e}")
            
    def _check_if_object_exists(self, DB_address: str):
        # Check if feature exists in the DB using SQLAlchemy engine
        query = sqlalchemy.text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"
        )
        with self.engine.connect() as conn:
            result = conn.execute(query, {"table_name": DB_address})
            return result.fetchone() is not None

    def _get_feature_schema(self, DB_address: str):
        # Query the DB for the schema of the feature using SQLAlchemy engine
        try:
            query = sqlalchemy.text(f"PRAGMA table_info({DB_address})")
            with self.engine.connect() as conn:
                result = conn.execute(query)
                schema = result.fetchall()
                return schema
        except Exception as e:
            print(f"Error retrieving schema for {DB_address}: {e}")
            return None
        
    def _compare_schemas(self, DB_address: str, df_to_upload: pd.DataFrame):
        # Use SQLAlchemy engine to check if table exists
        query_tables = sqlalchemy.text("SELECT name FROM sqlite_master WHERE type='table'")
        with self.engine.connect() as conn:
            table_list = conn.execute(query_tables).fetchall()
            table_names = [table[0] for table in table_list]

            if DB_address not in table_names:
                print(f"Table '{DB_address}' does not exist. No schema comparison needed.")
                return

            # Check number of rows in table
            query_count = sqlalchemy.text(f"SELECT COUNT(*) FROM {DB_address}")
            num_rows = conn.execute(query_count).fetchone()[0]
            if num_rows == 0:
                print(f"Table '{DB_address}' is empty. No schema comparison needed.")
                return

        feature_db_schema = self._get_feature_schema(DB_address)
        feature_df_schema = schema_from_dataframe(df_to_upload)

        # compare the schema of the DB feature with the DataFrame schema
        db_schema_dict = {col[1]: self.typedict_sqlite_to_pandas[col[2].lower()] for col in feature_db_schema}
        if db_schema_dict.keys() != feature_df_schema.keys():
            raise SchemaMismatchError(f"Schema mismatch for feature '{DB_address}'. Expected {feature_db_schema}, got {feature_df_schema}.")

        for key, val in feature_df_schema.items():
            db_type = db_schema_dict.get(key)
            if isinstance(db_type, list):
                if val not in db_type:
                    raise SchemaMismatchError(f"Schema mismatch for feature '{DB_address}'. Expected {feature_db_schema}, got {feature_df_schema}.")
            else:
                if val != db_type:
                    raise SchemaMismatchError(f"Schema mismatch for feature '{DB_address}'. Expected {feature_db_schema}, got {feature_df_schema}.")

    def query(self, sql_query: str):
        # query the DB with a custom SQL query using SQLAlchemy engine
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(sql_query))
                rows = result.fetchall()
            return rows
        except Exception as e:
            print(f"Error querying database: {e}")
            return None

    def _feature_and_module_name_to_DB_address(self, feature_name : str, module_name : str) -> str:
        # convert feature and module name to table name
        return f"{module_name}__{feature_name}"


    def _to_sql(self, df : pd.DataFrame, DB_address: str, engine : sqlalchemy.engine.base.Engine):
        dot_split = DB_address.split('.')
        kwargs = {}
        if len(dot_split) == 2:
            raise ValueError(f"SQLite does not support schemas. Expected 'table_name' only (no dots).")
        elif len(dot_split) == 1:
            table_name = dot_split[0]
        else:
            raise ValueError(f"Invalid DB_address format: {DB_address}. Expected 'table_name' only (no dots).")
        
        df.to_sql(table_name, engine, if_exists='append', index=False, **kwargs)