from sqlalchemy import Engine, create_engine
from config.settings import Settings
import pandas as pd
from sqlalchemy import text
from loguru import logger

class DatabaseService:
    """
    A service class to manage database connections and operations.
    """
    def __init__(self):
        self.engine = self.get_database_engine()
    
    def fetch_data(self, query: str, params: dict = None) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a DataFrame.

        Args:
            query (str): The SQL query to execute.
            params (dict, optional): Parameters to pass to the SQL query.
        Returns:
            pd.DataFrame: The results of the query.
        """
        with self.engine.connect() as connection:
            sql_query = text(query)
            df = pd.read_sql(sql_query, connection, params=params).drop_duplicates()
            length = len(df)
            logger.info(f"Executed query with length: {length}") 
        return df
    
    def get_database_engine(self) -> Engine:
        """
        Generates a connection engine for the database. Adds additional SSL requirements for Cloud-based
        database connections.

        Args: None

        Returns:
            Engine: Sqlalchemy engine used to make connections with the database
        """

        if Settings.ENVIRONMENT in ["PRODUCTION", "STAGING", "DEVELOPMENT"]:
            engine = create_engine(
                Settings.POSTGRES_CONN_STR, connect_args={"sslmode": "require"}
            )
        else:
            engine = create_engine(Settings.POSTGRES_CONN_STR)

        return engine

