"""
Funciones para cargar datos desde Excel o SQL (abstracta).
Devuelve DataFrame crudo.
"""
import pandas as pd
import sqlalchemy
import os

def load_from_excel(path, sheet_name="DATA"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel no encontrado: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name).dropna(how="all")
    return df

def create_sql_engine(server: str, database: str, driver="ODBC Driver 17 for SQL Server"):
    conn_str = f"mssql+pyodbc://@{server}/{database}?driver={driver}&trusted_connection=yes"
    return sqlalchemy.create_engine(conn_str)

def load_from_sql(engine, query: str):
    return pd.read_sql(query, engine)
