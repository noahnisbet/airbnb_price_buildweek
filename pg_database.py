import sqlite3
import psycopg2



def connect_to_pg(pg_dbname, pg_user, pg_password, pg_host, extraction_db='rpg_db.sqlite3'):
    """ Connects to DB - return sl_conn & pg_conn """
    sl_conn = sqlite3.connect(extraction_db)
    pg_conn = psycopg2.connect(
        dbname=pg_dbname, user=pg_user, password=pg_password, host=pg_host) 
    return sl_conn, pg_conn


def execute_query(curs, query):
    return curs.execute(query)


