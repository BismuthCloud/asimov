# BISMUTH FILE: db_manager.py

import psycopg2
from threading import Lock
import contextlib
from psycopg2.extras import RealDictCursor
import psycopg2.pool
import opentelemetry.instrumentation.psycopg2

opentelemetry.instrumentation.psycopg2.Psycopg2Instrumentor().instrument()


class DatabaseManager:
    _instances: dict[str, dict] = {}
    _lock = Lock()
    _initialized = False

    def __new__(cls, dsn: str):
        with cls._lock:
            if cls._instances.get(dsn) is None:
                cls._instances[dsn] = {
                    "instance": super(DatabaseManager, cls).__new__(cls),
                    "initialized": False,
                }
        return cls._instances[dsn]["instance"]

    def __init__(self, dsn: str):
        if not DatabaseManager._instances[dsn]["initialized"]:
            DatabaseManager._instances[dsn]["instance"].initialize(dsn)

    def initialize(self, dsn: str = ""):
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=20,  # Adjust based on your needs
            dsn=dsn,
        )

        DatabaseManager._instances[dsn]["initialized"] = True

    @contextlib.contextmanager
    def get_connection(self):
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)

    @contextlib.contextmanager
    def get_cursor(self):
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            try:
                yield cursor
            finally:
                conn.commit()
                cursor.close()

    def execute_query(self, query, params=None):
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description:
                return cursor.fetchall()
            return None

    def execute_and_fetch_one(self, query, params=None):
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def execute_and_return_id(self, query, params=None):
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()["id"]
