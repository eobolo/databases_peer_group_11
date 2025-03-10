# api/database.py
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException
import pymongo
from bson.objectid import ObjectId
import certifi

# PostgreSQL Configuration
DATABASE_URL = "postgresql://neondb_owner:npg_S8dxteJ7yYmo@ep-round-river-a59jaxkv-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"

def get_db_connection():
    """Establishes a connection to the Neon database."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def resolve_category(cur, table_info: dict, value: str | None) -> int | None:
    """Resolves a categorical value to its corresponding ID."""
    if value is None:
        return None
    table = table_info["table"]
    name_column = table_info["name_column"]
    id_column = table_info["id_column"]  # Use explicit id_column from table_info
    cur.execute(f"SELECT {id_column} FROM {table} WHERE {name_column} = %s", (value,))
    result = cur.fetchone()
    if result is None:
        raise HTTPException(status_code=400, detail=f"Invalid {name_column}: {value}")
    return result[id_column]

def get_unique_values(cur, column: str, table_map: dict) -> list[str]:
    """Fetches unique values for a categorical column."""
    if column not in table_map:
        raise HTTPException(status_code=400, detail=f"Column '{column}' is not categorical")
    table_info = table_map[column]
    table = table_info["table"]
    name_column = table_info["name_column"]
    cur.execute(f"SELECT DISTINCT {name_column} FROM {table} ORDER BY {name_column}")
    results = cur.fetchall()
    return [row[name_column] for row in results]

# MongoDB Configuration
MONGO_URI = "mongodb+srv://kihiupurity29:Kyla2054%40@cluster0.uqqy5.mongodb.net/database_and_apis?retryWrites=true&w=majority"

try:
    mongo_client = pymongo.MongoClient(
        MONGO_URI,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000
    )
    # Test the connection
    mongo_client.admin.command('ping')
    mongo_db = mongo_client["database_and_apis"]
    print("MongoDB connection successful!")
except Exception as e:
    raise Exception(f"MongoDB connection failed: {e}")

def resolve_mongo_category(collection_name: str, field_name: str, value: str) -> ObjectId:
    """Resolves a categorical value to its corresponding ObjectId in MongoDB."""
    doc = mongo_db[collection_name].find_one({field_name: value})
    if doc:
        return doc["_id"]
    else:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}: {value}")