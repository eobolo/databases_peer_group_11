# api/main.py
from api.database import get_db_connection, resolve_category, get_unique_values, mongo_db, resolve_mongo_category
from api.models import IndividualCreate, IndividualUpdate, IncomeLogCreate, IncomeLogUpdate, MongoIncomeLogCreate, MongoIncomeLogUpdate, IncomeData
from bson.objectid import ObjectId
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
import re
from tensorflow.keras.models import load_model
from typing import List, Optional

app = FastAPI()

# Load preprocessors and models
l_encoders = {}
# Update this block after app = FastAPI()
categorical_cols = ['gender', 'workclass', 'education', 'marital-status', 
                    'occupation', 'relationship', 'race', 'native-country']
numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 
                  'capital-loss', 'hours-per-week']
expected_cols = [
    'age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week',
    'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 
    'gender', 'native-country'
]

for col in categorical_cols:
    encoder_file = f'encoders/label_encoder_{col}.pkl'  # Matches your ls output
    l_encoders[col] = joblib.load(encoder_file)
scaler = joblib.load('encoders/standard_scaler.pkl')

models = {
    'Model 1': load_model('saved_models/Model 1.keras'),
    'Model 2': load_model('saved_models/Model 2.keras'),
    'Model 3': load_model('saved_models/Model 3.keras')
}
available_models = list(models.keys())

# Column metadata for validation
COLUMN_TYPES = {
    "individual_id": "INTEGER",
    "age": "INTEGER",
    "fnlwgt": "INTEGER",
    "educational_num": "INTEGER",
    "capital_gain": "INTEGER",
    "capital_loss": "INTEGER",
    "hours_per_week": "INTEGER",
    "income_greater_50k": "BOOLEAN",
    "gender": "VARCHAR",
    "workclass": "VARCHAR",
    "education": "VARCHAR",
    "marital_status": "VARCHAR",
    "occupation": "VARCHAR",
    "relationship": "VARCHAR",
    "race": "VARCHAR",
    "country": "VARCHAR"
}

OPERATORS = {
    "INTEGER": ["=", ">", "<", ">=", "<=", "!="],
    "BOOLEAN": ["="],
    "VARCHAR": ["=", "!=", "LIKE"]
}

CATEGORICAL_TABLES = {
    "gender": {"table": "Gender", "name_column": "gender", "id_column": "gender_id"},
    "workclass": {"table": "Workclass", "name_column": "workclass_name", "id_column": "workclass_id"},
    "education": {"table": "Education", "name_column": "education_level", "id_column": "education_id"},
    "marital_status": {"table": "MaritalStatus", "name_column": "marital_status", "id_column": "marital_status_id"},
    "occupation": {"table": "Occupation", "name_column": "occupation_name", "id_column": "occupation_id"},
    "relationship": {"table": "Relationship", "name_column": "relationship_type", "id_column": "relationship_id"},
    "race": {"table": "Race", "name_column": "race_name", "id_column": "race_id"},
    "country": {"table": "NativeCountry", "name_column": "country_name", "id_column": "country_id"}
}

NUMERIC_BOUNDS = {
    "age": (13, 120),
    "fnlwgt": (0, 10_000_000),
    "educational_num": (1, 16),
    "capital_gain": (0, 999_999),
    "capital_loss": (0, 999_999),
    "hours_per_week": (1, 168)
}

# Helper functions for PostgreSQL
def validate_numeric_bounds(column: str, value: int) -> int:
    if column in NUMERIC_BOUNDS:
        min_val, max_val = NUMERIC_BOUNDS[column]
        if not (min_val <= value <= max_val):
            raise HTTPException(400, f"{column} must be between {min_val} and {max_val}")
    return value

def parse_filter(filter_str: str) -> tuple[str, str, any]:
    print(f"Parsing filter: {filter_str}")
    match = re.match(r"(\w+)\s*(>=|<=|!=|=|>|<|LIKE)\s*(.+)", filter_str.strip())
    if not match:
        raise HTTPException(400, f"Invalid filter format. Use 'column operator value' (e.g., 'age > 30'). Got: {filter_str}")
    column, operator, value = match.groups()
    print(f"Parsed: column={column}, operator={operator}, value={value}")
    if column not in COLUMN_TYPES:
        raise HTTPException(400, f"Invalid column: {column}")
    col_type = COLUMN_TYPES[column]
    if operator not in OPERATORS[col_type]:
        raise HTTPException(400, f"Operator '{operator}' not allowed for {col_type}")
    if col_type == "INTEGER":
        try:
            value = int(value)
            value = validate_numeric_bounds(column, value)
        except ValueError:
            raise HTTPException(400, f"Invalid integer value for {column}: {value}")
    elif col_type == "BOOLEAN":
        value = value.lower() in ["true", "1", "t"]
    return column, operator, value

def build_where_clause(filters: List[str], cur, use_alias: bool = True) -> tuple[str, list]:
    if not filters:
        return "", []
    conditions = []
    params = []
    prefix = "i." if use_alias else ""
    for filter_str in filters:
        column, operator, value = parse_filter(filter_str)
        if column in CATEGORICAL_TABLES:
            table_info = CATEGORICAL_TABLES[column]
            id_column = table_info["id_column"]
            resolved_id = resolve_category(cur, table_info, value)
            if resolved_id is None:
                raise HTTPException(400, f"Invalid {column} value: {value}")
            conditions.append(f"{prefix}{id_column} = %s")
            params.append(resolved_id)
        else:
            conditions.append(f"{prefix}{column} {operator} %s")
            params.append(value)
    return "WHERE " + " AND ".join(conditions), params

# MongoDB Helper Function
def build_mongo_query(filters: List[str]) -> dict:
    """Builds a MongoDB query from filter strings."""
    query = {}
    if not filters:
        return query
    for filter_str in filters:
        column, operator, value = parse_filter(filter_str)
        if column in CATEGORICAL_TABLES:
            collection_name = CATEGORICAL_TABLES[column]["table"].lower()
            field_name = CATEGORICAL_TABLES[column]["name_column"]
            category_id = resolve_mongo_category(collection_name, field_name, value)
            query[f"{column}_id"] = category_id
        else:
            col_type = COLUMN_TYPES[column]
            if col_type == "INTEGER":
                try:
                    value = int(value)  # Convert string to int if needed (e.g., from URL)
                except ValueError:
                    raise HTTPException(400, f"Invalid integer value for {column}: {value}")
            # For BOOLEAN, value is already True/False from parse_filter, so no further conversion
            if operator == "=":
                query[column] = value
            elif operator == ">":
                query[column] = {"$gt": value}
            elif operator == "<":
                query[column] = {"$lt": value}
            elif operator == ">=":
                query[column] = {"$gte": value}
            elif operator == "<=":
                query[column] = {"$lte": value}
            elif operator == "!=":
                query[column] = {"$ne": value}
    return query

# preprocessing data
def preprocess_data(data: pd.DataFrame):
    training_order = [
        'age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week',
        'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 
        'gender', 'native-country'
    ]
    for col in training_order:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")
    for col in categorical_cols:
        data[col] = data[col].apply(lambda x: l_encoders[col].transform([x])[0] 
                                   if x in l_encoders[col].classes_ 
                                   else l_encoders[col].transform([l_encoders[col].classes_[0]])[0])
    # Reorder columns to match training order
    data = data[training_order]
    data_scaled = scaler.transform(data)
    return data_scaled

@app.get("/columns/", response_model=list[str])
def get_all_columns():
    """Returns a list of all available column names."""
    return list(COLUMN_TYPES.keys())

# PostgreSQL CRUD Endpoints (Unchanged)
@app.get("/individuals/", response_model=list[dict])
def read_individuals(
    filter: List[str] = Query(None, description="Filters: 'column operator value'"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    order_by: Optional[str] = Query(None, description="Sort: 'column direction'")
):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        where_clause, params = build_where_clause(filter, cur, use_alias=True)
        order_clause = ""
        if order_by:
            match = re.match(r"(\w+)\s*(asc|desc)?", order_by.lower())
            if not match or match.group(1) not in COLUMN_TYPES:
                raise HTTPException(400, "Invalid order_by format or column")
            col, direction = match.groups()
            order_clause = f"ORDER BY {col} {direction or 'asc'}"
        query = f"""
            SELECT 
                i.individual_id,
                i.age,
                i.fnlwgt,
                i.educational_num,
                i.capital_gain,
                i.capital_loss,
                i.hours_per_week,
                i.income_greater_50k,
                g.gender AS gender,
                w.workclass_name AS workclass,
                e.education_level AS education,
                m.marital_status AS marital_status,
                o.occupation_name AS occupation,
                r.relationship_type AS relationship,
                ra.race_name AS race,
                c.country_name AS country
            FROM Individuals i
            LEFT JOIN Gender g ON i.gender_id = g.gender_id
            LEFT JOIN Workclass w ON i.workclass_id = w.workclass_id
            LEFT JOIN Education e ON i.education_id = e.education_id
            LEFT JOIN MaritalStatus m ON i.marital_status_id = m.marital_status_id
            LEFT JOIN Occupation o ON i.occupation_id = o.occupation_id
            LEFT JOIN Relationship r ON i.relationship_id = r.relationship_id
            LEFT JOIN Race ra ON i.race_id = ra.race_id
            LEFT JOIN NativeCountry c ON i.country_id = c.country_id
            {where_clause}
            {order_clause}
            LIMIT %s OFFSET %s
        """
        cur.execute(query, params + [limit, offset])
        results = cur.fetchall()
        return [dict(row) for row in results]
    except Exception as e:
        raise HTTPException(400, f"Query failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.get("/individuals/{individual_id}", response_model=dict)
def read_individual_by_id(individual_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = """
            SELECT 
                i.individual_id,
                i.age,
                i.fnlwgt,
                i.educational_num,
                i.capital_gain,
                i.capital_loss,
                i.hours_per_week,
                i.income_greater_50k,
                g.gender AS gender,
                w.workclass_name AS workclass,
                e.education_level AS education,
                m.marital_status AS marital_status,
                o.occupation_name AS occupation,
                r.relationship_type AS relationship,
                ra.race_name AS race,
                c.country_name AS country
            FROM Individuals i
            LEFT JOIN Gender g ON i.gender_id = g.gender_id
            LEFT JOIN Workclass w ON i.workclass_id = w.workclass_id
            LEFT JOIN Education e ON i.education_id = e.education_id
            LEFT JOIN MaritalStatus m ON i.marital_status_id = m.marital_status_id
            LEFT JOIN Occupation o ON i.occupation_id = o.occupation_id
            LEFT JOIN Relationship r ON i.relationship_id = r.relationship_id
            LEFT JOIN Race ra ON i.race_id = ra.race_id
            LEFT JOIN NativeCountry c ON i.country_id = c.country_id
            WHERE i.individual_id = %s
        """
        cur.execute(query, (individual_id,))
        result = cur.fetchone()
        if result is None:
            raise HTTPException(404, f"Individual with ID {individual_id} not found")
        return dict(result)
    except Exception as e:
        raise HTTPException(400, f"Query failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.post("/individuals/", response_model=dict)
def create_individual(individual: IndividualCreate):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        gender_id = resolve_category(cur, CATEGORICAL_TABLES["gender"], individual.gender)
        workclass_id = resolve_category(cur, CATEGORICAL_TABLES["workclass"], individual.workclass)
        education_id = resolve_category(cur, CATEGORICAL_TABLES["education"], individual.education)
        marital_status_id = resolve_category(cur, CATEGORICAL_TABLES["marital_status"], individual.marital_status)
        occupation_id = resolve_category(cur, CATEGORICAL_TABLES["occupation"], individual.occupation)
        relationship_id = resolve_category(cur, CATEGORICAL_TABLES["relationship"], individual.relationship)
        race_id = resolve_category(cur, CATEGORICAL_TABLES["race"], individual.race)
        country_id = resolve_category(cur, CATEGORICAL_TABLES["country"], individual.country)
        
        cur.execute("""
            INSERT INTO Individuals (age, fnlwgt, educational_num, capital_gain, capital_loss, hours_per_week, income_greater_50k,
                                    gender_id, workclass_id, education_id, marital_status_id, occupation_id, relationship_id, race_id, country_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING individual_id
        """, (individual.age, individual.fnlwgt, individual.educational_num, individual.capital_gain, individual.capital_loss,
              individual.hours_per_week, individual.income_greater_50k, gender_id, workclass_id, education_id,
              marital_status_id, occupation_id, relationship_id, race_id, country_id))
        individual_id = cur.fetchone()["individual_id"]
        conn.commit()
        return {"individual_id": individual_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(400, f"Creation failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.put("/individuals/", response_model=dict)
def update_individuals(
    update_data: IndividualUpdate,
    filter: List[str] = Query(..., description="Filters: 'column operator value' (required)"),
):
    if not filter:
        raise HTTPException(400, "At least one filter required for update")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        where_clause, where_params = build_where_clause(filter, cur, use_alias=False)
        updates = {k: v for k, v in update_data.dict().items() if v is not None}
        if not updates:
            raise HTTPException(400, "No fields to update")
        if "gender" in updates:
            updates["gender_id"] = resolve_category(cur, CATEGORICAL_TABLES["gender"], updates.pop("gender"))
        if "workclass" in updates:
            updates["workclass_id"] = resolve_category(cur, CATEGORICAL_TABLES["workclass"], updates.pop("workclass"))
        if "education" in updates:
            updates["education_id"] = resolve_category(cur, CATEGORICAL_TABLES["education"], updates.pop("education"))
        if "marital_status" in updates:
            updates["marital_status_id"] = resolve_category(cur, CATEGORICAL_TABLES["marital_status"], updates.pop("marital_status"))
        if "occupation" in updates:
            updates["occupation_id"] = resolve_category(cur, CATEGORICAL_TABLES["occupation"], updates.pop("occupation"))
        if "relationship" in updates:
            updates["relationship_id"] = resolve_category(cur, CATEGORICAL_TABLES["relationship"], updates.pop("relationship"))
        if "race" in updates:
            updates["race_id"] = resolve_category(cur, CATEGORICAL_TABLES["race"], updates.pop("race"))
        if "country" in updates:
            updates["country_id"] = resolve_category(cur, CATEGORICAL_TABLES["country"], updates.pop("country"))
        set_clause = ", ".join(f"{k} = %s" for k in updates.keys())
        params = list(updates.values()) + where_params
        cur.execute(f"UPDATE Individuals SET {set_clause} {where_clause} RETURNING individual_id", params)
        updated = cur.fetchall()
        conn.commit()
        return {"updated_count": len(updated), "individual_ids": [row["individual_id"] for row in updated]}
    except Exception as e:
        conn.rollback()
        raise HTTPException(400, f"Update failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.put("/individuals/{individual_id}", response_model=dict)
def update_individual_by_id(individual_id: int, update_data: IndividualUpdate):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        updates = {k: v for k, v in update_data.dict().items() if v is not None}
        if not updates:
            raise HTTPException(400, "No fields to update")
        if "gender" in updates:
            updates["gender_id"] = resolve_category(cur, CATEGORICAL_TABLES["gender"], updates.pop("gender"))
        if "workclass" in updates:
            updates["workclass_id"] = resolve_category(cur, CATEGORICAL_TABLES["workclass"], updates.pop("workclass"))
        if "education" in updates:
            updates["education_id"] = resolve_category(cur, CATEGORICAL_TABLES["education"], updates.pop("education"))
        if "marital_status" in updates:
            updates["marital_status_id"] = resolve_category(cur, CATEGORICAL_TABLES["marital_status"], updates.pop("marital_status"))
        if "occupation" in updates:
            updates["occupation_id"] = resolve_category(cur, CATEGORICAL_TABLES["occupation"], updates.pop("occupation"))
        if "relationship" in updates:
            updates["relationship_id"] = resolve_category(cur, CATEGORICAL_TABLES["relationship"], updates.pop("relationship"))
        if "race" in updates:
            updates["race_id"] = resolve_category(cur, CATEGORICAL_TABLES["race"], updates.pop("race"))
        if "country" in updates:
            updates["country_id"] = resolve_category(cur, CATEGORICAL_TABLES["country"], updates.pop("country"))
        set_clause = ", ".join(f"{k} = %s" for k in updates.keys())
        params = list(updates.values()) + [individual_id]
        cur.execute(f"UPDATE Individuals SET {set_clause} WHERE individual_id = %s RETURNING individual_id", params)
        updated = cur.fetchone()
        if updated is None:
            raise HTTPException(404, f"Individual with ID {individual_id} not found")
        conn.commit()
        return {"individual_id": updated["individual_id"]}
    except Exception as e:
        conn.rollback()
        raise HTTPException(400, f"Update failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.delete("/individuals/", response_model=dict)
def delete_individuals(filter: List[str] = Query(..., description="Filters: 'column operator value' (required)")):
    if not filter:
        raise HTTPException(400, "At least one filter required for deletion")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        where_clause, params = build_where_clause(filter, cur, use_alias=False)
        cur.execute(f"DELETE FROM Individuals {where_clause} RETURNING individual_id", params)
        deleted = cur.fetchall()
        conn.commit()
        return {"deleted_count": len(deleted), "individual_ids": [row["individual_id"] for row in deleted]}
    except Exception as e:
        conn.rollback()
        raise HTTPException(400, f"Deletion failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.delete("/individuals/{individual_id}", response_model=dict)
def delete_individual_by_id(individual_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM Individuals WHERE individual_id = %s RETURNING individual_id", (individual_id,))
        deleted = cur.fetchone()
        if deleted is None:
            raise HTTPException(404, f"Individual with ID {individual_id} not found")
        conn.commit()
        return {"individual_id": deleted["individual_id"]}
    except Exception as e:
        conn.rollback()
        raise HTTPException(400, f"Deletion failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.get("/income_logs/", response_model=list[dict])
def read_income_logs(
    individual_id: Optional[int] = Query(None, description="Filter by individual_id"),
    action: Optional[str] = Query(None, description="Filter by action")
):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        conditions = []
        params = []
        if individual_id is not None:
            conditions.append("individual_id = %s")
            params.append(individual_id)
        if action is not None:
            conditions.append("action_taken = %s")
            params.append(action)
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        cur.execute(f"SELECT * FROM Income_Log {where_clause}", params)
        logs = cur.fetchall()
        return [dict(log) for log in logs]
    except Exception as e:
        raise HTTPException(400, f"Query failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.post("/income_logs/", response_model=dict)
def create_income_log(income_log: IncomeLogCreate):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO Income_Log (individual_id, action_taken, log_timestamp)
            VALUES (%s, %s, NOW())
            RETURNING log_id
        """, (income_log.individual_id, income_log.action))
        log_id = cur.fetchone()["log_id"]
        conn.commit()
        return {"log_id": log_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(400, f"Creation failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.put("/income_logs/{log_id}", response_model=dict)
def update_income_log(log_id: int, update_data: IncomeLogUpdate):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        updates = {k: v for k, v in update_data.dict().items() if v is not None}
        if not updates:
            raise HTTPException(400, "No fields to update")
        if "action" in updates:
            updates["action_taken"] = updates.pop("action")
        set_clause = ", ".join(f"{k} = %s" for k in updates.keys())
        params = list(updates.values()) + [log_id]
        cur.execute(f"UPDATE Income_Log SET {set_clause} WHERE log_id = %s RETURNING log_id", params)
        updated = cur.fetchone()
        if updated is None:
            raise HTTPException(404, f"Income log with ID {log_id} not found")
        conn.commit()
        return {"log_id": updated["log_id"]}
    except Exception as e:
        conn.rollback()
        raise HTTPException(400, f"Update failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.delete("/income_logs/", response_model=dict)
def delete_income_logs(
    individual_id: Optional[int] = Query(None, description="Filter by individual_id"),
    action: Optional[str] = Query(None, description="Filter by action")
):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        conditions = []
        params = []
        if individual_id is not None:
            conditions.append("individual_id = %s")
            params.append(individual_id)
        if action is not None:
            conditions.append("action_taken = %s")
            params.append(action)
        if not conditions:
            raise HTTPException(400, "At least one filter required for deletion")
        where_clause = "WHERE " + " AND ".join(conditions)
        cur.execute(f"DELETE FROM Income_Log {where_clause} RETURNING log_id", params)
        deleted = cur.fetchall()
        conn.commit()
        return {"deleted_count": len(deleted), "log_ids": [row["log_id"] for row in deleted]}
    except Exception as e:
        conn.rollback()
        raise HTTPException(400, f"Deletion failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.delete("/income_logs/{log_id}", response_model=dict)
def delete_income_log_by_id(log_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM Income_Log WHERE log_id = %s RETURNING log_id", (log_id,))
        deleted = cur.fetchone()
        if deleted is None:
            raise HTTPException(404, f"Income log with ID {log_id} not found")
        conn.commit()
        return {"log_id": deleted["log_id"]}
    except Exception as e:
        conn.rollback()
        raise HTTPException(400, f"Deletion failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.get("/unique/{column}/", response_model=list[str])
def get_unique_categorical_values(column: str):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        unique_values = get_unique_values(cur, column, CATEGORICAL_TABLES)
        return unique_values
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch unique values: {str(e)}")
    finally:
        cur.close()
        conn.close()

# MongoDB CRUD Endpoints
@app.get("/mongo/individuals/", response_model=list[dict])
def read_mongo_individuals(
    filter: List[str] = Query(None, description="Filters: 'column operator value'"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    order_by: Optional[str] = Query(None, description="Sort: 'column direction'")
):
    try:
        query = build_mongo_query(filter) if filter else {}
        pipeline = [{"$match": query}]
        
        # Add lookups for categorical fields
        for cat_field in CATEGORICAL_TABLES:
            collection_name = CATEGORICAL_TABLES[cat_field]["table"].lower()
            id_field = f"{cat_field}_id"
            name_field = CATEGORICAL_TABLES[cat_field]["name_column"]
            pipeline.extend([
                {"$lookup": {
                    "from": collection_name,
                    "localField": id_field,
                    "foreignField": "_id",
                    "as": f"{cat_field}_doc"
                }},
                {"$addFields": {
                    cat_field: {"$arrayElemAt": [f"${cat_field}_doc.{name_field}", 0]}
                }}
            ])
        
        # Project fields
        project = {
            "_id": 0,
            "individual_id": "$_id",
            "age": 1,
            "fnlwgt": 1,
            "educational_num": 1,
            "capital_gain": 1,
            "capital_loss": 1,
            "hours_per_week": 1,
            "income_greater_50k": 1,
        }
        for cat_field in CATEGORICAL_TABLES:
            project[cat_field] = 1
        pipeline.append({"$project": project})
        
        # Add sorting
        if order_by:
            match = re.match(r"(\w+)\s*(asc|desc)?", order_by.lower())
            if not match or match.group(1) not in COLUMN_TYPES:
                raise HTTPException(400, "Invalid order_by format or column")
            col, direction = match.groups()
            sort_order = 1 if direction == "asc" or not direction else -1
            pipeline.append({"$sort": {col: sort_order}})
        
        # Add skip and limit
        pipeline.extend([{"$skip": offset}, {"$limit": limit}])
        
        results = list(mongo_db.individuals.aggregate(pipeline))
        for doc in results:
            doc["individual_id"] = str(doc["individual_id"])
        return results
    except Exception as e:
        raise HTTPException(400, f"Query failed: {str(e)}")

@app.get("/mongo/individuals/{individual_id}", response_model=dict)
def read_mongo_individual_by_id(individual_id: str):
    try:
        pipeline = [{"$match": {"_id": ObjectId(individual_id)}}]
        for cat_field in CATEGORICAL_TABLES:
            collection_name = CATEGORICAL_TABLES[cat_field]["table"].lower()
            id_field = f"{cat_field}_id"
            name_field = CATEGORICAL_TABLES[cat_field]["name_column"]
            pipeline.extend([
                {"$lookup": {
                    "from": collection_name,
                    "localField": id_field,
                    "foreignField": "_id",
                    "as": f"{cat_field}_doc"
                }},
                {"$addFields": {
                    cat_field: {"$arrayElemAt": [f"${cat_field}_doc.{name_field}", 0]}
                }}
            ])
        project = {
            "_id": 0,
            "individual_id": "$_id",
            "age": 1,
            "fnlwgt": 1,
            "educational_num": 1,
            "capital_gain": 1,
            "capital_loss": 1,
            "hours_per_week": 1,
            "income_greater_50k": 1,
        }
        for cat_field in CATEGORICAL_TABLES:
            project[cat_field] = 1
        pipeline.append({"$project": project})
        
        result = list(mongo_db.individuals.aggregate(pipeline))
        if not result:
            raise HTTPException(404, f"Individual with ID {individual_id} not found")
        doc = result[0]
        doc["individual_id"] = str(doc["individual_id"])
        return doc
    except Exception as e:
        raise HTTPException(400, f"Query failed: {str(e)}")

@app.post("/mongo/individuals/", response_model=dict)
def create_mongo_individual(individual: IndividualCreate):
    try:
        doc = {
            "age": individual.age,
            "fnlwgt": individual.fnlwgt,
            "educational_num": individual.educational_num,
            "capital_gain": individual.capital_gain or 0,
            "capital_loss": individual.capital_loss or 0,
            "hours_per_week": individual.hours_per_week or 40,
            "income_greater_50k": individual.income_greater_50k or False,
        }
        for cat_field in CATEGORICAL_TABLES:
            value = getattr(individual, cat_field, None)
            if value:
                collection_name = CATEGORICAL_TABLES[cat_field]["table"].lower()
                field_name = CATEGORICAL_TABLES[cat_field]["name_column"]
                category_id = resolve_mongo_category(collection_name, field_name, value)
                doc[f"{cat_field}_id"] = category_id
            else:
                doc[f"{cat_field}_id"] = None
        result = mongo_db.individuals.insert_one(doc)
        return {"individual_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(400, f"Creation failed: {str(e)}")

@app.put("/mongo/individuals/{individual_id}", response_model=dict)
def update_mongo_individual_by_id(individual_id: str, update_data: IndividualUpdate):
    try:
        # Fetch the current document to check income_greater_50k before update
        current_doc = mongo_db.individuals.find_one({"_id": ObjectId(individual_id)})
        if not current_doc:
            raise HTTPException(404, f"Individual with ID {individual_id} not found")
        
        # Store the old income_greater_50k value (default to False if not present)
        old_income = current_doc.get("income_greater_50k", False)

        # Build the updates dictionary
        updates = {}
        for field, value in update_data.dict(exclude_unset=True).items():
            if field in CATEGORICAL_TABLES:
                if value is not None:
                    collection_name = CATEGORICAL_TABLES[field]["table"].lower()
                    field_name = CATEGORICAL_TABLES[field]["name_column"]
                    category_id = resolve_mongo_category(collection_name, field_name, value)
                    updates[f"{field}_id"] = category_id
                else:
                    updates[f"{field}_id"] = None
            else:
                updates[field] = value
        if not updates:
            raise HTTPException(400, "No fields to update")

        # Check if income_greater_50k is being updated and differs from the old value
        new_income = updates.get("income_greater_50k")
        if new_income is not None and new_income != old_income:
            # Log the change in income_log collection
            action = "Income updated to >50k" if new_income else "Income updated to <=50k"
            log_entry = {
                "individual_id": ObjectId(individual_id),
                "action_taken": action,
                "log_timestamp": datetime.utcnow()
            }
            mongo_db.income_log.insert_one(log_entry)

        # Perform the update
        result = mongo_db.individuals.update_one(
            {"_id": ObjectId(individual_id)},
            {"$set": updates}
        )
        if result.matched_count == 0:
            raise HTTPException(404, f"Individual with ID {individual_id} not found")
        return {"individual_id": individual_id}
    except Exception as e:
        raise HTTPException(400, f"Update failed: {str(e)}")

@app.delete("/mongo/individuals/{individual_id}", response_model=dict)
def delete_mongo_individual_by_id(individual_id: str):
    try:
        result = mongo_db.individuals.delete_one({"_id": ObjectId(individual_id)})
        if result.deleted_count == 0:
            raise HTTPException(404, f"Individual with ID {individual_id} not found")
        return {"individual_id": individual_id}
    except Exception as e:
        raise HTTPException(400, f"Deletion failed: {str(e)}")

@app.get("/mongo/income_logs/", response_model=list[dict])
def read_mongo_income_logs(
    individual_id: Optional[str] = Query(None, description="Filter by individual_id"),
    action: Optional[str] = Query(None, description="Filter by action")
):
    try:
        query = {}
        if individual_id:
            query["individual_id"] = ObjectId(individual_id)
        if action:
            query["action_taken"] = action
        logs = list(mongo_db.income_log.find(query))
        for log in logs:
            log["_id"] = str(log["_id"])
            log["individual_id"] = str(log["individual_id"])
            log["log_timestamp"] = log["log_timestamp"].isoformat()
        return logs
    except Exception as e:
        raise HTTPException(400, f"Query failed: {str(e)}")

@app.post("/mongo/income_logs/", response_model=dict)
def create_mongo_income_log(income_log: MongoIncomeLogCreate):
    try:
        doc = {
            "individual_id": ObjectId(income_log.individual_id),
            "action_taken": income_log.action,
            "log_timestamp": datetime.utcnow()
        }
        result = mongo_db.income_log.insert_one(doc)
        return {"log_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(400, f"Creation failed: {str(e)}")

@app.put("/mongo/income_logs/{log_id}", response_model=dict)
def update_mongo_income_log(log_id: str, update_data: MongoIncomeLogUpdate):
    try:
        updates = {}
        if update_data.individual_id is not None:
            updates["individual_id"] = ObjectId(update_data.individual_id)
        if update_data.action is not None:
            updates["action_taken"] = update_data.action
        if not updates:
            raise HTTPException(400, "No fields to update")
        result = mongo_db.income_log.update_one(
            {"_id": ObjectId(log_id)},
            {"$set": updates}
        )
        if result.matched_count == 0:
            raise HTTPException(404, f"Income log with ID {log_id} not found")
        return {"log_id": log_id}
    except Exception as e:
        raise HTTPException(400, f"Update failed: {str(e)}")

@app.delete("/mongo/income_logs/{log_id}", response_model=dict)
def delete_mongo_income_log_by_id(log_id: str):
    try:
        result = mongo_db.income_log.delete_one({"_id": ObjectId(log_id)})
        if result.deleted_count == 0:
            raise HTTPException(404, f"Income log with ID {log_id} not found")
        return {"log_id": log_id}
    except Exception as e:
        raise HTTPException(400, f"Deletion failed: {str(e)}")

@app.get("/mongo/unique/{column}/", response_model=list[str])
def get_mongo_unique_categorical_values(column: str):
    if column not in CATEGORICAL_TABLES:
        raise HTTPException(400, f"Column '{column}' is not categorical")
    collection_name = CATEGORICAL_TABLES[column]["table"].lower()
    field_name = CATEGORICAL_TABLES[column]["name_column"]
    try:
        values = mongo_db[collection_name].distinct(field_name)
        return sorted(values)
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch unique values: {str(e)}")
    
@app.post("/predict")
async def predict(data: IncomeData, model_name: str = Query("Model 3", enum=available_models)):
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Invalid model_name. Choose from {available_models}")
    selected_model = models[model_name]
    input_dict = data.model_dump()
    # Rename columns to match training data
    input_dict_renamed = {
        'age': input_dict['age'],
        'fnlwgt': input_dict['fnlwgt'],
        'educational-num': input_dict['educational_num'],
        'capital-gain': input_dict['capital_gain'],
        'capital-loss': input_dict['capital_loss'],
        'hours-per-week': input_dict['hours_per_week'],
        'gender': input_dict['gender'],
        'workclass': input_dict['workclass'],
        'education': input_dict['education'],
        'marital-status': input_dict['marital_status'],
        'occupation': input_dict['occupation'],
        'relationship': input_dict['relationship'],
        'race': input_dict['race'],
        'native-country': input_dict['country']
    }
    df = pd.DataFrame([input_dict_renamed])
    input_data = preprocess_data(df)
    prediction_prob = selected_model.predict(input_data)[0][0]
    prediction_class = int(prediction_prob >= 0.5)
    return {
        "model_used": model_name,
        "prediction_probability": float(prediction_prob),
        "prediction_class": prediction_class,
        "income_prediction": ">50K" if prediction_class == 1 else "<=50K"
    }

@app.get("/predict-latest")
async def predict_latest(model_name: str = Query("Model 3", enum=available_models)):
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Invalid model_name. Choose from {available_models}")
    selected_model = models[model_name]

    # Try PostgreSQL first
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Fetch the latest entry from PostgreSQL
        query = """
            SELECT 
                i.individual_id,
                i.age,
                i.fnlwgt,
                i.educational_num,
                i.capital_gain,
                i.capital_loss,
                i.hours_per_week,
                g.gender AS gender,
                w.workclass_name AS workclass,
                e.education_level AS education,
                m.marital_status AS marital_status,
                o.occupation_name AS occupation,
                r.relationship_type AS relationship,
                ra.race_name AS race,
                c.country_name AS country
            FROM Individuals i
            LEFT JOIN Gender g ON i.gender_id = g.gender_id
            LEFT JOIN Workclass w ON i.workclass_id = w.workclass_id
            LEFT JOIN Education e ON i.education_id = e.education_id
            LEFT JOIN MaritalStatus m ON i.marital_status_id = m.marital_status_id
            LEFT JOIN Occupation o ON i.occupation_id = o.occupation_id
            LEFT JOIN Relationship r ON i.relationship_id = r.relationship_id
            LEFT JOIN Race ra ON i.race_id = ra.race_id
            LEFT JOIN NativeCountry c ON i.country_id = c.country_id
            ORDER BY i.individual_id DESC
            LIMIT 1
        """
        cur.execute(query)
        postgres_entry = cur.fetchone()
        if postgres_entry:
            entry = dict(postgres_entry)
            source = "PostgreSQL"
        else:
            # If no entry in PostgreSQL, try MongoDB
            mongo_entry = mongo_db.individuals.find().sort('_id', -1).limit(1)
            mongo_entry = next(mongo_entry, None)
            if mongo_entry:
                entry = {
                    'individual_id': str(mongo_entry['_id']),
                    'age': mongo_entry['age'],
                    'fnlwgt': mongo_entry['fnlwgt'],
                    'educational_num': mongo_entry['educational_num'],
                    'capital_gain': mongo_entry.get('capital_gain', 0),
                    'capital_loss': mongo_entry.get('capital_loss', 0),
                    'hours_per_week': mongo_entry.get('hours_per_week', 40),
                    'gender': mongo_db.gender.find_one({'_id': mongo_entry.get('gender_id')}, {'gender': 1})['gender'] if mongo_entry.get('gender_id') else 'Male',
                    'workclass': mongo_db.workclass.find_one({'_id': mongo_entry.get('workclass_id')}, {'workclass_name': 1})['workclass_name'] if mongo_entry.get('workclass_id') else 'Private',
                    'education': mongo_db.education.find_one({'_id': mongo_entry.get('education_id')}, {'education_level': 1})['education_level'] if mongo_entry.get('education_id') else 'HS-grad',
                    'marital_status': mongo_db.maritalstatus.find_one({'_id': mongo_entry.get('marital_status_id')}, {'marital_status': 1})['marital_status'] if mongo_entry.get('marital_status_id') else 'Never-married',
                    'occupation': mongo_db.occupation.find_one({'_id': mongo_entry.get('occupation_id')}, {'occupation_name': 1})['occupation_name'] if mongo_entry.get('occupation_id') else 'Other-service',
                    'relationship': mongo_db.relationship.find_one({'_id': mongo_entry.get('relationship_id')}, {'relationship_type': 1})['relationship_type'] if mongo_entry.get('relationship_id') else 'Not-in-family',
                    'race': mongo_db.race.find_one({'_id': mongo_entry.get('race_id')}, {'race_name': 1})['race_name'] if mongo_entry.get('race_id') else 'White',
                    'country': mongo_db.nativecountry.find_one({'_id': mongo_entry.get('country_id')}, {'country_name': 1})['country_name'] if mongo_entry.get('country_id') else 'United-States'
                }
                source = "MongoDB"
            else:
                raise HTTPException(status_code=404, detail="No entries found in either PostgreSQL or MongoDB individuals collection")

        # Prepare data for prediction
        input_dict = {
            'age': entry['age'],
            'fnlwgt': entry['fnlwgt'],
            'educational-num': entry['educational_num'],
            'capital-gain': entry['capital_gain'],
            'capital-loss': entry['capital_loss'],
            'hours-per-week': entry['hours_per_week'],
            'gender': entry['gender'],
            'workclass': entry['workclass'],
            'education': entry['education'],
            'marital-status': entry['marital_status'],
            'occupation': entry['occupation'],
            'relationship': entry['relationship'],
            'race': entry['race'],
            'native-country': entry['country']
        }
        df = pd.DataFrame([input_dict])
        input_data = preprocess_data(df)
        prediction_prob = selected_model.predict(input_data)[0][0]
        prediction_class = int(prediction_prob >= 0.5)

        return {
            "source": source,
            "latest_entry": entry,
            "model_used": model_name,
            "prediction_probability": float(prediction_prob),
            "prediction_class": prediction_class,
            "income_prediction": ">50K" if prediction_class == 1 else "<=50K"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict latest: {str(e)}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)