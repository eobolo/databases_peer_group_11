# api/main.py
from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import re
from api.database import get_db_connection, resolve_category, get_unique_values
from api.models import IndividualCreate, IndividualUpdate, IncomeLogCreate, IncomeLogUpdate

app = FastAPI()

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
    "age": (0, 120),
    "fnlwgt": (0, 10_000_000),
    "educational_num": (1, 16),
    "capital_gain": (0, 999_999),
    "capital_loss": (0, 999_999),
    "hours_per_week": (1, 168)
}

# Helper functions
def validate_numeric_bounds(column: str, value: int) -> int:
    if column in NUMERIC_BOUNDS:
        min_val, max_val = NUMERIC_BOUNDS[column]
        if not (min_val <= value <= max_val):
            raise HTTPException(400, f"{column} must be between {min_val} and {max_val}")
    return value

def parse_filter(filter_str: str) -> tuple[str, str, any]:
    print(f"Parsing filter: {filter_str}")  # Debug log
    match = re.match(r"(\w+)\s*(>=|<=|!=|=|>|<|LIKE)\s*(.+)", filter_str.strip())
    if not match:
        raise HTTPException(400, f"Invalid filter format. Use 'column operator value' (e.g., 'age > 30'). Got: {filter_str}")
    column, operator, value = match.groups()
    print(f"Parsed: column={column}, operator={operator}, value={value}")  # Debug log
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
    prefix = "i." if use_alias else ""  # Use alias only for SELECT queries
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

# CRUD for Individuals
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
        where_clause, where_params = build_where_clause(filter, cur, use_alias=False)  # No alias for UPDATE
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
        where_clause, params = build_where_clause(filter, cur, use_alias=False)  # No alias for DELETE
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

# CRUD for Income_Log
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
            conditions.append("action_taken = %s")  # Use the correct column name
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

# Endpoint for unique categorical values
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)