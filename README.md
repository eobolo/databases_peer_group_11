# Income Dataset Database Project (Group 11)

This project implements a PostgreSQL database in Neon (`neondb`) for an ML-relevant income dataset with 43,957 rows. I’ve completed the schema, ERD, and data loading script (not executed yet). This README is your complete guide to setting up, connecting to, and verifying the database. Follow these steps to get started and contribute.

## Project Overview
- **Database**: PostgreSQL hosted on Neon (`neondb`).
- **Schema**: 10 tables (`Workclass`, `Education`, ..., `Individuals`, `Income_Log`) with `CHECK` constraints, a stored procedure (`flag_high_capital_gains`), and a trigger (`income_update_trigger`).
- **Data**: Script to load 43,957 rows from the Adult Income dataset (optional, not loaded yet).
- **Files**:
  - `scripts/schema.sql`: Full schema definition.
  - `scripts/load_data.py`: Python script to load data.
  - `docs/erd.png`: ERD diagram.
  - `docs/README.md`: This file.
- **Dataset**: [Kaggle Adult Income](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset).

---

## Step 1: Prerequisites
Install these tools to work with the project.

### Install PostgreSQL Client (psql)
- **Why**: Run SQL scripts via terminal.
- **Windows**:
  1. Download from [postgresql.org/download/windows/](https://www.postgresql.org/download/windows/) (e.g., v17.x).
  2. Run installer → Select “Command Line Tools” → Install to `C:\Program Files\PostgreSQL\17\`.
- **macOS**: `brew install postgresql` (via Homebrew).
- **Linux**: `sudo apt install postgresql-client` (Ubuntu).
- **Verify**: `psql --version` → Expect `psql (PostgreSQL) 17.x`.

### Set PostgreSQL as Environment Variable (Windows)
- **Why**: Access `psql` from any terminal.
- **Steps**:
  1. Right-click “This PC” → “Properties” → “Advanced system settings” → “Environment Variables”.
  2. Under “System variables” → `Path` → “Edit” → Add: `C:\Program Files\PostgreSQL\17\bin`.
  3. OK → Restart terminal → `psql --version`.

### Install Python and Dependencies
- **Why**: Run `load_data.py` (Don't run).
- **Steps**:
  1. Install Python 3.11/3.12 from [python.org](https://www.python.org/downloads/) → Check “Add Python to PATH”.
  2. Verify: `python --version`.
  3. Install:
     ```bash
     pip install psycopg2-binary pandas numpy
     ```

### Install Git
- **Why**: Clone this repo.
- **Steps**: Download from [git-scm.com](https://git-scm.com/) → Install → `git --version`.

### Install pgAdmin
- **Why**: GUI for `neondb`.
- **Steps**: Included with PostgreSQL installer or download from [pgadmin.org](https://www.pgadmin.org/) (v8.x).

---

## Step 2: Clone the Repository
Get the project files:
```bash
git clone https://github.com/eobolo/databases_peer_group_11.git
cd databases_peer_group_11
```

## Step 3: Create virtual env
```terminal
py -m venv databases
cd databases
Scripts\activate.bat
cd ..
```

### Project Structure

```plaintext
databases_peer_group_11/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── database.py
│   ├── models.py
├── scripts/
│   ├── __init__.py
│   ├── predict_from_api.py
│   ├── schema.sql
│   ├── load_data.py
├── docs/
│   ├── income_dataset_erd.png
├── encoders/
│   ├── label_encoder_education.pkl
│   ├── label_encoder_gender.pkl
│   ├── label_encoder_marital-status.pkl
│   ├── label_encoder_native-country.pkl
│   ├── label_encoder_occupation.pkl
│   ├── label_encoder_race.pkl
│   ├── label_encoder_relationship.pkl
│   ├── label_encoder_workclass.pkl
│   ├── standard_scaler.pkl
├── model/
│   ├── Income_Prediction_Model.ipynb
│   ├── Income_Prediction_Model.py
├── saved_models/
│   ├── Model 1.keras
│   ├── Model 2.keras
│   ├── Model 3.keras
├── data/
│   ├── income_dataset.csv
    ├── train.csv (contains NAN)
└── .gitignore
└── README.md
└── requirements.txt
```

## Step 4: Connect to Neon (neondb)

The database is live—here’s how to access it.

### Connection String
```plaintext
postgresql://neondb_owner:npg_S8dxteJ7yYmo@ep-round-river-a59jaxkv-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require
```


### Connection Details

- **Host:** `ep-round-river-a59jaxkv-pooler.us-east-2.aws.neon.tech`
- **Database:** `neondb`
- **Username:** `neondb_owner`
- **Password:** `npg_S8dxteJ7yYmo`
- **SSL:** Required
---

### Via Terminal (psql)

#### Test Connection

```bash
psql "postgresql://neondb_owner:npg_S8dxteJ7yYmo@ep-round-river-a59jaxkv-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
```
If successful, you should see the prompt:
```plaintext
neondb=>
```

List Tables:
```sql
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
```

## How to run the sever and Script
- Sever: python -m api.main
- Script: python -m scripts.predict_from_api

## PROJECT API DOCUMENTATIONS
- [Swagger Documentation](https://databases-and-apis.onrender.com/docs)
- [Redoc Documentation](https://databases-and-apis.onrender.com/redoc)
