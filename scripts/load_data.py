import psycopg2
import pandas as pd

# Connect to Neon
conn = psycopg2.connect("postgresql://neondb_owner:npg_S8dxteJ7yYmo@ep-round-river-a59jaxkv-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require")
cur = conn.cursor()

# Load CSV (adjust path if needed)
df = pd.read_csv('data/income_dataset.csv')

# Insert Categorical Values
for workclass in df['workclass'].dropna().unique():
    cur.execute("INSERT INTO Workclass (workclass_name) VALUES (%s) ON CONFLICT DO NOTHING", (workclass,))
for education in df['education'].dropna().unique():
    cur.execute("INSERT INTO Education (education_level) VALUES (%s) ON CONFLICT DO NOTHING", (education,))
for marital in df['marital-status'].dropna().unique():
    cur.execute("INSERT INTO MaritalStatus (marital_status) VALUES (%s) ON CONFLICT DO NOTHING", (marital,))
for occupation in df['occupation'].dropna().unique():
    cur.execute("INSERT INTO Occupation (occupation_name) VALUES (%s) ON CONFLICT DO NOTHING", (occupation,))
for relationship in df['relationship'].dropna().unique():
    cur.execute("INSERT INTO Relationship (relationship_type) VALUES (%s) ON CONFLICT DO NOTHING", (relationship,))
for race in df['race'].dropna().unique():
    cur.execute("INSERT INTO Race (race_name) VALUES (%s) ON CONFLICT DO NOTHING", (race,))
for gender in df['gender'].dropna().unique():
    cur.execute("INSERT INTO Gender (gender) VALUES (%s) ON CONFLICT DO NOTHING", (gender,))
for country in df['native-country'].dropna().unique():
    cur.execute("INSERT INTO NativeCountry (country_name) VALUES (%s) ON CONFLICT DO NOTHING", (country,))

# # Insert Individuals with corrected boolean conversion
# for i, row in df.iterrows():
#     cur.execute("""
#         INSERT INTO Individuals (age, fnlwgt, educational_num, capital_gain, capital_loss, hours_per_week, income_greater_50k,
#                                  workclass_id, education_id, marital_status_id, occupation_id, relationship_id, race_id, gender_id, country_id)
#         VALUES (%s, %s, %s, %s, %s, %s, %s,
#                 (SELECT workclass_id FROM Workclass WHERE workclass_name = %s),
#                 (SELECT education_id FROM Education WHERE education_level = %s),
#                 (SELECT marital_status_id FROM MaritalStatus WHERE marital_status = %s),
#                 (SELECT occupation_id FROM Occupation WHERE occupation_name = %s),
#                 (SELECT relationship_id FROM Relationship WHERE relationship_type = %s),
#                 (SELECT race_id FROM Race WHERE race_name = %s),
#                 (SELECT gender_id FROM Gender WHERE gender = %s),
#                 (SELECT country_id FROM NativeCountry WHERE country_name = %s))
#     """, (row['age'], row['fnlwgt'], row['educational-num'], row['capital-gain'], row['capital-loss'], row['hours-per-week'], bool(row['income_>50K']),
#           row['workclass'], row['education'], row['marital-status'], row['occupation'], row['relationship'], row['race'], row['gender'], row['native-country']))

conn.commit()
cur.close()
conn.close()