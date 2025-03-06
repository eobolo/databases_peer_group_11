-- Categorical Tables with CHECK Constraints

CREATE TABLE Workclass (
    workclass_id SERIAL PRIMARY KEY,
    workclass_name VARCHAR(50) UNIQUE,
    CONSTRAINT valid_workclass CHECK (workclass_name IN ('Private', 'State-gov', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Never-worked', 'Without-pay') OR workclass_name IS NULL)
);

CREATE TABLE Education (
    education_id SERIAL PRIMARY KEY,
    education_level VARCHAR(50) UNIQUE,
    CONSTRAINT valid_education CHECK (education_level IN ('Doctorate', '12th', 'Bachelors', '7th-8th', 'Some-college', 'HS-grad', '9th', '10th', '11th', 'Masters', 'Preschool', '5th-6th', 'Prof-school', 'Assoc-voc', '1st-4th', 'Assoc-acdm') OR education_level IS NULL)
);

CREATE TABLE MaritalStatus (
    marital_status_id SERIAL PRIMARY KEY,
    marital_status VARCHAR(50) UNIQUE,
    CONSTRAINT valid_marital_status CHECK (marital_status IN ('Divorced', 'Never-married', 'Married-civ-spouse', 'Widowed', 'Separated', 'Married-spouse-absent', 'Married-AF-spouse') OR marital_status IS NULL)
);

CREATE TABLE Occupation (
    occupation_id SERIAL PRIMARY KEY,
    occupation_name VARCHAR(50) UNIQUE,
    CONSTRAINT valid_occupation CHECK (occupation_name IN ('Exec-managerial', 'Other-service', 'Transport-moving', 'Adm-clerical', 'Machine-op-inspct', 'Sales', 'Handlers-cleaners', 'Farming-fishing', 'Protective-serv', 'Prof-specialty', 'Craft-repair', 'Tech-support', 'Priv-house-serv', 'Armed-Forces') OR occupation_name IS NULL)
);

CREATE TABLE Relationship (
    relationship_id SERIAL PRIMARY KEY,
    relationship_type VARCHAR(50) UNIQUE,
    CONSTRAINT valid_relationship CHECK (relationship_type IN ('Not-in-family', 'Own-child', 'Husband', 'Wife', 'Unmarried', 'Other-relative') OR relationship_type IS NULL)
);

CREATE TABLE Race (
    race_id SERIAL PRIMARY KEY,
    race_name VARCHAR(50) UNIQUE,
    CONSTRAINT valid_race CHECK (race_name IN ('White', 'Black', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo') OR race_name IS NULL)
);

CREATE TABLE Gender (
    gender_id SERIAL PRIMARY KEY,
    gender VARCHAR(20) UNIQUE,
    CONSTRAINT valid_gender CHECK (gender IN ('Male', 'Female') OR gender IS NULL)
);

CREATE TABLE NativeCountry (
    country_id SERIAL PRIMARY KEY,
    country_name VARCHAR(50) UNIQUE,
    CONSTRAINT valid_country CHECK (country_name IN ('United-States', 'Japan', 'South', 'Portugal', 'Italy', 'Mexico', 'Ecuador', 'England', 'Philippines', 'China', 'Germany', 'Dominican-Republic', 'Jamaica', 'Vietnam', 'Thailand', 'Puerto-Rico', 'Cuba', 'India', 'Cambodia', 'Yugoslavia', 'Iran', 'El-Salvador', 'Poland', 'Greece', 'Ireland', 'Canada', 'Guatemala', 'Scotland', 'Columbia', 'Outlying-US(Guam-USVI-etc)', 'Haiti', 'Peru', 'Nicaragua', 'Taiwan', 'France', 'Trinadad&Tobago', 'Laos', 'Hungary', 'Honduras', 'Hong', 'Holand-Netherlands') OR country_name IS NULL)
);

-- Main Table: Individuals
CREATE TABLE Individuals (
    individual_id SERIAL PRIMARY KEY,
    age INTEGER,
    fnlwgt INTEGER,
    educational_num INTEGER,
    capital_gain INTEGER,
    capital_loss INTEGER,
    hours_per_week INTEGER,
    income_greater_50k BOOLEAN,
    workclass_id INTEGER REFERENCES Workclass(workclass_id),
    education_id INTEGER REFERENCES Education(education_id),
    marital_status_id INTEGER REFERENCES MaritalStatus(marital_status_id),
    occupation_id INTEGER REFERENCES Occupation(occupation_id),
    relationship_id INTEGER REFERENCES Relationship(relationship_id),
    race_id INTEGER REFERENCES Race(race_id),
    gender_id INTEGER REFERENCES Gender(gender_id),
    country_id INTEGER REFERENCES NativeCountry(country_id)
);

-- Logging Table: Income_Log
CREATE TABLE Income_Log (
    log_id SERIAL PRIMARY KEY,
    individual_id INTEGER REFERENCES Individuals(individual_id),
    log_timestamp TIMESTAMP,
    action_taken VARCHAR(50)
);

-- Stored Procedure: Flag high-income individuals with significant capital gain
CREATE OR REPLACE PROCEDURE flag_high_capital_gains()
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE Individuals
    SET income_greater_50k = TRUE
    WHERE capital_gain > 50000 AND income_greater_50k IS NULL;
END;
$$;

-- Trigger: Log changes to income_greater_50k
CREATE OR REPLACE FUNCTION log_income_change()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.income_greater_50k IS DISTINCT FROM OLD.income_greater_50k THEN
        INSERT INTO Income_Log (individual_id, log_timestamp, action_taken)
        VALUES (NEW.individual_id, NOW(), 'Income Updated');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER income_update_trigger
AFTER UPDATE OF income_greater_50k ON Individuals
FOR EACH ROW EXECUTE FUNCTION log_income_change();