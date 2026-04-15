-- Customer Churn Database Schema
-- Database: customer_churn_db

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    gender TEXT NOT NULL CHECK (gender IN ('Male', 'Female')),
    age INTEGER NOT NULL CHECK (age >= 18 AND age <= 100),
    tenure INTEGER NOT NULL CHECK (tenure >= 0),
    monthly_charges REAL NOT NULL CHECK (monthly_charges >= 0),
    total_charges REAL NOT NULL CHECK (total_charges >= 0),
    contract_type TEXT NOT NULL CHECK (contract_type IN ('Month-to-month', 'One year', 'Two year')),
    payment_method TEXT NOT NULL CHECK (payment_method IN ('Electronic check', 'Mailed check', 'Bank transfer', 'Credit card')),
    internet_service TEXT NOT NULL CHECK (internet_service IN ('DSL', 'Fiber optic', 'No')),
    churn TEXT NOT NULL CHECK (churn IN ('Yes', 'No'))
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_churn ON customers(churn);
CREATE INDEX IF NOT EXISTS idx_contract ON customers(contract_type);
CREATE INDEX IF NOT EXISTS idx_tenure ON customers(tenure);
