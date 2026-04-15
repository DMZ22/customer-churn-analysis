-- Customer Churn Analysis Queries

-- 1. Overall churn rate
SELECT
    churn,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage
FROM customers
GROUP BY churn;

-- 2. Average monthly charges by churn status
SELECT
    churn,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(MIN(monthly_charges), 2) AS min_monthly_charges,
    ROUND(MAX(monthly_charges), 2) AS max_monthly_charges
FROM customers
GROUP BY churn;

-- 3. Tenure vs churn analysis
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_group,
    churn,
    COUNT(*) AS count,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customers
GROUP BY tenure_group, churn
ORDER BY tenure_group, churn;

-- 4. Contract type churn analysis
SELECT
    contract_type,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM customers
GROUP BY contract_type
ORDER BY churn_rate DESC;

-- 5. Payment method churn analysis
SELECT
    payment_method,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM customers
GROUP BY payment_method
ORDER BY churn_rate DESC;

-- 6. Internet service churn analysis
SELECT
    internet_service,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM customers
GROUP BY internet_service
ORDER BY churn_rate DESC;

-- 7. Gender-based churn
SELECT
    gender,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM customers
GROUP BY gender;

-- 8. High-risk customers (month-to-month, high charges, low tenure)
SELECT
    customer_id,
    tenure,
    monthly_charges,
    contract_type,
    internet_service
FROM customers
WHERE churn = 'No'
    AND contract_type = 'Month-to-month'
    AND tenure < 12
    AND monthly_charges > 70
ORDER BY monthly_charges DESC
LIMIT 20;

-- 9. Average tenure by contract type and churn
SELECT
    contract_type,
    churn,
    ROUND(AVG(tenure), 1) AS avg_tenure,
    ROUND(AVG(age), 1) AS avg_age
FROM customers
GROUP BY contract_type, churn
ORDER BY contract_type;

-- 10. Revenue impact of churn
SELECT
    churn,
    COUNT(*) AS customers,
    ROUND(SUM(monthly_charges), 2) AS total_monthly_revenue,
    ROUND(AVG(total_charges), 2) AS avg_lifetime_value
FROM customers
GROUP BY churn;
