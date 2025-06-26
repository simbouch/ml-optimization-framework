-- Initialize Optuna database
-- This script sets up the database for Optuna studies

-- Create database if not exists (already created by POSTGRES_DB)
-- CREATE DATABASE IF NOT EXISTS optuna;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE optuna TO optuna;

-- Create schema for Optuna (Optuna will create tables automatically)
-- The Optuna library will handle table creation when first used
