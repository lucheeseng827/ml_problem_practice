-- Initialize ML Practice Database
-- This script runs automatically when PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schema for ML experiments
CREATE SCHEMA IF NOT EXISTS ml_experiments;

-- Table for tracking model experiments
CREATE TABLE IF NOT EXISTS ml_experiments.experiments (
    experiment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    dataset_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    description TEXT,
    tags JSONB,
    UNIQUE(experiment_name)
);

-- Table for model metrics
CREATE TABLE IF NOT EXISTS ml_experiments.metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES ml_experiments.experiments(experiment_id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    step INTEGER,
    epoch INTEGER,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Table for model hyperparameters
CREATE TABLE IF NOT EXISTS ml_experiments.hyperparameters (
    param_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES ml_experiments.experiments(experiment_id) ON DELETE CASCADE,
    param_name VARCHAR(100) NOT NULL,
    param_value TEXT,
    param_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for datasets metadata
CREATE TABLE IF NOT EXISTS ml_experiments.datasets (
    dataset_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(255) NOT NULL UNIQUE,
    dataset_path TEXT,
    dataset_size BIGINT,
    num_samples INTEGER,
    num_features INTEGER,
    target_column VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    schema_info JSONB
);

-- Table for model artifacts
CREATE TABLE IF NOT EXISTS ml_experiments.artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES ml_experiments.experiments(experiment_id) ON DELETE CASCADE,
    artifact_name VARCHAR(255) NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_type VARCHAR(100),
    file_size BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes for better query performance
CREATE INDEX idx_experiments_name ON ml_experiments.experiments(experiment_name);
CREATE INDEX idx_experiments_created_at ON ml_experiments.experiments(created_at);
CREATE INDEX idx_experiments_model_type ON ml_experiments.experiments(model_type);
CREATE INDEX idx_metrics_experiment ON ml_experiments.metrics(experiment_id);
CREATE INDEX idx_metrics_name ON ml_experiments.metrics(metric_name);
CREATE INDEX idx_hyperparams_experiment ON ml_experiments.hyperparameters(experiment_id);
CREATE INDEX idx_datasets_name ON ml_experiments.datasets(dataset_name);
CREATE INDEX idx_artifacts_experiment ON ml_experiments.artifacts(experiment_id);

-- Create a view for experiment summary
CREATE OR REPLACE VIEW ml_experiments.experiment_summary AS
SELECT
    e.experiment_id,
    e.experiment_name,
    e.model_type,
    e.dataset_name,
    e.created_at,
    COUNT(DISTINCT m.metric_id) as num_metrics,
    COUNT(DISTINCT h.param_id) as num_hyperparameters,
    COUNT(DISTINCT a.artifact_id) as num_artifacts
FROM ml_experiments.experiments e
LEFT JOIN ml_experiments.metrics m ON e.experiment_id = m.experiment_id
LEFT JOIN ml_experiments.hyperparameters h ON e.experiment_id = h.experiment_id
LEFT JOIN ml_experiments.artifacts a ON e.experiment_id = a.experiment_id
GROUP BY e.experiment_id, e.experiment_name, e.model_type, e.dataset_name, e.created_at;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA ml_experiments TO mluser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_experiments TO mluser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_experiments TO mluser;

-- Insert sample experiment for testing
INSERT INTO ml_experiments.experiments (
    experiment_name,
    model_type,
    dataset_name,
    description,
    tags
) VALUES (
    'sample_experiment',
    'random_forest',
    'iris',
    'Sample experiment to test database setup',
    '{"status": "test", "category": "classification"}'::jsonb
) ON CONFLICT (experiment_name) DO NOTHING;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'ML Practice database initialized successfully!';
END $$;
