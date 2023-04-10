CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- Create entity table
CREATE TABLE IF NOT EXISTS Data (
  id uuid DEFAULT uuid_generate_v4 (),
  name VARCHAR,
  requestId VARCHAR,
  PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS Prediction (
  id uuid DEFAULT uuid_generate_v4 (),
  name VARCHAR,
  requestId VARCHAR,
  PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS DataQualityReport (
  id uuid DEFAULT uuid_generate_v4 (),
  value json  NOT NULL,
  requestId VARCHAR,
  PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS PredictionQuality (
  id uuid DEFAULT uuid_generate_v4 (),
  value json  NOT NULL,
  requestId VARCHAR,
  PRIMARY KEY(id)
);
-- Create activity table

CREATE TABLE IF NOT EXISTS Predict (
  id uuid DEFAULT uuid_generate_v4 (),
  start_time timestamp, 
  end_time timestamp,
  requestId VARCHAR,
  PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS AssessDataQuality (
  id uuid DEFAULT uuid_generate_v4 (),
  name VARCHAR,
  requestId VARCHAR,
  PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS Preprocess (
  id uuid DEFAULT uuid_generate_v4 (),
  name VARCHAR,
  requestId VARCHAR,
  PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS EnsembleFunction (
  id uuid DEFAULT uuid_generate_v4 (),
  name VARCHAR,
  requestId VARCHAR,
  PRIMARY KEY(id)
);
-- Create agent table
CREATE TABLE IF NOT EXISTS Model (
  id uuid DEFAULT uuid_generate_v4 (),
  name VARCHAR,
  parameter json NOT NULL, 
  PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS PreprocessingUtil (
  id uuid DEFAULT uuid_generate_v4 (),
  name VARCHAR,
  PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS DataQualityAssessor (
  id uuid DEFAULT uuid_generate_v4 (),
  name VARCHAR,
  PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS EnsembleUtil (
  id uuid DEFAULT uuid_generate_v4 (),
  name VARCHAR,
  PRIMARY KEY(id)
);
-- Create relationship table

CREATE TABLE IF NOT EXISTS WasAssociatedWith (
  activityId uuid,
  agentId uuid
);

CREATE TABLE IF NOT EXISTS Used (
  activityId uuid,
  entityId uuid
);

CREATE TABLE IF NOT EXISTS WasGeneratedBy (
  activityId uuid,
  entityId uuid
);


CREATE TABLE IF NOT EXISTS WasAttributedTo (
  activityId uuid,
  entityId uuid
);

CREATE TABLE IF NOT EXISTS WasDerivedFrom (
  sourceEntityId uuid,
  derivedEntityID uuid
);
