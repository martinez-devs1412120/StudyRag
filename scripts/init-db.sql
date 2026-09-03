-- PostgreSQL init script: runs once on first container startup.
-- Mounted into /docker-entrypoint-initdb.d/ by docker-compose.yml.
CREATE EXTENSION IF NOT EXISTS vector;
