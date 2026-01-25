FROM mongo:7.0

# Copy initialization script
COPY init-mongo.js /docker-entrypoint-initdb.d/

# Create directory for MongoDB data
RUN mkdir -p /data/db

# Expose MongoDB port
EXPOSE 27017

# Use default MongoDB entrypoint
# It will automatically run scripts in /docker-entrypoint-initdb.d/
