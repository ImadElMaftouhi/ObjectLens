# db/Dockerfile
FROM mongo:7

# Default database name (optional but convenient)
ENV MONGO_INITDB_DATABASE=objectlens

# Initialization script: create "images" and "objects" collections
COPY init-mongo.js /docker-entrypoint-initdb.d/

EXPOSE 27017

CMD ["mongod", "--bind_ip_all"]
