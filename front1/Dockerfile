# frontend/Dockerfile
FROM node:22-alpine

WORKDIR /usr/src/app

# Install deps (good caching)
COPY package*.json ./
RUN npm install

# Copy the rest of the Vite app
COPY . .

# Vite dev server default port
EXPOSE 5173

# Run dev server accessible from outside container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
