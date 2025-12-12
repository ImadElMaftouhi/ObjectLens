FROM node:22

WORKDIR /usr/src/app

# Install deps first (better caching)
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
