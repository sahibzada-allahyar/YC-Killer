FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Build TypeScript
RUN npm run build

# Create reports directory
RUN mkdir -p /app/reports

# Set environment variables
ENV NODE_ENV=production
ENV OPENAI_MODEL=o3-mini

# Run the application
CMD ["node", "--experimental-vm-modules", "dist/run.js"]
