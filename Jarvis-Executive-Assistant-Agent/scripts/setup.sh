#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Jarvis Executive Assistant development environment...${NC}\n"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed. Please install Node.js v18 or higher.${NC}"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2)
if (( ${NODE_VERSION%%.*} < 18 )); then
    echo -e "${RED}Node.js version must be 18 or higher. Current version: ${NODE_VERSION}${NC}"
    exit 1
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
npm install

# Copy environment files
echo -e "${BLUE}Setting up environment files...${NC}"

# Backend environment
if [ ! -f "packages/backend/.env" ]; then
    cp packages/backend/.env.example packages/backend/.env
    echo -e "${GREEN}Created packages/backend/.env${NC}"
fi

# Create development certificates for HTTPS
echo -e "${BLUE}Setting up development certificates...${NC}"
mkdir -p .cert
if [ ! -f ".cert/dev.key" ]; then
    openssl req -x509 -newkey rsa:4096 -keyout .cert/dev.key -out .cert/dev.crt -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
    echo -e "${GREEN}Created development certificates${NC}"
fi

# Build packages
echo -e "${BLUE}Building packages...${NC}"
npm run build

echo -e "\n${GREEN}Setup complete! 🎉${NC}"
echo -e "\nNext steps:"
echo -e "1. Edit packages/backend/.env with your API keys and configuration"
echo -e "2. Start the development server with: ${BLUE}npm run dev${NC}"
echo -e "3. Visit http://localhost:3000 in your browser"
echo -e "\nFor more information, see the README.md file."
