const fs = require('fs');
const path = require('path');

// Define cache directories to create
const directories = [
  'backend/cache',
  'backend/cache/voice',
  'backend/cache/images',
  'backend/cache/wikipedia',
  'backend/logs'
];

// Create directories if they don't exist
directories.forEach(dir => {
  const dirPath = path.join(process.cwd(), dir);
  
  if (!fs.existsSync(dirPath)) {
    console.log(`Creating directory: ${dir}`);
    fs.mkdirSync(dirPath, { recursive: true });
    
    // Add .gitkeep file
    fs.writeFileSync(path.join(dirPath, '.gitkeep'), '');
  } else {
    console.log(`Directory already exists: ${dir}`);
  }
});

console.log('\nCache directories initialized successfully!');
console.log('\nNote: These directories are git-ignored except for .gitkeep files');
console.log('This ensures the directories exist when cloning the repository.');
