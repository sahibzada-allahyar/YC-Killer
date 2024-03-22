import { createServer } from './server';

// Handle uncaught errors
process.on('uncaughtException', (error: Error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason: unknown) => {
  console.error('Unhandled Rejection:', reason);
  process.exit(1);
});

// Start the server
async function start(): Promise<void> {
  try {
    await createServer();
    console.log('Server started successfully');
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Initialize the application
start().catch(error => {
  console.error('Application startup failed:', error);
  process.exit(1);
});
