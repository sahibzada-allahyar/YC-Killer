import fs from 'fs';
import path from 'path';
import os from 'os';
import { v4 as uuidv4 } from 'uuid';

/**
 * Creates a temporary file from a buffer and returns a file object compatible with OpenAI's API
 * @param buffer The buffer containing the file data
 * @param filename The desired filename
 * @param mimeType The MIME type of the file
 * @returns A file object that can be used with OpenAI's API
 */
export async function createTempFile(
  buffer: Buffer,
  filename: string,
  mimeType: string
): Promise<{
  path: string;
  cleanup: () => Promise<void>;
}> {
  // Create a unique temporary file path
  const tempDir = os.tmpdir();
  const uniqueId = uuidv4();
  const tempPath = path.join(tempDir, `${uniqueId}-${filename}`);

  // Write the buffer to the temporary file
  await fs.promises.writeFile(tempPath, buffer);

  // Return the file path and a cleanup function
  return {
    path: tempPath,
    cleanup: async () => {
      try {
        await fs.promises.unlink(tempPath);
      } catch (error) {
        console.error('Failed to cleanup temporary file:', error);
      }
    },
  };
}
