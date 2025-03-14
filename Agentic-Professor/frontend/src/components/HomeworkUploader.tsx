import React, { useState, useRef } from 'react';
import { Box, Button, TextField, Typography, CircularProgress } from '@mui/material';
import { CloudUpload as CloudUploadIcon } from '@mui/icons-material';
import { useStore } from '../store/tutorStore';

interface HomeworkUploaderProps {
  onUploadComplete?: () => void;
}

export const HomeworkUploader: React.FC<HomeworkUploaderProps> = ({ onUploadComplete }) => {
  const [subject, setSubject] = useState('');
  const [notes, setNotes] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const submitHomework = useStore(state => state.submitHomework);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      setError(null);

      // Convert image to base64
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64String = reader.result as string;
        const imageData = base64String.split(',')[1]; // Remove data URL prefix

        try {
          await submitHomework({
            subject,
            imageData,
            additionalNotes: notes
          });
          
          // Reset form
          setSubject('');
          setNotes('');
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
          
          onUploadComplete?.();
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to submit homework');
        }
      };

      reader.readAsDataURL(file);
    } catch (err) {
      setError('Failed to process image');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Box sx={{ 
      maxWidth: 600,
      mx: 'auto',
      p: 3,
      display: 'flex',
      flexDirection: 'column',
      gap: 2
    }}>
      <Typography variant="h5" gutterBottom>
        Submit Homework
      </Typography>

      <TextField
        label="Subject"
        value={subject}
        onChange={(e) => setSubject(e.target.value)}
        fullWidth
        margin="normal"
      />

      <TextField
        label="Additional Notes"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        multiline
        rows={3}
        fullWidth
        margin="normal"
      />

      <input
        ref={fileInputRef}
        accept="image/*"
        type="file"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      <Button
        variant="contained"
        startIcon={isUploading ? <CircularProgress size={20} /> : <CloudUploadIcon />}
        onClick={() => fileInputRef.current?.click()}
        disabled={isUploading}
        sx={{ mt: 2 }}
      >
        {isUploading ? 'Uploading...' : 'Upload Image'}
      </Button>

      {error && (
        <Typography color="error" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}
    </Box>
  );
};
