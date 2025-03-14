import React from 'react';
import { Container, CssBaseline, ThemeProvider } from '@mui/material';
import { HomeworkUploader } from './components/HomeworkUploader';
import { TutorResponse } from './components/TutorResponse';
import { theme } from './theme';

export const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <HomeworkUploader />
        <TutorResponse />
      </Container>
    </ThemeProvider>
  );
};
