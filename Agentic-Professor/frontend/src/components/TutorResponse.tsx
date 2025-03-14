import React, { useEffect, useRef } from 'react';
import { Box, Paper, Typography } from '@mui/material';
import { useStore } from '../store/tutorStore';
import { AudioControls } from './AudioControls';
import { LoadingSpinner } from './LoadingSpinner';

declare global {
  interface Window {
    MathJax: {
      Hub?: {
        Queue: (args: any[]) => void;
      };
    };
  }
}

export const TutorResponse: React.FC = () => {
  const { currentResponse, isLoading, error } = useStore();
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Load MathJax script if not already loaded
    if (!window.MathJax) {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js?config=TeX-MML-AM_CHTML';
      script.async = true;
      document.head.appendChild(script);
    }
  }, []);

  useEffect(() => {
    if (contentRef.current && currentResponse?.htmlContent && window.MathJax?.Hub) {
      window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub, contentRef.current]);
    }
  }, [currentResponse?.htmlContent]);

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography color="error">
          {error}
        </Typography>
      </Box>
    );
  }

  if (!currentResponse) {
    return null;
  }

  return (
    <Paper 
      elevation={2}
      sx={{
        p: 3,
        mt: 3,
        maxWidth: '100%',
        overflow: 'auto'
      }}
    >
      <Box ref={contentRef}>
        <div 
          dangerouslySetInnerHTML={{ __html: currentResponse.htmlContent }}
          style={{ fontSize: '1.1rem', lineHeight: 1.6 }}
        />
      </Box>

      {currentResponse.references && currentResponse.references.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            References
          </Typography>
          <ul>
            {currentResponse.references.map((ref, index) => (
              <li key={index}>
                <a href={ref} target="_blank" rel="noopener noreferrer">
                  {ref}
                </a>
              </li>
            ))}
          </ul>
        </Box>
      )}

      {currentResponse.diagrams && currentResponse.diagrams.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Diagrams
          </Typography>
          {currentResponse.diagrams.map((diagram, index) => (
            <Box key={index} sx={{ mt: 2 }}>
              {diagram.type === 'image' ? (
                <img 
                  src={`data:image/png;base64,${diagram.data}`}
                  alt={diagram.caption || `Diagram ${index + 1}`}
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
              ) : (
                <div 
                  dangerouslySetInnerHTML={{ __html: diagram.data }}
                  style={{ maxWidth: '100%', overflow: 'auto' }}
                />
              )}
              {diagram.caption && (
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  {diagram.caption}
                </Typography>
              )}
            </Box>
          ))}
        </Box>
      )}

      {currentResponse.audioUrl && (
        <Box sx={{ mt: 3 }}>
          <AudioControls audioUrl={currentResponse.audioUrl} />
        </Box>
      )}
    </Paper>
  );
};
