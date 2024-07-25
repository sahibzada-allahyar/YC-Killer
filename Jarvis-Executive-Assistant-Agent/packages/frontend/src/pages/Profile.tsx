import { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Avatar,
  Grid,
  Divider,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  CardActions,
} from '@mui/material';
import {
  Save as SaveIcon,
  Edit as EditIcon,
  Google as GoogleIcon,
} from '@mui/icons-material';
import { useAuth } from '@/hooks/useAuth';
import type { User, UserIntegrations, UserPreferences } from '@jarvis-executive-assistant/shared/types/user';
import axios from 'axios';

interface ExtendedUser extends Omit<User, 'createdAt' | 'updatedAt'> {
  picture?: string;
  name: string;
  email: string;
  preferences: UserPreferences;
  integrations: UserIntegrations;
  createdAt: string;
  updatedAt: string;
}

interface Integration {
  accessToken: string;
  refreshToken: string;
  scope: string[];
  expiresAt: Date;
  createdAt: Date;
}

type IntegrationEntry = [provider: string, integration: Integration];

const Profile = () => {
  const handleDisconnectService = async (provider: string) => {
    try {
      await axios.delete(`/api/v1/users/integrations/${provider}`);
      setSuccess(`Successfully disconnected ${provider}`);
      // Refresh user data
      window.location.reload();
    } catch (error) {
      setError(`Failed to disconnect ${provider}. Please try again.`);
      console.error('Disconnect service error:', error);
    }
  };

  const { user } = useAuth() as { user: ExtendedUser | null };
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    name: user?.name || '',
    timezone: user?.preferences?.timezone || 'UTC',
    language: user?.preferences?.language || 'en',
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      await axios.patch('/api/v1/users/profile', formData);
      setSuccess('Profile updated successfully');
      setIsEditing(false);
    } catch (error) {
      setError('Failed to update profile. Please try again.');
      console.error('Profile update error:', error);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
        <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
          <Avatar
            src={user?.picture}
            alt={user?.name || user?.email}
            sx={{ width: 80, height: 80 }}
          />
          <Box>
            <Typography variant="h4" gutterBottom>
              Profile Settings
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Manage your personal information and preferences
            </Typography>
          </Box>
        </Box>

        <Divider sx={{ mb: 4 }} />

        {(error || success) && (
          <Alert
            severity={error ? 'error' : 'success'}
            sx={{ mb: 3 }}
            onClose={() => {
              setError(null);
              setSuccess(null);
            }}
          >
            {error || success}
          </Alert>
        )}

        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Email"
                value={user?.email}
                disabled
                variant="filled"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                disabled={!isEditing}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Timezone"
                name="timezone"
                value={formData.timezone}
                onChange={handleInputChange}
                disabled={!isEditing}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Language"
                name="language"
                value={formData.language}
                onChange={handleInputChange}
                disabled={!isEditing}
                required
              />
            </Grid>
          </Grid>

          <Box sx={{ mt: 4, display: 'flex', gap: 2 }}>
            {!isEditing ? (
              <Button
                variant="contained"
                startIcon={<EditIcon />}
                onClick={() => setIsEditing(true)}
              >
                Edit Profile
              </Button>
            ) : (
              <>
                <Button
                  variant="contained"
                  type="submit"
                  disabled={isSaving}
                  startIcon={
                    isSaving ? <CircularProgress size={20} /> : <SaveIcon />
                  }
                >
                  Save Changes
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => {
                    setIsEditing(false);
                    setFormData({
                      name: user?.name || '',
                      timezone: 'UTC',
                      language: 'en',
                    });
                  }}
                  disabled={isSaving}
                >
                  Cancel
                </Button>
              </>
            )}
          </Box>
        </form>

        <Box sx={{ mt: 6 }}>
          <Typography variant="h6" gutterBottom>
            Connected Services
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Manage your connected third-party services and integrations
          </Typography>
          <Grid container spacing={3}>
            {(Object.entries(user?.integrations || {}) as IntegrationEntry[]).map(([provider, integration]) => (
              <Grid item xs={12} sm={6} key={provider}>
                <Card variant="outlined">
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      {provider === 'google' && <GoogleIcon sx={{ mr: 1 }} />}
                      <Typography variant="h6" sx={{ textTransform: 'capitalize' }}>
                        {provider}
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Connected since {new Date(integration.createdAt).toLocaleDateString()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Expires {new Date(integration.expiresAt).toLocaleDateString()}
                    </Typography>
                  </CardContent>
                  <CardActions>
                    <Button
                      variant="outlined"
                      color="error"
                      onClick={() => handleDisconnectService(provider)}
                    >
                      Disconnect
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      </Paper>
    </Container>
  );
};

export default Profile;
