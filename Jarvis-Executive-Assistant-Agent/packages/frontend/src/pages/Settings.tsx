import { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Switch,
  FormControlLabel,
  Button,
  Divider,
  Alert,
  Grid,
  Card,
  CardContent,
  CardActions,
  CircularProgress,
} from '@mui/material';
import {
  Google as GoogleIcon,
  NotificationsActive as NotificationsIcon,
  Security as SecurityIcon,
  Save as SaveIcon,
} from '@mui/icons-material';
import { useAuth } from '@/hooks/useAuth';
import type { User, UserIntegrations, NotificationPreferences } from '@jarvis-executive-assistant/shared/types/user';
import axios from 'axios';

interface ExtendedUser extends User {
  integrations: UserIntegrations & {
    google?: {
      connected: boolean;
      email: string;
    };
  };
}

const Settings = () => {
  const { user } = useAuth() as { user: ExtendedUser | null };
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [preferences, setPreferences] = useState<{ notificationPreferences: NotificationPreferences }>({
    notificationPreferences: {
      email: user?.preferences?.notificationPreferences?.email ?? true,
      push: user?.preferences?.notificationPreferences?.push ?? true,
      reminders: user?.preferences?.notificationPreferences?.reminders ?? true,
    },
  });

  const handleNotificationChange = (type: keyof NotificationPreferences) => {
    setPreferences(prev => ({
      ...prev,
      notificationPreferences: {
        ...prev.notificationPreferences,
        [type]: !prev.notificationPreferences[type],
      },
    }));
  };

  const handleSavePreferences = async () => {
    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      await axios.patch('/api/v1/users/preferences', preferences);
      setSuccess('Preferences updated successfully');
    } catch (error) {
      setError('Failed to update preferences. Please try again.');
      console.error('Preferences update error:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleConnectGoogle = async () => {
    try {
      if (user?.integrations?.google?.connected) {
        // Disconnect Google account
        await axios.delete('/api/v1/users/integrations/google');
        setSuccess('Successfully disconnected Google account');
        window.location.reload();
      } else {
        // Start OAuth flow
        const response = await axios.post('/api/v1/auth/google/init', {
          redirectUri: window.location.href,
        });

        if (response.data?.authUrl) {
          window.location.href = response.data.authUrl;
        } else {
          throw new Error('Failed to get auth URL');
        }
      }
    } catch (error) {
      setError('Failed to manage Google connection. Please try again.');
      console.error('Google connection error:', error);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom>
          Settings
        </Typography>

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

        <Grid container spacing={4}>
          {/* Notifications Section */}
          <Grid item xs={12}>
            <Paper elevation={2} sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <NotificationsIcon sx={{ mr: 1 }} />
                <Typography variant="h6">Notifications</Typography>
              </Box>
              <Divider sx={{ mb: 3 }} />

              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={preferences.notificationPreferences.email}
                      onChange={() => handleNotificationChange('email')}
                    />
                  }
                  label="Email Notifications"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={preferences.notificationPreferences.push}
                      onChange={() => handleNotificationChange('push')}
                    />
                  }
                  label="Push Notifications"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={preferences.notificationPreferences.reminders}
                      onChange={() => handleNotificationChange('reminders')}
                    />
                  }
                  label="Reminder Notifications"
                />
              </Box>
            </Paper>
          </Grid>

          {/* Integrations Section */}
          <Grid item xs={12}>
            <Paper elevation={2} sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SecurityIcon sx={{ mr: 1 }} />
                <Typography variant="h6">Connected Services</Typography>
              </Box>
              <Divider sx={{ mb: 3 }} />

              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <GoogleIcon sx={{ mr: 1 }} />
                        <Typography variant="h6">Google</Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {user?.integrations?.google?.connected
                          ? `Connected as ${user.integrations.google.email}`
                          : 'Connect your Google account to access calendar, email, and contacts.'}
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button
                        variant={user?.integrations?.google?.connected ? 'outlined' : 'contained'}
                        onClick={handleConnectGoogle}
                        startIcon={<GoogleIcon />}
                      >
                        {user?.integrations?.google?.connected ? 'Disconnect' : 'Connect'}
                      </Button>
                    </CardActions>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>

        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            onClick={handleSavePreferences}
            disabled={isSaving}
            startIcon={isSaving ? <CircularProgress size={20} /> : <SaveIcon />}
          >
            Save Changes
          </Button>
        </Box>
      </Box>
    </Container>
  );
};

export default Settings;
