export interface GoogleSignInResponse {
  clientId: string;
  client_id: string;
  credential: string;
  select_by: string;
}

export interface User {
  id: string;
  email: string;
  name?: string;
  picture?: string;
  preferences?: {
    timezone: string;
    language: string;
    notifications: {
      email: boolean;
      push: boolean;
      reminders: boolean;
    };
  };
  integrations?: {
    google?: {
      connected: boolean;
      email?: string;
      scope?: string[];
    };
    // Add other integration types as needed
  };
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface AuthState {
  isInitialized: boolean;
  isAuthenticated: boolean;
  user: User | null;
}

export interface AuthContextType extends AuthState {
  login: (idToken: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
}
