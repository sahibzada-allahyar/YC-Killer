import { User } from './user';

export interface GoogleSignInResponse {
  clientId: string;
  client_id: string;
  credential: string;
  select_by: string;
}

export interface AuthResponse {
  token: string;
  refreshToken: string;
  user: User;
}

export interface RefreshTokenRequest {
  refreshToken: string;
}

export interface RefreshTokenResponse {
  token: string;
  refreshToken: string;
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

export interface ConnectIntegrationRequest {
  provider: string;
  code: string;
  redirectUri: string;
}

export interface ConnectIntegrationResponse {
  provider: string;
  connected: boolean;
  email?: string;
  scope?: string[];
}
