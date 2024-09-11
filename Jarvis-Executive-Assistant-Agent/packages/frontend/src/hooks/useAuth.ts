import { createContext, useContext, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

interface User {
  id: string;
  email: string;
  name?: string;
}

interface AuthContextType {
  isInitialized: boolean;
  isAuthenticated: boolean;
  user: User | null;
  login: (idToken: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const initAuth = async () => {
      try {
        const token = localStorage.getItem('token');
        if (token) {
          // Set default authorization header
          axios.defaults.headers.common.Authorization = `Bearer ${token}`;
          
          // Fetch user profile
          const response = await axios.get('/api/v1/users/profile');
          setUser(response.data.data);
          setIsAuthenticated(true);
        }
      } catch (error) {
        console.error('Auth initialization error:', error);
        localStorage.removeItem('token');
        axios.defaults.headers.common.Authorization = '';
      } finally {
        setIsInitialized(true);
      }
    };

    initAuth();
  }, []);

  const login = async (idToken: string) => {
    try {
      const response = await axios.post('/api/v1/auth/google', { idToken });
      const { token, user } = response.data.data;
      
      localStorage.setItem('token', token);
      axios.defaults.headers.common.Authorization = `Bearer ${token}`;
      
      setUser(user);
      setIsAuthenticated(true);
      navigate('/chat');
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };

  const logout = async () => {
    try {
      await axios.post('/api/v1/auth/logout');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      localStorage.removeItem('token');
      axios.defaults.headers.common.Authorization = '';
      setUser(null);
      setIsAuthenticated(false);
      navigate('/login');
    }
  };

  const refreshToken = async () => {
    try {
      const refreshToken = localStorage.getItem('refreshToken');
      if (!refreshToken) throw new Error('No refresh token');

      const response = await axios.post('/api/v1/auth/refresh', { refreshToken });
      const { token } = response.data.data;
      
      localStorage.setItem('token', token);
      axios.defaults.headers.common.Authorization = `Bearer ${token}`;
    } catch (error) {
      console.error('Token refresh error:', error);
      await logout();
    }
  };

  return (
    <AuthContext.Provider
      value={{
        isInitialized,
        isAuthenticated,
        user,
        login,
        logout,
        refreshToken,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
