import axios from 'axios';
import { Call, CallFeedback, CallMetrics, APIResponse } from '@/types';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add auth token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const callsApi = {
  // Get active calls
  getActiveCalls: async (): Promise<Call[]> => {
    const response = await api.get<APIResponse<Call[]>>('/api/calls/active');
    return response.data.data || [];
  },

  // Get recent calls (for training/review)
  getRecentCalls: async (limit: number = 10): Promise<Call[]> => {
    const response = await api.get<APIResponse<Call[]>>('/api/calls/recent', {
      params: { limit }
    });
    return response.data.data || [];
  },

  // Get call details
  getCall: async (callId: string): Promise<Call> => {
    const response = await api.get<APIResponse<Call>>(`/api/calls/${callId}`);
    if (!response.data.success) {
      throw new Error(response.data.error);
    }
    return response.data.data!;
  },

  // Get call recording
  getCallRecording: async (callId: string): Promise<string> => {
    const response = await api.get<APIResponse<{ url: string }>>(`/api/calls/${callId}/recording`);
    if (!response.data.success) {
      throw new Error(response.data.error);
    }
    return response.data.data!.url;
  },

  // Submit feedback for a call
  submitFeedback: async (
    callId: string,
    feedback: Omit<CallFeedback, 'id' | 'callId' | 'createdAt'>
  ): Promise<void> => {
    const response = await api.post<APIResponse<void>>(`/api/calls/${callId}/feedback`, feedback);
    if (!response.data.success) {
      throw new Error(response.data.error);
    }
  },

  // Initiate outbound call
  initiateCall: async (phoneNumber: string, type: 'sales' | 'support'): Promise<string> => {
    const response = await api.post<APIResponse<{ callId: string }>>('/api/calls/initiate', {
      phoneNumber,
      type
    });
    if (!response.data.success) {
      throw new Error(response.data.error);
    }
    return response.data.data!.callId;
  },

  // Get call metrics
  getMetrics: async (startDate?: Date, endDate?: Date): Promise<CallMetrics> => {
    const params = new URLSearchParams();
    if (startDate) params.append('startDate', startDate.toISOString());
    if (endDate) params.append('endDate', endDate.toISOString());

    const response = await api.get<APIResponse<CallMetrics>>('/api/calls/metrics', {
      params
    });
    if (!response.data.success) {
      throw new Error(response.data.error);
    }
    return response.data.data!;
  }
};

export const authApi = {
  // Login
  login: async (email: string, password: string): Promise<string> => {
    const response = await api.post<APIResponse<{ token: string }>>('/api/auth/login', {
      email,
      password
    });
    if (!response.data.success) {
      throw new Error(response.data.error);
    }
    const token = response.data.data!.token;
    localStorage.setItem('auth_token', token);
    return token;
  },

  // Logout
  logout: () => {
    localStorage.removeItem('auth_token');
  },

  // Check if user is authenticated
  isAuthenticated: (): boolean => {
    return !!localStorage.getItem('auth_token');
  }
};

// Error handling
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      // Unauthorized - clear token and redirect to login
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
