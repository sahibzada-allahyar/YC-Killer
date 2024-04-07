export interface User {
  id: string;
  email: string;
  name?: string;
  preferences?: UserPreferences;
  integrations: UserIntegrations;
  permissions?: UserPermission[];
  createdAt: Date;
  updatedAt: Date;
}

export interface UserPermission {
  id: string;
  userId: string;
  permission: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface UserPreferences {
  timezone: string;
  language: string;
  notificationPreferences: NotificationPreferences;
}

export interface NotificationPreferences {
  email: boolean;
  push: boolean;
  reminders: boolean;
}

export interface UserIntegrations {
  google?: {
    accessToken: string;
    refreshToken: string;
    scope: string[];
    expiresAt: Date;
  };
  // Add other integration types as needed
}
