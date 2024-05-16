export * from './api';
export * from './auth';
export * from './chat';
export * from './skills';
export * from './user';

// Re-export commonly used types with more specific names
export type {
  ApiResponse as BaseApiResponse,
  PaginatedResponse as BasePaginatedResponse,
  ApiError as BaseApiError,
} from './api';

export type {
  User as BaseUser,
  UserPreferences as BaseUserPreferences,
  UserIntegrations as BaseUserIntegrations,
} from './user';

export type {
  Message as ChatMessage,
  Conversation as ChatConversation,
  ConversationContext as ChatContext,
} from './chat';

export type {
  Skill as BaseSkill,
  SkillContext as BaseSkillContext,
  SkillResult as BaseSkillResult,
} from './skills';
