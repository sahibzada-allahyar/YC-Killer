declare module 'http-status-codes' {
  export enum StatusCodes {
    OK = 200,
    CREATED = 201,
    ACCEPTED = 202,
    NO_CONTENT = 204,
    BAD_REQUEST = 400,
    UNAUTHORIZED = 401,
    FORBIDDEN = 403,
    NOT_FOUND = 404,
    METHOD_NOT_ALLOWED = 405,
    CONFLICT = 409,
    UNPROCESSABLE_ENTITY = 422,
    TOO_MANY_REQUESTS = 429,
    INTERNAL_SERVER_ERROR = 500,
    BAD_GATEWAY = 502,
    SERVICE_UNAVAILABLE = 503,
    GATEWAY_TIMEOUT = 504,
  }

  export enum ReasonPhrases {
    OK = 'OK',
    CREATED = 'Created',
    ACCEPTED = 'Accepted',
    NO_CONTENT = 'No Content',
    BAD_REQUEST = 'Bad Request',
    UNAUTHORIZED = 'Unauthorized',
    FORBIDDEN = 'Forbidden',
    NOT_FOUND = 'Not Found',
    METHOD_NOT_ALLOWED = 'Method Not Allowed',
    CONFLICT = 'Conflict',
    UNPROCESSABLE_ENTITY = 'Unprocessable Entity',
    TOO_MANY_REQUESTS = 'Too Many Requests',
    INTERNAL_SERVER_ERROR = 'Internal Server Error',
    BAD_GATEWAY = 'Bad Gateway',
    SERVICE_UNAVAILABLE = 'Service Unavailable',
    GATEWAY_TIMEOUT = 'Gateway Timeout',
  }

  export function getStatusCode(reasonPhrase: string): number;
  export function getReasonPhrase(statusCode: number): string;
}
