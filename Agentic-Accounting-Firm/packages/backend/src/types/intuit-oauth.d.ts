declare module '@intuit/oauth' {
  export interface Token {
    access_token: string;
    refresh_token: string;
    expires_in: number;
    x_refresh_token_expires_in: number;
    token_type: string;
  }

  export interface OAuthResponse {
    getJson(): Token;
  }

  export interface OAuthClientConfig {
    clientId: string;
    clientSecret: string;
    environment: 'sandbox' | 'production';
    redirectUri: string;
  }

  export class OAuthClient {
    constructor(config: OAuthClientConfig);
    refreshUsingToken(refreshToken?: string): Promise<OAuthResponse>;
  }

  export default OAuthClient;
}
