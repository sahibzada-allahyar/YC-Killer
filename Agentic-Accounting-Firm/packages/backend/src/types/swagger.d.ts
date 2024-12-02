declare module 'swagger-jsdoc' {
  interface SwaggerDefinition {
    openapi?: string;
    swagger?: string;
    info: {
      title: string;
      version: string;
      description?: string;
      termsOfService?: string;
      contact?: {
        name?: string;
        url?: string;
        email?: string;
      };
      license?: {
        name: string;
        url?: string;
      };
    };
    servers?: Array<{
      url: string;
      description?: string;
      variables?: Record<string, {
        enum?: string[];
        default: string;
        description?: string;
      }>;
    }>;
    basePath?: string;
    schemes?: string[];
    consumes?: string[];
    produces?: string[];
    tags?: Array<{
      name: string;
      description?: string;
      externalDocs?: {
        description?: string;
        url: string;
      };
    }>;
    securityDefinitions?: Record<string, any>;
    security?: Array<Record<string, string[]>>;
  }

  interface Options {
    swaggerDefinition: SwaggerDefinition;
    apis: string[];
    definition?: SwaggerDefinition;
  }

  function swaggerJsdoc(options: Options): any;
  export = swaggerJsdoc;
}

declare module 'swagger-ui-express' {
  import { Request, Response, RequestHandler } from 'express';

  interface SwaggerUiOptions {
    customCss?: string;
    customfavIcon?: string;
    customSiteTitle?: string;
    customJs?: string;
    swaggerUrl?: string;
    swaggerUrls?: Array<{ url: string; name: string }>;
    explorer?: boolean;
    customCssUrl?: string;
    customJsUrl?: string;
    swaggerOptions?: Record<string, any>;
  }

  interface SwaggerUiExpressOptions extends SwaggerUiOptions {
    explorer?: boolean;
    swaggerUrl?: string;
    customCss?: string;
    swaggerOptions?: Record<string, any>;
    customSiteTitle?: string;
  }

  export const serve: RequestHandler[];
  export function setup(
    swaggerDoc: any,
    swaggerUiOpts?: SwaggerUiExpressOptions,
    swaggerOptions?: Record<string, any>,
    customCss?: string,
    customfavIcon?: string,
    swaggerUrl?: string,
    customSiteTitle?: string
  ): RequestHandler;
  export function serveFiles(swaggerDoc?: any, opts?: SwaggerUiOptions): RequestHandler;
  export function generateHTML(swaggerDoc?: any, opts?: SwaggerUiOptions): string;
}
