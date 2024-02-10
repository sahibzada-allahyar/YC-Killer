/// <reference types="node" />
/// <reference types="node" />
import { IncomingHttpHeaders } from "http2";
export interface Request {
    protocol: string;
    header(name: string): string | undefined;
    headers: IncomingHttpHeaders;
    originalUrl: string;
    rawBody?: any;
    body: any;
}
export interface RequestValidatorOptions {
    /**
     * The full URL (with query string) you used to configure the webhook with Twilio - overrides host/protocol options
     */
    url?: string;
    /**
     * Manually specify the host name used by Twilio in a number's webhook config
     */
    host?: string;
    /**
     * Manually specify the protocol used by Twilio in a number's webhook config
     */
    protocol?: string;
}
export interface WebhookOptions {
    /**
     * Whether or not the middleware should validate the request
     * came from Twilio.  Default true. If the request does not originate from
     * Twilio, we will return a text body and a 403.  If there is no configured
     * auth token and validate=true, this is an error condition, so we will return
     * a 500.
     */
    validate?: boolean;
    /**
     * Add helpers to the response object to improve support for XML (TwiML) rendering.  Default true.
     */
    includeHelpers?: boolean;
    /**
     * The full URL (with query string) you used to configure the webhook with Twilio - overrides host/protocol options
     */
    url?: string;
    /**
     * Manually specify the host name used by Twilio in a number's webhook config
     */
    host?: string;
    /**
     * Manually specify the protocol used by Twilio in a number's webhook config
     */
    protocol?: string;
    /**
     * Authentication token
     */
    authToken?: string;
}
/**
 Utility function to get the expected signature for a given request

 @param authToken - The auth token, as seen in the Twilio portal
 @param url - The full URL (with query string) you configured to handle
 this request
 @param params - the parameters sent with this request
 @returns signature
 */
export declare function getExpectedTwilioSignature(authToken: string, url: string, params: Record<string, any>): string;
/**
 Utility function to get the expected body hash for a given request's body

 @param body - The plain-text body of the request
 */
export declare function getExpectedBodyHash(body: string): string;
/**
 Utility function to validate an incoming request is indeed from Twilio

 @param authToken - The auth token, as seen in the Twilio portal
 @param twilioHeader - The value of the X-Twilio-Signature header from the request
 @param url - The full URL (with query string) you configured to handle this request
 @param params - the parameters sent with this request
 @returns valid
 */
export declare function validateRequest(authToken: string, twilioHeader: string, url: string, params: Record<string, any>): boolean;
export declare function validateBody(body: string, bodyHash: any[] | string | Buffer): boolean;
/**
 Utility function to validate an incoming request is indeed from Twilio. This also validates
 the request body against the bodySHA256 post parameter.

 @param authToken - The auth token, as seen in the Twilio portal
 @param twilioHeader - The value of the X-Twilio-Signature header from the request
 @param url - The full URL (with query string) you configured to handle this request
 @param body - The body of the request
 @returns valid
 */
export declare function validateRequestWithBody(authToken: string, twilioHeader: string, url: string, body: string): boolean;
/**
 Utility function to validate an incoming request is indeed from Twilio.
 adapted from https://github.com/crabasa/twiliosig

 @param request - A request object (based on Express implementation http://expressjs.com/api.html#req.params)
 @param authToken - The auth token, as seen in the Twilio portal
 @param opts - options for request validation:
    -> url: The full URL (with query string) you used to configure the webhook with Twilio - overrides host/protocol options
    -> host: manually specify the host name used by Twilio in a number's webhook config
    -> protocol: manually specify the protocol used by Twilio in a number's webhook config
 */
export declare function validateIncomingRequest(request: Request, authToken: string, opts?: RequestValidatorOptions): boolean;
export declare function validateExpressRequest(request: Request, authToken: string, opts?: RequestValidatorOptions): boolean;
/**
Express middleware to accompany a Twilio webhook. Provides Twilio
request validation, and makes the response a little more friendly for our
TwiML generator.  Request validation requires the express.urlencoded middleware
to have been applied (e.g. app.use(express.urlencoded()); in your app config).

Options:
- validate: {Boolean} whether or not the middleware should validate the request
    came from Twilio.  Default true. If the request does not originate from
    Twilio, we will return a text body and a 403.  If there is no configured
    auth token and validate=true, this is an error condition, so we will return
    a 500.
- host: manually specify the host name used by Twilio in a number's webhook config
- protocol: manually specify the protocol used by Twilio in a number's webhook config
- url: The full URL (with query string) you used to configure the webhook with Twilio - overrides host/protocol options

Returns a middleware function.

Examples:
var webhookMiddleware = twilio.webhook();
var webhookMiddleware = twilio.webhook('asdha9dhjasd'); //init with auth token
var webhookMiddleware = twilio.webhook({
    validate:false // don't attempt request validation
});
var webhookMiddleware = twilio.webhook({
    host: 'hook.twilio.com',
    protocol: 'https'
});
 */
export declare function webhook(opts?: string | WebhookOptions, authToken?: string | WebhookOptions): (req: any, res: any, next: any) => void;
