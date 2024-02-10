/// <reference types="node" />
import RequestClient from "./RequestClient";
import { HttpMethod } from "../interfaces";
import { Headers } from "../http/request";
declare namespace Twilio {
    interface ClientOpts {
        httpClient?: RequestClient;
        accountSid?: string;
        env?: NodeJS.ProcessEnv;
        edge?: string;
        region?: string;
        lazyLoading?: boolean;
        logLevel?: string;
        userAgentExtensions?: string[];
        autoRetry?: boolean;
        maxRetries?: number;
    }
    interface RequestOpts {
        method?: HttpMethod;
        uri?: string;
        username?: string;
        password?: string;
        headers?: Headers;
        params?: object;
        data?: object;
        timeout?: number;
        allowRedirects?: boolean;
        logLevel?: string;
    }
    /**
     * Parent class for Twilio Client that implements request & validation logic
     */
    class Client {
        username: string;
        password: string;
        accountSid: string;
        opts?: ClientOpts;
        env?: NodeJS.ProcessEnv;
        edge?: string;
        region?: string;
        logLevel?: string;
        autoRetry: boolean;
        maxRetries?: number;
        userAgentExtensions?: string[];
        _httpClient?: RequestClient;
        /**
         * Create a BaseTwilio instance
         *
         * @param username -
         *          The username used for authentication. This is normally account sid, but if using key/secret auth will be
         *          the api key sid.
         * @param password -
         *          The password used for authentication. This is normally auth token, but if using key/secret auth will be
         *          the secret.
         * @param opts - The options argument
         *
         * @returns A new instance of BaseTwilio
         */
        constructor(username?: string, password?: string, opts?: ClientOpts);
        get httpClient(): RequestClient;
        /**
         * Makes a request to the Twilio API using the configured http client.
         * Authentication information is automatically added if none is provided.
         *
         * @param opts - The options argument
         */
        request(opts: RequestOpts): Promise<any>;
        /**
         * Adds a region and/or edge to a given hostname
         *
         * @param hostname - A URI hostname (e.g. api.twilio.com)
         * @param targetEdge - The targeted edge location (e.g. sydney)
         * @param targetRegion - The targeted region location (e.g. au1)
         */
        getHostname(hostname: string, targetEdge: string | undefined, targetRegion: string | undefined): string;
        /**
         * Test if your environment is impacted by a TLS or certificate
         * change is by sending an HTTP request to the test endpoint
         *
         * @throws RestException if the request fails
         *
         */
        validateSslCert(): Promise<any>;
    }
}
export = Twilio;
