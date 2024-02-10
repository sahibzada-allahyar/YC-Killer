export interface TaskRouterCapabilityOptions {
    accountSid: string;
    authToken: string;
    workspaceSid: string;
    channelId: string;
    friendlyName?: string;
    ttl?: number;
    version?: string;
}
export interface PolicyOptions {
    /** Policy URL */
    url?: string;
    /** HTTP Method */
    method?: string;
    /** Request query filter allowances */
    queryFilter?: object;
    /** Request post filter allowances */
    postFilter?: object;
    /** Allow the policy */
    allow?: boolean;
}
export interface PolicyPayload {
    url: string;
    method: string;
    query_filter: object;
    post_filter: object;
    allow: boolean;
}
/**
 * Create a new Policy
 */
export declare class Policy {
    url: string;
    method: string;
    queryFilter: object;
    postFilter: object;
    allow: boolean;
    /**
     * Create a new Policy instance
     *
     * @param options - ...
     * @param options.url - Policy URL
     * @param options.method - HTTP Method
     * @param options.queryFilter - Request query filter allowances
     * @param options.postFilter - Request post filter allowances
     * @param options.allowed - Allow the policy
     */
    constructor(options?: PolicyOptions);
    payload(): PolicyPayload;
}
export default class TaskRouterCapability {
    accountSid: string;
    authToken: string;
    workspaceSid: string;
    channelId: string;
    ttl: number;
    version: string;
    policies: Policy[];
    friendlyName?: string;
    /**
     * @param options - ...
     * @param options.accountSid - account sid
     * @param options.authToken - auth token
     * @param options.workspaceSid - workspace sid
     * @param options.channelId - taskrouter channel id
     * @param options.friendlyName - friendly name for the jwt
     * @param options.ttl - time to live
     * @param options.version - taskrouter version
     */
    constructor(options: TaskRouterCapabilityOptions);
    static Policy: typeof Policy;
    addPolicy(policy: Policy): void;
    toJwt(): string;
}
