declare class AccessToken implements AccessToken.AccessTokenOptions {
    static DEFAULT_ALGORITHM: "HS256";
    static ALGORITHMS: string[];
    accountSid: string;
    keySid: string;
    secret: string;
    ttl: number;
    identity: string;
    nbf?: number;
    region?: string;
    grants: AccessToken.Grant<any, any, any>[];
    /**
     * @param accountSid - The account's unique ID to which access is scoped
     * @param keySid - The signing key's unique ID
     * @param secret - The secret to sign the token with
     * @param options - ...
     * @param options.ttl - Time to live in seconds (default 3600)
     * @param options.identity - The identity of the first person. Required.
     * @param options.nbf - Time from epoch in seconds for not before value
     * @param options.region - The region value associated with this account
     */
    constructor(accountSid: string, keySid: string, secret: string, options: AccessToken.AccessTokenOptions);
    addGrant<T extends AccessToken.Grant<any, any, any>>(grant: T): void;
    toJwt(algorithm?: "HS256" | "HS384" | "HS512"): string;
}
declare namespace AccessToken {
    abstract class Grant<TOptions, TPayload, TKey> {
        key: TKey;
        protected constructor(key: TKey);
        abstract toPayload(): TPayload;
    }
    interface TaskRouterGrantOptions {
        workspaceSid?: string;
        workerSid?: string;
        role?: string;
    }
    interface TaskRouterGrantPayload {
        workspace_sid?: string;
        worker_sid?: string;
        role?: string;
    }
    interface ChatGrantOptions {
        serviceSid?: string;
        endpointId?: string;
        deploymentRoleSid?: string;
        pushCredentialSid?: string;
    }
    interface ChatGrantPayload {
        service_sid?: string;
        endpoint_id?: string;
        deployment_role_sid?: string;
        push_credential_sid?: string;
    }
    interface VideoGrantOptions {
        room?: string;
    }
    interface VideoGrantPayload {
        room?: string;
    }
    interface SyncGrantOptions {
        serviceSid?: string;
        endpointId?: string;
    }
    interface SyncGrantPayload {
        service_sid?: string;
        endpoint_id?: string;
    }
    interface VoiceGrantOptions {
        incomingAllow?: boolean;
        outgoingApplicationSid?: string;
        outgoingApplicationParams?: object;
        pushCredentialSid?: string;
        endpointId?: string;
    }
    interface VoiceGrantPayload {
        incoming?: {
            allow: boolean;
        };
        outgoing?: {
            application_sid: string;
            params?: object;
        };
        push_credential_sid?: string;
        endpoint_id?: string;
    }
    interface PlaybackGrantOptions {
        grant?: object;
    }
    interface PlaybackGrantPayload {
        grant?: object;
    }
    interface AccessTokenOptions {
        /**
         * Time to live in seconds
         */
        ttl?: number;
        /**
         * The identity of the first person. Required.
         */
        identity: string;
        /**
         * Time from epoch in seconds for not before value
         */
        nbf?: number;
        /**
         * The region value associated with this account
         */
        region?: string;
    }
    class TaskRouterGrant extends Grant<TaskRouterGrantOptions, TaskRouterGrantPayload, "task_router"> implements TaskRouterGrantOptions {
        workspaceSid?: string;
        workerSid?: string;
        role?: string;
        /**
         * @param options - ...
         * @param options.workspaceSid - The workspace unique ID
         * @param options.workerSid - The worker unique ID
         * @param options.role - The role of the grant
         */
        constructor(options?: TaskRouterGrantOptions);
        toPayload(): TaskRouterGrantPayload;
    }
    class ChatGrant extends Grant<ChatGrantOptions, ChatGrantPayload, "chat"> implements ChatGrantOptions {
        serviceSid?: string;
        endpointId?: string;
        deploymentRoleSid?: string;
        pushCredentialSid?: string;
        /**
         * @param options - ...
         * @param options.serviceSid - The service unique ID
         * @param options.endpointId - The endpoint ID
         * @param options.deploymentRoleSid - SID of the deployment role to be
         *                 assigned to the user
         * @param options.pushCredentialSid - The Push Credentials SID
         */
        constructor(options?: ChatGrantOptions);
        toPayload(): ChatGrantPayload;
    }
    class VideoGrant extends Grant<VideoGrantOptions, VideoGrantPayload, "video"> implements VideoGrantOptions {
        room?: string;
        /**
         * @param options - ...
         * @param options.room - The Room name or Room sid.
         */
        constructor(options?: VideoGrantOptions);
        toPayload(): VideoGrantPayload;
    }
    class SyncGrant extends Grant<SyncGrantOptions, SyncGrantPayload, "data_sync"> implements SyncGrantOptions {
        serviceSid?: string;
        endpointId?: string;
        /**
         * @param options.serviceSid - The service unique ID
         * @param options.endpointId - The endpoint ID
         */
        constructor(options?: SyncGrantOptions);
        toPayload(): SyncGrantPayload;
    }
    class VoiceGrant extends Grant<VoiceGrantOptions, VoiceGrantPayload, "voice"> implements VoiceGrantOptions {
        incomingAllow?: boolean;
        outgoingApplicationSid?: string;
        outgoingApplicationParams?: object;
        pushCredentialSid?: string;
        endpointId?: string;
        /**
         * @param options - ...
         * @param options.incomingAllow - Whether or not this endpoint is allowed to receive incoming calls as grants.identity
         * @param options.outgoingApplicationSid - application sid to call when placing outgoing call
         * @param options.outgoingApplicationParams - request params to pass to the application
         * @param options.pushCredentialSid - Push Credential Sid to use when registering to receive incoming call notifications
         * @param options.endpointId - Specify an endpoint identifier for this device, which will allow the developer
         *                 to direct calls to a specific endpoint when multiple devices are associated with a single identity
         */
        constructor(options?: VoiceGrantOptions);
        toPayload(): VoiceGrantPayload;
    }
    class PlaybackGrant extends Grant<PlaybackGrantOptions, PlaybackGrantPayload, "player"> implements PlaybackGrantOptions {
        grant?: object;
        /**
         * @param options - ...
         * @param options.grant - The PlaybackGrant retrieved from Twilio's API
         */
        constructor(options?: PlaybackGrantOptions);
        toPayload(): PlaybackGrantPayload;
    }
}
export = AccessToken;
