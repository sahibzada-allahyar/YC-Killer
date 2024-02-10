/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to update a UsAppToPersonInstance
 */
export interface UsAppToPersonContextUpdateOptions {
    /** Indicates that this SMS campaign will send messages that contain links. */
    hasEmbeddedLinks: boolean;
    /** Indicates that this SMS campaign will send messages that contain phone numbers. */
    hasEmbeddedPhone: boolean;
    /** An array of sample message strings, min two and max five. Min length for each sample: 20 chars. Max length for each sample: 1024 chars. */
    messageSamples: Array<string>;
    /** Required for all Campaigns. Details around how a consumer opts-in to their campaign, therefore giving consent to receive their messages. If multiple opt-in methods can be used for the same campaign, they must all be listed. 40 character minimum. 2048 character maximum. */
    messageFlow: string;
    /** A short description of what this SMS campaign does. Min length: 40 characters. Max length: 4096 characters. */
    description: string;
    /** A boolean that specifies whether campaign requires age gate for federally legal content. */
    ageGated: boolean;
    /** A boolean that specifies whether campaign allows direct lending or not. */
    directLending: boolean;
}
/**
 * Options to pass to create a UsAppToPersonInstance
 */
export interface UsAppToPersonListInstanceCreateOptions {
    /** A2P Brand Registration SID */
    brandRegistrationSid: string;
    /** A short description of what this SMS campaign does. Min length: 40 characters. Max length: 4096 characters. */
    description: string;
    /** Required for all Campaigns. Details around how a consumer opts-in to their campaign, therefore giving consent to receive their messages. If multiple opt-in methods can be used for the same campaign, they must all be listed. 40 character minimum. 2048 character maximum. */
    messageFlow: string;
    /** An array of sample message strings, min two and max five. Min length for each sample: 20 chars. Max length for each sample: 1024 chars. */
    messageSamples: Array<string>;
    /** A2P Campaign Use Case. Examples: [ 2FA, EMERGENCY, MARKETING..] */
    usAppToPersonUsecase: string;
    /** Indicates that this SMS campaign will send messages that contain links. */
    hasEmbeddedLinks: boolean;
    /** Indicates that this SMS campaign will send messages that contain phone numbers. */
    hasEmbeddedPhone: boolean;
    /** If end users can text in a keyword to start receiving messages from this campaign, the auto-reply messages sent to the end users must be provided. The opt-in response should include the Brand name, confirmation of opt-in enrollment to a recurring message campaign, how to get help, and clear description of how to opt-out. This field is required if end users can text in a keyword to start receiving messages from this campaign. 20 character minimum. 320 character maximum. */
    optInMessage?: string;
    /** Upon receiving the opt-out keywords from the end users, Twilio customers are expected to send back an auto-generated response, which must provide acknowledgment of the opt-out request and confirmation that no further messages will be sent. It is also recommended that these opt-out messages include the brand name. This field is required if managing opt out keywords yourself (i.e. not using Twilio\\\'s Default or Advanced Opt Out features). 20 character minimum. 320 character maximum. */
    optOutMessage?: string;
    /** When customers receive the help keywords from their end users, Twilio customers are expected to send back an auto-generated response; this may include the brand name and additional support contact information. This field is required if managing help keywords yourself (i.e. not using Twilio\\\'s Default or Advanced Opt Out features). 20 character minimum. 320 character maximum. */
    helpMessage?: string;
    /** If end users can text in a keyword to start receiving messages from this campaign, those keywords must be provided. This field is required if end users can text in a keyword to start receiving messages from this campaign. Values must be alphanumeric. 255 character maximum. */
    optInKeywords?: Array<string>;
    /** End users should be able to text in a keyword to stop receiving messages from this campaign. Those keywords must be provided. This field is required if managing opt out keywords yourself (i.e. not using Twilio\\\'s Default or Advanced Opt Out features). Values must be alphanumeric. 255 character maximum. */
    optOutKeywords?: Array<string>;
    /** End users should be able to text in a keyword to receive help. Those keywords must be provided as part of the campaign registration request. This field is required if managing help keywords yourself (i.e. not using Twilio\\\'s Default or Advanced Opt Out features). Values must be alphanumeric. 255 character maximum. */
    helpKeywords?: Array<string>;
    /** A boolean that specifies whether campaign has Subscriber Optin or not. */
    subscriberOptIn?: boolean;
    /** A boolean that specifies whether campaign is age gated or not. */
    ageGated?: boolean;
    /** A boolean that specifies whether campaign allows direct lending or not. */
    directLending?: boolean;
}
/**
 * Options to pass to each
 */
export interface UsAppToPersonListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: UsAppToPersonInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface UsAppToPersonListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface UsAppToPersonListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface UsAppToPersonContext {
    /**
     * Remove a UsAppToPersonInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a UsAppToPersonInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsAppToPersonInstance
     */
    fetch(callback?: (error: Error | null, item?: UsAppToPersonInstance) => any): Promise<UsAppToPersonInstance>;
    /**
     * Update a UsAppToPersonInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsAppToPersonInstance
     */
    update(params: UsAppToPersonContextUpdateOptions, callback?: (error: Error | null, item?: UsAppToPersonInstance) => any): Promise<UsAppToPersonInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface UsAppToPersonContextSolution {
    messagingServiceSid: string;
    sid: string;
}
export declare class UsAppToPersonContextImpl implements UsAppToPersonContext {
    protected _version: V1;
    protected _solution: UsAppToPersonContextSolution;
    protected _uri: string;
    constructor(_version: V1, messagingServiceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: UsAppToPersonInstance) => any): Promise<UsAppToPersonInstance>;
    update(params: UsAppToPersonContextUpdateOptions, callback?: (error: Error | null, item?: UsAppToPersonInstance) => any): Promise<UsAppToPersonInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): UsAppToPersonContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface UsAppToPersonPayload extends TwilioResponsePayload {
    compliance: UsAppToPersonResource[];
}
interface UsAppToPersonResource {
    sid: string;
    account_sid: string;
    brand_registration_sid: string;
    messaging_service_sid: string;
    description: string;
    message_samples: Array<string>;
    us_app_to_person_usecase: string;
    has_embedded_links: boolean;
    has_embedded_phone: boolean;
    subscriber_opt_in: boolean;
    age_gated: boolean;
    direct_lending: boolean;
    campaign_status: string;
    campaign_id: string;
    is_externally_registered: boolean;
    rate_limits: any;
    message_flow: string;
    opt_in_message: string;
    opt_out_message: string;
    help_message: string;
    opt_in_keywords: Array<string>;
    opt_out_keywords: Array<string>;
    help_keywords: Array<string>;
    date_created: Date;
    date_updated: Date;
    url: string;
    mock: boolean;
    errors: Array<any>;
}
export declare class UsAppToPersonInstance {
    protected _version: V1;
    protected _solution: UsAppToPersonContextSolution;
    protected _context?: UsAppToPersonContext;
    constructor(_version: V1, payload: UsAppToPersonResource, messagingServiceSid: string, sid?: string);
    /**
     * The unique string that identifies a US A2P Compliance resource `QE2c6890da8086d771620e9b13fadeba0b`.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that the Campaign belongs to.
     */
    accountSid: string;
    /**
     * The unique string to identify the A2P brand.
     */
    brandRegistrationSid: string;
    /**
     * The SID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) that the resource is associated with.
     */
    messagingServiceSid: string;
    /**
     * A short description of what this SMS campaign does. Min length: 40 characters. Max length: 4096 characters.
     */
    description: string;
    /**
     * An array of sample message strings, min two and max five. Min length for each sample: 20 chars. Max length for each sample: 1024 chars.
     */
    messageSamples: Array<string>;
    /**
     * A2P Campaign Use Case. Examples: [ 2FA, EMERGENCY, MARKETING, SOLE_PROPRIETOR...]. SOLE_PROPRIETOR campaign use cases can only be created by SOLE_PROPRIETOR Brands, and there can only be one SOLE_PROPRIETOR campaign created per SOLE_PROPRIETOR Brand.
     */
    usAppToPersonUsecase: string;
    /**
     * Indicate that this SMS campaign will send messages that contain links.
     */
    hasEmbeddedLinks: boolean;
    /**
     * Indicates that this SMS campaign will send messages that contain phone numbers.
     */
    hasEmbeddedPhone: boolean;
    /**
     * A boolean that specifies whether campaign has Subscriber Optin or not.
     */
    subscriberOptIn: boolean;
    /**
     * A boolean that specifies whether campaign is age gated or not.
     */
    ageGated: boolean;
    /**
     * A boolean that specifies whether campaign allows direct lending or not.
     */
    directLending: boolean;
    /**
     * Campaign status. Examples: IN_PROGRESS, VERIFIED, FAILED.
     */
    campaignStatus: string;
    /**
     * The Campaign Registry (TCR) Campaign ID.
     */
    campaignId: string;
    /**
     * Indicates whether the campaign was registered externally or not.
     */
    isExternallyRegistered: boolean;
    /**
     * Rate limit and/or classification set by each carrier, Ex. AT&T or T-Mobile.
     */
    rateLimits: any;
    /**
     * Details around how a consumer opts-in to their campaign, therefore giving consent to receive their messages. If multiple opt-in methods can be used for the same campaign, they must all be listed. 40 character minimum. 2048 character maximum.
     */
    messageFlow: string;
    /**
     * If end users can text in a keyword to start receiving messages from this campaign, the auto-reply messages sent to the end users must be provided. The opt-in response should include the Brand name, confirmation of opt-in enrollment to a recurring message campaign, how to get help, and clear description of how to opt-out. This field is required if end users can text in a keyword to start receiving messages from this campaign. 20 character minimum. 320 character maximum.
     */
    optInMessage: string;
    /**
     * Upon receiving the opt-out keywords from the end users, Twilio customers are expected to send back an auto-generated response, which must provide acknowledgment of the opt-out request and confirmation that no further messages will be sent. It is also recommended that these opt-out messages include the brand name. This field is required if managing opt out keywords yourself (i.e. not using Twilio\'s Default or Advanced Opt Out features). 20 character minimum. 320 character maximum.
     */
    optOutMessage: string;
    /**
     * When customers receive the help keywords from their end users, Twilio customers are expected to send back an auto-generated response; this may include the brand name and additional support contact information. This field is required if managing help keywords yourself (i.e. not using Twilio\'s Default or Advanced Opt Out features). 20 character minimum. 320 character maximum.
     */
    helpMessage: string;
    /**
     * If end users can text in a keyword to start receiving messages from this campaign, those keywords must be provided. This field is required if end users can text in a keyword to start receiving messages from this campaign. Values must be alphanumeric. 255 character maximum.
     */
    optInKeywords: Array<string>;
    /**
     * End users should be able to text in a keyword to stop receiving messages from this campaign. Those keywords must be provided. This field is required if managing opt out keywords yourself (i.e. not using Twilio\'s Default or Advanced Opt Out features). Values must be alphanumeric. 255 character maximum.
     */
    optOutKeywords: Array<string>;
    /**
     * End users should be able to text in a keyword to receive help. Those keywords must be provided as part of the campaign registration request. This field is required if managing help keywords yourself (i.e. not using Twilio\'s Default or Advanced Opt Out features). Values must be alphanumeric. 255 character maximum.
     */
    helpKeywords: Array<string>;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the US App to Person resource.
     */
    url: string;
    /**
     * A boolean that specifies whether campaign is a mock or not. Mock campaigns will be automatically created if using a mock brand. Mock campaigns should only be used for testing purposes.
     */
    mock: boolean;
    /**
     * Details indicating why a campaign registration failed. These errors can indicate one or more fields that were incorrect or did not meet review requirements.
     */
    errors: Array<any>;
    private get _proxy();
    /**
     * Remove a UsAppToPersonInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a UsAppToPersonInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsAppToPersonInstance
     */
    fetch(callback?: (error: Error | null, item?: UsAppToPersonInstance) => any): Promise<UsAppToPersonInstance>;
    /**
     * Update a UsAppToPersonInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsAppToPersonInstance
     */
    update(params: UsAppToPersonContextUpdateOptions, callback?: (error: Error | null, item?: UsAppToPersonInstance) => any): Promise<UsAppToPersonInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        brandRegistrationSid: string;
        messagingServiceSid: string;
        description: string;
        messageSamples: string[];
        usAppToPersonUsecase: string;
        hasEmbeddedLinks: boolean;
        hasEmbeddedPhone: boolean;
        subscriberOptIn: boolean;
        ageGated: boolean;
        directLending: boolean;
        campaignStatus: string;
        campaignId: string;
        isExternallyRegistered: boolean;
        rateLimits: any;
        messageFlow: string;
        optInMessage: string;
        optOutMessage: string;
        helpMessage: string;
        optInKeywords: string[];
        optOutKeywords: string[];
        helpKeywords: string[];
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        mock: boolean;
        errors: any[];
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface UsAppToPersonSolution {
    messagingServiceSid: string;
}
export interface UsAppToPersonListInstance {
    _version: V1;
    _solution: UsAppToPersonSolution;
    _uri: string;
    (sid: string): UsAppToPersonContext;
    get(sid: string): UsAppToPersonContext;
    /**
     * Create a UsAppToPersonInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsAppToPersonInstance
     */
    create(params: UsAppToPersonListInstanceCreateOptions, callback?: (error: Error | null, item?: UsAppToPersonInstance) => any): Promise<UsAppToPersonInstance>;
    /**
     * Streams UsAppToPersonInstance records from the API.
     *
     * This operation lazily loads records as efficiently as possible until the limit
     * is reached.
     *
     * The results are passed into the callback function, so this operation is memory
     * efficient.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UsAppToPersonListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: UsAppToPersonInstance, done: (err?: Error) => void) => void): void;
    each(params: UsAppToPersonListInstanceEachOptions, callback?: (item: UsAppToPersonInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of UsAppToPersonInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: UsAppToPersonPage) => any): Promise<UsAppToPersonPage>;
    /**
     * Lists UsAppToPersonInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UsAppToPersonListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: UsAppToPersonInstance[]) => any): Promise<UsAppToPersonInstance[]>;
    list(params: UsAppToPersonListInstanceOptions, callback?: (error: Error | null, items: UsAppToPersonInstance[]) => any): Promise<UsAppToPersonInstance[]>;
    /**
     * Retrieve a single page of UsAppToPersonInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UsAppToPersonListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: UsAppToPersonPage) => any): Promise<UsAppToPersonPage>;
    page(params: UsAppToPersonListInstancePageOptions, callback?: (error: Error | null, items: UsAppToPersonPage) => any): Promise<UsAppToPersonPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UsAppToPersonListInstance(version: V1, messagingServiceSid: string): UsAppToPersonListInstance;
export declare class UsAppToPersonPage extends Page<V1, UsAppToPersonPayload, UsAppToPersonResource, UsAppToPersonInstance> {
    /**
     * Initialize the UsAppToPersonPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: UsAppToPersonSolution);
    /**
     * Build an instance of UsAppToPersonInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: UsAppToPersonResource): UsAppToPersonInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
