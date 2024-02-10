/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
/**
 * Options to pass to update a MemberInstance
 */
export interface MemberContextUpdateOptions {
    /** The SID of the [Role](https://www.twilio.com/docs/api/chat/rest/roles) to assign to the member. The default roles are those specified on the [Service](https://www.twilio.com/docs/chat/api/services). */
    roleSid?: string;
    /** The index of the last [Message](https://www.twilio.com/docs/api/chat/rest/messages) that the Member has read within the [Channel](https://www.twilio.com/docs/api/chat/rest/channels). */
    lastConsumedMessageIndex?: number;
}
/**
 * Options to pass to create a MemberInstance
 */
export interface MemberListInstanceCreateOptions {
    /** The `identity` value that uniquely identifies the new resource\\\'s [User](https://www.twilio.com/docs/api/chat/rest/v1/user) within the [Service](https://www.twilio.com/docs/api/chat/rest/services). See [access tokens](https://www.twilio.com/docs/api/chat/guides/create-tokens) for more details. */
    identity: string;
    /** The SID of the [Role](https://www.twilio.com/docs/api/chat/rest/roles) to assign to the member. The default roles are those specified on the [Service](https://www.twilio.com/docs/chat/api/services). */
    roleSid?: string;
}
/**
 * Options to pass to each
 */
export interface MemberListInstanceEachOptions {
    /** The [User](https://www.twilio.com/docs/api/chat/rest/v1/user)\'s `identity` value of the resources to read. See [access tokens](https://www.twilio.com/docs/api/chat/guides/create-tokens) for more details. */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: MemberInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface MemberListInstanceOptions {
    /** The [User](https://www.twilio.com/docs/api/chat/rest/v1/user)\'s `identity` value of the resources to read. See [access tokens](https://www.twilio.com/docs/api/chat/guides/create-tokens) for more details. */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface MemberListInstancePageOptions {
    /** The [User](https://www.twilio.com/docs/api/chat/rest/v1/user)\'s `identity` value of the resources to read. See [access tokens](https://www.twilio.com/docs/api/chat/guides/create-tokens) for more details. */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface MemberContext {
    /**
     * Remove a MemberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MemberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MemberInstance
     */
    fetch(callback?: (error: Error | null, item?: MemberInstance) => any): Promise<MemberInstance>;
    /**
     * Update a MemberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MemberInstance
     */
    update(callback?: (error: Error | null, item?: MemberInstance) => any): Promise<MemberInstance>;
    /**
     * Update a MemberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MemberInstance
     */
    update(params: MemberContextUpdateOptions, callback?: (error: Error | null, item?: MemberInstance) => any): Promise<MemberInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface MemberContextSolution {
    serviceSid: string;
    channelSid: string;
    sid: string;
}
export declare class MemberContextImpl implements MemberContext {
    protected _version: V1;
    protected _solution: MemberContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, channelSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: MemberInstance) => any): Promise<MemberInstance>;
    update(params?: MemberContextUpdateOptions | ((error: Error | null, item?: MemberInstance) => any), callback?: (error: Error | null, item?: MemberInstance) => any): Promise<MemberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): MemberContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface MemberPayload extends TwilioResponsePayload {
    members: MemberResource[];
}
interface MemberResource {
    sid: string;
    account_sid: string;
    channel_sid: string;
    service_sid: string;
    identity: string;
    date_created: Date;
    date_updated: Date;
    role_sid: string;
    last_consumed_message_index: number;
    last_consumption_timestamp: Date;
    url: string;
}
export declare class MemberInstance {
    protected _version: V1;
    protected _solution: MemberContextSolution;
    protected _context?: MemberContext;
    constructor(_version: V1, payload: MemberResource, serviceSid: string, channelSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Member resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/api/rest/account) that created the Member resource.
     */
    accountSid: string;
    /**
     * The unique ID of the [Channel](https://www.twilio.com/docs/api/chat/rest/channels) for the member.
     */
    channelSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/api/chat/rest/services) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The application-defined string that uniquely identifies the resource\'s [User](https://www.twilio.com/docs/api/chat/rest/users) within the [Service](https://www.twilio.com/docs/api/chat/rest/services). See [access tokens](https://www.twilio.com/docs/api/chat/guides/create-tokens) for more info.
     */
    identity: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The SID of the [Role](https://www.twilio.com/docs/api/chat/rest/roles) assigned to the member.
     */
    roleSid: string;
    /**
     * The index of the last [Message](https://www.twilio.com/docs/api/chat/rest/messages) in the [Channel](https://www.twilio.com/docs/api/chat/rest/channels) that the Member has read.
     */
    lastConsumedMessageIndex: number;
    /**
     * The ISO 8601 timestamp string that represents the date-time of the last [Message](https://www.twilio.com/docs/api/chat/rest/messages) read event for the Member within the [Channel](https://www.twilio.com/docs/api/chat/rest/channels).
     */
    lastConsumptionTimestamp: Date;
    /**
     * The absolute URL of the Member resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a MemberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MemberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MemberInstance
     */
    fetch(callback?: (error: Error | null, item?: MemberInstance) => any): Promise<MemberInstance>;
    /**
     * Update a MemberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MemberInstance
     */
    update(callback?: (error: Error | null, item?: MemberInstance) => any): Promise<MemberInstance>;
    /**
     * Update a MemberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MemberInstance
     */
    update(params: MemberContextUpdateOptions, callback?: (error: Error | null, item?: MemberInstance) => any): Promise<MemberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        channelSid: string;
        serviceSid: string;
        identity: string;
        dateCreated: Date;
        dateUpdated: Date;
        roleSid: string;
        lastConsumedMessageIndex: number;
        lastConsumptionTimestamp: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface MemberSolution {
    serviceSid: string;
    channelSid: string;
}
export interface MemberListInstance {
    _version: V1;
    _solution: MemberSolution;
    _uri: string;
    (sid: string): MemberContext;
    get(sid: string): MemberContext;
    /**
     * Create a MemberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MemberInstance
     */
    create(params: MemberListInstanceCreateOptions, callback?: (error: Error | null, item?: MemberInstance) => any): Promise<MemberInstance>;
    /**
     * Streams MemberInstance records from the API.
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
     * @param { MemberListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: MemberInstance, done: (err?: Error) => void) => void): void;
    each(params: MemberListInstanceEachOptions, callback?: (item: MemberInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of MemberInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: MemberPage) => any): Promise<MemberPage>;
    /**
     * Lists MemberInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MemberListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: MemberInstance[]) => any): Promise<MemberInstance[]>;
    list(params: MemberListInstanceOptions, callback?: (error: Error | null, items: MemberInstance[]) => any): Promise<MemberInstance[]>;
    /**
     * Retrieve a single page of MemberInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MemberListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: MemberPage) => any): Promise<MemberPage>;
    page(params: MemberListInstancePageOptions, callback?: (error: Error | null, items: MemberPage) => any): Promise<MemberPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MemberListInstance(version: V1, serviceSid: string, channelSid: string): MemberListInstance;
export declare class MemberPage extends Page<V1, MemberPayload, MemberResource, MemberInstance> {
    /**
     * Initialize the MemberPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: MemberSolution);
    /**
     * Build an instance of MemberInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: MemberResource): MemberInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
