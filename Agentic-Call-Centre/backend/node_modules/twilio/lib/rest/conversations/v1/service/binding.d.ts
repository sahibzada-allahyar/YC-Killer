/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type BindingBindingType = "apn" | "gcm" | "fcm";
/**
 * Options to pass to each
 */
export interface BindingListInstanceEachOptions {
    /** The push technology used by the Binding resources to read.  Can be: `apn`, `gcm`, or `fcm`.  See [push notification configuration](https://www.twilio.com/docs/chat/push-notification-configuration) for more info. */
    bindingType?: Array<BindingBindingType>;
    /** The identity of a [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource) this binding belongs to. See [access tokens](https://www.twilio.com/docs/conversations/create-tokens) for more details. */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: BindingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface BindingListInstanceOptions {
    /** The push technology used by the Binding resources to read.  Can be: `apn`, `gcm`, or `fcm`.  See [push notification configuration](https://www.twilio.com/docs/chat/push-notification-configuration) for more info. */
    bindingType?: Array<BindingBindingType>;
    /** The identity of a [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource) this binding belongs to. See [access tokens](https://www.twilio.com/docs/conversations/create-tokens) for more details. */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface BindingListInstancePageOptions {
    /** The push technology used by the Binding resources to read.  Can be: `apn`, `gcm`, or `fcm`.  See [push notification configuration](https://www.twilio.com/docs/chat/push-notification-configuration) for more info. */
    bindingType?: Array<BindingBindingType>;
    /** The identity of a [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource) this binding belongs to. See [access tokens](https://www.twilio.com/docs/conversations/create-tokens) for more details. */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface BindingContext {
    /**
     * Remove a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BindingInstance
     */
    fetch(callback?: (error: Error | null, item?: BindingInstance) => any): Promise<BindingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BindingContextSolution {
    chatServiceSid: string;
    sid: string;
}
export declare class BindingContextImpl implements BindingContext {
    protected _version: V1;
    protected _solution: BindingContextSolution;
    protected _uri: string;
    constructor(_version: V1, chatServiceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: BindingInstance) => any): Promise<BindingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BindingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BindingPayload extends TwilioResponsePayload {
    bindings: BindingResource[];
}
interface BindingResource {
    sid: string;
    account_sid: string;
    chat_service_sid: string;
    credential_sid: string;
    date_created: Date;
    date_updated: Date;
    endpoint: string;
    identity: string;
    binding_type: BindingBindingType;
    message_types: Array<string>;
    url: string;
}
export declare class BindingInstance {
    protected _version: V1;
    protected _solution: BindingContextSolution;
    protected _context?: BindingContext;
    constructor(_version: V1, payload: BindingResource, chatServiceSid: string, sid?: string);
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this binding.
     */
    accountSid: string;
    /**
     * The SID of the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) the Binding resource is associated with.
     */
    chatServiceSid: string;
    /**
     * The SID of the [Credential](https://www.twilio.com/docs/conversations/api/credential-resource) for the binding. See [push notification configuration](https://www.twilio.com/docs/chat/push-notification-configuration) for more info.
     */
    credentialSid: string;
    /**
     * The date that this resource was created.
     */
    dateCreated: Date;
    /**
     * The date that this resource was last updated.
     */
    dateUpdated: Date;
    /**
     * The unique endpoint identifier for the Binding. The format of this value depends on the `binding_type`.
     */
    endpoint: string;
    /**
     * The application-defined string that uniquely identifies the [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource) within the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource). See [access tokens](https://www.twilio.com/docs/conversations/create-tokens) for more info.
     */
    identity: string;
    bindingType: BindingBindingType;
    /**
     * The [Conversation message types](https://www.twilio.com/docs/chat/push-notification-configuration#push-types) the binding is subscribed to.
     */
    messageTypes: Array<string>;
    /**
     * An absolute API resource URL for this binding.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BindingInstance
     */
    fetch(callback?: (error: Error | null, item?: BindingInstance) => any): Promise<BindingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        chatServiceSid: string;
        credentialSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        endpoint: string;
        identity: string;
        bindingType: BindingBindingType;
        messageTypes: string[];
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BindingSolution {
    chatServiceSid: string;
}
export interface BindingListInstance {
    _version: V1;
    _solution: BindingSolution;
    _uri: string;
    (sid: string): BindingContext;
    get(sid: string): BindingContext;
    /**
     * Streams BindingInstance records from the API.
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
     * @param { BindingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: BindingInstance, done: (err?: Error) => void) => void): void;
    each(params: BindingListInstanceEachOptions, callback?: (item: BindingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of BindingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: BindingPage) => any): Promise<BindingPage>;
    /**
     * Lists BindingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BindingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: BindingInstance[]) => any): Promise<BindingInstance[]>;
    list(params: BindingListInstanceOptions, callback?: (error: Error | null, items: BindingInstance[]) => any): Promise<BindingInstance[]>;
    /**
     * Retrieve a single page of BindingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BindingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: BindingPage) => any): Promise<BindingPage>;
    page(params: BindingListInstancePageOptions, callback?: (error: Error | null, items: BindingPage) => any): Promise<BindingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BindingListInstance(version: V1, chatServiceSid: string): BindingListInstance;
export declare class BindingPage extends Page<V1, BindingPayload, BindingResource, BindingInstance> {
    /**
     * Initialize the BindingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: BindingSolution);
    /**
     * Build an instance of BindingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: BindingResource): BindingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
