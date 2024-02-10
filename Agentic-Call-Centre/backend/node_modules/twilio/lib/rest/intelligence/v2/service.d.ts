/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V2 from "../V2";
export type ServiceHttpMethod = "GET" | "POST" | "NULL";
/**
 * Options to pass to update a ServiceInstance
 */
export interface ServiceContextUpdateOptions {
    /** The If-Match HTTP request header */
    ifMatch?: string;
    /** Instructs the Speech Recognition service to automatically transcribe all recordings made on the account. */
    autoTranscribe?: boolean;
    /** Data logging allows Twilio to improve the quality of the speech recognition & language understanding services through using customer data to refine, fine tune and evaluate machine learning models. Note: Data logging cannot be activated via API, only via www.twilio.com, as it requires additional consent. */
    dataLogging?: boolean;
    /** A human readable description of this resource, up to 64 characters. */
    friendlyName?: string;
    /** The default language code of the audio. */
    languageCode?: string;
    /** Provides a unique and addressable name to be assigned to this Service, assigned by the developer, to be optionally used in addition to SID. */
    uniqueName?: string;
    /** Instructs the Speech Recognition service to automatically redact PII from all transcripts made on this service. */
    autoRedaction?: boolean;
    /** Instructs the Speech Recognition service to automatically redact PII from all transcripts media made on this service. The auto_redaction flag must be enabled, results in error otherwise. */
    mediaRedaction?: boolean;
    /** The URL Twilio will request when executing the Webhook. */
    webhookUrl?: string;
    /**  */
    webhookHttpMethod?: ServiceHttpMethod;
}
/**
 * Options to pass to create a ServiceInstance
 */
export interface ServiceListInstanceCreateOptions {
    /** Provides a unique and addressable name to be assigned to this Service, assigned by the developer, to be optionally used in addition to SID. */
    uniqueName: string;
    /** Instructs the Speech Recognition service to automatically transcribe all recordings made on the account. */
    autoTranscribe?: boolean;
    /** Data logging allows Twilio to improve the quality of the speech recognition & language understanding services through using customer data to refine, fine tune and evaluate machine learning models. Note: Data logging cannot be activated via API, only via www.twilio.com, as it requires additional consent. */
    dataLogging?: boolean;
    /** A human readable description of this resource, up to 64 characters. */
    friendlyName?: string;
    /** The default language code of the audio. */
    languageCode?: string;
    /** Instructs the Speech Recognition service to automatically redact PII from all transcripts made on this service. */
    autoRedaction?: boolean;
    /** Instructs the Speech Recognition service to automatically redact PII from all transcripts media made on this service. The auto_redaction flag must be enabled, results in error otherwise. */
    mediaRedaction?: boolean;
    /** The URL Twilio will request when executing the Webhook. */
    webhookUrl?: string;
    /**  */
    webhookHttpMethod?: ServiceHttpMethod;
}
/**
 * Options to pass to each
 */
export interface ServiceListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ServiceInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ServiceListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ServiceListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ServiceContext {
    /**
     * Remove a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    fetch(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Update a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    update(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Update a ServiceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    update(params: ServiceContextUpdateOptions, callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ServiceContextSolution {
    sid: string;
}
export declare class ServiceContextImpl implements ServiceContext {
    protected _version: V2;
    protected _solution: ServiceContextSolution;
    protected _uri: string;
    constructor(_version: V2, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    update(params?: ServiceContextUpdateOptions | ((error: Error | null, item?: ServiceInstance) => any), callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ServiceContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ServicePayload extends TwilioResponsePayload {
    services: ServiceResource[];
}
interface ServiceResource {
    account_sid: string;
    auto_redaction: boolean;
    media_redaction: boolean;
    auto_transcribe: boolean;
    data_logging: boolean;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    language_code: string;
    sid: string;
    unique_name: string;
    url: string;
    webhook_url: string;
    webhook_http_method: ServiceHttpMethod;
    version: number;
}
export declare class ServiceInstance {
    protected _version: V2;
    protected _solution: ServiceContextSolution;
    protected _context?: ServiceContext;
    constructor(_version: V2, payload: ServiceResource, sid?: string);
    /**
     * The unique SID identifier of the Account the Service belongs to.
     */
    accountSid: string;
    /**
     * Instructs the Speech Recognition service to automatically redact PII from all transcripts made on this service.
     */
    autoRedaction: boolean;
    /**
     * Instructs the Speech Recognition service to automatically redact PII from all transcripts media made on this service. The auto_redaction flag must be enabled, results in error otherwise.
     */
    mediaRedaction: boolean;
    /**
     * Instructs the Speech Recognition service to automatically transcribe all recordings made on the account.
     */
    autoTranscribe: boolean;
    /**
     * Data logging allows Twilio to improve the quality of the speech recognition & language understanding services through using customer data to refine, fine tune and evaluate machine learning models. Note: Data logging cannot be activated via API, only via www.twilio.com, as it requires additional consent.
     */
    dataLogging: boolean;
    /**
     * The date that this Service was created, given in ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * The date that this Service was updated, given in ISO 8601 format.
     */
    dateUpdated: Date;
    /**
     * A human readable description of this resource, up to 64 characters.
     */
    friendlyName: string;
    /**
     * The default language code of the audio.
     */
    languageCode: string;
    /**
     * A 34 character string that uniquely identifies this Service.
     */
    sid: string;
    /**
     * Provides a unique and addressable name to be assigned to this Service, assigned by the developer, to be optionally used in addition to SID.
     */
    uniqueName: string;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * The URL Twilio will request when executing the Webhook.
     */
    webhookUrl: string;
    webhookHttpMethod: ServiceHttpMethod;
    /**
     * The version number of this Service.
     */
    version: number;
    private get _proxy();
    /**
     * Remove a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    fetch(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Update a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    update(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Update a ServiceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    update(params: ServiceContextUpdateOptions, callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        autoRedaction: boolean;
        mediaRedaction: boolean;
        autoTranscribe: boolean;
        dataLogging: boolean;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        languageCode: string;
        sid: string;
        uniqueName: string;
        url: string;
        webhookUrl: string;
        webhookHttpMethod: ServiceHttpMethod;
        version: number;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ServiceSolution {
}
export interface ServiceListInstance {
    _version: V2;
    _solution: ServiceSolution;
    _uri: string;
    (sid: string): ServiceContext;
    get(sid: string): ServiceContext;
    /**
     * Create a ServiceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    create(params: ServiceListInstanceCreateOptions, callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Streams ServiceInstance records from the API.
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
     * @param { ServiceListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ServiceInstance, done: (err?: Error) => void) => void): void;
    each(params: ServiceListInstanceEachOptions, callback?: (item: ServiceInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ServiceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ServicePage) => any): Promise<ServicePage>;
    /**
     * Lists ServiceInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ServiceListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ServiceInstance[]) => any): Promise<ServiceInstance[]>;
    list(params: ServiceListInstanceOptions, callback?: (error: Error | null, items: ServiceInstance[]) => any): Promise<ServiceInstance[]>;
    /**
     * Retrieve a single page of ServiceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ServiceListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ServicePage) => any): Promise<ServicePage>;
    page(params: ServiceListInstancePageOptions, callback?: (error: Error | null, items: ServicePage) => any): Promise<ServicePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ServiceListInstance(version: V2): ServiceListInstance;
export declare class ServicePage extends Page<V2, ServicePayload, ServiceResource, ServiceInstance> {
    /**
     * Initialize the ServicePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: ServiceSolution);
    /**
     * Build an instance of ServiceInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ServiceResource): ServiceInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
