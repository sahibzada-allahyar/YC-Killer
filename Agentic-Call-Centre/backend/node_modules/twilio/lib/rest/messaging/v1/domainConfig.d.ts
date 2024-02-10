/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to update a DomainConfigInstance
 */
export interface DomainConfigContextUpdateOptions {
    /** Any requests we receive to this domain that do not match an existing shortened message will be redirected to the fallback url. These will likely be either expired messages, random misdirected traffic, or intentional scraping. */
    fallbackUrl?: string;
    /** URL to receive click events to your webhook whenever the recipients click on the shortened links */
    callbackUrl?: string;
    /** Boolean field to set customer delivery preference when there is a failure in linkShortening service */
    continueOnFailure?: boolean;
    /** Customer\\\'s choice to send links with/without \\\"https://\\\" attached to shortened url. If true, messages will not be sent with https:// at the beginning of the url. If false, messages will be sent with https:// at the beginning of the url. False is the default behavior if it is not specified. */
    disableHttps?: boolean;
}
export interface DomainConfigContext {
    /**
     * Fetch a DomainConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainConfigInstance
     */
    fetch(callback?: (error: Error | null, item?: DomainConfigInstance) => any): Promise<DomainConfigInstance>;
    /**
     * Update a DomainConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainConfigInstance
     */
    update(callback?: (error: Error | null, item?: DomainConfigInstance) => any): Promise<DomainConfigInstance>;
    /**
     * Update a DomainConfigInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainConfigInstance
     */
    update(params: DomainConfigContextUpdateOptions, callback?: (error: Error | null, item?: DomainConfigInstance) => any): Promise<DomainConfigInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DomainConfigContextSolution {
    domainSid: string;
}
export declare class DomainConfigContextImpl implements DomainConfigContext {
    protected _version: V1;
    protected _solution: DomainConfigContextSolution;
    protected _uri: string;
    constructor(_version: V1, domainSid: string);
    fetch(callback?: (error: Error | null, item?: DomainConfigInstance) => any): Promise<DomainConfigInstance>;
    update(params?: DomainConfigContextUpdateOptions | ((error: Error | null, item?: DomainConfigInstance) => any), callback?: (error: Error | null, item?: DomainConfigInstance) => any): Promise<DomainConfigInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DomainConfigContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DomainConfigResource {
    domain_sid: string;
    config_sid: string;
    fallback_url: string;
    callback_url: string;
    continue_on_failure: boolean;
    date_created: Date;
    date_updated: Date;
    url: string;
    disable_https: boolean;
}
export declare class DomainConfigInstance {
    protected _version: V1;
    protected _solution: DomainConfigContextSolution;
    protected _context?: DomainConfigContext;
    constructor(_version: V1, payload: DomainConfigResource, domainSid?: string);
    /**
     * The unique string that we created to identify the Domain resource.
     */
    domainSid: string;
    /**
     * The unique string that we created to identify the Domain config (prefix ZK).
     */
    configSid: string;
    /**
     * Any requests we receive to this domain that do not match an existing shortened message will be redirected to the fallback url. These will likely be either expired messages, random misdirected traffic, or intentional scraping.
     */
    fallbackUrl: string;
    /**
     * URL to receive click events to your webhook whenever the recipients click on the shortened links.
     */
    callbackUrl: string;
    /**
     * Boolean field to set customer delivery preference when there is a failure in linkShortening service
     */
    continueOnFailure: boolean;
    /**
     * Date this Domain Config was created.
     */
    dateCreated: Date;
    /**
     * Date that this Domain Config was last updated.
     */
    dateUpdated: Date;
    url: string;
    /**
     * Customer\'s choice to send links with/without \"https://\" attached to shortened url. If true, messages will not be sent with https:// at the beginning of the url. If false, messages will be sent with https:// at the beginning of the url. False is the default behavior if it is not specified.
     */
    disableHttps: boolean;
    private get _proxy();
    /**
     * Fetch a DomainConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainConfigInstance
     */
    fetch(callback?: (error: Error | null, item?: DomainConfigInstance) => any): Promise<DomainConfigInstance>;
    /**
     * Update a DomainConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainConfigInstance
     */
    update(callback?: (error: Error | null, item?: DomainConfigInstance) => any): Promise<DomainConfigInstance>;
    /**
     * Update a DomainConfigInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainConfigInstance
     */
    update(params: DomainConfigContextUpdateOptions, callback?: (error: Error | null, item?: DomainConfigInstance) => any): Promise<DomainConfigInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        domainSid: string;
        configSid: string;
        fallbackUrl: string;
        callbackUrl: string;
        continueOnFailure: boolean;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        disableHttps: boolean;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DomainConfigSolution {
}
export interface DomainConfigListInstance {
    _version: V1;
    _solution: DomainConfigSolution;
    _uri: string;
    (domainSid: string): DomainConfigContext;
    get(domainSid: string): DomainConfigContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DomainConfigListInstance(version: V1): DomainConfigListInstance;
export {};
