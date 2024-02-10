/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export interface DomainConfigMessagingServiceContext {
    /**
     * Fetch a DomainConfigMessagingServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainConfigMessagingServiceInstance
     */
    fetch(callback?: (error: Error | null, item?: DomainConfigMessagingServiceInstance) => any): Promise<DomainConfigMessagingServiceInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DomainConfigMessagingServiceContextSolution {
    messagingServiceSid: string;
}
export declare class DomainConfigMessagingServiceContextImpl implements DomainConfigMessagingServiceContext {
    protected _version: V1;
    protected _solution: DomainConfigMessagingServiceContextSolution;
    protected _uri: string;
    constructor(_version: V1, messagingServiceSid: string);
    fetch(callback?: (error: Error | null, item?: DomainConfigMessagingServiceInstance) => any): Promise<DomainConfigMessagingServiceInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DomainConfigMessagingServiceContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DomainConfigMessagingServiceResource {
    domain_sid: string;
    config_sid: string;
    messaging_service_sid: string;
    fallback_url: string;
    callback_url: string;
    continue_on_failure: boolean;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class DomainConfigMessagingServiceInstance {
    protected _version: V1;
    protected _solution: DomainConfigMessagingServiceContextSolution;
    protected _context?: DomainConfigMessagingServiceContext;
    constructor(_version: V1, payload: DomainConfigMessagingServiceResource, messagingServiceSid?: string);
    /**
     * The unique string that we created to identify the Domain resource.
     */
    domainSid: string;
    /**
     * The unique string that we created to identify the Domain config (prefix ZK).
     */
    configSid: string;
    /**
     * The unique string that identifies the messaging service
     */
    messagingServiceSid: string;
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
    private get _proxy();
    /**
     * Fetch a DomainConfigMessagingServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainConfigMessagingServiceInstance
     */
    fetch(callback?: (error: Error | null, item?: DomainConfigMessagingServiceInstance) => any): Promise<DomainConfigMessagingServiceInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        domainSid: string;
        configSid: string;
        messagingServiceSid: string;
        fallbackUrl: string;
        callbackUrl: string;
        continueOnFailure: boolean;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DomainConfigMessagingServiceSolution {
}
export interface DomainConfigMessagingServiceListInstance {
    _version: V1;
    _solution: DomainConfigMessagingServiceSolution;
    _uri: string;
    (messagingServiceSid: string): DomainConfigMessagingServiceContext;
    get(messagingServiceSid: string): DomainConfigMessagingServiceContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DomainConfigMessagingServiceListInstance(version: V1): DomainConfigMessagingServiceListInstance;
export {};
