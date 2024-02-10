/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export interface LinkshorteningMessagingServiceContext {
    /**
     * Create a LinkshorteningMessagingServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed LinkshorteningMessagingServiceInstance
     */
    create(callback?: (error: Error | null, item?: LinkshorteningMessagingServiceInstance) => any): Promise<LinkshorteningMessagingServiceInstance>;
    /**
     * Remove a LinkshorteningMessagingServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface LinkshorteningMessagingServiceContextSolution {
    domainSid: string;
    messagingServiceSid: string;
}
export declare class LinkshorteningMessagingServiceContextImpl implements LinkshorteningMessagingServiceContext {
    protected _version: V1;
    protected _solution: LinkshorteningMessagingServiceContextSolution;
    protected _uri: string;
    constructor(_version: V1, domainSid: string, messagingServiceSid: string);
    create(callback?: (error: Error | null, item?: LinkshorteningMessagingServiceInstance) => any): Promise<LinkshorteningMessagingServiceInstance>;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): LinkshorteningMessagingServiceContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface LinkshorteningMessagingServiceResource {
    domain_sid: string;
    messaging_service_sid: string;
    url: string;
}
export declare class LinkshorteningMessagingServiceInstance {
    protected _version: V1;
    protected _solution: LinkshorteningMessagingServiceContextSolution;
    protected _context?: LinkshorteningMessagingServiceContext;
    constructor(_version: V1, payload: LinkshorteningMessagingServiceResource, domainSid?: string, messagingServiceSid?: string);
    /**
     * The unique string identifies the domain resource
     */
    domainSid: string;
    /**
     * The unique string that identifies the messaging service
     */
    messagingServiceSid: string;
    url: string;
    private get _proxy();
    /**
     * Create a LinkshorteningMessagingServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed LinkshorteningMessagingServiceInstance
     */
    create(callback?: (error: Error | null, item?: LinkshorteningMessagingServiceInstance) => any): Promise<LinkshorteningMessagingServiceInstance>;
    /**
     * Remove a LinkshorteningMessagingServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        domainSid: string;
        messagingServiceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface LinkshorteningMessagingServiceSolution {
}
export interface LinkshorteningMessagingServiceListInstance {
    _version: V1;
    _solution: LinkshorteningMessagingServiceSolution;
    _uri: string;
    (domainSid: string, messagingServiceSid: string): LinkshorteningMessagingServiceContext;
    get(domainSid: string, messagingServiceSid: string): LinkshorteningMessagingServiceContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function LinkshorteningMessagingServiceListInstance(version: V1): LinkshorteningMessagingServiceListInstance;
export {};
