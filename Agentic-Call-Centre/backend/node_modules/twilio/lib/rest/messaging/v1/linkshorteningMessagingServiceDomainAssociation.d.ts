/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export interface LinkshorteningMessagingServiceDomainAssociationContext {
    /**
     * Fetch a LinkshorteningMessagingServiceDomainAssociationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed LinkshorteningMessagingServiceDomainAssociationInstance
     */
    fetch(callback?: (error: Error | null, item?: LinkshorteningMessagingServiceDomainAssociationInstance) => any): Promise<LinkshorteningMessagingServiceDomainAssociationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface LinkshorteningMessagingServiceDomainAssociationContextSolution {
    messagingServiceSid: string;
}
export declare class LinkshorteningMessagingServiceDomainAssociationContextImpl implements LinkshorteningMessagingServiceDomainAssociationContext {
    protected _version: V1;
    protected _solution: LinkshorteningMessagingServiceDomainAssociationContextSolution;
    protected _uri: string;
    constructor(_version: V1, messagingServiceSid: string);
    fetch(callback?: (error: Error | null, item?: LinkshorteningMessagingServiceDomainAssociationInstance) => any): Promise<LinkshorteningMessagingServiceDomainAssociationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): LinkshorteningMessagingServiceDomainAssociationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface LinkshorteningMessagingServiceDomainAssociationResource {
    domain_sid: string;
    messaging_service_sid: string;
    url: string;
}
export declare class LinkshorteningMessagingServiceDomainAssociationInstance {
    protected _version: V1;
    protected _solution: LinkshorteningMessagingServiceDomainAssociationContextSolution;
    protected _context?: LinkshorteningMessagingServiceDomainAssociationContext;
    constructor(_version: V1, payload: LinkshorteningMessagingServiceDomainAssociationResource, messagingServiceSid?: string);
    /**
     * The unique string that we created to identify the Domain resource.
     */
    domainSid: string;
    /**
     * The unique string that identifies the messaging service
     */
    messagingServiceSid: string;
    url: string;
    private get _proxy();
    /**
     * Fetch a LinkshorteningMessagingServiceDomainAssociationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed LinkshorteningMessagingServiceDomainAssociationInstance
     */
    fetch(callback?: (error: Error | null, item?: LinkshorteningMessagingServiceDomainAssociationInstance) => any): Promise<LinkshorteningMessagingServiceDomainAssociationInstance>;
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
export interface LinkshorteningMessagingServiceDomainAssociationSolution {
}
export interface LinkshorteningMessagingServiceDomainAssociationListInstance {
    _version: V1;
    _solution: LinkshorteningMessagingServiceDomainAssociationSolution;
    _uri: string;
    (messagingServiceSid: string): LinkshorteningMessagingServiceDomainAssociationContext;
    get(messagingServiceSid: string): LinkshorteningMessagingServiceDomainAssociationContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function LinkshorteningMessagingServiceDomainAssociationListInstance(version: V1): LinkshorteningMessagingServiceDomainAssociationListInstance;
export {};
