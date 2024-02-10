/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V3 from "../V3";
export type ChannelChannelType = "public" | "private";
export type ChannelWebhookEnabledType = "true" | "false";
/**
 * Options to pass to update a ChannelInstance
 */
export interface ChannelContextUpdateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: ChannelWebhookEnabledType;
    /**  */
    type?: ChannelChannelType;
    /** The unique ID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) this channel belongs to. */
    messagingServiceSid?: string;
}
export interface ChannelContext {
    /**
     * Update a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    update(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Update a ChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    update(params: ChannelContextUpdateOptions, callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ChannelContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class ChannelContextImpl implements ChannelContext {
    protected _version: V3;
    protected _solution: ChannelContextSolution;
    protected _uri: string;
    constructor(_version: V3, serviceSid: string, sid: string);
    update(params?: ChannelContextUpdateOptions | ((error: Error | null, item?: ChannelInstance) => any), callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ChannelContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ChannelResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    friendly_name: string;
    unique_name: string;
    attributes: string;
    type: ChannelChannelType;
    date_created: Date;
    date_updated: Date;
    created_by: string;
    members_count: number;
    messages_count: number;
    messaging_service_sid: string;
    url: string;
}
export declare class ChannelInstance {
    protected _version: V3;
    protected _solution: ChannelContextSolution;
    protected _context?: ChannelContext;
    constructor(_version: V3, payload: ChannelResource, serviceSid?: string, sid?: string);
    /**
     * The unique string that we created to identify the Channel resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Channel resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/chat/rest/service-resource) the Channel resource is associated with.
     */
    serviceSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used to address the resource in place of the resource\'s `sid` in the URL.
     */
    uniqueName: string;
    /**
     * The JSON string that stores application-specific data. If attributes have not been set, `{}` is returned.
     */
    attributes: string;
    type: ChannelChannelType;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The `identity` of the User that created the channel. If the Channel was created by using the API, the value is `system`.
     */
    createdBy: string;
    /**
     * The number of Members in the Channel.
     */
    membersCount: number;
    /**
     * The number of Messages that have been passed in the Channel.
     */
    messagesCount: number;
    /**
     * The unique ID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) this channel belongs to.
     */
    messagingServiceSid: string;
    /**
     * The absolute URL of the Channel resource.
     */
    url: string;
    private get _proxy();
    /**
     * Update a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    update(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Update a ChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    update(params: ChannelContextUpdateOptions, callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        friendlyName: string;
        uniqueName: string;
        attributes: string;
        type: ChannelChannelType;
        dateCreated: Date;
        dateUpdated: Date;
        createdBy: string;
        membersCount: number;
        messagesCount: number;
        messagingServiceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ChannelSolution {
}
export interface ChannelListInstance {
    _version: V3;
    _solution: ChannelSolution;
    _uri: string;
    (serviceSid: string, sid: string): ChannelContext;
    get(serviceSid: string, sid: string): ChannelContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ChannelListInstance(version: V3): ChannelListInstance;
export {};
