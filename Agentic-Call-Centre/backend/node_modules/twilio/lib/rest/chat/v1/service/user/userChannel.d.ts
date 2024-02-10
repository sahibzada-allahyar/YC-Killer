/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type UserChannelChannelStatus = "joined" | "invited" | "not_participating";
/**
 * Options to pass to each
 */
export interface UserChannelListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: UserChannelInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface UserChannelListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface UserChannelListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface UserChannelSolution {
    serviceSid: string;
    userSid: string;
}
export interface UserChannelListInstance {
    _version: V1;
    _solution: UserChannelSolution;
    _uri: string;
    /**
     * Streams UserChannelInstance records from the API.
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
     * @param { UserChannelListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: UserChannelInstance, done: (err?: Error) => void) => void): void;
    each(params: UserChannelListInstanceEachOptions, callback?: (item: UserChannelInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of UserChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: UserChannelPage) => any): Promise<UserChannelPage>;
    /**
     * Lists UserChannelInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UserChannelListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: UserChannelInstance[]) => any): Promise<UserChannelInstance[]>;
    list(params: UserChannelListInstanceOptions, callback?: (error: Error | null, items: UserChannelInstance[]) => any): Promise<UserChannelInstance[]>;
    /**
     * Retrieve a single page of UserChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UserChannelListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: UserChannelPage) => any): Promise<UserChannelPage>;
    page(params: UserChannelListInstancePageOptions, callback?: (error: Error | null, items: UserChannelPage) => any): Promise<UserChannelPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UserChannelListInstance(version: V1, serviceSid: string, userSid: string): UserChannelListInstance;
interface UserChannelPayload extends TwilioResponsePayload {
    channels: UserChannelResource[];
}
interface UserChannelResource {
    account_sid: string;
    service_sid: string;
    channel_sid: string;
    member_sid: string;
    status: UserChannelChannelStatus;
    last_consumed_message_index: number;
    unread_messages_count: number;
    links: Record<string, string>;
}
export declare class UserChannelInstance {
    protected _version: V1;
    constructor(_version: V1, payload: UserChannelResource, serviceSid: string, userSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/api/rest/account) that created the User Channel resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/api/chat/rest/services) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the [Channel](https://www.twilio.com/docs/api/chat/rest/channels) the resource belongs to.
     */
    channelSid: string;
    /**
     * The SID of a [Member](https://www.twilio.com/docs/api/chat/rest/members) that represents the User on the Channel.
     */
    memberSid: string;
    status: UserChannelChannelStatus;
    /**
     * The index of the last [Message](https://www.twilio.com/docs/api/chat/rest/messages) in the [Channel](https://www.twilio.com/docs/api/chat/rest/channels) that the Member has read.
     */
    lastConsumedMessageIndex: number;
    /**
     * The number of unread Messages in the Channel for the User. Note that retrieving messages on a client endpoint does not mean that messages are consumed or read. See [Consumption Horizon feature](/docs/api/chat/guides/consumption-horizon) to learn how to mark messages as consumed.
     */
    unreadMessagesCount: number;
    /**
     * The absolute URLs of the [Members](https://www.twilio.com/docs/chat/api/members), [Messages](https://www.twilio.com/docs/chat/api/messages) , [Invites](https://www.twilio.com/docs/chat/api/invites) and, if it exists, the last [Message](https://www.twilio.com/docs/chat/api/messages) for the Channel.
     */
    links: Record<string, string>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        serviceSid: string;
        channelSid: string;
        memberSid: string;
        status: UserChannelChannelStatus;
        lastConsumedMessageIndex: number;
        unreadMessagesCount: number;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class UserChannelPage extends Page<V1, UserChannelPayload, UserChannelResource, UserChannelInstance> {
    /**
     * Initialize the UserChannelPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: UserChannelSolution);
    /**
     * Build an instance of UserChannelInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: UserChannelResource): UserChannelInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
