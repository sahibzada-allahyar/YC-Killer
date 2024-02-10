/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type CommandCommandMode = "text" | "binary";
export type CommandDirection = "from_sim" | "to_sim";
export type CommandStatus = "queued" | "sent" | "delivered" | "received" | "failed";
export type CommandTransport = "sms" | "ip";
/**
 * Options to pass to create a CommandInstance
 */
export interface CommandListInstanceCreateOptions {
    /** The message body of the Command. Can be plain text in text mode or a Base64 encoded byte string in binary mode. */
    command: string;
    /** The `sid` or `unique_name` of the [SIM](https://www.twilio.com/docs/iot/wireless/api/sim-resource) to send the Command to. */
    sim?: string;
    /** The HTTP method we use to call `callback_url`. Can be: `POST` or `GET`, and the default is `POST`. */
    callbackMethod?: string;
    /** The URL we call using the `callback_url` when the Command has finished sending, whether the command was delivered or it failed. */
    callbackUrl?: string;
    /**  */
    commandMode?: CommandCommandMode;
    /** Whether to include the SID of the command in the message body. Can be: `none`, `start`, or `end`, and the default behavior is `none`. When sending a Command to a SIM in text mode, we can automatically include the SID of the Command in the message body, which could be used to ensure that the device does not process the same Command more than once.  A value of `start` will prepend the message with the Command SID, and `end` will append it to the end, separating the Command SID from the message body with a space. The length of the Command SID is included in the 160 character limit so the SMS body must be 128 characters or less before the Command SID is included. */
    includeSid?: string;
    /** Whether to request delivery receipt from the recipient. For Commands that request delivery receipt, the Command state transitions to \\\'delivered\\\' once the server has received a delivery receipt from the device. The default value is `true`. */
    deliveryReceiptRequested?: boolean;
}
/**
 * Options to pass to each
 */
export interface CommandListInstanceEachOptions {
    /** The `sid` or `unique_name` of the [Sim resources](https://www.twilio.com/docs/iot/wireless/api/sim-resource) to read. */
    sim?: string;
    /** The status of the resources to read. Can be: `queued`, `sent`, `delivered`, `received`, or `failed`. */
    status?: CommandStatus;
    /** Only return Commands with this direction value. */
    direction?: CommandDirection;
    /** Only return Commands with this transport value. Can be: `sms` or `ip`. */
    transport?: CommandTransport;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CommandInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CommandListInstanceOptions {
    /** The `sid` or `unique_name` of the [Sim resources](https://www.twilio.com/docs/iot/wireless/api/sim-resource) to read. */
    sim?: string;
    /** The status of the resources to read. Can be: `queued`, `sent`, `delivered`, `received`, or `failed`. */
    status?: CommandStatus;
    /** Only return Commands with this direction value. */
    direction?: CommandDirection;
    /** Only return Commands with this transport value. Can be: `sms` or `ip`. */
    transport?: CommandTransport;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CommandListInstancePageOptions {
    /** The `sid` or `unique_name` of the [Sim resources](https://www.twilio.com/docs/iot/wireless/api/sim-resource) to read. */
    sim?: string;
    /** The status of the resources to read. Can be: `queued`, `sent`, `delivered`, `received`, or `failed`. */
    status?: CommandStatus;
    /** Only return Commands with this direction value. */
    direction?: CommandDirection;
    /** Only return Commands with this transport value. Can be: `sms` or `ip`. */
    transport?: CommandTransport;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CommandContext {
    /**
     * Remove a CommandInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CommandInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CommandInstance
     */
    fetch(callback?: (error: Error | null, item?: CommandInstance) => any): Promise<CommandInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CommandContextSolution {
    sid: string;
}
export declare class CommandContextImpl implements CommandContext {
    protected _version: V1;
    protected _solution: CommandContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: CommandInstance) => any): Promise<CommandInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CommandContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CommandPayload extends TwilioResponsePayload {
    commands: CommandResource[];
}
interface CommandResource {
    sid: string;
    account_sid: string;
    sim_sid: string;
    command: string;
    command_mode: CommandCommandMode;
    transport: CommandTransport;
    delivery_receipt_requested: boolean;
    status: CommandStatus;
    direction: CommandDirection;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class CommandInstance {
    protected _version: V1;
    protected _solution: CommandContextSolution;
    protected _context?: CommandContext;
    constructor(_version: V1, payload: CommandResource, sid?: string);
    /**
     * The unique string that we created to identify the Command resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Command resource.
     */
    accountSid: string;
    /**
     * The SID of the [Sim resource](https://www.twilio.com/docs/iot/wireless/api/sim-resource) that the Command was sent to or from.
     */
    simSid: string;
    /**
     * The message being sent to or from the SIM. For text mode messages, this can be up to 160 characters. For binary mode messages, this is a series of up to 140 bytes of data encoded using base64.
     */
    command: string;
    commandMode: CommandCommandMode;
    transport: CommandTransport;
    /**
     * Whether to request a delivery receipt.
     */
    deliveryReceiptRequested: boolean;
    status: CommandStatus;
    direction: CommandDirection;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a CommandInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CommandInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CommandInstance
     */
    fetch(callback?: (error: Error | null, item?: CommandInstance) => any): Promise<CommandInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        simSid: string;
        command: string;
        commandMode: CommandCommandMode;
        transport: CommandTransport;
        deliveryReceiptRequested: boolean;
        status: CommandStatus;
        direction: CommandDirection;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CommandSolution {
}
export interface CommandListInstance {
    _version: V1;
    _solution: CommandSolution;
    _uri: string;
    (sid: string): CommandContext;
    get(sid: string): CommandContext;
    /**
     * Create a CommandInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CommandInstance
     */
    create(params: CommandListInstanceCreateOptions, callback?: (error: Error | null, item?: CommandInstance) => any): Promise<CommandInstance>;
    /**
     * Streams CommandInstance records from the API.
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
     * @param { CommandListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CommandInstance, done: (err?: Error) => void) => void): void;
    each(params: CommandListInstanceEachOptions, callback?: (item: CommandInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CommandInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CommandPage) => any): Promise<CommandPage>;
    /**
     * Lists CommandInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CommandListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CommandInstance[]) => any): Promise<CommandInstance[]>;
    list(params: CommandListInstanceOptions, callback?: (error: Error | null, items: CommandInstance[]) => any): Promise<CommandInstance[]>;
    /**
     * Retrieve a single page of CommandInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CommandListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CommandPage) => any): Promise<CommandPage>;
    page(params: CommandListInstancePageOptions, callback?: (error: Error | null, items: CommandPage) => any): Promise<CommandPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CommandListInstance(version: V1): CommandListInstance;
export declare class CommandPage extends Page<V1, CommandPayload, CommandResource, CommandInstance> {
    /**
     * Initialize the CommandPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: CommandSolution);
    /**
     * Build an instance of CommandInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CommandResource): CommandInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
