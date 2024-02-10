/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import Wireless from "../Wireless";
/**
 * Options to pass to create a CommandInstance
 */
export interface CommandListInstanceCreateOptions {
    /**  */
    command: string;
    /**  */
    device?: string;
    /**  */
    sim?: string;
    /**  */
    callbackMethod?: string;
    /**  */
    callbackUrl?: string;
    /**  */
    commandMode?: string;
    /**  */
    includeSid?: string;
}
/**
 * Options to pass to each
 */
export interface CommandListInstanceEachOptions {
    /**  */
    device?: string;
    /**  */
    sim?: string;
    /**  */
    status?: string;
    /**  */
    direction?: string;
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
    /**  */
    device?: string;
    /**  */
    sim?: string;
    /**  */
    status?: string;
    /**  */
    direction?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CommandListInstancePageOptions {
    /**  */
    device?: string;
    /**  */
    sim?: string;
    /**  */
    status?: string;
    /**  */
    direction?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CommandContext {
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
    protected _version: Wireless;
    protected _solution: CommandContextSolution;
    protected _uri: string;
    constructor(_version: Wireless, sid: string);
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
    device_sid: string;
    sim_sid: string;
    command: string;
    command_mode: string;
    status: string;
    direction: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class CommandInstance {
    protected _version: Wireless;
    protected _solution: CommandContextSolution;
    protected _context?: CommandContext;
    constructor(_version: Wireless, payload: CommandResource, sid?: string);
    sid: string;
    accountSid: string;
    deviceSid: string;
    simSid: string;
    command: string;
    commandMode: string;
    status: string;
    direction: string;
    dateCreated: Date;
    dateUpdated: Date;
    url: string;
    private get _proxy();
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
        deviceSid: string;
        simSid: string;
        command: string;
        commandMode: string;
        status: string;
        direction: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CommandSolution {
}
export interface CommandListInstance {
    _version: Wireless;
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
export declare function CommandListInstance(version: Wireless): CommandListInstance;
export declare class CommandPage extends Page<Wireless, CommandPayload, CommandResource, CommandInstance> {
    /**
     * Initialize the CommandPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Wireless, response: Response<string>, solution: CommandSolution);
    /**
     * Build an instance of CommandInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CommandResource): CommandInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
