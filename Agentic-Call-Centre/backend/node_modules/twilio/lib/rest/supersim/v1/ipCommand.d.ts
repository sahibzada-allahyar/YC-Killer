/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type IpCommandDirection = "to_sim" | "from_sim";
export type IpCommandPayloadType = "text" | "binary";
export type IpCommandStatus = "queued" | "sent" | "received" | "failed";
/**
 * Options to pass to create a IpCommandInstance
 */
export interface IpCommandListInstanceCreateOptions {
    /** The `sid` or `unique_name` of the [Super SIM](https://www.twilio.com/docs/iot/supersim/api/sim-resource) to send the IP Command to. */
    sim: string;
    /** The data that will be sent to the device. The payload cannot exceed 1300 bytes. If the PayloadType is set to text, the payload is encoded in UTF-8. If PayloadType is set to binary, the payload is encoded in Base64. */
    payload: string;
    /** The device port to which the IP Command will be sent. */
    devicePort: number;
    /**  */
    payloadType?: IpCommandPayloadType;
    /** The URL we should call using the `callback_method` after we have sent the IP Command. */
    callbackUrl?: string;
    /** The HTTP method we should use to call `callback_url`. Can be `GET` or `POST`, and the default is `POST`. */
    callbackMethod?: string;
}
/**
 * Options to pass to each
 */
export interface IpCommandListInstanceEachOptions {
    /** The SID or unique name of the Sim resource that IP Command was sent to or from. */
    sim?: string;
    /** The ICCID of the Sim resource that IP Command was sent to or from. */
    simIccid?: string;
    /** The status of the IP Command. Can be: `queued`, `sent`, `received` or `failed`. See the [IP Command Status Values](https://www.twilio.com/docs/iot/supersim/api/ipcommand-resource#status-values) for a description of each. */
    status?: IpCommandStatus;
    /** The direction of the IP Command. Can be `to_sim` or `from_sim`. The value of `to_sim` is synonymous with the term `mobile terminated`, and `from_sim` is synonymous with the term `mobile originated`. */
    direction?: IpCommandDirection;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: IpCommandInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface IpCommandListInstanceOptions {
    /** The SID or unique name of the Sim resource that IP Command was sent to or from. */
    sim?: string;
    /** The ICCID of the Sim resource that IP Command was sent to or from. */
    simIccid?: string;
    /** The status of the IP Command. Can be: `queued`, `sent`, `received` or `failed`. See the [IP Command Status Values](https://www.twilio.com/docs/iot/supersim/api/ipcommand-resource#status-values) for a description of each. */
    status?: IpCommandStatus;
    /** The direction of the IP Command. Can be `to_sim` or `from_sim`. The value of `to_sim` is synonymous with the term `mobile terminated`, and `from_sim` is synonymous with the term `mobile originated`. */
    direction?: IpCommandDirection;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface IpCommandListInstancePageOptions {
    /** The SID or unique name of the Sim resource that IP Command was sent to or from. */
    sim?: string;
    /** The ICCID of the Sim resource that IP Command was sent to or from. */
    simIccid?: string;
    /** The status of the IP Command. Can be: `queued`, `sent`, `received` or `failed`. See the [IP Command Status Values](https://www.twilio.com/docs/iot/supersim/api/ipcommand-resource#status-values) for a description of each. */
    status?: IpCommandStatus;
    /** The direction of the IP Command. Can be `to_sim` or `from_sim`. The value of `to_sim` is synonymous with the term `mobile terminated`, and `from_sim` is synonymous with the term `mobile originated`. */
    direction?: IpCommandDirection;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface IpCommandContext {
    /**
     * Fetch a IpCommandInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpCommandInstance
     */
    fetch(callback?: (error: Error | null, item?: IpCommandInstance) => any): Promise<IpCommandInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface IpCommandContextSolution {
    sid: string;
}
export declare class IpCommandContextImpl implements IpCommandContext {
    protected _version: V1;
    protected _solution: IpCommandContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: IpCommandInstance) => any): Promise<IpCommandInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): IpCommandContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface IpCommandPayload extends TwilioResponsePayload {
    ip_commands: IpCommandResource[];
}
interface IpCommandResource {
    sid: string;
    account_sid: string;
    sim_sid: string;
    sim_iccid: string;
    status: IpCommandStatus;
    direction: IpCommandDirection;
    device_ip: string;
    device_port: number;
    payload_type: IpCommandPayloadType;
    payload: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class IpCommandInstance {
    protected _version: V1;
    protected _solution: IpCommandContextSolution;
    protected _context?: IpCommandContext;
    constructor(_version: V1, payload: IpCommandResource, sid?: string);
    /**
     * The unique string that we created to identify the IP Command resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the IP Command resource.
     */
    accountSid: string;
    /**
     * The SID of the [Super SIM](https://www.twilio.com/docs/iot/supersim/api/sim-resource) that this IP Command was sent to or from.
     */
    simSid: string;
    /**
     * The [ICCID](https://en.wikipedia.org/wiki/Subscriber_identity_module#ICCID) of the [Super SIM](https://www.twilio.com/docs/iot/supersim/api/sim-resource) that this IP Command was sent to or from.
     */
    simIccid: string;
    status: IpCommandStatus;
    direction: IpCommandDirection;
    /**
     * The IP address of the device that the IP Command was sent to or received from. For an IP Command sent to a Super SIM, `device_ip` starts out as `null`, and once the IP Command is “sent”, the `device_ip` will be filled out. An IP Command sent from a Super SIM have its `device_ip` always set.
     */
    deviceIp: string;
    /**
     * For an IP Command sent to a Super SIM, it would be the destination port of the IP message. For an IP Command sent from a Super SIM, it would be the source port of the IP message.
     */
    devicePort: number;
    payloadType: IpCommandPayloadType;
    /**
     * The payload that is carried in the IP/UDP message. The payload can be encoded in either text or binary format. For text payload, UTF-8 encoding must be used.  For an IP Command sent to a Super SIM, the payload is appended to the IP/UDP message “as is”. The payload should not exceed 1300 bytes.  For an IP Command sent from a Super SIM, the payload from the received IP/UDP message is extracted and sent in binary encoding. For an IP Command sent from a Super SIM, the payload should not exceed 1300 bytes. If it is larger than 1300 bytes, there might be fragmentation on the upstream and the message may appear truncated.
     */
    payload: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the IP Command resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a IpCommandInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpCommandInstance
     */
    fetch(callback?: (error: Error | null, item?: IpCommandInstance) => any): Promise<IpCommandInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        simSid: string;
        simIccid: string;
        status: IpCommandStatus;
        direction: IpCommandDirection;
        deviceIp: string;
        devicePort: number;
        payloadType: IpCommandPayloadType;
        payload: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface IpCommandSolution {
}
export interface IpCommandListInstance {
    _version: V1;
    _solution: IpCommandSolution;
    _uri: string;
    (sid: string): IpCommandContext;
    get(sid: string): IpCommandContext;
    /**
     * Create a IpCommandInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpCommandInstance
     */
    create(params: IpCommandListInstanceCreateOptions, callback?: (error: Error | null, item?: IpCommandInstance) => any): Promise<IpCommandInstance>;
    /**
     * Streams IpCommandInstance records from the API.
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
     * @param { IpCommandListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: IpCommandInstance, done: (err?: Error) => void) => void): void;
    each(params: IpCommandListInstanceEachOptions, callback?: (item: IpCommandInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of IpCommandInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: IpCommandPage) => any): Promise<IpCommandPage>;
    /**
     * Lists IpCommandInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { IpCommandListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: IpCommandInstance[]) => any): Promise<IpCommandInstance[]>;
    list(params: IpCommandListInstanceOptions, callback?: (error: Error | null, items: IpCommandInstance[]) => any): Promise<IpCommandInstance[]>;
    /**
     * Retrieve a single page of IpCommandInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { IpCommandListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: IpCommandPage) => any): Promise<IpCommandPage>;
    page(params: IpCommandListInstancePageOptions, callback?: (error: Error | null, items: IpCommandPage) => any): Promise<IpCommandPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function IpCommandListInstance(version: V1): IpCommandListInstance;
export declare class IpCommandPage extends Page<V1, IpCommandPayload, IpCommandResource, IpCommandInstance> {
    /**
     * Initialize the IpCommandPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: IpCommandSolution);
    /**
     * Build an instance of IpCommandInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: IpCommandResource): IpCommandInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
