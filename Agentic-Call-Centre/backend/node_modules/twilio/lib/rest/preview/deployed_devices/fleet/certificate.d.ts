/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import DeployedDevices from "../../DeployedDevices";
/**
 * Options to pass to update a CertificateInstance
 */
export interface CertificateContextUpdateOptions {
    /** Provides a human readable descriptive text for this Certificate credential, up to 256 characters long. */
    friendlyName?: string;
    /** Provides the unique string identifier of an existing Device to become authenticated with this Certificate credential. */
    deviceSid?: string;
}
/**
 * Options to pass to create a CertificateInstance
 */
export interface CertificateListInstanceCreateOptions {
    /** Provides a URL encoded representation of the public certificate in PEM format. */
    certificateData: string;
    /** Provides a human readable descriptive text for this Certificate credential, up to 256 characters long. */
    friendlyName?: string;
    /** Provides the unique string identifier of an existing Device to become authenticated with this Certificate credential. */
    deviceSid?: string;
}
/**
 * Options to pass to each
 */
export interface CertificateListInstanceEachOptions {
    /** Filters the resulting list of Certificates by a unique string identifier of an authenticated Device. */
    deviceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CertificateInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CertificateListInstanceOptions {
    /** Filters the resulting list of Certificates by a unique string identifier of an authenticated Device. */
    deviceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CertificateListInstancePageOptions {
    /** Filters the resulting list of Certificates by a unique string identifier of an authenticated Device. */
    deviceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CertificateContext {
    /**
     * Remove a CertificateInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CertificateInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CertificateInstance
     */
    fetch(callback?: (error: Error | null, item?: CertificateInstance) => any): Promise<CertificateInstance>;
    /**
     * Update a CertificateInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CertificateInstance
     */
    update(callback?: (error: Error | null, item?: CertificateInstance) => any): Promise<CertificateInstance>;
    /**
     * Update a CertificateInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CertificateInstance
     */
    update(params: CertificateContextUpdateOptions, callback?: (error: Error | null, item?: CertificateInstance) => any): Promise<CertificateInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CertificateContextSolution {
    fleetSid: string;
    sid: string;
}
export declare class CertificateContextImpl implements CertificateContext {
    protected _version: DeployedDevices;
    protected _solution: CertificateContextSolution;
    protected _uri: string;
    constructor(_version: DeployedDevices, fleetSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: CertificateInstance) => any): Promise<CertificateInstance>;
    update(params?: CertificateContextUpdateOptions | ((error: Error | null, item?: CertificateInstance) => any), callback?: (error: Error | null, item?: CertificateInstance) => any): Promise<CertificateInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CertificateContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CertificatePayload extends TwilioResponsePayload {
    certificates: CertificateResource[];
}
interface CertificateResource {
    sid: string;
    url: string;
    friendly_name: string;
    fleet_sid: string;
    account_sid: string;
    device_sid: string;
    thumbprint: string;
    date_created: Date;
    date_updated: Date;
}
export declare class CertificateInstance {
    protected _version: DeployedDevices;
    protected _solution: CertificateContextSolution;
    protected _context?: CertificateContext;
    constructor(_version: DeployedDevices, payload: CertificateResource, fleetSid: string, sid?: string);
    /**
     * Contains a 34 character string that uniquely identifies this Certificate credential resource.
     */
    sid: string;
    /**
     * Contains an absolute URL for this Certificate credential resource.
     */
    url: string;
    /**
     * Contains a human readable descriptive text for this Certificate credential, up to 256 characters long.
     */
    friendlyName: string;
    /**
     * Specifies the unique string identifier of the Fleet that the given Certificate credential belongs to.
     */
    fleetSid: string;
    /**
     * Specifies the unique string identifier of the Account responsible for this Certificate credential.
     */
    accountSid: string;
    /**
     * Specifies the unique string identifier of a Device authenticated with this Certificate credential.
     */
    deviceSid: string;
    /**
     * Contains a unique hash of the payload of this Certificate credential, used to authenticate the Device.
     */
    thumbprint: string;
    /**
     * Specifies the date this Certificate credential was created, given in UTC ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * Specifies the date this Certificate credential was last updated, given in UTC ISO 8601 format.
     */
    dateUpdated: Date;
    private get _proxy();
    /**
     * Remove a CertificateInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CertificateInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CertificateInstance
     */
    fetch(callback?: (error: Error | null, item?: CertificateInstance) => any): Promise<CertificateInstance>;
    /**
     * Update a CertificateInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CertificateInstance
     */
    update(callback?: (error: Error | null, item?: CertificateInstance) => any): Promise<CertificateInstance>;
    /**
     * Update a CertificateInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CertificateInstance
     */
    update(params: CertificateContextUpdateOptions, callback?: (error: Error | null, item?: CertificateInstance) => any): Promise<CertificateInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        url: string;
        friendlyName: string;
        fleetSid: string;
        accountSid: string;
        deviceSid: string;
        thumbprint: string;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CertificateSolution {
    fleetSid: string;
}
export interface CertificateListInstance {
    _version: DeployedDevices;
    _solution: CertificateSolution;
    _uri: string;
    (sid: string): CertificateContext;
    get(sid: string): CertificateContext;
    /**
     * Create a CertificateInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CertificateInstance
     */
    create(params: CertificateListInstanceCreateOptions, callback?: (error: Error | null, item?: CertificateInstance) => any): Promise<CertificateInstance>;
    /**
     * Streams CertificateInstance records from the API.
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
     * @param { CertificateListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CertificateInstance, done: (err?: Error) => void) => void): void;
    each(params: CertificateListInstanceEachOptions, callback?: (item: CertificateInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CertificateInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CertificatePage) => any): Promise<CertificatePage>;
    /**
     * Lists CertificateInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CertificateListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CertificateInstance[]) => any): Promise<CertificateInstance[]>;
    list(params: CertificateListInstanceOptions, callback?: (error: Error | null, items: CertificateInstance[]) => any): Promise<CertificateInstance[]>;
    /**
     * Retrieve a single page of CertificateInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CertificateListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CertificatePage) => any): Promise<CertificatePage>;
    page(params: CertificateListInstancePageOptions, callback?: (error: Error | null, items: CertificatePage) => any): Promise<CertificatePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CertificateListInstance(version: DeployedDevices, fleetSid: string): CertificateListInstance;
export declare class CertificatePage extends Page<DeployedDevices, CertificatePayload, CertificateResource, CertificateInstance> {
    /**
     * Initialize the CertificatePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: DeployedDevices, response: Response<string>, solution: CertificateSolution);
    /**
     * Build an instance of CertificateInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CertificateResource): CertificateInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
