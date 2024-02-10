/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type EsimProfileStatus = "new" | "reserving" | "available" | "downloaded" | "installed" | "failed";
/**
 * Options to pass to create a EsimProfileInstance
 */
export interface EsimProfileListInstanceCreateOptions {
    /** The URL we should call using the `callback_method` when the status of the eSIM Profile changes. At this stage of the eSIM Profile pilot, the a request to the URL will only be called when the ESimProfile resource changes from `reserving` to `available`. */
    callbackUrl?: string;
    /** The HTTP method we should use to call `callback_url`. Can be: `GET` or `POST` and the default is POST. */
    callbackMethod?: string;
    /** When set to `true`, a value for `Eid` does not need to be provided. Instead, when the eSIM profile is reserved, a matching ID will be generated and returned via the `matching_id` property. This identifies the specific eSIM profile that can be used by any capable device to claim and download the profile. */
    generateMatchingId?: boolean;
    /** Identifier of the eUICC that will claim the eSIM Profile. */
    eid?: string;
}
/**
 * Options to pass to each
 */
export interface EsimProfileListInstanceEachOptions {
    /** List the eSIM Profiles that have been associated with an EId. */
    eid?: string;
    /** Find the eSIM Profile resource related to a [Sim](https://www.twilio.com/docs/iot/supersim/api/sim-resource) resource by providing the SIM SID. Will always return an array with either 1 or 0 records. */
    simSid?: string;
    /** List the eSIM Profiles that are in a given status. */
    status?: EsimProfileStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: EsimProfileInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface EsimProfileListInstanceOptions {
    /** List the eSIM Profiles that have been associated with an EId. */
    eid?: string;
    /** Find the eSIM Profile resource related to a [Sim](https://www.twilio.com/docs/iot/supersim/api/sim-resource) resource by providing the SIM SID. Will always return an array with either 1 or 0 records. */
    simSid?: string;
    /** List the eSIM Profiles that are in a given status. */
    status?: EsimProfileStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface EsimProfileListInstancePageOptions {
    /** List the eSIM Profiles that have been associated with an EId. */
    eid?: string;
    /** Find the eSIM Profile resource related to a [Sim](https://www.twilio.com/docs/iot/supersim/api/sim-resource) resource by providing the SIM SID. Will always return an array with either 1 or 0 records. */
    simSid?: string;
    /** List the eSIM Profiles that are in a given status. */
    status?: EsimProfileStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface EsimProfileContext {
    /**
     * Fetch a EsimProfileInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EsimProfileInstance
     */
    fetch(callback?: (error: Error | null, item?: EsimProfileInstance) => any): Promise<EsimProfileInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface EsimProfileContextSolution {
    sid: string;
}
export declare class EsimProfileContextImpl implements EsimProfileContext {
    protected _version: V1;
    protected _solution: EsimProfileContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: EsimProfileInstance) => any): Promise<EsimProfileInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): EsimProfileContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface EsimProfilePayload extends TwilioResponsePayload {
    esim_profiles: EsimProfileResource[];
}
interface EsimProfileResource {
    sid: string;
    account_sid: string;
    iccid: string;
    sim_sid: string;
    status: EsimProfileStatus;
    eid: string;
    smdp_plus_address: string;
    matching_id: string;
    activation_code: string;
    error_code: string;
    error_message: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class EsimProfileInstance {
    protected _version: V1;
    protected _solution: EsimProfileContextSolution;
    protected _context?: EsimProfileContext;
    constructor(_version: V1, payload: EsimProfileResource, sid?: string);
    /**
     * The unique string that we created to identify the eSIM Profile resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) to which the eSIM Profile resource belongs.
     */
    accountSid: string;
    /**
     * The [ICCID](https://en.wikipedia.org/wiki/Subscriber_identity_module#ICCID) associated with the Sim resource.
     */
    iccid: string;
    /**
     * The SID of the [Sim](https://www.twilio.com/docs/iot/supersim/api/sim-resource) resource that this eSIM Profile controls.
     */
    simSid: string;
    status: EsimProfileStatus;
    /**
     * Identifier of the eUICC that can claim the eSIM Profile.
     */
    eid: string;
    /**
     * Address of the SM-DP+ server from which the Profile will be downloaded. The URL will appear once the eSIM Profile reaches the status `available`.
     */
    smdpPlusAddress: string;
    /**
     * Unique identifier of the eSIM profile that can be used to identify and download the eSIM profile from the SM-DP+ server. Populated if `generate_matching_id` is set to `true` when creating the eSIM profile reservation.
     */
    matchingId: string;
    /**
     * Combined machine-readable activation code for acquiring an eSIM Profile with the Activation Code download method. Can be used in a QR code to download an eSIM profile.
     */
    activationCode: string;
    /**
     * Code indicating the failure if the download of the SIM Profile failed and the eSIM Profile is in `failed` state.
     */
    errorCode: string;
    /**
     * Error message describing the failure if the download of the SIM Profile failed and the eSIM Profile is in `failed` state.
     */
    errorMessage: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the eSIM Profile resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a EsimProfileInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EsimProfileInstance
     */
    fetch(callback?: (error: Error | null, item?: EsimProfileInstance) => any): Promise<EsimProfileInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        iccid: string;
        simSid: string;
        status: EsimProfileStatus;
        eid: string;
        smdpPlusAddress: string;
        matchingId: string;
        activationCode: string;
        errorCode: string;
        errorMessage: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface EsimProfileSolution {
}
export interface EsimProfileListInstance {
    _version: V1;
    _solution: EsimProfileSolution;
    _uri: string;
    (sid: string): EsimProfileContext;
    get(sid: string): EsimProfileContext;
    /**
     * Create a EsimProfileInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EsimProfileInstance
     */
    create(callback?: (error: Error | null, item?: EsimProfileInstance) => any): Promise<EsimProfileInstance>;
    /**
     * Create a EsimProfileInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EsimProfileInstance
     */
    create(params: EsimProfileListInstanceCreateOptions, callback?: (error: Error | null, item?: EsimProfileInstance) => any): Promise<EsimProfileInstance>;
    /**
     * Streams EsimProfileInstance records from the API.
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
     * @param { EsimProfileListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: EsimProfileInstance, done: (err?: Error) => void) => void): void;
    each(params: EsimProfileListInstanceEachOptions, callback?: (item: EsimProfileInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of EsimProfileInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: EsimProfilePage) => any): Promise<EsimProfilePage>;
    /**
     * Lists EsimProfileInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EsimProfileListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: EsimProfileInstance[]) => any): Promise<EsimProfileInstance[]>;
    list(params: EsimProfileListInstanceOptions, callback?: (error: Error | null, items: EsimProfileInstance[]) => any): Promise<EsimProfileInstance[]>;
    /**
     * Retrieve a single page of EsimProfileInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EsimProfileListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: EsimProfilePage) => any): Promise<EsimProfilePage>;
    page(params: EsimProfileListInstancePageOptions, callback?: (error: Error | null, items: EsimProfilePage) => any): Promise<EsimProfilePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function EsimProfileListInstance(version: V1): EsimProfileListInstance;
export declare class EsimProfilePage extends Page<V1, EsimProfilePayload, EsimProfileResource, EsimProfileInstance> {
    /**
     * Initialize the EsimProfilePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: EsimProfileSolution);
    /**
     * Build an instance of EsimProfileInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: EsimProfileResource): EsimProfileInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
