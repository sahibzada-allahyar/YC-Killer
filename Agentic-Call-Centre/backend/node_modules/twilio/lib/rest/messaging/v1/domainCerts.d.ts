/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to update a DomainCertsInstance
 */
export interface DomainCertsContextUpdateOptions {
    /** Contains the full TLS certificate and private for this domain in PEM format: https://en.wikipedia.org/wiki/Privacy-Enhanced_Mail. Twilio uses this information to process HTTPS traffic sent to your domain. */
    tlsCert: string;
}
export interface DomainCertsContext {
    /**
     * Remove a DomainCertsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DomainCertsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainCertsInstance
     */
    fetch(callback?: (error: Error | null, item?: DomainCertsInstance) => any): Promise<DomainCertsInstance>;
    /**
     * Update a DomainCertsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainCertsInstance
     */
    update(params: DomainCertsContextUpdateOptions, callback?: (error: Error | null, item?: DomainCertsInstance) => any): Promise<DomainCertsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DomainCertsContextSolution {
    domainSid: string;
}
export declare class DomainCertsContextImpl implements DomainCertsContext {
    protected _version: V1;
    protected _solution: DomainCertsContextSolution;
    protected _uri: string;
    constructor(_version: V1, domainSid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: DomainCertsInstance) => any): Promise<DomainCertsInstance>;
    update(params: DomainCertsContextUpdateOptions, callback?: (error: Error | null, item?: DomainCertsInstance) => any): Promise<DomainCertsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DomainCertsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DomainCertsResource {
    domain_sid: string;
    date_updated: Date;
    date_expires: Date;
    date_created: Date;
    domain_name: string;
    certificate_sid: string;
    url: string;
    cert_in_validation: any;
}
export declare class DomainCertsInstance {
    protected _version: V1;
    protected _solution: DomainCertsContextSolution;
    protected _context?: DomainCertsContext;
    constructor(_version: V1, payload: DomainCertsResource, domainSid?: string);
    /**
     * The unique string that we created to identify the Domain resource.
     */
    domainSid: string;
    /**
     * Date that this Domain was last updated.
     */
    dateUpdated: Date;
    /**
     * Date that the private certificate associated with this domain expires. You will need to update the certificate before that date to ensure your shortened links will continue to work.
     */
    dateExpires: Date;
    /**
     * Date that this Domain was registered to the Twilio platform to create a new Domain object.
     */
    dateCreated: Date;
    /**
     * Full url path for this domain.
     */
    domainName: string;
    /**
     * The unique string that we created to identify this Certificate resource.
     */
    certificateSid: string;
    url: string;
    /**
     * Optional JSON field describing the status and upload date of a new certificate in the process of validation
     */
    certInValidation: any;
    private get _proxy();
    /**
     * Remove a DomainCertsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DomainCertsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainCertsInstance
     */
    fetch(callback?: (error: Error | null, item?: DomainCertsInstance) => any): Promise<DomainCertsInstance>;
    /**
     * Update a DomainCertsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DomainCertsInstance
     */
    update(params: DomainCertsContextUpdateOptions, callback?: (error: Error | null, item?: DomainCertsInstance) => any): Promise<DomainCertsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        domainSid: string;
        dateUpdated: Date;
        dateExpires: Date;
        dateCreated: Date;
        domainName: string;
        certificateSid: string;
        url: string;
        certInValidation: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DomainCertsSolution {
}
export interface DomainCertsListInstance {
    _version: V1;
    _solution: DomainCertsSolution;
    _uri: string;
    (domainSid: string): DomainCertsContext;
    get(domainSid: string): DomainCertsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DomainCertsListInstance(version: V1): DomainCertsListInstance;
export {};
