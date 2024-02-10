/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to update a ExportConfigurationInstance
 */
export interface ExportConfigurationContextUpdateOptions {
    /** If true, Twilio will automatically generate every day\\\'s file when the day is over. */
    enabled?: boolean;
    /** Stores the URL destination for the method specified in webhook_method. */
    webhookUrl?: string;
    /** Sets whether Twilio should call a webhook URL when the automatic generation is complete, using GET or POST. The actual destination is set in the webhook_url */
    webhookMethod?: string;
}
export interface ExportConfigurationContext {
    /**
     * Fetch a ExportConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExportConfigurationInstance
     */
    fetch(callback?: (error: Error | null, item?: ExportConfigurationInstance) => any): Promise<ExportConfigurationInstance>;
    /**
     * Update a ExportConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExportConfigurationInstance
     */
    update(callback?: (error: Error | null, item?: ExportConfigurationInstance) => any): Promise<ExportConfigurationInstance>;
    /**
     * Update a ExportConfigurationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExportConfigurationInstance
     */
    update(params: ExportConfigurationContextUpdateOptions, callback?: (error: Error | null, item?: ExportConfigurationInstance) => any): Promise<ExportConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ExportConfigurationContextSolution {
    resourceType: string;
}
export declare class ExportConfigurationContextImpl implements ExportConfigurationContext {
    protected _version: V1;
    protected _solution: ExportConfigurationContextSolution;
    protected _uri: string;
    constructor(_version: V1, resourceType: string);
    fetch(callback?: (error: Error | null, item?: ExportConfigurationInstance) => any): Promise<ExportConfigurationInstance>;
    update(params?: ExportConfigurationContextUpdateOptions | ((error: Error | null, item?: ExportConfigurationInstance) => any), callback?: (error: Error | null, item?: ExportConfigurationInstance) => any): Promise<ExportConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ExportConfigurationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ExportConfigurationResource {
    enabled: boolean;
    webhook_url: string;
    webhook_method: string;
    resource_type: string;
    url: string;
}
export declare class ExportConfigurationInstance {
    protected _version: V1;
    protected _solution: ExportConfigurationContextSolution;
    protected _context?: ExportConfigurationContext;
    constructor(_version: V1, payload: ExportConfigurationResource, resourceType?: string);
    /**
     * If true, Twilio will automatically generate every day\'s file when the day is over.
     */
    enabled: boolean;
    /**
     * Stores the URL destination for the method specified in webhook_method.
     */
    webhookUrl: string;
    /**
     * Sets whether Twilio should call a webhook URL when the automatic generation is complete, using GET or POST. The actual destination is set in the webhook_url
     */
    webhookMethod: string;
    /**
     * The type of communication – Messages, Calls, Conferences, and Participants
     */
    resourceType: string;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a ExportConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExportConfigurationInstance
     */
    fetch(callback?: (error: Error | null, item?: ExportConfigurationInstance) => any): Promise<ExportConfigurationInstance>;
    /**
     * Update a ExportConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExportConfigurationInstance
     */
    update(callback?: (error: Error | null, item?: ExportConfigurationInstance) => any): Promise<ExportConfigurationInstance>;
    /**
     * Update a ExportConfigurationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExportConfigurationInstance
     */
    update(params: ExportConfigurationContextUpdateOptions, callback?: (error: Error | null, item?: ExportConfigurationInstance) => any): Promise<ExportConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        enabled: boolean;
        webhookUrl: string;
        webhookMethod: string;
        resourceType: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ExportConfigurationSolution {
}
export interface ExportConfigurationListInstance {
    _version: V1;
    _solution: ExportConfigurationSolution;
    _uri: string;
    (resourceType: string): ExportConfigurationContext;
    get(resourceType: string): ExportConfigurationContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ExportConfigurationListInstance(version: V1): ExportConfigurationListInstance;
export {};
