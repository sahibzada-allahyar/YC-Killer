/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
export interface AppManifestContext {
    /**
     * Fetch a AppManifestInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AppManifestInstance
     */
    fetch(callback?: (error: Error | null, item?: AppManifestInstance) => any): Promise<AppManifestInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AppManifestContextSolution {
    appSid: string;
}
export declare class AppManifestContextImpl implements AppManifestContext {
    protected _version: V1;
    protected _solution: AppManifestContextSolution;
    protected _uri: string;
    constructor(_version: V1, appSid: string);
    fetch(callback?: (error: Error | null, item?: AppManifestInstance) => any): Promise<AppManifestInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AppManifestContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AppManifestResource {
    app_sid: string;
    hash: string;
    encoded_bytes: string;
    url: string;
}
export declare class AppManifestInstance {
    protected _version: V1;
    protected _solution: AppManifestContextSolution;
    protected _context?: AppManifestContext;
    constructor(_version: V1, payload: AppManifestResource, appSid: string);
    /**
     * A 34-character string that uniquely identifies this App.
     */
    appSid: string;
    /**
     * App manifest hash represented as `hash_algorithm:hash_value`.
     */
    hash: string;
    /**
     * The base-64 encoded manifest
     */
    encodedBytes: string;
    /**
     * The absolute URL of this Manifest.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a AppManifestInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AppManifestInstance
     */
    fetch(callback?: (error: Error | null, item?: AppManifestInstance) => any): Promise<AppManifestInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        appSid: string;
        hash: string;
        encodedBytes: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AppManifestSolution {
    appSid: string;
}
export interface AppManifestListInstance {
    _version: V1;
    _solution: AppManifestSolution;
    _uri: string;
    (): AppManifestContext;
    get(): AppManifestContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AppManifestListInstance(version: V1, appSid: string): AppManifestListInstance;
export {};
