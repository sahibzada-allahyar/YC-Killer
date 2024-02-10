/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
import { SchemaVersionListInstance } from "./schema/schemaVersion";
export interface SchemaContext {
    versions: SchemaVersionListInstance;
    /**
     * Fetch a SchemaInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SchemaInstance
     */
    fetch(callback?: (error: Error | null, item?: SchemaInstance) => any): Promise<SchemaInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SchemaContextSolution {
    id: string;
}
export declare class SchemaContextImpl implements SchemaContext {
    protected _version: V1;
    protected _solution: SchemaContextSolution;
    protected _uri: string;
    protected _versions?: SchemaVersionListInstance;
    constructor(_version: V1, id: string);
    get versions(): SchemaVersionListInstance;
    fetch(callback?: (error: Error | null, item?: SchemaInstance) => any): Promise<SchemaInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SchemaContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SchemaResource {
    id: string;
    url: string;
    links: Record<string, string>;
    latest_version_date_created: Date;
    latest_version: number;
}
export declare class SchemaInstance {
    protected _version: V1;
    protected _solution: SchemaContextSolution;
    protected _context?: SchemaContext;
    constructor(_version: V1, payload: SchemaResource, id?: string);
    /**
     * The unique identifier of the schema. Each schema can have multiple versions, that share the same id.
     */
    id: string;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * Contains a dictionary of URL links to nested resources of this schema.
     */
    links: Record<string, string>;
    /**
     * The date that the latest schema version was created, given in ISO 8601 format.
     */
    latestVersionDateCreated: Date;
    /**
     * The latest version published of this schema.
     */
    latestVersion: number;
    private get _proxy();
    /**
     * Fetch a SchemaInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SchemaInstance
     */
    fetch(callback?: (error: Error | null, item?: SchemaInstance) => any): Promise<SchemaInstance>;
    /**
     * Access the versions.
     */
    versions(): SchemaVersionListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        id: string;
        url: string;
        links: Record<string, string>;
        latestVersionDateCreated: Date;
        latestVersion: number;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SchemaSolution {
}
export interface SchemaListInstance {
    _version: V1;
    _solution: SchemaSolution;
    _uri: string;
    (id: string): SchemaContext;
    get(id: string): SchemaContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SchemaListInstance(version: V1): SchemaListInstance;
export {};
