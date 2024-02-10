/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
import { DayListInstance } from "./export/day";
import { ExportCustomJobListInstance } from "./export/exportCustomJob";
import { JobListInstance } from "./export/job";
export interface ExportContext {
    days: DayListInstance;
    exportCustomJobs: ExportCustomJobListInstance;
    /**
     * Fetch a ExportInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExportInstance
     */
    fetch(callback?: (error: Error | null, item?: ExportInstance) => any): Promise<ExportInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ExportContextSolution {
    resourceType: string;
}
export declare class ExportContextImpl implements ExportContext {
    protected _version: V1;
    protected _solution: ExportContextSolution;
    protected _uri: string;
    protected _days?: DayListInstance;
    protected _exportCustomJobs?: ExportCustomJobListInstance;
    constructor(_version: V1, resourceType: string);
    get days(): DayListInstance;
    get exportCustomJobs(): ExportCustomJobListInstance;
    fetch(callback?: (error: Error | null, item?: ExportInstance) => any): Promise<ExportInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ExportContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ExportResource {
    resource_type: string;
    url: string;
    links: Record<string, string>;
}
export declare class ExportInstance {
    protected _version: V1;
    protected _solution: ExportContextSolution;
    protected _context?: ExportContext;
    constructor(_version: V1, payload: ExportResource, resourceType?: string);
    /**
     * The type of communication – Messages, Calls, Conferences, and Participants
     */
    resourceType: string;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * Contains a dictionary of URL links to nested resources of this Export.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a ExportInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExportInstance
     */
    fetch(callback?: (error: Error | null, item?: ExportInstance) => any): Promise<ExportInstance>;
    /**
     * Access the days.
     */
    days(): DayListInstance;
    /**
     * Access the exportCustomJobs.
     */
    exportCustomJobs(): ExportCustomJobListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        resourceType: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ExportSolution {
}
export interface ExportListInstance {
    _version: V1;
    _solution: ExportSolution;
    _uri: string;
    (resourceType: string): ExportContext;
    get(resourceType: string): ExportContext;
    _jobs?: JobListInstance;
    jobs: JobListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ExportListInstance(version: V1): ExportListInstance;
export {};
