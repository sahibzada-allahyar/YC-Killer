/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
export interface JobContext {
    /**
     * Remove a JobInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a JobInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed JobInstance
     */
    fetch(callback?: (error: Error | null, item?: JobInstance) => any): Promise<JobInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface JobContextSolution {
    jobSid: string;
}
export declare class JobContextImpl implements JobContext {
    protected _version: V1;
    protected _solution: JobContextSolution;
    protected _uri: string;
    constructor(_version: V1, jobSid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: JobInstance) => any): Promise<JobInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): JobContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface JobResource {
    resource_type: string;
    friendly_name: string;
    details: any;
    start_day: string;
    end_day: string;
    job_sid: string;
    webhook_url: string;
    webhook_method: string;
    email: string;
    url: string;
    job_queue_position: string;
    estimated_completion_time: string;
}
export declare class JobInstance {
    protected _version: V1;
    protected _solution: JobContextSolution;
    protected _context?: JobContext;
    constructor(_version: V1, payload: JobResource, jobSid?: string);
    /**
     * The type of communication – Messages, Calls, Conferences, and Participants
     */
    resourceType: string;
    /**
     * The friendly name specified when creating the job
     */
    friendlyName: string;
    /**
     * The details of a job which is an object that contains an array of status grouped by `status` state.  Each `status` object has a `status` string, a count which is the number of days in that `status`, and list of days in that `status`. The day strings are in the format yyyy-MM-dd. As an example, a currently running job may have a status object for COMPLETED and a `status` object for SUBMITTED each with its own count and list of days.
     */
    details: any;
    /**
     * The start time for the export specified when creating the job
     */
    startDay: string;
    /**
     * The end time for the export specified when creating the job
     */
    endDay: string;
    /**
     * The job_sid returned when the export was created
     */
    jobSid: string;
    /**
     * The optional webhook url called on completion
     */
    webhookUrl: string;
    /**
     * This is the method used to call the webhook
     */
    webhookMethod: string;
    /**
     * The optional email to send the completion notification to
     */
    email: string;
    url: string;
    /**
     * This is the job position from the 1st in line. Your queue position will never increase. As jobs ahead of yours in the queue are processed, the queue position number will decrease
     */
    jobQueuePosition: string;
    /**
     * this is the time estimated until your job is complete. This is calculated each time you request the job list. The time is calculated based on the current rate of job completion (which may vary) and your job queue position
     */
    estimatedCompletionTime: string;
    private get _proxy();
    /**
     * Remove a JobInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a JobInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed JobInstance
     */
    fetch(callback?: (error: Error | null, item?: JobInstance) => any): Promise<JobInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        resourceType: string;
        friendlyName: string;
        details: any;
        startDay: string;
        endDay: string;
        jobSid: string;
        webhookUrl: string;
        webhookMethod: string;
        email: string;
        url: string;
        jobQueuePosition: string;
        estimatedCompletionTime: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface JobSolution {
}
export interface JobListInstance {
    _version: V1;
    _solution: JobSolution;
    _uri: string;
    (jobSid: string): JobContext;
    get(jobSid: string): JobContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function JobListInstance(version: V1): JobListInstance;
export {};
