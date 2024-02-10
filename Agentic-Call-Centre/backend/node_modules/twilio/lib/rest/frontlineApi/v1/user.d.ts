/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export type UserStateType = "active" | "deactivated";
/**
 * Options to pass to update a UserInstance
 */
export interface UserContextUpdateOptions {
    /** The string that you assigned to describe the User. */
    friendlyName?: string;
    /** The avatar URL which will be shown in Frontline application. */
    avatar?: string;
    /**  */
    state?: UserStateType;
    /** Whether the User is available for new conversations. Set to `false` to prevent User from receiving new inbound conversations if you are using [Pool Routing](https://www.twilio.com/docs/frontline/handle-incoming-conversations#3-pool-routing). */
    isAvailable?: boolean;
}
export interface UserContext {
    /**
     * Fetch a UserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    fetch(callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Update a UserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    update(callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Update a UserInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    update(params: UserContextUpdateOptions, callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface UserContextSolution {
    sid: string;
}
export declare class UserContextImpl implements UserContext {
    protected _version: V1;
    protected _solution: UserContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    update(params?: UserContextUpdateOptions | ((error: Error | null, item?: UserInstance) => any), callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): UserContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface UserResource {
    sid: string;
    identity: string;
    friendly_name: string;
    avatar: string;
    state: UserStateType;
    is_available: boolean;
    url: string;
}
export declare class UserInstance {
    protected _version: V1;
    protected _solution: UserContextSolution;
    protected _context?: UserContext;
    constructor(_version: V1, payload: UserResource, sid?: string);
    /**
     * The unique string that we created to identify the User resource.
     */
    sid: string;
    /**
     * The application-defined string that uniquely identifies the resource\'s User. This value is often a username or an email address, and is case-sensitive.
     */
    identity: string;
    /**
     * The string that you assigned to describe the User.
     */
    friendlyName: string;
    /**
     * The avatar URL which will be shown in Frontline application.
     */
    avatar: string;
    state: UserStateType;
    /**
     * Whether the User is available for new conversations. Defaults to `false` for new users.
     */
    isAvailable: boolean;
    /**
     * An absolute API resource URL for this user.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a UserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    fetch(callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Update a UserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    update(callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Update a UserInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    update(params: UserContextUpdateOptions, callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        identity: string;
        friendlyName: string;
        avatar: string;
        state: UserStateType;
        isAvailable: boolean;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface UserSolution {
}
export interface UserListInstance {
    _version: V1;
    _solution: UserSolution;
    _uri: string;
    (sid: string): UserContext;
    get(sid: string): UserContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UserListInstance(version: V1): UserListInstance;
export {};
