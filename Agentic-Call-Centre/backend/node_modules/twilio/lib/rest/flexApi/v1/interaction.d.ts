/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
import { InteractionChannelListInstance } from "./interaction/interactionChannel";
/**
 * Options to pass to create a InteractionInstance
 */
export interface InteractionListInstanceCreateOptions {
    /** The Interaction\\\'s channel. */
    channel: any;
    /** The Interaction\\\'s routing logic. */
    routing: any;
    /** The Interaction context sid is used for adding a context lookup sid */
    interactionContextSid?: string;
}
export interface InteractionContext {
    channels: InteractionChannelListInstance;
    /**
     * Fetch a InteractionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionInstance
     */
    fetch(callback?: (error: Error | null, item?: InteractionInstance) => any): Promise<InteractionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface InteractionContextSolution {
    sid: string;
}
export declare class InteractionContextImpl implements InteractionContext {
    protected _version: V1;
    protected _solution: InteractionContextSolution;
    protected _uri: string;
    protected _channels?: InteractionChannelListInstance;
    constructor(_version: V1, sid: string);
    get channels(): InteractionChannelListInstance;
    fetch(callback?: (error: Error | null, item?: InteractionInstance) => any): Promise<InteractionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): InteractionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface InteractionResource {
    sid: string;
    channel: any;
    routing: any;
    url: string;
    links: Record<string, string>;
    interaction_context_sid: string;
}
export declare class InteractionInstance {
    protected _version: V1;
    protected _solution: InteractionContextSolution;
    protected _context?: InteractionContext;
    constructor(_version: V1, payload: InteractionResource, sid?: string);
    /**
     * The unique string created by Twilio to identify an Interaction resource, prefixed with KD.
     */
    sid: string;
    /**
     * A JSON object that defines the Interaction’s communication channel and includes details about the channel. See the [Outbound SMS](https://www.twilio.com/docs/flex/developer/conversations/interactions-api/interactions#agent-initiated-outbound-interactions) and [inbound (API-initiated)](https://www.twilio.com/docs/flex/developer/conversations/interactions-api/interactions#api-initiated-contact) Channel object examples.
     */
    channel: any;
    /**
     * A JSON Object representing the routing rules for the Interaction Channel. See [Outbound SMS Example](https://www.twilio.com/docs/flex/developer/conversations/interactions-api/interactions#agent-initiated-outbound-interactions) for an example Routing object. The Interactions resource uses TaskRouter for all routing functionality.   All attributes in the Routing object on your Interaction request body are added “as is” to the task. For a list of known attributes consumed by the Flex UI and/or Flex Insights, see [Known Task Attributes](https://www.twilio.com/docs/flex/developer/conversations/interactions-api#task-attributes).
     */
    routing: any;
    url: string;
    links: Record<string, string>;
    interactionContextSid: string;
    private get _proxy();
    /**
     * Fetch a InteractionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionInstance
     */
    fetch(callback?: (error: Error | null, item?: InteractionInstance) => any): Promise<InteractionInstance>;
    /**
     * Access the channels.
     */
    channels(): InteractionChannelListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        channel: any;
        routing: any;
        url: string;
        links: Record<string, string>;
        interactionContextSid: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface InteractionSolution {
}
export interface InteractionListInstance {
    _version: V1;
    _solution: InteractionSolution;
    _uri: string;
    (sid: string): InteractionContext;
    get(sid: string): InteractionContext;
    /**
     * Create a InteractionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionInstance
     */
    create(params: InteractionListInstanceCreateOptions, callback?: (error: Error | null, item?: InteractionInstance) => any): Promise<InteractionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InteractionListInstance(version: V1): InteractionListInstance;
export {};
