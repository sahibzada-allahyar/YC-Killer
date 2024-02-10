/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to create a ExternalCampaignInstance
 */
export interface ExternalCampaignListInstanceCreateOptions {
    /** ID of the preregistered campaign. */
    campaignId: string;
    /** The SID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) that the resource is associated with. */
    messagingServiceSid: string;
}
export interface ExternalCampaignSolution {
}
export interface ExternalCampaignListInstance {
    _version: V1;
    _solution: ExternalCampaignSolution;
    _uri: string;
    /**
     * Create a ExternalCampaignInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExternalCampaignInstance
     */
    create(params: ExternalCampaignListInstanceCreateOptions, callback?: (error: Error | null, item?: ExternalCampaignInstance) => any): Promise<ExternalCampaignInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ExternalCampaignListInstance(version: V1): ExternalCampaignListInstance;
interface ExternalCampaignResource {
    sid: string;
    account_sid: string;
    campaign_id: string;
    messaging_service_sid: string;
    date_created: Date;
}
export declare class ExternalCampaignInstance {
    protected _version: V1;
    constructor(_version: V1, payload: ExternalCampaignResource);
    /**
     * The unique string that identifies a US A2P Compliance resource `QE2c6890da8086d771620e9b13fadeba0b`.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that the Campaign belongs to.
     */
    accountSid: string;
    /**
     * ID of the preregistered campaign.
     */
    campaignId: string;
    /**
     * The SID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) that the resource is associated with.
     */
    messagingServiceSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        campaignId: string;
        messagingServiceSid: string;
        dateCreated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
