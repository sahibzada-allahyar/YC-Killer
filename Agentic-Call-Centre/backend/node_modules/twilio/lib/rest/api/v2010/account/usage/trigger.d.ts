/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
export type TriggerRecurring = "daily" | "monthly" | "yearly" | "alltime";
export type TriggerTriggerField = "count" | "usage" | "price";
export type TriggerUsageCategory = "a2p-registration-fees" | "agent-conference" | "amazon-polly" | "answering-machine-detection" | "authy-authentications" | "authy-calls-outbound" | "authy-monthly-fees" | "authy-phone-intelligence" | "authy-phone-verifications" | "authy-sms-outbound" | "call-progess-events" | "calleridlookups" | "calls" | "calls-client" | "calls-globalconference" | "calls-inbound" | "calls-inbound-local" | "calls-inbound-mobile" | "calls-inbound-tollfree" | "calls-outbound" | "calls-pay-verb-transactions" | "calls-recordings" | "calls-sip" | "calls-sip-inbound" | "calls-sip-outbound" | "calls-transfers" | "carrier-lookups" | "conversations" | "conversations-api-requests" | "conversations-conversation-events" | "conversations-endpoint-connectivity" | "conversations-events" | "conversations-participant-events" | "conversations-participants" | "cps" | "flex-usage" | "fraud-lookups" | "group-rooms" | "group-rooms-data-track" | "group-rooms-encrypted-media-recorded" | "group-rooms-media-downloaded" | "group-rooms-media-recorded" | "group-rooms-media-routed" | "group-rooms-media-stored" | "group-rooms-participant-minutes" | "group-rooms-recorded-minutes" | "imp-v1-usage" | "lookups" | "marketplace" | "marketplace-algorithmia-named-entity-recognition" | "marketplace-cadence-transcription" | "marketplace-cadence-translation" | "marketplace-capio-speech-to-text" | "marketplace-convriza-ababa" | "marketplace-deepgram-phrase-detector" | "marketplace-digital-segment-business-info" | "marketplace-facebook-offline-conversions" | "marketplace-google-speech-to-text" | "marketplace-ibm-watson-message-insights" | "marketplace-ibm-watson-message-sentiment" | "marketplace-ibm-watson-recording-analysis" | "marketplace-ibm-watson-tone-analyzer" | "marketplace-icehook-systems-scout" | "marketplace-infogroup-dataaxle-bizinfo" | "marketplace-keen-io-contact-center-analytics" | "marketplace-marchex-cleancall" | "marketplace-marchex-sentiment-analysis-for-sms" | "marketplace-marketplace-nextcaller-social-id" | "marketplace-mobile-commons-opt-out-classifier" | "marketplace-nexiwave-voicemail-to-text" | "marketplace-nextcaller-advanced-caller-identification" | "marketplace-nomorobo-spam-score" | "marketplace-payfone-tcpa-compliance" | "marketplace-remeeting-automatic-speech-recognition" | "marketplace-tcpa-defense-solutions-blacklist-feed" | "marketplace-telo-opencnam" | "marketplace-truecnam-true-spam" | "marketplace-twilio-caller-name-lookup-us" | "marketplace-twilio-carrier-information-lookup" | "marketplace-voicebase-pci" | "marketplace-voicebase-transcription" | "marketplace-voicebase-transcription-custom-vocabulary" | "marketplace-whitepages-pro-caller-identification" | "marketplace-whitepages-pro-phone-intelligence" | "marketplace-whitepages-pro-phone-reputation" | "marketplace-wolfarm-spoken-results" | "marketplace-wolfram-short-answer" | "marketplace-ytica-contact-center-reporting-analytics" | "mediastorage" | "mms" | "mms-inbound" | "mms-inbound-longcode" | "mms-inbound-shortcode" | "mms-messages-carrierfees" | "mms-outbound" | "mms-outbound-longcode" | "mms-outbound-shortcode" | "monitor-reads" | "monitor-storage" | "monitor-writes" | "notify" | "notify-actions-attempts" | "notify-channels" | "number-format-lookups" | "pchat" | "pchat-users" | "peer-to-peer-rooms-participant-minutes" | "pfax" | "pfax-minutes" | "pfax-minutes-inbound" | "pfax-minutes-outbound" | "pfax-pages" | "phonenumbers" | "phonenumbers-cps" | "phonenumbers-emergency" | "phonenumbers-local" | "phonenumbers-mobile" | "phonenumbers-setups" | "phonenumbers-tollfree" | "premiumsupport" | "proxy" | "proxy-active-sessions" | "pstnconnectivity" | "pv" | "pv-composition-media-downloaded" | "pv-composition-media-encrypted" | "pv-composition-media-stored" | "pv-composition-minutes" | "pv-recording-compositions" | "pv-room-participants" | "pv-room-participants-au1" | "pv-room-participants-br1" | "pv-room-participants-ie1" | "pv-room-participants-jp1" | "pv-room-participants-sg1" | "pv-room-participants-us1" | "pv-room-participants-us2" | "pv-rooms" | "pv-sip-endpoint-registrations" | "recordings" | "recordingstorage" | "rooms-group-bandwidth" | "rooms-group-minutes" | "rooms-peer-to-peer-minutes" | "shortcodes" | "shortcodes-customerowned" | "shortcodes-mms-enablement" | "shortcodes-mps" | "shortcodes-random" | "shortcodes-uk" | "shortcodes-vanity" | "small-group-rooms" | "small-group-rooms-data-track" | "small-group-rooms-participant-minutes" | "sms" | "sms-inbound" | "sms-inbound-longcode" | "sms-inbound-shortcode" | "sms-messages-carrierfees" | "sms-messages-features" | "sms-messages-features-senderid" | "sms-outbound" | "sms-outbound-content-inspection" | "sms-outbound-longcode" | "sms-outbound-shortcode" | "speech-recognition" | "studio-engagements" | "sync" | "sync-actions" | "sync-endpoint-hours" | "sync-endpoint-hours-above-daily-cap" | "taskrouter-tasks" | "totalprice" | "transcriptions" | "trunking-cps" | "trunking-emergency-calls" | "trunking-origination" | "trunking-origination-local" | "trunking-origination-mobile" | "trunking-origination-tollfree" | "trunking-recordings" | "trunking-secure" | "trunking-termination" | "tts-google" | "turnmegabytes" | "turnmegabytes-australia" | "turnmegabytes-brasil" | "turnmegabytes-germany" | "turnmegabytes-india" | "turnmegabytes-ireland" | "turnmegabytes-japan" | "turnmegabytes-singapore" | "turnmegabytes-useast" | "turnmegabytes-uswest" | "twilio-interconnect" | "verify-push" | "verify-totp" | "verify-whatsapp-conversations-business-initiated" | "video-recordings" | "virtual-agent" | "voice-insights" | "voice-insights-client-insights-on-demand-minute" | "voice-insights-ptsn-insights-on-demand-minute" | "voice-insights-sip-interface-insights-on-demand-minute" | "voice-insights-sip-trunking-insights-on-demand-minute" | "voice-intelligence" | "voice-intelligence-transcription" | "voice-intelligence-operators" | "wireless" | "wireless-orders" | "wireless-orders-artwork" | "wireless-orders-bulk" | "wireless-orders-esim" | "wireless-orders-starter" | "wireless-usage" | "wireless-usage-commands" | "wireless-usage-commands-africa" | "wireless-usage-commands-asia" | "wireless-usage-commands-centralandsouthamerica" | "wireless-usage-commands-europe" | "wireless-usage-commands-home" | "wireless-usage-commands-northamerica" | "wireless-usage-commands-oceania" | "wireless-usage-commands-roaming" | "wireless-usage-data" | "wireless-usage-data-africa" | "wireless-usage-data-asia" | "wireless-usage-data-centralandsouthamerica" | "wireless-usage-data-custom-additionalmb" | "wireless-usage-data-custom-first5mb" | "wireless-usage-data-domestic-roaming" | "wireless-usage-data-europe" | "wireless-usage-data-individual-additionalgb" | "wireless-usage-data-individual-firstgb" | "wireless-usage-data-international-roaming-canada" | "wireless-usage-data-international-roaming-india" | "wireless-usage-data-international-roaming-mexico" | "wireless-usage-data-northamerica" | "wireless-usage-data-oceania" | "wireless-usage-data-pooled" | "wireless-usage-data-pooled-downlink" | "wireless-usage-data-pooled-uplink" | "wireless-usage-mrc" | "wireless-usage-mrc-custom" | "wireless-usage-mrc-individual" | "wireless-usage-mrc-pooled" | "wireless-usage-mrc-suspended" | "wireless-usage-sms" | "wireless-usage-voice";
/**
 * Options to pass to update a TriggerInstance
 */
export interface TriggerContextUpdateOptions {
    /** The HTTP method we should use to call `callback_url`. Can be: `GET` or `POST` and the default is `POST`. */
    callbackMethod?: string;
    /** The URL we should call using `callback_method` when the trigger fires. */
    callbackUrl?: string;
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
}
/**
 * Options to pass to create a TriggerInstance
 */
export interface TriggerListInstanceCreateOptions {
    /** The URL we should call using `callback_method` when the trigger fires. */
    callbackUrl: string;
    /** The usage value at which the trigger should fire.  For convenience, you can use an offset value such as `+30` to specify a trigger_value that is 30 units more than the current usage value. Be sure to urlencode a `+` as `%2B`. */
    triggerValue: string;
    /**  */
    usageCategory: TriggerUsageCategory;
    /** The HTTP method we should use to call `callback_url`. Can be: `GET` or `POST` and the default is `POST`. */
    callbackMethod?: string;
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /**  */
    recurring?: TriggerRecurring;
    /**  */
    triggerBy?: TriggerTriggerField;
}
/**
 * Options to pass to each
 */
export interface TriggerListInstanceEachOptions {
    /** The frequency of recurring UsageTriggers to read. Can be: `daily`, `monthly`, or `yearly` to read recurring UsageTriggers. An empty value or a value of `alltime` reads non-recurring UsageTriggers. */
    recurring?: TriggerRecurring;
    /** The trigger field of the UsageTriggers to read.  Can be: `count`, `usage`, or `price` as described in the [UsageRecords documentation](https://www.twilio.com/docs/usage/api/usage-record#usage-count-price). */
    triggerBy?: TriggerTriggerField;
    /** The usage category of the UsageTriggers to read. Must be a supported [usage categories](https://www.twilio.com/docs/usage/api/usage-record#usage-categories). */
    usageCategory?: TriggerUsageCategory;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TriggerInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TriggerListInstanceOptions {
    /** The frequency of recurring UsageTriggers to read. Can be: `daily`, `monthly`, or `yearly` to read recurring UsageTriggers. An empty value or a value of `alltime` reads non-recurring UsageTriggers. */
    recurring?: TriggerRecurring;
    /** The trigger field of the UsageTriggers to read.  Can be: `count`, `usage`, or `price` as described in the [UsageRecords documentation](https://www.twilio.com/docs/usage/api/usage-record#usage-count-price). */
    triggerBy?: TriggerTriggerField;
    /** The usage category of the UsageTriggers to read. Must be a supported [usage categories](https://www.twilio.com/docs/usage/api/usage-record#usage-categories). */
    usageCategory?: TriggerUsageCategory;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TriggerListInstancePageOptions {
    /** The frequency of recurring UsageTriggers to read. Can be: `daily`, `monthly`, or `yearly` to read recurring UsageTriggers. An empty value or a value of `alltime` reads non-recurring UsageTriggers. */
    recurring?: TriggerRecurring;
    /** The trigger field of the UsageTriggers to read.  Can be: `count`, `usage`, or `price` as described in the [UsageRecords documentation](https://www.twilio.com/docs/usage/api/usage-record#usage-count-price). */
    triggerBy?: TriggerTriggerField;
    /** The usage category of the UsageTriggers to read. Must be a supported [usage categories](https://www.twilio.com/docs/usage/api/usage-record#usage-categories). */
    usageCategory?: TriggerUsageCategory;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TriggerContext {
    /**
     * Remove a TriggerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TriggerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TriggerInstance
     */
    fetch(callback?: (error: Error | null, item?: TriggerInstance) => any): Promise<TriggerInstance>;
    /**
     * Update a TriggerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TriggerInstance
     */
    update(callback?: (error: Error | null, item?: TriggerInstance) => any): Promise<TriggerInstance>;
    /**
     * Update a TriggerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TriggerInstance
     */
    update(params: TriggerContextUpdateOptions, callback?: (error: Error | null, item?: TriggerInstance) => any): Promise<TriggerInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TriggerContextSolution {
    accountSid: string;
    sid: string;
}
export declare class TriggerContextImpl implements TriggerContext {
    protected _version: V2010;
    protected _solution: TriggerContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: TriggerInstance) => any): Promise<TriggerInstance>;
    update(params?: TriggerContextUpdateOptions | ((error: Error | null, item?: TriggerInstance) => any), callback?: (error: Error | null, item?: TriggerInstance) => any): Promise<TriggerInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TriggerContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TriggerPayload extends TwilioResponsePayload {
    usage_triggers: TriggerResource[];
}
interface TriggerResource {
    account_sid: string;
    api_version: string;
    callback_method: string;
    callback_url: string;
    current_value: string;
    date_created: Date;
    date_fired: Date;
    date_updated: Date;
    friendly_name: string;
    recurring: TriggerRecurring;
    sid: string;
    trigger_by: TriggerTriggerField;
    trigger_value: string;
    uri: string;
    usage_category: TriggerUsageCategory;
    usage_record_uri: string;
}
export declare class TriggerInstance {
    protected _version: V2010;
    protected _solution: TriggerContextSolution;
    protected _context?: TriggerContext;
    constructor(_version: V2010, payload: TriggerResource, accountSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that the trigger monitors.
     */
    accountSid: string;
    /**
     * The API version used to create the resource.
     */
    apiVersion: string;
    /**
     * The HTTP method we use to call `callback_url`. Can be: `GET` or `POST`.
     */
    callbackMethod: string;
    /**
     * The URL we call using the `callback_method` when the trigger fires.
     */
    callbackUrl: string;
    /**
     * The current value of the field the trigger is watching.
     */
    currentValue: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the trigger was last fired specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateFired: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The string that you assigned to describe the trigger.
     */
    friendlyName: string;
    recurring: TriggerRecurring;
    /**
     * The unique string that that we created to identify the UsageTrigger resource.
     */
    sid: string;
    triggerBy: TriggerTriggerField;
    /**
     * The value at which the trigger will fire.  Must be a positive, numeric value.
     */
    triggerValue: string;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    usageCategory: TriggerUsageCategory;
    /**
     * The URI of the [UsageRecord](https://www.twilio.com/docs/usage/api/usage-record) resource this trigger watches, relative to `https://api.twilio.com`.
     */
    usageRecordUri: string;
    private get _proxy();
    /**
     * Remove a TriggerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TriggerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TriggerInstance
     */
    fetch(callback?: (error: Error | null, item?: TriggerInstance) => any): Promise<TriggerInstance>;
    /**
     * Update a TriggerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TriggerInstance
     */
    update(callback?: (error: Error | null, item?: TriggerInstance) => any): Promise<TriggerInstance>;
    /**
     * Update a TriggerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TriggerInstance
     */
    update(params: TriggerContextUpdateOptions, callback?: (error: Error | null, item?: TriggerInstance) => any): Promise<TriggerInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        apiVersion: string;
        callbackMethod: string;
        callbackUrl: string;
        currentValue: string;
        dateCreated: Date;
        dateFired: Date;
        dateUpdated: Date;
        friendlyName: string;
        recurring: TriggerRecurring;
        sid: string;
        triggerBy: TriggerTriggerField;
        triggerValue: string;
        uri: string;
        usageCategory: TriggerUsageCategory;
        usageRecordUri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TriggerSolution {
    accountSid: string;
}
export interface TriggerListInstance {
    _version: V2010;
    _solution: TriggerSolution;
    _uri: string;
    (sid: string): TriggerContext;
    get(sid: string): TriggerContext;
    /**
     * Create a TriggerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TriggerInstance
     */
    create(params: TriggerListInstanceCreateOptions, callback?: (error: Error | null, item?: TriggerInstance) => any): Promise<TriggerInstance>;
    /**
     * Streams TriggerInstance records from the API.
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
     * @param { TriggerListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TriggerInstance, done: (err?: Error) => void) => void): void;
    each(params: TriggerListInstanceEachOptions, callback?: (item: TriggerInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TriggerInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TriggerPage) => any): Promise<TriggerPage>;
    /**
     * Lists TriggerInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TriggerListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TriggerInstance[]) => any): Promise<TriggerInstance[]>;
    list(params: TriggerListInstanceOptions, callback?: (error: Error | null, items: TriggerInstance[]) => any): Promise<TriggerInstance[]>;
    /**
     * Retrieve a single page of TriggerInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TriggerListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TriggerPage) => any): Promise<TriggerPage>;
    page(params: TriggerListInstancePageOptions, callback?: (error: Error | null, items: TriggerPage) => any): Promise<TriggerPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TriggerListInstance(version: V2010, accountSid: string): TriggerListInstance;
export declare class TriggerPage extends Page<V2010, TriggerPayload, TriggerResource, TriggerInstance> {
    /**
     * Initialize the TriggerPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: TriggerSolution);
    /**
     * Build an instance of TriggerInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TriggerResource): TriggerInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
