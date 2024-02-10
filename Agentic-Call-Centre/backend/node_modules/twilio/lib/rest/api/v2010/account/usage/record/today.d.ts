/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../../base/Page";
import Response from "../../../../../../http/response";
import V2010 from "../../../../V2010";
export type TodayCategory = "a2p-registration-fees" | "agent-conference" | "amazon-polly" | "answering-machine-detection" | "authy-authentications" | "authy-calls-outbound" | "authy-monthly-fees" | "authy-phone-intelligence" | "authy-phone-verifications" | "authy-sms-outbound" | "call-progess-events" | "calleridlookups" | "calls" | "calls-client" | "calls-globalconference" | "calls-inbound" | "calls-inbound-local" | "calls-inbound-mobile" | "calls-inbound-tollfree" | "calls-outbound" | "calls-pay-verb-transactions" | "calls-recordings" | "calls-sip" | "calls-sip-inbound" | "calls-sip-outbound" | "calls-transfers" | "carrier-lookups" | "conversations" | "conversations-api-requests" | "conversations-conversation-events" | "conversations-endpoint-connectivity" | "conversations-events" | "conversations-participant-events" | "conversations-participants" | "cps" | "flex-usage" | "fraud-lookups" | "group-rooms" | "group-rooms-data-track" | "group-rooms-encrypted-media-recorded" | "group-rooms-media-downloaded" | "group-rooms-media-recorded" | "group-rooms-media-routed" | "group-rooms-media-stored" | "group-rooms-participant-minutes" | "group-rooms-recorded-minutes" | "imp-v1-usage" | "lookups" | "marketplace" | "marketplace-algorithmia-named-entity-recognition" | "marketplace-cadence-transcription" | "marketplace-cadence-translation" | "marketplace-capio-speech-to-text" | "marketplace-convriza-ababa" | "marketplace-deepgram-phrase-detector" | "marketplace-digital-segment-business-info" | "marketplace-facebook-offline-conversions" | "marketplace-google-speech-to-text" | "marketplace-ibm-watson-message-insights" | "marketplace-ibm-watson-message-sentiment" | "marketplace-ibm-watson-recording-analysis" | "marketplace-ibm-watson-tone-analyzer" | "marketplace-icehook-systems-scout" | "marketplace-infogroup-dataaxle-bizinfo" | "marketplace-keen-io-contact-center-analytics" | "marketplace-marchex-cleancall" | "marketplace-marchex-sentiment-analysis-for-sms" | "marketplace-marketplace-nextcaller-social-id" | "marketplace-mobile-commons-opt-out-classifier" | "marketplace-nexiwave-voicemail-to-text" | "marketplace-nextcaller-advanced-caller-identification" | "marketplace-nomorobo-spam-score" | "marketplace-payfone-tcpa-compliance" | "marketplace-remeeting-automatic-speech-recognition" | "marketplace-tcpa-defense-solutions-blacklist-feed" | "marketplace-telo-opencnam" | "marketplace-truecnam-true-spam" | "marketplace-twilio-caller-name-lookup-us" | "marketplace-twilio-carrier-information-lookup" | "marketplace-voicebase-pci" | "marketplace-voicebase-transcription" | "marketplace-voicebase-transcription-custom-vocabulary" | "marketplace-whitepages-pro-caller-identification" | "marketplace-whitepages-pro-phone-intelligence" | "marketplace-whitepages-pro-phone-reputation" | "marketplace-wolfarm-spoken-results" | "marketplace-wolfram-short-answer" | "marketplace-ytica-contact-center-reporting-analytics" | "mediastorage" | "mms" | "mms-inbound" | "mms-inbound-longcode" | "mms-inbound-shortcode" | "mms-messages-carrierfees" | "mms-outbound" | "mms-outbound-longcode" | "mms-outbound-shortcode" | "monitor-reads" | "monitor-storage" | "monitor-writes" | "notify" | "notify-actions-attempts" | "notify-channels" | "number-format-lookups" | "pchat" | "pchat-users" | "peer-to-peer-rooms-participant-minutes" | "pfax" | "pfax-minutes" | "pfax-minutes-inbound" | "pfax-minutes-outbound" | "pfax-pages" | "phonenumbers" | "phonenumbers-cps" | "phonenumbers-emergency" | "phonenumbers-local" | "phonenumbers-mobile" | "phonenumbers-setups" | "phonenumbers-tollfree" | "premiumsupport" | "proxy" | "proxy-active-sessions" | "pstnconnectivity" | "pv" | "pv-composition-media-downloaded" | "pv-composition-media-encrypted" | "pv-composition-media-stored" | "pv-composition-minutes" | "pv-recording-compositions" | "pv-room-participants" | "pv-room-participants-au1" | "pv-room-participants-br1" | "pv-room-participants-ie1" | "pv-room-participants-jp1" | "pv-room-participants-sg1" | "pv-room-participants-us1" | "pv-room-participants-us2" | "pv-rooms" | "pv-sip-endpoint-registrations" | "recordings" | "recordingstorage" | "rooms-group-bandwidth" | "rooms-group-minutes" | "rooms-peer-to-peer-minutes" | "shortcodes" | "shortcodes-customerowned" | "shortcodes-mms-enablement" | "shortcodes-mps" | "shortcodes-random" | "shortcodes-uk" | "shortcodes-vanity" | "small-group-rooms" | "small-group-rooms-data-track" | "small-group-rooms-participant-minutes" | "sms" | "sms-inbound" | "sms-inbound-longcode" | "sms-inbound-shortcode" | "sms-messages-carrierfees" | "sms-messages-features" | "sms-messages-features-senderid" | "sms-outbound" | "sms-outbound-content-inspection" | "sms-outbound-longcode" | "sms-outbound-shortcode" | "speech-recognition" | "studio-engagements" | "sync" | "sync-actions" | "sync-endpoint-hours" | "sync-endpoint-hours-above-daily-cap" | "taskrouter-tasks" | "totalprice" | "transcriptions" | "trunking-cps" | "trunking-emergency-calls" | "trunking-origination" | "trunking-origination-local" | "trunking-origination-mobile" | "trunking-origination-tollfree" | "trunking-recordings" | "trunking-secure" | "trunking-termination" | "tts-google" | "turnmegabytes" | "turnmegabytes-australia" | "turnmegabytes-brasil" | "turnmegabytes-germany" | "turnmegabytes-india" | "turnmegabytes-ireland" | "turnmegabytes-japan" | "turnmegabytes-singapore" | "turnmegabytes-useast" | "turnmegabytes-uswest" | "twilio-interconnect" | "verify-push" | "verify-totp" | "verify-whatsapp-conversations-business-initiated" | "video-recordings" | "virtual-agent" | "voice-insights" | "voice-insights-client-insights-on-demand-minute" | "voice-insights-ptsn-insights-on-demand-minute" | "voice-insights-sip-interface-insights-on-demand-minute" | "voice-insights-sip-trunking-insights-on-demand-minute" | "voice-intelligence" | "voice-intelligence-transcription" | "voice-intelligence-operators" | "wireless" | "wireless-orders" | "wireless-orders-artwork" | "wireless-orders-bulk" | "wireless-orders-esim" | "wireless-orders-starter" | "wireless-usage" | "wireless-usage-commands" | "wireless-usage-commands-africa" | "wireless-usage-commands-asia" | "wireless-usage-commands-centralandsouthamerica" | "wireless-usage-commands-europe" | "wireless-usage-commands-home" | "wireless-usage-commands-northamerica" | "wireless-usage-commands-oceania" | "wireless-usage-commands-roaming" | "wireless-usage-data" | "wireless-usage-data-africa" | "wireless-usage-data-asia" | "wireless-usage-data-centralandsouthamerica" | "wireless-usage-data-custom-additionalmb" | "wireless-usage-data-custom-first5mb" | "wireless-usage-data-domestic-roaming" | "wireless-usage-data-europe" | "wireless-usage-data-individual-additionalgb" | "wireless-usage-data-individual-firstgb" | "wireless-usage-data-international-roaming-canada" | "wireless-usage-data-international-roaming-india" | "wireless-usage-data-international-roaming-mexico" | "wireless-usage-data-northamerica" | "wireless-usage-data-oceania" | "wireless-usage-data-pooled" | "wireless-usage-data-pooled-downlink" | "wireless-usage-data-pooled-uplink" | "wireless-usage-mrc" | "wireless-usage-mrc-custom" | "wireless-usage-mrc-individual" | "wireless-usage-mrc-pooled" | "wireless-usage-mrc-suspended" | "wireless-usage-sms" | "wireless-usage-voice";
/**
 * Options to pass to each
 */
export interface TodayListInstanceEachOptions {
    /** The [usage category](https://www.twilio.com/docs/usage/api/usage-record#usage-categories) of the UsageRecord resources to read. Only UsageRecord resources in the specified category are retrieved. */
    category?: TodayCategory;
    /** Only include usage that has occurred on or after this date. Specify the date in GMT and format as `YYYY-MM-DD`. You can also specify offsets from the current date, such as: `-30days`, which will set the start date to be 30 days before the current date. */
    startDate?: Date;
    /** Only include usage that occurred on or before this date. Specify the date in GMT and format as `YYYY-MM-DD`.  You can also specify offsets from the current date, such as: `+30days`, which will set the end date to 30 days from the current date. */
    endDate?: Date;
    /** Whether to include usage from the master account and all its subaccounts. Can be: `true` (the default) to include usage from the master account and all subaccounts or `false` to retrieve usage from only the specified account. */
    includeSubaccounts?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TodayInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TodayListInstanceOptions {
    /** The [usage category](https://www.twilio.com/docs/usage/api/usage-record#usage-categories) of the UsageRecord resources to read. Only UsageRecord resources in the specified category are retrieved. */
    category?: TodayCategory;
    /** Only include usage that has occurred on or after this date. Specify the date in GMT and format as `YYYY-MM-DD`. You can also specify offsets from the current date, such as: `-30days`, which will set the start date to be 30 days before the current date. */
    startDate?: Date;
    /** Only include usage that occurred on or before this date. Specify the date in GMT and format as `YYYY-MM-DD`.  You can also specify offsets from the current date, such as: `+30days`, which will set the end date to 30 days from the current date. */
    endDate?: Date;
    /** Whether to include usage from the master account and all its subaccounts. Can be: `true` (the default) to include usage from the master account and all subaccounts or `false` to retrieve usage from only the specified account. */
    includeSubaccounts?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TodayListInstancePageOptions {
    /** The [usage category](https://www.twilio.com/docs/usage/api/usage-record#usage-categories) of the UsageRecord resources to read. Only UsageRecord resources in the specified category are retrieved. */
    category?: TodayCategory;
    /** Only include usage that has occurred on or after this date. Specify the date in GMT and format as `YYYY-MM-DD`. You can also specify offsets from the current date, such as: `-30days`, which will set the start date to be 30 days before the current date. */
    startDate?: Date;
    /** Only include usage that occurred on or before this date. Specify the date in GMT and format as `YYYY-MM-DD`.  You can also specify offsets from the current date, such as: `+30days`, which will set the end date to 30 days from the current date. */
    endDate?: Date;
    /** Whether to include usage from the master account and all its subaccounts. Can be: `true` (the default) to include usage from the master account and all subaccounts or `false` to retrieve usage from only the specified account. */
    includeSubaccounts?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TodaySolution {
    accountSid: string;
}
export interface TodayListInstance {
    _version: V2010;
    _solution: TodaySolution;
    _uri: string;
    /**
     * Streams TodayInstance records from the API.
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
     * @param { TodayListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TodayInstance, done: (err?: Error) => void) => void): void;
    each(params: TodayListInstanceEachOptions, callback?: (item: TodayInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TodayInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TodayPage) => any): Promise<TodayPage>;
    /**
     * Lists TodayInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TodayListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TodayInstance[]) => any): Promise<TodayInstance[]>;
    list(params: TodayListInstanceOptions, callback?: (error: Error | null, items: TodayInstance[]) => any): Promise<TodayInstance[]>;
    /**
     * Retrieve a single page of TodayInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TodayListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TodayPage) => any): Promise<TodayPage>;
    page(params: TodayListInstancePageOptions, callback?: (error: Error | null, items: TodayPage) => any): Promise<TodayPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TodayListInstance(version: V2010, accountSid: string): TodayListInstance;
interface TodayPayload extends TwilioResponsePayload {
    usage_records: TodayResource[];
}
interface TodayResource {
    account_sid: string;
    api_version: string;
    as_of: string;
    category: TodayCategory;
    count: string;
    count_unit: string;
    description: string;
    end_date: Date;
    price: number;
    price_unit: string;
    start_date: Date;
    subresource_uris: Record<string, string>;
    uri: string;
    usage: string;
    usage_unit: string;
}
export declare class TodayInstance {
    protected _version: V2010;
    constructor(_version: V2010, payload: TodayResource, accountSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that accrued the usage.
     */
    accountSid: string;
    /**
     * The API version used to create the resource.
     */
    apiVersion: string;
    /**
     * Usage records up to date as of this timestamp, formatted as YYYY-MM-DDTHH:MM:SS+00:00. All timestamps are in GMT
     */
    asOf: string;
    category: TodayCategory;
    /**
     * The number of usage events, such as the number of calls.
     */
    count: string;
    /**
     * The units in which `count` is measured, such as `calls` for calls or `messages` for SMS.
     */
    countUnit: string;
    /**
     * A plain-language description of the usage category.
     */
    description: string;
    /**
     * The last date for which usage is included in the UsageRecord. The date is specified in GMT and formatted as `YYYY-MM-DD`.
     */
    endDate: Date;
    /**
     * The total price of the usage in the currency specified in `price_unit` and associated with the account.
     */
    price: number;
    /**
     * The currency in which `price` is measured, in [ISO 4127](https://www.iso.org/iso/home/standards/currency_codes.htm) format, such as `usd`, `eur`, and `jpy`.
     */
    priceUnit: string;
    /**
     * The first date for which usage is included in this UsageRecord. The date is specified in GMT and formatted as `YYYY-MM-DD`.
     */
    startDate: Date;
    /**
     * A list of related resources identified by their URIs. For more information, see [List Subresources](https://www.twilio.com/docs/usage/api/usage-record#list-subresources).
     */
    subresourceUris: Record<string, string>;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * The amount used to bill usage and measured in units described in `usage_unit`.
     */
    usage: string;
    /**
     * The units in which `usage` is measured, such as `minutes` for calls or `messages` for SMS.
     */
    usageUnit: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        apiVersion: string;
        asOf: string;
        category: TodayCategory;
        count: string;
        countUnit: string;
        description: string;
        endDate: Date;
        price: number;
        priceUnit: string;
        startDate: Date;
        subresourceUris: Record<string, string>;
        uri: string;
        usage: string;
        usageUnit: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class TodayPage extends Page<V2010, TodayPayload, TodayResource, TodayInstance> {
    /**
     * Initialize the TodayPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: TodaySolution);
    /**
     * Build an instance of TodayInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TodayResource): TodayInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
