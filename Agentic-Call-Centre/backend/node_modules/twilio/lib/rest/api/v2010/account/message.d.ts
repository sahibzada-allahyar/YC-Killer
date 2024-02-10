/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
import { FeedbackListInstance } from "./message/feedback";
import { MediaListInstance } from "./message/media";
export type MessageAddressRetention = "retain" | "obfuscate";
export type MessageContentRetention = "retain" | "discard";
export type MessageDirection = "inbound" | "outbound-api" | "outbound-call" | "outbound-reply";
export type MessageRiskCheck = "enable" | "disable";
export type MessageScheduleType = "fixed";
export type MessageStatus = "queued" | "sending" | "sent" | "failed" | "delivered" | "undelivered" | "receiving" | "received" | "accepted" | "scheduled" | "read" | "partially_delivered" | "canceled";
export type MessageUpdateStatus = "canceled";
/**
 * Options to pass to update a MessageInstance
 */
export interface MessageContextUpdateOptions {
    /** The new `body` of the Message resource. To redact the text content of a Message, this parameter\\\'s value must be an empty string */
    body?: string;
    /**  */
    status?: MessageUpdateStatus;
}
/**
 * Options to pass to create a MessageInstance
 */
export interface MessageListInstanceCreateOptions {
    /** The recipient\\\'s phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format (for SMS/MMS) or [channel address](https://www.twilio.com/docs/messaging/channels), e.g. `whatsapp:+15552229999`. */
    to: string;
    /** The URL of the endpoint to which Twilio sends [Message status callback requests](https://www.twilio.com/docs/sms/api/message-resource#twilios-request-to-the-statuscallback-url). URL must contain a valid hostname and underscores are not allowed. If you include this parameter with the `messaging_service_sid`, Twilio uses this URL instead of the Status Callback URL of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource).  */
    statusCallback?: string;
    /** The SID of the associated [TwiML Application](https://www.twilio.com/docs/usage/api/applications). If this parameter is provided, the `status_callback` parameter of this request is ignored; [Message status callback requests](https://www.twilio.com/docs/sms/api/message-resource#twilios-request-to-the-statuscallback-url) are sent to the TwiML App\\\'s `message_status_callback` URL. */
    applicationSid?: string;
    /** The maximum price in US dollars that you are willing to pay for this Message\\\'s delivery. The value can have up to four decimal places. When the `max_price` parameter is provided, the cost of a message is checked before it is sent. If the cost exceeds `max_price`, the message is not sent and the Message `status` is `failed`. */
    maxPrice?: number;
    /** Boolean indicating whether or not you intend to provide delivery confirmation feedback to Twilio (used in conjunction with the [Message Feedback subresource](https://www.twilio.com/docs/sms/api/message-feedback-resource)). Default value is `false`. */
    provideFeedback?: boolean;
    /** Total number of attempts made (including this request) to send the message regardless of the provider used */
    attempt?: number;
    /** The maximum length in seconds that the Message can remain in Twilio\\\'s outgoing message queue. If a queued Message exceeds the `validity_period`, the Message is not sent. Accepted values are integers from `1` to `14400`. Default value is `14400`. A `validity_period` greater than `5` is recommended. [Learn more about the validity period](https://www.twilio.com/blog/take-more-control-of-outbound-messages-using-validity-period-html) */
    validityPeriod?: number;
    /** Reserved */
    forceDelivery?: boolean;
    /**  */
    contentRetention?: MessageContentRetention;
    /**  */
    addressRetention?: MessageAddressRetention;
    /** Whether to detect Unicode characters that have a similar GSM-7 character and replace them. Can be: `true` or `false`. */
    smartEncoded?: boolean;
    /** Rich actions for non-SMS/MMS channels. Used for [sending location in WhatsApp messages](https://www.twilio.com/docs/whatsapp/message-features#location-messages-with-whatsapp). */
    persistentAction?: Array<string>;
    /** For Messaging Services with [Link Shortening configured](https://www.twilio.com/docs/messaging/features/link-shortening) only: A Boolean indicating whether or not Twilio should shorten links in the `body` of the Message. Default value is `false`. If `true`, the `messaging_service_sid` parameter must also be provided. */
    shortenUrls?: boolean;
    /**  */
    scheduleType?: MessageScheduleType;
    /** The time that Twilio will send the message. Must be in ISO 8601 format. */
    sendAt?: Date;
    /** If set to `true`, Twilio delivers the message as a single MMS message, regardless of the presence of media. */
    sendAsMms?: boolean;
    /** For [Content Editor/API](https://www.twilio.com/docs/content) only: Key-value pairs of [Template variables](https://www.twilio.com/docs/content/using-variables-with-content-api) and their substitution values. `content_sid` parameter must also be provided. If values are not defined in the `content_variables` parameter, the [Template\\\'s default placeholder values](https://www.twilio.com/docs/content/content-api-resources#create-templates) are used. */
    contentVariables?: string;
    /**  */
    riskCheck?: MessageRiskCheck;
    /** The sender\\\'s Twilio phone number (in [E.164](https://en.wikipedia.org/wiki/E.164) format), [alphanumeric sender ID](https://www.twilio.com/docs/sms/quickstart), [Wireless SIM](https://www.twilio.com/docs/iot/wireless/programmable-wireless-send-machine-machine-sms-commands), [short code](https://www.twilio.com/en-us/messaging/channels/sms/short-codes), or [channel address](https://www.twilio.com/docs/messaging/channels) (e.g., `whatsapp:+15554449999`). The value of the `from` parameter must be a sender that is hosted within Twilio and belongs to the Account creating the Message. If you are using `messaging_service_sid`, this parameter can be empty (Twilio assigns a `from` value from the Messaging Service\\\'s Sender Pool) or you can provide a specific sender from your Sender Pool. */
    from?: string;
    /** The SID of the [Messaging Service](https://www.twilio.com/docs/messaging/services) you want to associate with the Message. When this parameter is provided and the `from` parameter is omitted, Twilio selects the optimal sender from the Messaging Service\\\'s Sender Pool. You may also provide a `from` parameter if you want to use a specific Sender from the Sender Pool. */
    messagingServiceSid?: string;
    /** The text content of the outgoing message. Can be up to 1,600 characters in length. SMS only: If the `body` contains more than 160 [GSM-7](https://www.twilio.com/docs/glossary/what-is-gsm-7-character-encoding) characters (or 70 [UCS-2](https://www.twilio.com/docs/glossary/what-is-ucs-2-character-encoding) characters), the message is segmented and charged accordingly. For long `body` text, consider using the [send_as_mms parameter](https://www.twilio.com/blog/mms-for-long-text-messages). */
    body?: string;
    /** The URL of media to include in the Message content. `jpeg`, `jpg`, `gif`, and `png` file types are fully supported by Twilio and content is formatted for delivery on destination devices. The media size limit is 5 MB for supported file types (`jpeg`, `jpg`, `png`, `gif`) and 500 KB for [other types](https://www.twilio.com/docs/messaging/guides/accepted-mime-types) of accepted media. To send more than one image in the message, provide multiple `media_url` parameters in the POST request. You can include up to ten `media_url` parameters per message. [International](https://support.twilio.com/hc/en-us/articles/223179808-Sending-and-receiving-MMS-messages) and [carrier](https://support.twilio.com/hc/en-us/articles/223133707-Is-MMS-supported-for-all-carriers-in-US-and-Canada-) limits apply. */
    mediaUrl?: Array<string>;
    /** For [Content Editor/API](https://www.twilio.com/docs/content) only: The SID of the Content Template to be used with the Message, e.g., `HXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`. If this parameter is not provided, a Content Template is not used. Find the SID in the Console on the Content Editor page. For Content API users, the SID is found in Twilio\\\'s response when [creating the Template](https://www.twilio.com/docs/content/content-api-resources#create-templates) or by [fetching your Templates](https://www.twilio.com/docs/content/content-api-resources#fetch-all-content-resources). */
    contentSid?: string;
}
/**
 * Options to pass to each
 */
export interface MessageListInstanceEachOptions {
    /** Filter by recipient. For example: Set this `to` parameter to `+15558881111` to retrieve a list of Message resources with `to` properties of `+15558881111` */
    to?: string;
    /** Filter by sender. For example: Set this `from` parameter to `+15552229999` to retrieve a list of Message resources with `from` properties of `+15552229999` */
    from?: string;
    /** Filter by Message `sent_date`. Accepts GMT dates in the following formats: `YYYY-MM-DD` (to find Messages with a specific `sent_date`), `<=YYYY-MM-DD` (to find Messages with `sent_date`s on and before a specific date), and `>=YYYY-MM-DD` (to find Messages with `sent_dates` on and after a specific date). */
    dateSent?: Date;
    /** Filter by Message `sent_date`. Accepts GMT dates in the following formats: `YYYY-MM-DD` (to find Messages with a specific `sent_date`), `<=YYYY-MM-DD` (to find Messages with `sent_date`s on and before a specific date), and `>=YYYY-MM-DD` (to find Messages with `sent_dates` on and after a specific date). */
    dateSentBefore?: Date;
    /** Filter by Message `sent_date`. Accepts GMT dates in the following formats: `YYYY-MM-DD` (to find Messages with a specific `sent_date`), `<=YYYY-MM-DD` (to find Messages with `sent_date`s on and before a specific date), and `>=YYYY-MM-DD` (to find Messages with `sent_dates` on and after a specific date). */
    dateSentAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: MessageInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface MessageListInstanceOptions {
    /** Filter by recipient. For example: Set this `to` parameter to `+15558881111` to retrieve a list of Message resources with `to` properties of `+15558881111` */
    to?: string;
    /** Filter by sender. For example: Set this `from` parameter to `+15552229999` to retrieve a list of Message resources with `from` properties of `+15552229999` */
    from?: string;
    /** Filter by Message `sent_date`. Accepts GMT dates in the following formats: `YYYY-MM-DD` (to find Messages with a specific `sent_date`), `<=YYYY-MM-DD` (to find Messages with `sent_date`s on and before a specific date), and `>=YYYY-MM-DD` (to find Messages with `sent_dates` on and after a specific date). */
    dateSent?: Date;
    /** Filter by Message `sent_date`. Accepts GMT dates in the following formats: `YYYY-MM-DD` (to find Messages with a specific `sent_date`), `<=YYYY-MM-DD` (to find Messages with `sent_date`s on and before a specific date), and `>=YYYY-MM-DD` (to find Messages with `sent_dates` on and after a specific date). */
    dateSentBefore?: Date;
    /** Filter by Message `sent_date`. Accepts GMT dates in the following formats: `YYYY-MM-DD` (to find Messages with a specific `sent_date`), `<=YYYY-MM-DD` (to find Messages with `sent_date`s on and before a specific date), and `>=YYYY-MM-DD` (to find Messages with `sent_dates` on and after a specific date). */
    dateSentAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface MessageListInstancePageOptions {
    /** Filter by recipient. For example: Set this `to` parameter to `+15558881111` to retrieve a list of Message resources with `to` properties of `+15558881111` */
    to?: string;
    /** Filter by sender. For example: Set this `from` parameter to `+15552229999` to retrieve a list of Message resources with `from` properties of `+15552229999` */
    from?: string;
    /** Filter by Message `sent_date`. Accepts GMT dates in the following formats: `YYYY-MM-DD` (to find Messages with a specific `sent_date`), `<=YYYY-MM-DD` (to find Messages with `sent_date`s on and before a specific date), and `>=YYYY-MM-DD` (to find Messages with `sent_dates` on and after a specific date). */
    dateSent?: Date;
    /** Filter by Message `sent_date`. Accepts GMT dates in the following formats: `YYYY-MM-DD` (to find Messages with a specific `sent_date`), `<=YYYY-MM-DD` (to find Messages with `sent_date`s on and before a specific date), and `>=YYYY-MM-DD` (to find Messages with `sent_dates` on and after a specific date). */
    dateSentBefore?: Date;
    /** Filter by Message `sent_date`. Accepts GMT dates in the following formats: `YYYY-MM-DD` (to find Messages with a specific `sent_date`), `<=YYYY-MM-DD` (to find Messages with `sent_date`s on and before a specific date), and `>=YYYY-MM-DD` (to find Messages with `sent_dates` on and after a specific date). */
    dateSentAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface MessageContext {
    feedback: FeedbackListInstance;
    media: MediaListInstance;
    /**
     * Remove a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    fetch(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Update a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    update(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Update a MessageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    update(params: MessageContextUpdateOptions, callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface MessageContextSolution {
    accountSid: string;
    sid: string;
}
export declare class MessageContextImpl implements MessageContext {
    protected _version: V2010;
    protected _solution: MessageContextSolution;
    protected _uri: string;
    protected _feedback?: FeedbackListInstance;
    protected _media?: MediaListInstance;
    constructor(_version: V2010, accountSid: string, sid: string);
    get feedback(): FeedbackListInstance;
    get media(): MediaListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    update(params?: MessageContextUpdateOptions | ((error: Error | null, item?: MessageInstance) => any), callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): MessageContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface MessagePayload extends TwilioResponsePayload {
    messages: MessageResource[];
}
interface MessageResource {
    body: string;
    num_segments: string;
    direction: MessageDirection;
    from: string;
    to: string;
    date_updated: Date;
    price: string;
    error_message: string;
    uri: string;
    account_sid: string;
    num_media: string;
    status: MessageStatus;
    messaging_service_sid: string;
    sid: string;
    date_sent: Date;
    date_created: Date;
    error_code: number;
    price_unit: string;
    api_version: string;
    subresource_uris: Record<string, string>;
}
export declare class MessageInstance {
    protected _version: V2010;
    protected _solution: MessageContextSolution;
    protected _context?: MessageContext;
    constructor(_version: V2010, payload: MessageResource, accountSid: string, sid?: string);
    /**
     * The text content of the message
     */
    body: string;
    /**
     * The number of segments that make up the complete message. SMS message bodies that exceed the [character limit](https://www.twilio.com/docs/glossary/what-sms-character-limit) are segmented and charged as multiple messages. Note: For messages sent via a Messaging Service, `num_segments` is initially `0`, since a sender hasn\'t yet been assigned.
     */
    numSegments: string;
    direction: MessageDirection;
    /**
     * The sender\'s phone number (in [E.164](https://en.wikipedia.org/wiki/E.164) format), [alphanumeric sender ID](https://www.twilio.com/docs/sms/quickstart), [Wireless SIM](https://www.twilio.com/docs/iot/wireless/programmable-wireless-send-machine-machine-sms-commands), [short code](https://www.twilio.com/en-us/messaging/channels/sms/short-codes), or  [channel address](https://www.twilio.com/docs/messaging/channels) (e.g., `whatsapp:+15554449999`). For incoming messages, this is the number or channel address of the sender. For outgoing messages, this value is a Twilio phone number, alphanumeric sender ID, short code, or channel address from which the message is sent.
     */
    from: string;
    /**
     * The recipient\'s phone number (in [E.164](https://en.wikipedia.org/wiki/E.164) format) or [channel address](https://www.twilio.com/docs/messaging/channels) (e.g. `whatsapp:+15552229999`)
     */
    to: string;
    /**
     * The [RFC 2822](https://datatracker.ietf.org/doc/html/rfc2822#section-3.3) timestamp (in GMT) of when the Message resource was last updated
     */
    dateUpdated: Date;
    /**
     * The amount billed for the message in the currency specified by `price_unit`. The `price` is populated after the message has been sent/received, and may not be immediately availalble. View the [Pricing page](https://www.twilio.com/en-us/pricing) for more details.
     */
    price: string;
    /**
     * The description of the `error_code` if the Message `status` is `failed` or `undelivered`. If no error was encountered, the value is `null`.
     */
    errorMessage: string;
    /**
     * The URI of the Message resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) associated with the Message resource
     */
    accountSid: string;
    /**
     * The number of media files associated with the Message resource.
     */
    numMedia: string;
    status: MessageStatus;
    /**
     * The SID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) associated with the Message resource. The value is `null` if a Messaging Service was not used.
     */
    messagingServiceSid: string;
    /**
     * The unique, Twilio-provided string that identifies the Message resource.
     */
    sid: string;
    /**
     * The [RFC 2822](https://datatracker.ietf.org/doc/html/rfc2822#section-3.3) timestamp (in GMT) of when the Message was sent. For an outgoing message, this is when Twilio sent the message. For an incoming message, this is when Twilio sent the HTTP request to your incoming message webhook URL.
     */
    dateSent: Date;
    /**
     * The [RFC 2822](https://datatracker.ietf.org/doc/html/rfc2822#section-3.3) timestamp (in GMT) of when the Message resource was created
     */
    dateCreated: Date;
    /**
     * The [error code](https://www.twilio.com/docs/api/errors) returned if the Message `status` is `failed` or `undelivered`. If no error was encountered, the value is `null`.
     */
    errorCode: number;
    /**
     * The currency in which `price` is measured, in [ISO 4127](https://www.iso.org/iso/home/standards/currency_codes.htm) format (e.g. `usd`, `eur`, `jpy`).
     */
    priceUnit: string;
    /**
     * The API version used to process the Message
     */
    apiVersion: string;
    /**
     * A list of related resources identified by their URIs relative to `https://api.twilio.com`
     */
    subresourceUris: Record<string, string>;
    private get _proxy();
    /**
     * Remove a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    fetch(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Update a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    update(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Update a MessageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    update(params: MessageContextUpdateOptions, callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Access the feedback.
     */
    feedback(): FeedbackListInstance;
    /**
     * Access the media.
     */
    media(): MediaListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        body: string;
        numSegments: string;
        direction: MessageDirection;
        from: string;
        to: string;
        dateUpdated: Date;
        price: string;
        errorMessage: string;
        uri: string;
        accountSid: string;
        numMedia: string;
        status: MessageStatus;
        messagingServiceSid: string;
        sid: string;
        dateSent: Date;
        dateCreated: Date;
        errorCode: number;
        priceUnit: string;
        apiVersion: string;
        subresourceUris: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface MessageSolution {
    accountSid: string;
}
export interface MessageListInstance {
    _version: V2010;
    _solution: MessageSolution;
    _uri: string;
    (sid: string): MessageContext;
    get(sid: string): MessageContext;
    /**
     * Create a MessageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    create(params: MessageListInstanceCreateOptions, callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Streams MessageInstance records from the API.
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
     * @param { MessageListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: MessageInstance, done: (err?: Error) => void) => void): void;
    each(params: MessageListInstanceEachOptions, callback?: (item: MessageInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of MessageInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: MessagePage) => any): Promise<MessagePage>;
    /**
     * Lists MessageInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MessageListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: MessageInstance[]) => any): Promise<MessageInstance[]>;
    list(params: MessageListInstanceOptions, callback?: (error: Error | null, items: MessageInstance[]) => any): Promise<MessageInstance[]>;
    /**
     * Retrieve a single page of MessageInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MessageListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: MessagePage) => any): Promise<MessagePage>;
    page(params: MessageListInstancePageOptions, callback?: (error: Error | null, items: MessagePage) => any): Promise<MessagePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MessageListInstance(version: V2010, accountSid: string): MessageListInstance;
export declare class MessagePage extends Page<V2010, MessagePayload, MessageResource, MessageInstance> {
    /**
     * Initialize the MessagePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: MessageSolution);
    /**
     * Build an instance of MessageInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: MessageResource): MessageInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
