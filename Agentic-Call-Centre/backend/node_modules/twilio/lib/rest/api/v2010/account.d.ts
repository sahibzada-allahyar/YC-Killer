/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V2010 from "../V2010";
import { AddressListInstance } from "./account/address";
import { ApplicationListInstance } from "./account/application";
import { AuthorizedConnectAppListInstance } from "./account/authorizedConnectApp";
import { AvailablePhoneNumberCountryListInstance } from "./account/availablePhoneNumberCountry";
import { BalanceListInstance } from "./account/balance";
import { CallListInstance } from "./account/call";
import { ConferenceListInstance } from "./account/conference";
import { ConnectAppListInstance } from "./account/connectApp";
import { IncomingPhoneNumberListInstance } from "./account/incomingPhoneNumber";
import { KeyListInstance } from "./account/key";
import { MessageListInstance } from "./account/message";
import { NewKeyListInstance } from "./account/newKey";
import { NewSigningKeyListInstance } from "./account/newSigningKey";
import { NotificationListInstance } from "./account/notification";
import { OutgoingCallerIdListInstance } from "./account/outgoingCallerId";
import { QueueListInstance } from "./account/queue";
import { RecordingListInstance } from "./account/recording";
import { ShortCodeListInstance } from "./account/shortCode";
import { SigningKeyListInstance } from "./account/signingKey";
import { SipListInstance } from "./account/sip";
import { TokenListInstance } from "./account/token";
import { TranscriptionListInstance } from "./account/transcription";
import { UsageListInstance } from "./account/usage";
import { ValidationRequestListInstance } from "./account/validationRequest";
export type AccountStatus = "active" | "suspended" | "closed";
export type AccountType = "Trial" | "Full";
/**
 * Options to pass to update a AccountInstance
 */
export interface AccountContextUpdateOptions {
    /** Update the human-readable description of this Account */
    friendlyName?: string;
    /**  */
    status?: AccountStatus;
}
/**
 * Options to pass to create a AccountInstance
 */
export interface AccountListInstanceCreateOptions {
    /** A human readable description of the account to create, defaults to `SubAccount Created at {YYYY-MM-DD HH:MM meridian}` */
    friendlyName?: string;
}
/**
 * Options to pass to each
 */
export interface AccountListInstanceEachOptions {
    /** Only return the Account resources with friendly names that exactly match this name. */
    friendlyName?: string;
    /** Only return Account resources with the given status. Can be `closed`, `suspended` or `active`. */
    status?: AccountStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AccountInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AccountListInstanceOptions {
    /** Only return the Account resources with friendly names that exactly match this name. */
    friendlyName?: string;
    /** Only return Account resources with the given status. Can be `closed`, `suspended` or `active`. */
    status?: AccountStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AccountListInstancePageOptions {
    /** Only return the Account resources with friendly names that exactly match this name. */
    friendlyName?: string;
    /** Only return Account resources with the given status. Can be `closed`, `suspended` or `active`. */
    status?: AccountStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AccountContext {
    addresses: AddressListInstance;
    applications: ApplicationListInstance;
    authorizedConnectApps: AuthorizedConnectAppListInstance;
    availablePhoneNumbers: AvailablePhoneNumberCountryListInstance;
    balance: BalanceListInstance;
    calls: CallListInstance;
    conferences: ConferenceListInstance;
    connectApps: ConnectAppListInstance;
    incomingPhoneNumbers: IncomingPhoneNumberListInstance;
    keys: KeyListInstance;
    messages: MessageListInstance;
    newKeys: NewKeyListInstance;
    newSigningKeys: NewSigningKeyListInstance;
    notifications: NotificationListInstance;
    outgoingCallerIds: OutgoingCallerIdListInstance;
    queues: QueueListInstance;
    recordings: RecordingListInstance;
    shortCodes: ShortCodeListInstance;
    signingKeys: SigningKeyListInstance;
    sip: SipListInstance;
    tokens: TokenListInstance;
    transcriptions: TranscriptionListInstance;
    usage: UsageListInstance;
    validationRequests: ValidationRequestListInstance;
    /**
     * Fetch a AccountInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountInstance
     */
    fetch(callback?: (error: Error | null, item?: AccountInstance) => any): Promise<AccountInstance>;
    /**
     * Update a AccountInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountInstance
     */
    update(callback?: (error: Error | null, item?: AccountInstance) => any): Promise<AccountInstance>;
    /**
     * Update a AccountInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountInstance
     */
    update(params: AccountContextUpdateOptions, callback?: (error: Error | null, item?: AccountInstance) => any): Promise<AccountInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AccountContextSolution {
    sid: string;
}
export declare class AccountContextImpl implements AccountContext {
    protected _version: V2010;
    protected _solution: AccountContextSolution;
    protected _uri: string;
    protected _addresses?: AddressListInstance;
    protected _applications?: ApplicationListInstance;
    protected _authorizedConnectApps?: AuthorizedConnectAppListInstance;
    protected _availablePhoneNumbers?: AvailablePhoneNumberCountryListInstance;
    protected _balance?: BalanceListInstance;
    protected _calls?: CallListInstance;
    protected _conferences?: ConferenceListInstance;
    protected _connectApps?: ConnectAppListInstance;
    protected _incomingPhoneNumbers?: IncomingPhoneNumberListInstance;
    protected _keys?: KeyListInstance;
    protected _messages?: MessageListInstance;
    protected _newKeys?: NewKeyListInstance;
    protected _newSigningKeys?: NewSigningKeyListInstance;
    protected _notifications?: NotificationListInstance;
    protected _outgoingCallerIds?: OutgoingCallerIdListInstance;
    protected _queues?: QueueListInstance;
    protected _recordings?: RecordingListInstance;
    protected _shortCodes?: ShortCodeListInstance;
    protected _signingKeys?: SigningKeyListInstance;
    protected _sip?: SipListInstance;
    protected _tokens?: TokenListInstance;
    protected _transcriptions?: TranscriptionListInstance;
    protected _usage?: UsageListInstance;
    protected _validationRequests?: ValidationRequestListInstance;
    constructor(_version: V2010, sid: string);
    get addresses(): AddressListInstance;
    get applications(): ApplicationListInstance;
    get authorizedConnectApps(): AuthorizedConnectAppListInstance;
    get availablePhoneNumbers(): AvailablePhoneNumberCountryListInstance;
    get balance(): BalanceListInstance;
    get calls(): CallListInstance;
    get conferences(): ConferenceListInstance;
    get connectApps(): ConnectAppListInstance;
    get incomingPhoneNumbers(): IncomingPhoneNumberListInstance;
    get keys(): KeyListInstance;
    get messages(): MessageListInstance;
    get newKeys(): NewKeyListInstance;
    get newSigningKeys(): NewSigningKeyListInstance;
    get notifications(): NotificationListInstance;
    get outgoingCallerIds(): OutgoingCallerIdListInstance;
    get queues(): QueueListInstance;
    get recordings(): RecordingListInstance;
    get shortCodes(): ShortCodeListInstance;
    get signingKeys(): SigningKeyListInstance;
    get sip(): SipListInstance;
    get tokens(): TokenListInstance;
    get transcriptions(): TranscriptionListInstance;
    get usage(): UsageListInstance;
    get validationRequests(): ValidationRequestListInstance;
    fetch(callback?: (error: Error | null, item?: AccountInstance) => any): Promise<AccountInstance>;
    update(params?: AccountContextUpdateOptions | ((error: Error | null, item?: AccountInstance) => any), callback?: (error: Error | null, item?: AccountInstance) => any): Promise<AccountInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AccountContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AccountPayload extends TwilioResponsePayload {
    accounts: AccountResource[];
}
interface AccountResource {
    auth_token: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    owner_account_sid: string;
    sid: string;
    status: AccountStatus;
    subresource_uris: Record<string, string>;
    type: AccountType;
    uri: string;
}
export declare class AccountInstance {
    protected _version: V2010;
    protected _solution: AccountContextSolution;
    protected _context?: AccountContext;
    constructor(_version: V2010, payload: AccountResource, sid?: string);
    /**
     * The authorization token for this account. This token should be kept a secret, so no sharing.
     */
    authToken: string;
    /**
     * The date that this account was created, in GMT in RFC 2822 format
     */
    dateCreated: Date;
    /**
     * The date that this account was last updated, in GMT in RFC 2822 format.
     */
    dateUpdated: Date;
    /**
     * A human readable description of this account, up to 64 characters long. By default the FriendlyName is your email address.
     */
    friendlyName: string;
    /**
     * The unique 34 character id that represents the parent of this account. The OwnerAccountSid of a parent account is it\'s own sid.
     */
    ownerAccountSid: string;
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    status: AccountStatus;
    /**
     * A Map of various subresources available for the given Account Instance
     */
    subresourceUris: Record<string, string>;
    type: AccountType;
    /**
     * The URI for this resource, relative to `https://api.twilio.com`
     */
    uri: string;
    private get _proxy();
    /**
     * Fetch a AccountInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountInstance
     */
    fetch(callback?: (error: Error | null, item?: AccountInstance) => any): Promise<AccountInstance>;
    /**
     * Update a AccountInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountInstance
     */
    update(callback?: (error: Error | null, item?: AccountInstance) => any): Promise<AccountInstance>;
    /**
     * Update a AccountInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountInstance
     */
    update(params: AccountContextUpdateOptions, callback?: (error: Error | null, item?: AccountInstance) => any): Promise<AccountInstance>;
    /**
     * Access the addresses.
     */
    addresses(): AddressListInstance;
    /**
     * Access the applications.
     */
    applications(): ApplicationListInstance;
    /**
     * Access the authorizedConnectApps.
     */
    authorizedConnectApps(): AuthorizedConnectAppListInstance;
    /**
     * Access the availablePhoneNumbers.
     */
    availablePhoneNumbers(): AvailablePhoneNumberCountryListInstance;
    /**
     * Access the balance.
     */
    balance(): BalanceListInstance;
    /**
     * Access the calls.
     */
    calls(): CallListInstance;
    /**
     * Access the conferences.
     */
    conferences(): ConferenceListInstance;
    /**
     * Access the connectApps.
     */
    connectApps(): ConnectAppListInstance;
    /**
     * Access the incomingPhoneNumbers.
     */
    incomingPhoneNumbers(): IncomingPhoneNumberListInstance;
    /**
     * Access the keys.
     */
    keys(): KeyListInstance;
    /**
     * Access the messages.
     */
    messages(): MessageListInstance;
    /**
     * Access the newKeys.
     */
    newKeys(): NewKeyListInstance;
    /**
     * Access the newSigningKeys.
     */
    newSigningKeys(): NewSigningKeyListInstance;
    /**
     * Access the notifications.
     */
    notifications(): NotificationListInstance;
    /**
     * Access the outgoingCallerIds.
     */
    outgoingCallerIds(): OutgoingCallerIdListInstance;
    /**
     * Access the queues.
     */
    queues(): QueueListInstance;
    /**
     * Access the recordings.
     */
    recordings(): RecordingListInstance;
    /**
     * Access the shortCodes.
     */
    shortCodes(): ShortCodeListInstance;
    /**
     * Access the signingKeys.
     */
    signingKeys(): SigningKeyListInstance;
    /**
     * Access the sip.
     */
    sip(): SipListInstance;
    /**
     * Access the tokens.
     */
    tokens(): TokenListInstance;
    /**
     * Access the transcriptions.
     */
    transcriptions(): TranscriptionListInstance;
    /**
     * Access the usage.
     */
    usage(): UsageListInstance;
    /**
     * Access the validationRequests.
     */
    validationRequests(): ValidationRequestListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        authToken: string;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        ownerAccountSid: string;
        sid: string;
        status: AccountStatus;
        subresourceUris: Record<string, string>;
        type: AccountType;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AccountSolution {
}
export interface AccountListInstance {
    _version: V2010;
    _solution: AccountSolution;
    _uri: string;
    (sid: string): AccountContext;
    get(sid: string): AccountContext;
    /**
     * Create a AccountInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountInstance
     */
    create(callback?: (error: Error | null, item?: AccountInstance) => any): Promise<AccountInstance>;
    /**
     * Create a AccountInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountInstance
     */
    create(params: AccountListInstanceCreateOptions, callback?: (error: Error | null, item?: AccountInstance) => any): Promise<AccountInstance>;
    /**
     * Streams AccountInstance records from the API.
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
     * @param { AccountListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AccountInstance, done: (err?: Error) => void) => void): void;
    each(params: AccountListInstanceEachOptions, callback?: (item: AccountInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AccountInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AccountPage) => any): Promise<AccountPage>;
    /**
     * Lists AccountInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AccountListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AccountInstance[]) => any): Promise<AccountInstance[]>;
    list(params: AccountListInstanceOptions, callback?: (error: Error | null, items: AccountInstance[]) => any): Promise<AccountInstance[]>;
    /**
     * Retrieve a single page of AccountInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AccountListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AccountPage) => any): Promise<AccountPage>;
    page(params: AccountListInstancePageOptions, callback?: (error: Error | null, items: AccountPage) => any): Promise<AccountPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AccountListInstance(version: V2010): AccountListInstance;
export declare class AccountPage extends Page<V2010, AccountPayload, AccountResource, AccountInstance> {
    /**
     * Initialize the AccountPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: AccountSolution);
    /**
     * Build an instance of AccountInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AccountResource): AccountInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
