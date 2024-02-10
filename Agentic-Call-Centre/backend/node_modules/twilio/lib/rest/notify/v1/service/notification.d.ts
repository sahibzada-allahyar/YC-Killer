/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
export type NotificationPriority = "high" | "low";
/**
 * Options to pass to create a NotificationInstance
 */
export interface NotificationListInstanceCreateOptions {
    /** The notification text. For FCM and GCM, translates to `data.twi_body`. For APNS, translates to `aps.alert.body`. For SMS, translates to `body`. SMS requires either this `body` value, or `media_urls` attribute defined in the `sms` parameter of the notification. */
    body?: string;
    /**  */
    priority?: NotificationPriority;
    /** How long, in seconds, the notification is valid. Can be an integer between 0 and 2,419,200, which is 4 weeks, the default and the maximum supported time to live (TTL). Delivery should be attempted if the device is offline until the TTL elapses. Zero means that the notification delivery is attempted immediately, only once, and is not stored for future delivery. SMS does not support this property. */
    ttl?: number;
    /** The notification title. For FCM and GCM, this translates to the `data.twi_title` value. For APNS, this translates to the `aps.alert.title` value. SMS does not support this property. This field is not visible on iOS phones and tablets but appears on Apple Watch and Android devices. */
    title?: string;
    /** The name of the sound to be played for the notification. For FCM and GCM, this Translates to `data.twi_sound`.  For APNS, this translates to `aps.sound`.  SMS does not support this property. */
    sound?: string;
    /** The actions to display for the notification. For APNS, translates to the `aps.category` value. For GCM, translates to the `data.twi_action` value. For SMS, this parameter is not supported and is omitted from deliveries to those channels. */
    action?: string;
    /** The custom key-value pairs of the notification\\\'s payload. For FCM and GCM, this value translates to `data` in the FCM and GCM payloads. FCM and GCM [reserve certain keys](https://firebase.google.com/docs/cloud-messaging/http-server-ref) that cannot be used in those channels. For APNS, attributes of `data` are inserted into the APNS payload as custom properties outside of the `aps` dictionary. In all channels, we reserve keys that start with `twi_` for future use. Custom keys that start with `twi_` are not allowed and are rejected as 400 Bad request with no delivery attempted. For SMS, this parameter is not supported and is omitted from deliveries to those channels. */
    data?: any;
    /** The APNS-specific payload that overrides corresponding attributes in the generic payload for APNS Bindings. This property maps to the APNS `Payload` item, therefore the `aps` key must be used to change standard attributes. Adds custom key-value pairs to the root of the dictionary. See the [APNS documentation](https://developer.apple.com/library/content/documentation/NetworkingInternet/Conceptual/RemoteNotificationsPG/CommunicatingwithAPNs.html) for more details. We reserve keys that start with `twi_` for future use. Custom keys that start with `twi_` are not allowed. */
    apn?: any;
    /** The GCM-specific payload that overrides corresponding attributes in the generic payload for GCM Bindings.  This property maps to the root JSON dictionary. See the [GCM documentation](https://firebase.google.com/docs/cloud-messaging/http-server-ref) for more details. Target parameters `to`, `registration_ids`, and `notification_key` are not allowed. We reserve keys that start with `twi_` for future use. Custom keys that start with `twi_` are not allowed. GCM also [reserves certain keys](https://firebase.google.com/docs/cloud-messaging/http-server-ref). */
    gcm?: any;
    /** The SMS-specific payload that overrides corresponding attributes in the generic payload for SMS Bindings.  Each attribute in this value maps to the corresponding `form` parameter of the Twilio [Message](https://www.twilio.com/docs/sms/quickstart) resource.  These parameters of the Message resource are supported in snake case format: `body`, `media_urls`, `status_callback`, and `max_price`.  The `status_callback` parameter overrides the corresponding parameter in the messaging service, if configured. The `media_urls` property expects a JSON array. */
    sms?: any;
    /** Deprecated. */
    facebookMessenger?: any;
    /** The FCM-specific payload that overrides corresponding attributes in the generic payload for FCM Bindings. This property maps to the root JSON dictionary. See the [FCM documentation](https://firebase.google.com/docs/cloud-messaging/http-server-ref#downstream) for more details. Target parameters `to`, `registration_ids`, `condition`, and `notification_key` are not allowed in this parameter. We reserve keys that start with `twi_` for future use. Custom keys that start with `twi_` are not allowed. FCM also [reserves certain keys](https://firebase.google.com/docs/cloud-messaging/http-server-ref), which cannot be used in that channel. */
    fcm?: any;
    /** The Segment resource is deprecated. Use the `tag` parameter, instead. */
    segment?: Array<string>;
    /** Deprecated. */
    alexa?: any;
    /** The destination address specified as a JSON string.  Multiple `to_binding` parameters can be included but the total size of the request entity should not exceed 1MB. This is typically sufficient for 10,000 phone numbers. */
    toBinding?: Array<string>;
    /** URL to send webhooks. */
    deliveryCallbackUrl?: string;
    /** The `identity` value that uniquely identifies the new resource\\\'s [User](https://www.twilio.com/docs/chat/rest/user-resource) within the [Service](https://www.twilio.com/docs/notify/api/service-resource). Delivery will be attempted only to Bindings with an Identity in this list. No more than 20 items are allowed in this list. */
    identity?: Array<string>;
    /** A tag that selects the Bindings to notify. Repeat this parameter to specify more than one tag, up to a total of 5 tags. The implicit tag `all` is available to notify all Bindings in a Service instance. Similarly, the implicit tags `apn`, `fcm`, `gcm`, `sms` and `facebook-messenger` are available to notify all Bindings in a specific channel. */
    tag?: Array<string>;
}
export interface NotificationSolution {
    serviceSid: string;
}
export interface NotificationListInstance {
    _version: V1;
    _solution: NotificationSolution;
    _uri: string;
    /**
     * Create a NotificationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NotificationInstance
     */
    create(callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Create a NotificationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NotificationInstance
     */
    create(params: NotificationListInstanceCreateOptions, callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function NotificationListInstance(version: V1, serviceSid: string): NotificationListInstance;
interface NotificationResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    date_created: Date;
    identities: Array<string>;
    tags: Array<string>;
    segments: Array<string>;
    priority: NotificationPriority;
    ttl: number;
    title: string;
    body: string;
    sound: string;
    action: string;
    data: any;
    apn: any;
    gcm: any;
    fcm: any;
    sms: any;
    facebook_messenger: any;
    alexa: any;
}
export declare class NotificationInstance {
    protected _version: V1;
    constructor(_version: V1, payload: NotificationResource, serviceSid: string);
    /**
     * The unique string that we created to identify the Notification resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Notification resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/notify/api/service-resource) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The list of `identity` values of the Users to notify. We will attempt to deliver notifications only to Bindings with an identity in this list.
     */
    identities: Array<string>;
    /**
     * The tags that select the Bindings to notify. Notifications will be attempted only to Bindings that have all of the tags listed in this property.
     */
    tags: Array<string>;
    /**
     * The list of Segments to notify. The [Segment](https://www.twilio.com/docs/notify/api/segment-resource) resource is deprecated. Use the `tags` property, instead.
     */
    segments: Array<string>;
    priority: NotificationPriority;
    /**
     * How long, in seconds, the notification is valid. Can be an integer between 0 and 2,419,200, which is 4 weeks, the default and the maximum supported time to live (TTL). Delivery should be attempted if the device is offline until the TTL elapses. Zero means that the notification delivery is attempted immediately, only once, and is not stored for future delivery. SMS does not support this property.
     */
    ttl: number;
    /**
     * The notification title. For FCM and GCM, this translates to the `data.twi_title` value. For APNS, this translates to the `aps.alert.title` value. SMS does not support this property. This field is not visible on iOS phones and tablets but appears on Apple Watch and Android devices.
     */
    title: string;
    /**
     * The notification text. For FCM and GCM, translates to `data.twi_body`. For APNS, translates to `aps.alert.body`. For SMS, translates to `body`. SMS requires either this `body` value, or `media_urls` attribute defined in the `sms` parameter of the notification.
     */
    body: string;
    /**
     * The name of the sound to be played for the notification. For FCM and GCM, this Translates to `data.twi_sound`.  For APNS, this translates to `aps.sound`.  SMS does not support this property.
     */
    sound: string;
    /**
     * The actions to display for the notification. For APNS, translates to the `aps.category` value. For GCM, translates to the `data.twi_action` value. For SMS, this parameter is not supported and is omitted from deliveries to those channels.
     */
    action: string;
    /**
     * The custom key-value pairs of the notification\'s payload. For FCM and GCM, this value translates to `data` in the FCM and GCM payloads. FCM and GCM [reserve certain keys](https://firebase.google.com/docs/cloud-messaging/http-server-ref) that cannot be used in those channels. For APNS, attributes of `data` are inserted into the APNS payload as custom properties outside of the `aps` dictionary. In all channels, we reserve keys that start with `twi_` for future use. Custom keys that start with `twi_` are not allowed and are rejected as 400 Bad request with no delivery attempted. For SMS, this parameter is not supported and is omitted from deliveries to those channels.
     */
    data: any;
    /**
     * The APNS-specific payload that overrides corresponding attributes in the generic payload for APNS Bindings. This property maps to the APNS `Payload` item, therefore the `aps` key must be used to change standard attributes. Adds custom key-value pairs to the root of the dictionary. See the [APNS documentation](https://developer.apple.com/library/content/documentation/NetworkingInternet/Conceptual/RemoteNotificationsPG/CommunicatingwithAPNs.html) for more details. We reserve keys that start with `twi_` for future use. Custom keys that start with `twi_` are not allowed.
     */
    apn: any;
    /**
     * The GCM-specific payload that overrides corresponding attributes in the generic payload for GCM Bindings.  This property maps to the root JSON dictionary. Target parameters `to`, `registration_ids`, and `notification_key` are not allowed. We reserve keys that start with `twi_` for future use. Custom keys that start with `twi_` are not allowed.
     */
    gcm: any;
    /**
     * The FCM-specific payload that overrides corresponding attributes in the generic payload for FCM Bindings. This property maps to the root JSON dictionary. See the [FCM documentation](https://firebase.google.com/docs/cloud-messaging/http-server-ref#downstream) for more details. Target parameters `to`, `registration_ids`, `condition`, and `notification_key` are not allowed in this parameter. We reserve keys that start with `twi_` for future use. Custom keys that start with `twi_` are not allowed. FCM also [reserves certain keys](https://firebase.google.com/docs/cloud-messaging/http-server-ref), which cannot be used in that channel.
     */
    fcm: any;
    /**
     * The SMS-specific payload that overrides corresponding attributes in the generic payload for SMS Bindings.  Each attribute in this value maps to the corresponding `form` parameter of the Twilio [Message](https://www.twilio.com/docs/sms/api/message-resource) resource.  These parameters of the Message resource are supported in snake case format: `body`, `media_urls`, `status_callback`, and `max_price`.  The `status_callback` parameter overrides the corresponding parameter in the messaging service, if configured. The `media_urls` property expects a JSON array.
     */
    sms: any;
    /**
     * Deprecated.
     */
    facebookMessenger: any;
    /**
     * Deprecated.
     */
    alexa: any;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        dateCreated: Date;
        identities: string[];
        tags: string[];
        segments: string[];
        priority: NotificationPriority;
        ttl: number;
        title: string;
        body: string;
        sound: string;
        action: string;
        data: any;
        apn: any;
        gcm: any;
        fcm: any;
        sms: any;
        facebookMessenger: any;
        alexa: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
