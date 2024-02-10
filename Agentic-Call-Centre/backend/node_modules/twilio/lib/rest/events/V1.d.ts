import EventsBase from "../EventsBase";
import Version from "../../base/Version";
import { EventTypeListInstance } from "./v1/eventType";
import { SchemaListInstance } from "./v1/schema";
import { SinkListInstance } from "./v1/sink";
import { SubscriptionListInstance } from "./v1/subscription";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Events
     *
     * @param domain - The Twilio (Twilio.Events) domain
     */
    constructor(domain: EventsBase);
    /** eventTypes - { Twilio.Events.V1.EventTypeListInstance } resource */
    protected _eventTypes?: EventTypeListInstance;
    /** schemas - { Twilio.Events.V1.SchemaListInstance } resource */
    protected _schemas?: SchemaListInstance;
    /** sinks - { Twilio.Events.V1.SinkListInstance } resource */
    protected _sinks?: SinkListInstance;
    /** subscriptions - { Twilio.Events.V1.SubscriptionListInstance } resource */
    protected _subscriptions?: SubscriptionListInstance;
    /** Getter for eventTypes resource */
    get eventTypes(): EventTypeListInstance;
    /** Getter for schemas resource */
    get schemas(): SchemaListInstance;
    /** Getter for sinks resource */
    get sinks(): SinkListInstance;
    /** Getter for subscriptions resource */
    get subscriptions(): SubscriptionListInstance;
}
