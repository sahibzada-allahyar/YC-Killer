import MonitorBase from "../MonitorBase";
import Version from "../../base/Version";
import { AlertListInstance } from "./v1/alert";
import { EventListInstance } from "./v1/event";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Monitor
     *
     * @param domain - The Twilio (Twilio.Monitor) domain
     */
    constructor(domain: MonitorBase);
    /** alerts - { Twilio.Monitor.V1.AlertListInstance } resource */
    protected _alerts?: AlertListInstance;
    /** events - { Twilio.Monitor.V1.EventListInstance } resource */
    protected _events?: EventListInstance;
    /** Getter for alerts resource */
    get alerts(): AlertListInstance;
    /** Getter for events resource */
    get events(): EventListInstance;
}
