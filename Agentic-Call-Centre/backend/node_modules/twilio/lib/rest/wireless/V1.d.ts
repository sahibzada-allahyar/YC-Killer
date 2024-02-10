import WirelessBase from "../WirelessBase";
import Version from "../../base/Version";
import { CommandListInstance } from "./v1/command";
import { RatePlanListInstance } from "./v1/ratePlan";
import { SimListInstance } from "./v1/sim";
import { UsageRecordListInstance } from "./v1/usageRecord";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Wireless
     *
     * @param domain - The Twilio (Twilio.Wireless) domain
     */
    constructor(domain: WirelessBase);
    /** commands - { Twilio.Wireless.V1.CommandListInstance } resource */
    protected _commands?: CommandListInstance;
    /** ratePlans - { Twilio.Wireless.V1.RatePlanListInstance } resource */
    protected _ratePlans?: RatePlanListInstance;
    /** sims - { Twilio.Wireless.V1.SimListInstance } resource */
    protected _sims?: SimListInstance;
    /** usageRecords - { Twilio.Wireless.V1.UsageRecordListInstance } resource */
    protected _usageRecords?: UsageRecordListInstance;
    /** Getter for commands resource */
    get commands(): CommandListInstance;
    /** Getter for ratePlans resource */
    get ratePlans(): RatePlanListInstance;
    /** Getter for sims resource */
    get sims(): SimListInstance;
    /** Getter for usageRecords resource */
    get usageRecords(): UsageRecordListInstance;
}
