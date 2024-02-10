import PreviewBase from "../PreviewBase";
import Version from "../../base/Version";
import { CommandListInstance } from "./wireless/command";
import { RatePlanListInstance } from "./wireless/ratePlan";
import { SimListInstance } from "./wireless/sim";
export default class Wireless extends Version {
    /**
     * Initialize the Wireless version of Preview
     *
     * @param domain - The Twilio (Twilio.Preview) domain
     */
    constructor(domain: PreviewBase);
    /** commands - { Twilio.Preview.Wireless.CommandListInstance } resource */
    protected _commands?: CommandListInstance;
    /** ratePlans - { Twilio.Preview.Wireless.RatePlanListInstance } resource */
    protected _ratePlans?: RatePlanListInstance;
    /** sims - { Twilio.Preview.Wireless.SimListInstance } resource */
    protected _sims?: SimListInstance;
    /** Getter for commands resource */
    get commands(): CommandListInstance;
    /** Getter for ratePlans resource */
    get ratePlans(): RatePlanListInstance;
    /** Getter for sims resource */
    get sims(): SimListInstance;
}
