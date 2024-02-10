import TrunkingBase from "../TrunkingBase";
import Version from "../../base/Version";
import { TrunkListInstance } from "./v1/trunk";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Trunking
     *
     * @param domain - The Twilio (Twilio.Trunking) domain
     */
    constructor(domain: TrunkingBase);
    /** trunks - { Twilio.Trunking.V1.TrunkListInstance } resource */
    protected _trunks?: TrunkListInstance;
    /** Getter for trunks resource */
    get trunks(): TrunkListInstance;
}
