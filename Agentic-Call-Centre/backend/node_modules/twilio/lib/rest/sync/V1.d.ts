import SyncBase from "../SyncBase";
import Version from "../../base/Version";
import { ServiceListInstance } from "./v1/service";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Sync
     *
     * @param domain - The Twilio (Twilio.Sync) domain
     */
    constructor(domain: SyncBase);
    /** services - { Twilio.Sync.V1.ServiceListInstance } resource */
    protected _services?: ServiceListInstance;
    /** Getter for services resource */
    get services(): ServiceListInstance;
}
