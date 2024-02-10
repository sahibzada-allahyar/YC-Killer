import PreviewBase from "../PreviewBase";
import Version from "../../base/Version";
import { ServiceListInstance } from "./sync/service";
export default class Sync extends Version {
    /**
     * Initialize the Sync version of Preview
     *
     * @param domain - The Twilio (Twilio.Preview) domain
     */
    constructor(domain: PreviewBase);
    /** services - { Twilio.Preview.Sync.ServiceListInstance } resource */
    protected _services?: ServiceListInstance;
    /** Getter for services resource */
    get services(): ServiceListInstance;
}
