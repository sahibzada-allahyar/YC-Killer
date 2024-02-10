import NotifyBase from "../NotifyBase";
import Version from "../../base/Version";
import { CredentialListInstance } from "./v1/credential";
import { ServiceListInstance } from "./v1/service";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Notify
     *
     * @param domain - The Twilio (Twilio.Notify) domain
     */
    constructor(domain: NotifyBase);
    /** credentials - { Twilio.Notify.V1.CredentialListInstance } resource */
    protected _credentials?: CredentialListInstance;
    /** services - { Twilio.Notify.V1.ServiceListInstance } resource */
    protected _services?: ServiceListInstance;
    /** Getter for credentials resource */
    get credentials(): CredentialListInstance;
    /** Getter for services resource */
    get services(): ServiceListInstance;
}
