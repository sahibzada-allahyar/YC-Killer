import IpMessagingBase from "../IpMessagingBase";
import Version from "../../base/Version";
import { CredentialListInstance } from "./v2/credential";
import { ServiceListInstance } from "./v2/service";
export default class V2 extends Version {
    /**
     * Initialize the V2 version of IpMessaging
     *
     * @param domain - The Twilio (Twilio.IpMessaging) domain
     */
    constructor(domain: IpMessagingBase);
    /** credentials - { Twilio.IpMessaging.V2.CredentialListInstance } resource */
    protected _credentials?: CredentialListInstance;
    /** services - { Twilio.IpMessaging.V2.ServiceListInstance } resource */
    protected _services?: ServiceListInstance;
    /** Getter for credentials resource */
    get credentials(): CredentialListInstance;
    /** Getter for services resource */
    get services(): ServiceListInstance;
}
