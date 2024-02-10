import IpMessagingBase from "../IpMessagingBase";
import Version from "../../base/Version";
import { CredentialListInstance } from "./v1/credential";
import { ServiceListInstance } from "./v1/service";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of IpMessaging
     *
     * @param domain - The Twilio (Twilio.IpMessaging) domain
     */
    constructor(domain: IpMessagingBase);
    /** credentials - { Twilio.IpMessaging.V1.CredentialListInstance } resource */
    protected _credentials?: CredentialListInstance;
    /** services - { Twilio.IpMessaging.V1.ServiceListInstance } resource */
    protected _services?: ServiceListInstance;
    /** Getter for credentials resource */
    get credentials(): CredentialListInstance;
    /** Getter for services resource */
    get services(): ServiceListInstance;
}
