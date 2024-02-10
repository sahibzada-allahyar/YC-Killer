import RoutesBase from "../RoutesBase";
import Version from "../../base/Version";
import { PhoneNumberListInstance } from "./v2/phoneNumber";
import { SipDomainListInstance } from "./v2/sipDomain";
import { TrunkListInstance } from "./v2/trunk";
export default class V2 extends Version {
    /**
     * Initialize the V2 version of Routes
     *
     * @param domain - The Twilio (Twilio.Routes) domain
     */
    constructor(domain: RoutesBase);
    /** phoneNumbers - { Twilio.Routes.V2.PhoneNumberListInstance } resource */
    protected _phoneNumbers?: PhoneNumberListInstance;
    /** sipDomains - { Twilio.Routes.V2.SipDomainListInstance } resource */
    protected _sipDomains?: SipDomainListInstance;
    /** trunks - { Twilio.Routes.V2.TrunkListInstance } resource */
    protected _trunks?: TrunkListInstance;
    /** Getter for phoneNumbers resource */
    get phoneNumbers(): PhoneNumberListInstance;
    /** Getter for sipDomains resource */
    get sipDomains(): SipDomainListInstance;
    /** Getter for trunks resource */
    get trunks(): TrunkListInstance;
}
