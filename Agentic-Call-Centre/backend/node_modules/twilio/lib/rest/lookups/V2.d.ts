import LookupsBase from "../LookupsBase";
import Version from "../../base/Version";
import { PhoneNumberListInstance } from "./v2/phoneNumber";
export default class V2 extends Version {
    /**
     * Initialize the V2 version of Lookups
     *
     * @param domain - The Twilio (Twilio.Lookups) domain
     */
    constructor(domain: LookupsBase);
    /** phoneNumbers - { Twilio.Lookups.V2.PhoneNumberListInstance } resource */
    protected _phoneNumbers?: PhoneNumberListInstance;
    /** Getter for phoneNumbers resource */
    get phoneNumbers(): PhoneNumberListInstance;
}
