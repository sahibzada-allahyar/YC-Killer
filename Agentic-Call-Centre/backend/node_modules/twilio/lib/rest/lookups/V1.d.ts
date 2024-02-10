import LookupsBase from "../LookupsBase";
import Version from "../../base/Version";
import { PhoneNumberListInstance } from "./v1/phoneNumber";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Lookups
     *
     * @param domain - The Twilio (Twilio.Lookups) domain
     */
    constructor(domain: LookupsBase);
    /** phoneNumbers - { Twilio.Lookups.V1.PhoneNumberListInstance } resource */
    protected _phoneNumbers?: PhoneNumberListInstance;
    /** Getter for phoneNumbers resource */
    get phoneNumbers(): PhoneNumberListInstance;
}
