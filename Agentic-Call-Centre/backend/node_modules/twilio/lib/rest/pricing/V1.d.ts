import PricingBase from "../PricingBase";
import Version from "../../base/Version";
import { MessagingListInstance } from "./v1/messaging";
import { PhoneNumberListInstance } from "./v1/phoneNumber";
import { VoiceListInstance } from "./v1/voice";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Pricing
     *
     * @param domain - The Twilio (Twilio.Pricing) domain
     */
    constructor(domain: PricingBase);
    /** messaging - { Twilio.Pricing.V1.MessagingListInstance } resource */
    protected _messaging?: MessagingListInstance;
    /** phoneNumbers - { Twilio.Pricing.V1.PhoneNumberListInstance } resource */
    protected _phoneNumbers?: PhoneNumberListInstance;
    /** voice - { Twilio.Pricing.V1.VoiceListInstance } resource */
    protected _voice?: VoiceListInstance;
    /** Getter for messaging resource */
    get messaging(): MessagingListInstance;
    /** Getter for phoneNumbers resource */
    get phoneNumbers(): PhoneNumberListInstance;
    /** Getter for voice resource */
    get voice(): VoiceListInstance;
}
