import PricingBase from "../PricingBase";
import Version from "../../base/Version";
import { CountryListInstance } from "./v2/country";
import { NumberListInstance } from "./v2/number";
import { VoiceListInstance } from "./v2/voice";
export default class V2 extends Version {
    /**
     * Initialize the V2 version of Pricing
     *
     * @param domain - The Twilio (Twilio.Pricing) domain
     */
    constructor(domain: PricingBase);
    /** countries - { Twilio.Pricing.V2.CountryListInstance } resource */
    protected _countries?: CountryListInstance;
    /** numbers - { Twilio.Pricing.V2.NumberListInstance } resource */
    protected _numbers?: NumberListInstance;
    /** voice - { Twilio.Pricing.V2.VoiceListInstance } resource */
    protected _voice?: VoiceListInstance;
    /** Getter for countries resource */
    get countries(): CountryListInstance;
    /** Getter for numbers resource */
    get numbers(): NumberListInstance;
    /** Getter for voice resource */
    get voice(): VoiceListInstance;
}
