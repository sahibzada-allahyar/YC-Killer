import { MessagingListInstance } from "./pricing/v1/messaging";
import { PhoneNumberListInstance } from "./pricing/v1/phoneNumber";
import { VoiceListInstance } from "./pricing/v2/voice";
import { CountryListInstance } from "./pricing/v2/country";
import { NumberListInstance } from "./pricing/v2/number";
import PricingBase from "./PricingBase";
declare class Pricing extends PricingBase {
    /**
     * @deprecated - Use v1.messaging instead
     */
    get messaging(): MessagingListInstance;
    /**
     * @deprecated - Use v1.phoneNumbers instead
     */
    get phoneNumbers(): PhoneNumberListInstance;
    /**
     * @deprecated - Use v2.voice instead
     */
    get voice(): VoiceListInstance;
    /**
     * @deprecated - Use v2.countries instead
     */
    get countries(): CountryListInstance;
    /**
     * @deprecated - Use v2.numbers instead
     */
    get numbers(): NumberListInstance;
}
export = Pricing;
