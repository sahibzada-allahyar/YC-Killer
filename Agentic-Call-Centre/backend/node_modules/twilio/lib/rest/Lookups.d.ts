import { PhoneNumberListInstance } from "./lookups/v1/phoneNumber";
import LookupsBase from "./LookupsBase";
declare class Lookups extends LookupsBase {
    /**
     * @deprecated - Use v1.phoneNumbers instead
     */
    get phoneNumbers(): PhoneNumberListInstance;
}
export = Lookups;
