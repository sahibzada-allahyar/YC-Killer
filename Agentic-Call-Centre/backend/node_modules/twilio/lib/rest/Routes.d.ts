import { PhoneNumberListInstance } from "./routes/v2/phoneNumber";
import { SipDomainListInstance } from "./routes/v2/sipDomain";
import { TrunkListInstance } from "./routes/v2/trunk";
import RoutesBase from "./RoutesBase";
declare class Routes extends RoutesBase {
    /**
     * @deprecated - Use v1.phoneNumbers instead
     */
    get phoneNumbers(): PhoneNumberListInstance;
    /**
     * @deprecated - Use v1.sipDomains instead
     */
    get sipDomains(): SipDomainListInstance;
    /**
     * @deprecated - Use v1.trunks instead
     */
    get trunks(): TrunkListInstance;
}
export = Routes;
