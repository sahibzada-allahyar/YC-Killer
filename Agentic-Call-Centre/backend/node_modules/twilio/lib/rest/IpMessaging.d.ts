import { CredentialListInstance } from "./ipMessaging/v2/credential";
import { ServiceListInstance } from "./ipMessaging/v2/service";
import IpMessagingBase from "./IpMessagingBase";
declare class IpMessaging extends IpMessagingBase {
    /**
     * @deprecated - Use v2.credentials instead
     */
    get credentials(): CredentialListInstance;
    /**
     * @deprecated - Use v2.services instead
     */
    get services(): ServiceListInstance;
}
export = IpMessaging;
