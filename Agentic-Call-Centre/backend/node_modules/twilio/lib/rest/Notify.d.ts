import { CredentialListInstance } from "./notify/v1/credential";
import { ServiceListInstance } from "./notify/v1/service";
import NotifyBase from "./NotifyBase";
declare class Notify extends NotifyBase {
    /**
     * @deprecated - Use v1.credentials instead
     */
    get credentials(): CredentialListInstance;
    /**
     * @deprecated - Use v1.services instead
     */
    get services(): ServiceListInstance;
}
export = Notify;
