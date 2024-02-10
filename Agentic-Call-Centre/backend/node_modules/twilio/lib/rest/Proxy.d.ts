import { ServiceListInstance } from "./proxy/v1/service";
import ProxyBase from "./ProxyBase";
declare class Proxy extends ProxyBase {
    /**
     * @deprecated - Use v1.services instead
     */
    get services(): ServiceListInstance;
}
export = Proxy;
