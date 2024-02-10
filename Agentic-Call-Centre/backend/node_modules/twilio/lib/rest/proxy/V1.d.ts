import ProxyBase from "../ProxyBase";
import Version from "../../base/Version";
import { ServiceListInstance } from "./v1/service";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Proxy
     *
     * @param domain - The Twilio (Twilio.Proxy) domain
     */
    constructor(domain: ProxyBase);
    /** services - { Twilio.Proxy.V1.ServiceListInstance } resource */
    protected _services?: ServiceListInstance;
    /** Getter for services resource */
    get services(): ServiceListInstance;
}
