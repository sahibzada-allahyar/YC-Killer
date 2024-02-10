import ServerlessBase from "../ServerlessBase";
import Version from "../../base/Version";
import { ServiceListInstance } from "./v1/service";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Serverless
     *
     * @param domain - The Twilio (Twilio.Serverless) domain
     */
    constructor(domain: ServerlessBase);
    /** services - { Twilio.Serverless.V1.ServiceListInstance } resource */
    protected _services?: ServiceListInstance;
    /** Getter for services resource */
    get services(): ServiceListInstance;
}
