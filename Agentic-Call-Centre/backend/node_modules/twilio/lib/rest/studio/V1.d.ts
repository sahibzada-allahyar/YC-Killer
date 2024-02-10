import StudioBase from "../StudioBase";
import Version from "../../base/Version";
import { FlowListInstance } from "./v1/flow";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Studio
     *
     * @param domain - The Twilio (Twilio.Studio) domain
     */
    constructor(domain: StudioBase);
    /** flows - { Twilio.Studio.V1.FlowListInstance } resource */
    protected _flows?: FlowListInstance;
    /** Getter for flows resource */
    get flows(): FlowListInstance;
}
