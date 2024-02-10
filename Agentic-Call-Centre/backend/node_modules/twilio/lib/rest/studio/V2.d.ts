import StudioBase from "../StudioBase";
import Version from "../../base/Version";
import { FlowListInstance } from "./v2/flow";
import { FlowValidateListInstance } from "./v2/flowValidate";
export default class V2 extends Version {
    /**
     * Initialize the V2 version of Studio
     *
     * @param domain - The Twilio (Twilio.Studio) domain
     */
    constructor(domain: StudioBase);
    /** flows - { Twilio.Studio.V2.FlowListInstance } resource */
    protected _flows?: FlowListInstance;
    /** flowValidate - { Twilio.Studio.V2.FlowValidateListInstance } resource */
    protected _flowValidate?: FlowValidateListInstance;
    /** Getter for flows resource */
    get flows(): FlowListInstance;
    /** Getter for flowValidate resource */
    get flowValidate(): FlowValidateListInstance;
}
