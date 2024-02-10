import { FlowListInstance } from "./studio/v2/flow";
import { FlowValidateListInstance } from "./studio/v2/flowValidate";
import StudioBase from "./StudioBase";
declare class Studio extends StudioBase {
    /**
     * @deprecated - Use v2.flows instead
     */
    get flows(): FlowListInstance;
    /**
     * @deprecated - Use v2.flowValidate instead
     */
    get flowValidate(): FlowValidateListInstance;
}
export = Studio;
