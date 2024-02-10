import { RegulatoryComplianceListInstance } from "./numbers/v2/regulatoryCompliance";
import NumbersBase from "./NumbersBase";
declare class Numbers extends NumbersBase {
    /**
     * @deprecated - Use v2.regulatoryCompliance instead
     */
    get regulatoryCompliance(): RegulatoryComplianceListInstance;
}
export = Numbers;
