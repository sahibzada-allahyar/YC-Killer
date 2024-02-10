import { TrunkListInstance } from "./trunking/v1/trunk";
import TrunkingBase from "./TrunkingBase";
declare class Trunking extends TrunkingBase {
    /**
     * @deprecated - Use v1.trunks instead
     */
    get trunks(): TrunkListInstance;
}
export = Trunking;
