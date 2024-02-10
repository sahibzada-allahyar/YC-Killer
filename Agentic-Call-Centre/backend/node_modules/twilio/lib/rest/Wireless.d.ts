import { CommandListInstance } from "./wireless/v1/command";
import { RatePlanListInstance } from "./wireless/v1/ratePlan";
import { SimListInstance } from "./wireless/v1/sim";
import { UsageRecordListInstance } from "./wireless/v1/usageRecord";
import WirelessBase from "./WirelessBase";
declare class Wireless extends WirelessBase {
    /**
     * @deprecated - Use v1.usageRecords instead
     */
    get usageRecords(): UsageRecordListInstance;
    /**
     * @deprecated - Use v1.commands instead
     */
    get commands(): CommandListInstance;
    /**
     * @deprecated - Use v1.ratePlans instead
     */
    get ratePlans(): RatePlanListInstance;
    /**
     * @deprecated - Use v1.sims instead
     */
    get sims(): SimListInstance;
}
export = Wireless;
