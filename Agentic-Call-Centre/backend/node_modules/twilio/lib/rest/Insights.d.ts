import { CallListInstance } from "./insights/v1/call";
import { CallSummariesListInstance } from "./insights/v1/callSummaries";
import { ConferenceListInstance } from "./insights/v1/conference";
import { RoomListInstance } from "./insights/v1/room";
import { SettingListInstance } from "./insights/v1/setting";
import InsightsBase from "./InsightsBase";
declare class Insights extends InsightsBase {
    /**
     * @deprecated - Use v1.settings instead
     */
    get settings(): SettingListInstance;
    /**
     * @deprecated - Use v1.calls instead
     */
    get calls(): CallListInstance;
    /**
     * @deprecated - Use v1.callSummaries instead
     */
    get callSummaries(): CallSummariesListInstance;
    /**
     * @deprecated - Use v1.conferences instead
     */
    get conferences(): ConferenceListInstance;
    /**
     * @deprecated - Use v1.rooms instead
     */
    get rooms(): RoomListInstance;
}
export = Insights;
