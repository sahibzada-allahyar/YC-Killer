import InsightsBase from "../InsightsBase";
import Version from "../../base/Version";
import { CallListInstance } from "./v1/call";
import { CallSummariesListInstance } from "./v1/callSummaries";
import { ConferenceListInstance } from "./v1/conference";
import { RoomListInstance } from "./v1/room";
import { SettingListInstance } from "./v1/setting";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Insights
     *
     * @param domain - The Twilio (Twilio.Insights) domain
     */
    constructor(domain: InsightsBase);
    /** calls - { Twilio.Insights.V1.CallListInstance } resource */
    protected _calls?: CallListInstance;
    /** callSummaries - { Twilio.Insights.V1.CallSummariesListInstance } resource */
    protected _callSummaries?: CallSummariesListInstance;
    /** conferences - { Twilio.Insights.V1.ConferenceListInstance } resource */
    protected _conferences?: ConferenceListInstance;
    /** rooms - { Twilio.Insights.V1.RoomListInstance } resource */
    protected _rooms?: RoomListInstance;
    /** settings - { Twilio.Insights.V1.SettingListInstance } resource */
    protected _settings?: SettingListInstance;
    /** Getter for calls resource */
    get calls(): CallListInstance;
    /** Getter for callSummaries resource */
    get callSummaries(): CallSummariesListInstance;
    /** Getter for conferences resource */
    get conferences(): ConferenceListInstance;
    /** Getter for rooms resource */
    get rooms(): RoomListInstance;
    /** Getter for settings resource */
    get settings(): SettingListInstance;
}
