import VideoBase from "../VideoBase";
import Version from "../../base/Version";
import { CompositionListInstance } from "./v1/composition";
import { CompositionHookListInstance } from "./v1/compositionHook";
import { CompositionSettingsListInstance } from "./v1/compositionSettings";
import { RecordingListInstance } from "./v1/recording";
import { RecordingSettingsListInstance } from "./v1/recordingSettings";
import { RoomListInstance } from "./v1/room";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Video
     *
     * @param domain - The Twilio (Twilio.Video) domain
     */
    constructor(domain: VideoBase);
    /** compositions - { Twilio.Video.V1.CompositionListInstance } resource */
    protected _compositions?: CompositionListInstance;
    /** compositionHooks - { Twilio.Video.V1.CompositionHookListInstance } resource */
    protected _compositionHooks?: CompositionHookListInstance;
    /** compositionSettings - { Twilio.Video.V1.CompositionSettingsListInstance } resource */
    protected _compositionSettings?: CompositionSettingsListInstance;
    /** recordings - { Twilio.Video.V1.RecordingListInstance } resource */
    protected _recordings?: RecordingListInstance;
    /** recordingSettings - { Twilio.Video.V1.RecordingSettingsListInstance } resource */
    protected _recordingSettings?: RecordingSettingsListInstance;
    /** rooms - { Twilio.Video.V1.RoomListInstance } resource */
    protected _rooms?: RoomListInstance;
    /** Getter for compositions resource */
    get compositions(): CompositionListInstance;
    /** Getter for compositionHooks resource */
    get compositionHooks(): CompositionHookListInstance;
    /** Getter for compositionSettings resource */
    get compositionSettings(): CompositionSettingsListInstance;
    /** Getter for recordings resource */
    get recordings(): RecordingListInstance;
    /** Getter for recordingSettings resource */
    get recordingSettings(): RecordingSettingsListInstance;
    /** Getter for rooms resource */
    get rooms(): RoomListInstance;
}
