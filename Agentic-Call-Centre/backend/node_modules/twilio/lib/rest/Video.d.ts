import { CompositionListInstance } from "./video/v1/composition";
import { CompositionHookListInstance } from "./video/v1/compositionHook";
import { CompositionSettingsListInstance } from "./video/v1/compositionSettings";
import { RecordingListInstance } from "./video/v1/recording";
import { RecordingSettingsListInstance } from "./video/v1/recordingSettings";
import { RoomListInstance } from "./video/v1/room";
import VideoBase from "./VideoBase";
declare class Video extends VideoBase {
    /**
     * @deprecated - Use v1.compositions instead
     */
    get compositions(): CompositionListInstance;
    /**
     * @deprecated - Use v1.compositionHooks instead
     */
    get compositionHooks(): CompositionHookListInstance;
    /**
     * @deprecated - Use v1.compositionSettings instead
     */
    get compositionSettings(): CompositionSettingsListInstance;
    /**
     * @deprecated - Use v1.recordings instead
     */
    get recordings(): RecordingListInstance;
    /**
     * @deprecated - Use v1.recordingSettings instead
     */
    get recordingSettings(): RecordingSettingsListInstance;
    /**
     * @deprecated - Use v1.rooms instead
     */
    get rooms(): RoomListInstance;
}
export = Video;
