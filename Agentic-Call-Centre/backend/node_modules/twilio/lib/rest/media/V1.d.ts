import MediaBase from "../MediaBase";
import Version from "../../base/Version";
import { MediaProcessorListInstance } from "./v1/mediaProcessor";
import { MediaRecordingListInstance } from "./v1/mediaRecording";
import { PlayerStreamerListInstance } from "./v1/playerStreamer";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Media
     *
     * @param domain - The Twilio (Twilio.Media) domain
     */
    constructor(domain: MediaBase);
    /** mediaProcessor - { Twilio.Media.V1.MediaProcessorListInstance } resource */
    protected _mediaProcessor?: MediaProcessorListInstance;
    /** mediaRecording - { Twilio.Media.V1.MediaRecordingListInstance } resource */
    protected _mediaRecording?: MediaRecordingListInstance;
    /** playerStreamer - { Twilio.Media.V1.PlayerStreamerListInstance } resource */
    protected _playerStreamer?: PlayerStreamerListInstance;
    /** Getter for mediaProcessor resource */
    get mediaProcessor(): MediaProcessorListInstance;
    /** Getter for mediaRecording resource */
    get mediaRecording(): MediaRecordingListInstance;
    /** Getter for playerStreamer resource */
    get playerStreamer(): PlayerStreamerListInstance;
}
