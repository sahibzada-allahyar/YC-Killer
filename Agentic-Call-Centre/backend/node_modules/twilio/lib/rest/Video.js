"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const VideoBase_1 = __importDefault(require("./VideoBase"));
class Video extends VideoBase_1.default {
    /**
     * @deprecated - Use v1.compositions instead
     */
    get compositions() {
        console.warn("compositions is deprecated. Use v1.compositions instead.");
        return this.v1.compositions;
    }
    /**
     * @deprecated - Use v1.compositionHooks instead
     */
    get compositionHooks() {
        console.warn("compositionHooks is deprecated. Use v1.compositionHooks instead.");
        return this.v1.compositionHooks;
    }
    /**
     * @deprecated - Use v1.compositionSettings instead
     */
    get compositionSettings() {
        console.warn("compositionSettings is deprecated. Use v1.compositionSettings instead.");
        return this.v1.compositionSettings;
    }
    /**
     * @deprecated - Use v1.recordings instead
     */
    get recordings() {
        console.warn("recordings is deprecated. Use v1.recordings instead.");
        return this.v1.recordings;
    }
    /**
     * @deprecated - Use v1.recordingSettings instead
     */
    get recordingSettings() {
        console.warn("recordingSettings is deprecated. Use v1.recordingSettings instead.");
        return this.v1.recordingSettings;
    }
    /**
     * @deprecated - Use v1.rooms instead
     */
    get rooms() {
        console.warn("rooms is deprecated. Use v1.rooms instead.");
        return this.v1.rooms;
    }
}
module.exports = Video;
