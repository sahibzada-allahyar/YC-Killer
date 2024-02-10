"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const TaskrouterBase_1 = __importDefault(require("./TaskrouterBase"));
class Taskrouter extends TaskrouterBase_1.default {
    /**
     * @deprecated - Use v1.workspaces instead
     */
    get workspaces() {
        console.warn("workspaces is deprecated. Use v1.workspaces instead.");
        return this.v1.workspaces;
    }
}
module.exports = Taskrouter;
