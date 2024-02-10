import TaskrouterBase from "../TaskrouterBase";
import Version from "../../base/Version";
import { WorkspaceListInstance } from "./v1/workspace";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Taskrouter
     *
     * @param domain - The Twilio (Twilio.Taskrouter) domain
     */
    constructor(domain: TaskrouterBase);
    /** workspaces - { Twilio.Taskrouter.V1.WorkspaceListInstance } resource */
    protected _workspaces?: WorkspaceListInstance;
    /** Getter for workspaces resource */
    get workspaces(): WorkspaceListInstance;
}
