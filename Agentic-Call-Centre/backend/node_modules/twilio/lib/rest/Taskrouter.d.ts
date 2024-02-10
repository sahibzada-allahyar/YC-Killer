import { WorkspaceListInstance } from "./taskrouter/v1/workspace";
import TaskrouterBase from "./TaskrouterBase";
declare class Taskrouter extends TaskrouterBase {
    /**
     * @deprecated - Use v1.workspaces instead
     */
    get workspaces(): WorkspaceListInstance;
}
export = Taskrouter;
