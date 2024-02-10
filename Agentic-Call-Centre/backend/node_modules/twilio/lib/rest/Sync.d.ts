import { ServiceListInstance } from "./sync/v1/service";
import SyncBase from "./SyncBase";
declare class Sync extends SyncBase {
    /**
     * @deprecated - Use v1.services instead
     */
    get services(): ServiceListInstance;
}
export = Sync;
