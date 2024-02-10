import { UserListInstance } from "./frontlineApi/v1/user";
import FrontlineApiBase from "./FrontlineApiBase";
declare class FrontlineApi extends FrontlineApiBase {
    /**
     * @deprecated - Use v1.users instead
     */
    get users(): UserListInstance;
}
export = FrontlineApi;
