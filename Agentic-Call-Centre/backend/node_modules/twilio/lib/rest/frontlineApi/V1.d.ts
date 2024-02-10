import FrontlineApiBase from "../FrontlineApiBase";
import Version from "../../base/Version";
import { UserListInstance } from "./v1/user";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of FrontlineApi
     *
     * @param domain - The Twilio (Twilio.FrontlineApi) domain
     */
    constructor(domain: FrontlineApiBase);
    /** users - { Twilio.FrontlineApi.V1.UserListInstance } resource */
    protected _users?: UserListInstance;
    /** Getter for users resource */
    get users(): UserListInstance;
}
