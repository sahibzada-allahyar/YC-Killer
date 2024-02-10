import ApiBase from "../ApiBase";
import Version from "../../base/Version";
import { AccountListInstance } from "./v2010/account";
import { AccountContext } from "./v2010/account";
export default class V2010 extends Version {
    /**
     * Initialize the V2010 version of Api
     *
     * @param domain - The Twilio (Twilio.Api) domain
     */
    constructor(domain: ApiBase);
    /** accounts - { Twilio.Api.V2010.AccountListInstance } resource */
    protected _accounts?: AccountListInstance;
    /** account - { Twilio.Api.V2010.AccountContext } resource */
    protected _account?: AccountContext;
    /** Getter for accounts resource */
    get accounts(): AccountListInstance;
    /** Getter for account resource */
    get account(): AccountContext;
}
