import MicrovisorBase from "../MicrovisorBase";
import Version from "../../base/Version";
import { AccountConfigListInstance } from "./v1/accountConfig";
import { AccountSecretListInstance } from "./v1/accountSecret";
import { AppListInstance } from "./v1/app";
import { DeviceListInstance } from "./v1/device";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Microvisor
     *
     * @param domain - The Twilio (Twilio.Microvisor) domain
     */
    constructor(domain: MicrovisorBase);
    /** accountConfigs - { Twilio.Microvisor.V1.AccountConfigListInstance } resource */
    protected _accountConfigs?: AccountConfigListInstance;
    /** accountSecrets - { Twilio.Microvisor.V1.AccountSecretListInstance } resource */
    protected _accountSecrets?: AccountSecretListInstance;
    /** apps - { Twilio.Microvisor.V1.AppListInstance } resource */
    protected _apps?: AppListInstance;
    /** devices - { Twilio.Microvisor.V1.DeviceListInstance } resource */
    protected _devices?: DeviceListInstance;
    /** Getter for accountConfigs resource */
    get accountConfigs(): AccountConfigListInstance;
    /** Getter for accountSecrets resource */
    get accountSecrets(): AccountSecretListInstance;
    /** Getter for apps resource */
    get apps(): AppListInstance;
    /** Getter for devices resource */
    get devices(): DeviceListInstance;
}
