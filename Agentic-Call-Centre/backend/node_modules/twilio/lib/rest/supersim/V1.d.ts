import SupersimBase from "../SupersimBase";
import Version from "../../base/Version";
import { EsimProfileListInstance } from "./v1/esimProfile";
import { FleetListInstance } from "./v1/fleet";
import { IpCommandListInstance } from "./v1/ipCommand";
import { NetworkListInstance } from "./v1/network";
import { NetworkAccessProfileListInstance } from "./v1/networkAccessProfile";
import { SettingsUpdateListInstance } from "./v1/settingsUpdate";
import { SimListInstance } from "./v1/sim";
import { SmsCommandListInstance } from "./v1/smsCommand";
import { UsageRecordListInstance } from "./v1/usageRecord";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Supersim
     *
     * @param domain - The Twilio (Twilio.Supersim) domain
     */
    constructor(domain: SupersimBase);
    /** esimProfiles - { Twilio.Supersim.V1.EsimProfileListInstance } resource */
    protected _esimProfiles?: EsimProfileListInstance;
    /** fleets - { Twilio.Supersim.V1.FleetListInstance } resource */
    protected _fleets?: FleetListInstance;
    /** ipCommands - { Twilio.Supersim.V1.IpCommandListInstance } resource */
    protected _ipCommands?: IpCommandListInstance;
    /** networks - { Twilio.Supersim.V1.NetworkListInstance } resource */
    protected _networks?: NetworkListInstance;
    /** networkAccessProfiles - { Twilio.Supersim.V1.NetworkAccessProfileListInstance } resource */
    protected _networkAccessProfiles?: NetworkAccessProfileListInstance;
    /** settingsUpdates - { Twilio.Supersim.V1.SettingsUpdateListInstance } resource */
    protected _settingsUpdates?: SettingsUpdateListInstance;
    /** sims - { Twilio.Supersim.V1.SimListInstance } resource */
    protected _sims?: SimListInstance;
    /** smsCommands - { Twilio.Supersim.V1.SmsCommandListInstance } resource */
    protected _smsCommands?: SmsCommandListInstance;
    /** usageRecords - { Twilio.Supersim.V1.UsageRecordListInstance } resource */
    protected _usageRecords?: UsageRecordListInstance;
    /** Getter for esimProfiles resource */
    get esimProfiles(): EsimProfileListInstance;
    /** Getter for fleets resource */
    get fleets(): FleetListInstance;
    /** Getter for ipCommands resource */
    get ipCommands(): IpCommandListInstance;
    /** Getter for networks resource */
    get networks(): NetworkListInstance;
    /** Getter for networkAccessProfiles resource */
    get networkAccessProfiles(): NetworkAccessProfileListInstance;
    /** Getter for settingsUpdates resource */
    get settingsUpdates(): SettingsUpdateListInstance;
    /** Getter for sims resource */
    get sims(): SimListInstance;
    /** Getter for smsCommands resource */
    get smsCommands(): SmsCommandListInstance;
    /** Getter for usageRecords resource */
    get usageRecords(): UsageRecordListInstance;
}
