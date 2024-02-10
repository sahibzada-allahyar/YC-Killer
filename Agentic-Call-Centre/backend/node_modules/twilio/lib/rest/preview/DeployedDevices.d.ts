import PreviewBase from "../PreviewBase";
import Version from "../../base/Version";
import { FleetListInstance } from "./deployed_devices/fleet";
export default class DeployedDevices extends Version {
    /**
     * Initialize the DeployedDevices version of Preview
     *
     * @param domain - The Twilio (Twilio.Preview) domain
     */
    constructor(domain: PreviewBase);
    /** fleets - { Twilio.Preview.DeployedDevices.FleetListInstance } resource */
    protected _fleets?: FleetListInstance;
    /** Getter for fleets resource */
    get fleets(): FleetListInstance;
}
