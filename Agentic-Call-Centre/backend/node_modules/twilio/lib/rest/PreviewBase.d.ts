import Domain from "../base/Domain";
import DeployedDevices from "./preview/DeployedDevices";
import HostedNumbers from "./preview/HostedNumbers";
import Sync from "./preview/Sync";
import Marketplace from "./preview/Marketplace";
import Wireless from "./preview/Wireless";
declare class PreviewBase extends Domain {
    _deployed_devices?: DeployedDevices;
    _hosted_numbers?: HostedNumbers;
    _sync?: Sync;
    _marketplace?: Marketplace;
    _wireless?: Wireless;
    /**
     * Initialize preview domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get deployed_devices(): DeployedDevices;
    get hosted_numbers(): HostedNumbers;
    get sync(): Sync;
    get marketplace(): Marketplace;
    get wireless(): Wireless;
}
export = PreviewBase;
