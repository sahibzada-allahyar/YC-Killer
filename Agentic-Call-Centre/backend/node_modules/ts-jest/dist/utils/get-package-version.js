"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getPackageVersion = getPackageVersion;
/**
 * @internal
 */
function getPackageVersion(moduleName) {
    try {
        return require("".concat(moduleName, "/package.json")).version;
    }
    catch (err) { }
    return;
}
