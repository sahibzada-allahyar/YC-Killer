"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.isValidPathParam = exports.trim = void 0;
const INVALID_PATH_PARAM_CHARS = ["/", "?"];
const trim = (str, c = "\\s") => str.replace(new RegExp(`^([${c}]*)(.*?)([${c}]*)$`), "$2");
exports.trim = trim;
function isValidPathParam(param) {
    if (param === null || param === undefined)
        return false;
    const paramString = param.toString();
    return INVALID_PATH_PARAM_CHARS.every((invalidChar) => !paramString.includes(invalidChar));
}
exports.isValidPathParam = isValidPathParam;
