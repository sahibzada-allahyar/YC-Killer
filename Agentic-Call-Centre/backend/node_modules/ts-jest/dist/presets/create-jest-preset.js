"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __read = (this && this.__read) || function (o, n) {
    var m = typeof Symbol === "function" && o[Symbol.iterator];
    if (!m) return o;
    var i = m.call(o), r, ar = [], e;
    try {
        while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
    }
    catch (error) { e = { error: error }; }
    finally {
        try {
            if (r && !r.done && (m = i["return"])) m.call(i);
        }
        finally { if (e) throw e.error; }
    }
    return ar;
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createJestPreset = createJestPreset;
exports.createDefaultPreset = createDefaultPreset;
exports.createDefaultLegacyPreset = createDefaultLegacyPreset;
exports.createJsWithTsPreset = createJsWithTsPreset;
exports.createJsWithTsLegacyPreset = createJsWithTsLegacyPreset;
exports.createJsWithBabelPreset = createJsWithBabelPreset;
exports.createJsWithBabelLegacyPreset = createJsWithBabelLegacyPreset;
exports.createDefaultEsmPreset = createDefaultEsmPreset;
exports.createDefaultEsmLegacyPreset = createDefaultEsmLegacyPreset;
exports.createJsWithTsEsmPreset = createJsWithTsEsmPreset;
exports.createJsWithTsEsmLegacyPreset = createJsWithTsEsmLegacyPreset;
exports.createJsWithBabelEsmPreset = createJsWithBabelEsmPreset;
exports.createJsWithBabelEsmLegacyPreset = createJsWithBabelEsmLegacyPreset;
var constants_1 = require("../constants");
var utils_1 = require("../utils");
var logger = utils_1.rootLogger.child({ namespace: 'jest-preset' });
/**
 * @deprecated use other functions below instead
 */
function createJestPreset(legacy, allowJs, extraOptions) {
    var _a;
    if (legacy === void 0) { legacy = false; }
    if (allowJs === void 0) { allowJs = false; }
    if (extraOptions === void 0) { extraOptions = {}; }
    logger.debug({ allowJs: allowJs }, 'creating jest presets', allowJs ? 'handling' : 'not handling', 'JavaScript files');
    var extensionsToTreatAsEsm = extraOptions.extensionsToTreatAsEsm, moduleFileExtensions = extraOptions.moduleFileExtensions, testMatch = extraOptions.testMatch;
    var supportESM = extensionsToTreatAsEsm === null || extensionsToTreatAsEsm === void 0 ? void 0 : extensionsToTreatAsEsm.length;
    var tsJestTransformOptions = supportESM ? { useESM: true } : {};
    return __assign(__assign(__assign(__assign({}, (extensionsToTreatAsEsm ? { extensionsToTreatAsEsm: extensionsToTreatAsEsm } : undefined)), (moduleFileExtensions ? { moduleFileExtensions: moduleFileExtensions } : undefined)), (testMatch ? { testMatch: testMatch } : undefined)), { transform: __assign(__assign({}, extraOptions.transform), (_a = {}, _a[allowJs ? (supportESM ? '^.+\\.m?[tj]sx?$' : '^.+\\.[tj]sx?$') : '^.+\\.tsx?$'] = legacy
            ? ['ts-jest/legacy', tsJestTransformOptions]
            : ['ts-jest', tsJestTransformOptions], _a)) });
}
function createDefaultPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating default CJS Jest preset');
    return {
        transform: (_a = {},
            _a[constants_1.TS_TRANSFORM_PATTERN] = ['ts-jest', tsJestTransformOptions],
            _a),
    };
}
function createDefaultLegacyPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating default CJS Jest preset');
    return {
        transform: (_a = {},
            _a[constants_1.TS_TRANSFORM_PATTERN] = ['ts-jest/legacy', tsJestTransformOptions],
            _a),
    };
}
function createJsWithTsPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating Js with Ts CJS Jest preset');
    return {
        transform: (_a = {},
            _a[constants_1.TS_JS_TRANSFORM_PATTERN] = ['ts-jest', tsJestTransformOptions],
            _a),
    };
}
function createJsWithTsLegacyPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating Js with Ts CJS Jest preset');
    return {
        transform: (_a = {},
            _a[constants_1.TS_JS_TRANSFORM_PATTERN] = ['ts-jest/legacy', tsJestTransformOptions],
            _a),
    };
}
function createJsWithBabelPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating JS with Babel CJS Jest preset');
    return {
        transform: (_a = {},
            _a[constants_1.JS_TRANSFORM_PATTERN] = 'babel-jest',
            _a[constants_1.TS_TRANSFORM_PATTERN] = ['ts-jest', tsJestTransformOptions],
            _a),
    };
}
function createJsWithBabelLegacyPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating JS with Babel CJS Jest preset');
    return {
        transform: (_a = {},
            _a[constants_1.JS_TRANSFORM_PATTERN] = 'babel-jest',
            _a[constants_1.TS_TRANSFORM_PATTERN] = ['ts-jest/legacy', tsJestTransformOptions],
            _a),
    };
}
function createDefaultEsmPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating default ESM Jest preset');
    return {
        extensionsToTreatAsEsm: __spreadArray([], __read(constants_1.TS_EXT_TO_TREAT_AS_ESM), false),
        transform: (_a = {},
            _a[constants_1.ESM_TS_TRANSFORM_PATTERN] = [
                'ts-jest',
                __assign(__assign({}, tsJestTransformOptions), { useESM: true }),
            ],
            _a),
    };
}
function createDefaultEsmLegacyPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating default ESM Jest preset');
    return {
        extensionsToTreatAsEsm: __spreadArray([], __read(constants_1.TS_EXT_TO_TREAT_AS_ESM), false),
        transform: (_a = {},
            _a[constants_1.ESM_TS_TRANSFORM_PATTERN] = [
                'ts-jest/legacy',
                __assign(__assign({}, tsJestTransformOptions), { useESM: true }),
            ],
            _a),
    };
}
function createJsWithTsEsmPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating Js with Ts ESM Jest preset');
    return {
        extensionsToTreatAsEsm: __spreadArray(__spreadArray([], __read(constants_1.JS_EXT_TO_TREAT_AS_ESM), false), __read(constants_1.TS_EXT_TO_TREAT_AS_ESM), false),
        transform: (_a = {},
            _a[constants_1.ESM_TS_JS_TRANSFORM_PATTERN] = [
                'ts-jest',
                __assign(__assign({}, tsJestTransformOptions), { useESM: true }),
            ],
            _a),
    };
}
function createJsWithTsEsmLegacyPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating Js with Ts ESM Jest preset');
    return {
        extensionsToTreatAsEsm: __spreadArray(__spreadArray([], __read(constants_1.JS_EXT_TO_TREAT_AS_ESM), false), __read(constants_1.TS_EXT_TO_TREAT_AS_ESM), false),
        transform: (_a = {},
            _a[constants_1.ESM_TS_JS_TRANSFORM_PATTERN] = [
                'ts-jest/legacy',
                __assign(__assign({}, tsJestTransformOptions), { useESM: true }),
            ],
            _a),
    };
}
function createJsWithBabelEsmPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating JS with Babel ESM Jest preset');
    return {
        extensionsToTreatAsEsm: __spreadArray(__spreadArray([], __read(constants_1.JS_EXT_TO_TREAT_AS_ESM), false), __read(constants_1.TS_EXT_TO_TREAT_AS_ESM), false),
        transform: (_a = {},
            _a[constants_1.ESM_JS_TRANSFORM_PATTERN] = 'babel-jest',
            _a[constants_1.ESM_TS_TRANSFORM_PATTERN] = [
                'ts-jest',
                __assign(__assign({}, tsJestTransformOptions), { useESM: true }),
            ],
            _a),
    };
}
function createJsWithBabelEsmLegacyPreset(tsJestTransformOptions) {
    var _a;
    if (tsJestTransformOptions === void 0) { tsJestTransformOptions = {}; }
    logger.debug('creating JS with Babel ESM Jest preset');
    return {
        extensionsToTreatAsEsm: __spreadArray(__spreadArray([], __read(constants_1.JS_EXT_TO_TREAT_AS_ESM), false), __read(constants_1.TS_EXT_TO_TREAT_AS_ESM), false),
        transform: (_a = {},
            _a[constants_1.ESM_JS_TRANSFORM_PATTERN] = 'babel-jest',
            _a[constants_1.ESM_TS_TRANSFORM_PATTERN] = [
                'ts-jest/legacy',
                __assign(__assign({}, tsJestTransformOptions), { useESM: true }),
            ],
            _a),
    };
}
