"use strict";
/**
 * This has been written quickly. While trying to improve I realised it'd be better to have it in Jest...
 * ...and I saw a merged PR with `jest --init` tool!
 * TODO: see what's the best path for this
 */
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
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.help = exports.run = void 0;
var fs_1 = require("fs");
var path_1 = require("path");
var ejs_1 = __importDefault(require("ejs"));
var json5_1 = require("json5");
var constants_1 = require("../../constants");
var create_jest_preset_1 = require("../../presets/create-jest-preset");
var ensureOnlyUsingDoubleQuotes = function (str) {
    return str
        .replace(/"'(.*?)'"/g, '"$1"')
        .replace(/'ts-jest'/g, '"ts-jest"')
        .replace(/'babel-jest'/g, '"babel-jest"');
};
/**
 * @internal
 */
var run = function (args /* , logger: Logger */) { return __awaiter(void 0, void 0, void 0, function () {
    var askedTsconfig, force, jsdom, jsFilesProcessor, shouldPostProcessWithBabel, file, filePath, name, isPackageJsonConfig, isJestConfigFileExisted, pkgFile, isPackageJsonExisted, tsconfig, pkgJsonContent, body, resolvedTsconfigOption, transformConfig, _a, transformPattern, transformValue;
    var _b, _c;
    return __generator(this, function (_d) {
        askedTsconfig = args.tsconfig, force = args.force, jsdom = args.jsdom, jsFilesProcessor = args.js, shouldPostProcessWithBabel = args.babel;
        file = (_c = (_b = args._[0]) === null || _b === void 0 ? void 0 : _b.toString()) !== null && _c !== void 0 ? _c : 'jest.config.js';
        filePath = (0, path_1.join)(process.cwd(), file);
        name = (0, path_1.basename)(file);
        isPackageJsonConfig = name === 'package.json';
        isJestConfigFileExisted = (0, fs_1.existsSync)(filePath);
        pkgFile = isPackageJsonConfig ? filePath : (0, path_1.join)(process.cwd(), 'package.json');
        isPackageJsonExisted = isPackageJsonConfig || (0, fs_1.existsSync)(pkgFile);
        tsconfig = askedTsconfig === 'tsconfig.json' ? undefined : askedTsconfig;
        pkgJsonContent = isPackageJsonExisted ? JSON.parse((0, fs_1.readFileSync)(pkgFile, 'utf8')) : {};
        if (shouldPostProcessWithBabel) {
            console.warn("The option --babel is deprecated and will be removed in the next major version." +
                " Please specify 'js' option value (see more with npx ts-jest help) if you wish 'ts-jest' to process 'js' with TypeScript API or Babel.");
        }
        if (isPackageJsonConfig && !isJestConfigFileExisted) {
            throw new Error("File ".concat(file, " does not exists."));
        }
        else if (!isPackageJsonConfig && isJestConfigFileExisted && !force) {
            throw new Error("Configuration file ".concat(file, " already exists."));
        }
        if (!isPackageJsonConfig && !name.endsWith('.js')) {
            throw new TypeError("Configuration file ".concat(file, " must be a .js file or the package.json."));
        }
        if (isPackageJsonExisted && pkgJsonContent.jest) {
            if (force && !isPackageJsonConfig) {
                delete pkgJsonContent.jest;
                (0, fs_1.writeFileSync)(pkgFile, JSON.stringify(pkgJsonContent, undefined, '  '));
            }
            else if (!force) {
                throw new Error("A Jest configuration is already set in ".concat(pkgFile, "."));
            }
        }
        resolvedTsconfigOption = tsconfig ? { tsconfig: "".concat((0, json5_1.stringify)(tsconfig)) } : undefined;
        if (jsFilesProcessor === 'babel' || shouldPostProcessWithBabel) {
            transformConfig = (0, create_jest_preset_1.createJsWithBabelPreset)(resolvedTsconfigOption);
        }
        else if (jsFilesProcessor === 'ts') {
            transformConfig = (0, create_jest_preset_1.createJsWithTsPreset)(resolvedTsconfigOption);
        }
        else {
            transformConfig = (0, create_jest_preset_1.createDefaultPreset)(resolvedTsconfigOption);
        }
        if (isPackageJsonConfig) {
            body = ensureOnlyUsingDoubleQuotes(JSON.stringify(__assign(__assign({}, pkgJsonContent), { jest: transformConfig }), undefined, '  '));
        }
        else {
            _a = __read(Object.entries(transformConfig.transform)[0], 2), transformPattern = _a[0], transformValue = _a[1];
            body = ejs_1.default.render(constants_1.JEST_CONFIG_EJS_TEMPLATE, {
                exportKind: pkgJsonContent.type === 'module' ? 'export default' : 'module.exports =',
                testEnvironment: jsdom ? 'jsdom' : 'node',
                transformPattern: transformPattern,
                transformValue: ensureOnlyUsingDoubleQuotes((0, json5_1.stringify)(transformValue)),
            });
        }
        (0, fs_1.writeFileSync)(filePath, body);
        process.stderr.write("\nJest configuration written to \"".concat(filePath, "\".\n"));
        return [2 /*return*/];
    });
}); };
exports.run = run;
/**
 * @internal
 */
var help = function () { return __awaiter(void 0, void 0, void 0, function () {
    return __generator(this, function (_a) {
        process.stdout.write("\nUsage:\n  ts-jest config:init [options] [<config-file>]\n\nArguments:\n  <config-file>         Can be a js or json Jest config file. If it is a\n                        package.json file, the configuration will be read from\n                        the \"jest\" property.\n                        Default: jest.config.js\n\nOptions:\n  --force               Discard any existing Jest config\n  --js ts|babel         Process '.js' files with ts-jest if 'ts' or with\n                        babel-jest if 'babel'\n  --no-jest-preset      Disable the use of Jest presets\n  --tsconfig <file>     Path to the tsconfig.json file\n  --babel               Enable using Babel to process 'js' resulted content from 'ts-jest' processing\n  --jsdom               Use 'jsdom' as test environment instead of 'node'\n");
        return [2 /*return*/];
    });
}); };
exports.help = help;
