import { makeRe } from "next/dist/compiled/picomatch";
// Modifying this function should also modify writeImagesManifest()
export function matchLocalPattern(pattern, url) {
    if (pattern.search !== undefined) {
        if (pattern.search !== url.search) {
            return false;
        }
    }
    var _pattern_pathname;
    if (!makeRe((_pattern_pathname = pattern.pathname) != null ? _pattern_pathname : "**", {
        dot: true
    }).test(url.pathname)) {
        return false;
    }
    return true;
}
export function hasLocalMatch(localPatterns, urlPathAndQuery) {
    if (!localPatterns) {
        // if the user didn't define "localPatterns", we allow all local images
        return true;
    }
    const url = new URL(urlPathAndQuery, "http://n");
    return localPatterns.some((p)=>matchLocalPattern(p, url));
}

//# sourceMappingURL=match-local-pattern.js.map