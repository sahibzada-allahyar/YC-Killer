"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class Request {
    constructor(opts) {
        opts = opts || {};
        this.method = opts.method || this.ANY;
        this.url = opts.url || this.ANY;
        this.auth = opts.auth || this.ANY;
        this.params = opts.params || this.ANY;
        this.data = opts.data || this.ANY;
        this.headers = opts.headers || this.ANY;
    }
    get ANY() {
        return "*";
    }
    attributeEqual(lhs, rhs) {
        if (lhs === this.ANY || rhs === this.ANY) {
            return true;
        }
        lhs = lhs || undefined;
        rhs = rhs || undefined;
        if (typeof lhs !== typeof rhs) {
            return false;
        }
        if (typeof lhs !== "object") {
            return lhs === rhs;
        }
        return (Object.entries(lhs)
            .sort((a, b) => a[0].localeCompare(b[0]))
            .toString() ===
            Object.entries(rhs)
                .sort((a, b) => a[0].localeCompare(b[0]))
                .toString());
    }
    isEqual(other) {
        return (this.attributeEqual(this.method, other.method) &&
            this.attributeEqual(this.url, other.url) &&
            this.attributeEqual(this.auth, other.auth) &&
            this.attributeEqual(this.params, other.params) &&
            this.attributeEqual(this.data, other.data) &&
            this.attributeEqual(this.headers, other.headers));
    }
    toString() {
        var auth = "";
        if (this.auth && this.auth !== this.ANY) {
            auth = this.auth + " ";
        }
        var params = "";
        if (this.params && this.params !== this.ANY) {
            params =
                "?" +
                    Object.keys(this.params)
                        .map((key) => function () {
                        return key + "=" + this.params[key];
                    }.bind(this)())
                        .join("&");
        }
        var data = "";
        if (this.data && this.data !== this.ANY) {
            if (this.method === "get") {
                data = "\n -G";
            }
            data =
                data +
                    "\n" +
                    Object.entries(this.data)
                        .map((d) => {
                        return " -d " + d[0] + "=" + d[1];
                    })
                        .join("\n");
        }
        var headers = "";
        if (this.headers && this.headers !== this.ANY) {
            headers =
                "\n" +
                    Object.entries(this.headers)
                        .map((header) => {
                        return " -H " + header[0] + "=" + header[1];
                    })
                        .join("\n");
        }
        return auth + this.method + " " + this.url + params + data + headers;
    }
}
exports.default = Request;
