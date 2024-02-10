# twilio-node

[![][test-workflow-image]][test-workflow-url]
[![][npm-version-image]][npm-url]
[![][npm-install-size-image]][npm-install-size-url]
[![][npm-downloads-image]][npm-downloads-url]

## Documentation

The documentation for the Twilio API can be found [here][apidocs].

The Node library documentation can be found [here][libdocs].

## Versions

`twilio-node` uses a modified version of [Semantic Versioning](https://semver.org) for all changes. [See this document](VERSIONS.md) for details.

### Supported Node.js Versions

This library supports the following Node.js implementations:

- Node.js 14
- Node.js 16
- Node.js 18

TypeScript is supported for TypeScript version 2.9 and above.

> **Warning**
> Do not use this Node.js library in a front-end application. Doing so can expose your Twilio credentials to end-users as part of the bundled HTML/JavaScript sent to their browser.

## Installation

`npm install twilio` or `yarn add twilio`

### Test your installation

To make sure the installation was successful, try sending yourself an SMS message, like this:

```js
// Your AccountSID and Auth Token from console.twilio.com
const accountSid = 'ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX';
const authToken = 'your_auth_token';

const client = require('twilio')(accountSid, authToken);

client.messages
  .create({
    body: 'Hello from twilio-node',
    to: '+12345678901', // Text your number
    from: '+12345678901', // From a valid Twilio number
  })
  .then((message) => console.log(message.sid));
```

After a brief delay, you will receive the text message on your phone.

> **Warning**
> It's okay to hardcode your credentials when testing locally, but you should use environment variables to keep them secret before committing any code or deploying to production. Check out [How to Set Environment Variables](https://www.twilio.com/blog/2017/01/how-to-set-environment-variables.html) for more information.

## Usage

Check out these [code examples](examples) in JavaScript and TypeScript to get up and running quickly.

### Environment Variables

`twilio-node` supports credential storage in environment variables. If no credentials are provided when instantiating the Twilio client (e.g., `const client = require('twilio')();`), the values in following env vars will be used: `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN`.

If your environment requires SSL decryption, you can set the path to CA bundle in the env var `TWILIO_CA_BUNDLE`.

### Client Initialization

If you invoke any V2010 operations without specifying an account SID, `twilio-node` will automatically use the `TWILIO_ACCOUNT_SID` value that the client was initialized with. This is useful for when you'd like to, for example, fetch resources for your main account but also your subaccount. See below:

```javascript
// Your Account SID, Subaccount SID Auth Token from console.twilio.com
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;
const subaccountSid = process.env.TWILIO_ACCOUNT_SUBACCOUNT_SID;

const client = require('twilio')(accountSid, authToken);
const mainAccountCalls = client.api.v2010.account.calls.list; // SID not specified, so defaults to accountSid
const subaccountCalls = client.api.v2010.account(subaccountSid).calls.list; // SID specified as subaccountSid
```

### Lazy Loading

`twilio-node` supports lazy loading required modules for faster loading time. Lazy loading is enabled by default. To disable lazy loading, simply instantiate the Twilio client with the `lazyLoading` flag set to `false`:

```javascript
// Your Account SID and Auth Token from console.twilio.com
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;

const client = require('twilio')(accountSid, authToken, {
  lazyLoading: false,
});
```

### Enable Auto-Retry with Exponential Backoff

`twilio-node` supports automatic retry with exponential backoff when API requests receive an [Error 429 response](https://support.twilio.com/hc/en-us/articles/360044308153-Twilio-API-response-Error-429-Too-Many-Requests-). This retry with exponential backoff feature is disabled by default. To enable this feature, instantiate the Twilio client with the `autoRetry` flag set to `true`.

Optionally, the maximum number of retries performed by this feature can be set with the `maxRetries` flag. The default maximum number of retries is `3`.

```javascript
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;

const client = require('twilio')(accountSid, authToken, {
  autoRetry: true,
  maxRetries: 3,
});
```

### Specify Region and/or Edge

To take advantage of Twilio's [Global Infrastructure](https://www.twilio.com/docs/global-infrastructure), specify the target Region and/or Edge for the client:

```javascript
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;

const client = require('twilio')(accountSid, authToken, {
  region: 'au1',
  edge: 'sydney',
});
```

Alternatively, specify the edge and/or region after constructing the Twilio client:

```javascript
const client = require('twilio')(accountSid, authToken);
client.region = 'au1';
client.edge = 'sydney';
```

This will result in the `hostname` transforming from `api.twilio.com` to `api.sydney.au1.twilio.com`.

### Iterate through records

The library automatically handles paging for you. Collections, such as `calls` and `messages`, have `list` and `each` methods that page under the hood. With both `list` and `each`, you can specify the number of records you want to receive (`limit`) and the maximum size you want each page fetch to be (`pageSize`). The library will then handle the task for you.

`list` eagerly fetches all records and returns them as a list, whereas `each` streams records and lazily retrieves pages of records as you iterate over the collection. You can also page manually using the `page` method.

For more information about these methods, view the [auto-generated library docs](https://www.twilio.com/docs/libraries/reference/twilio-node/).

```js
// Your Account SID and Auth Token from console.twilio.com
const accountSid = 'ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX';
const authToken = 'your_auth_token';
const client = require('twilio')(accountSid, authToken);

client.calls.each((call) => console.log(call.direction));
```

### Enable Debug Logging

There are two ways to enable debug logging in the default HTTP client. You can create an environment variable called `TWILIO_LOG_LEVEL` and set it to `debug` or you can set the logLevel variable on the client as debug:

```javascript
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;

const client = require('twilio')(accountSid, authToken, {
  logLevel: 'debug',
});
```

You can also set the logLevel variable on the client after constructing the Twilio client:

```javascript
const client = require('twilio')(accountSid, authToken);
client.logLevel = 'debug';
```

### Debug API requests

To assist with debugging, the library allows you to access the underlying request and response objects. This capability is built into the default HTTP client that ships with the library.

For example, you can retrieve the status code of the last response like so:

```js
const accountSid = 'ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX';
const authToken = 'your_auth_token';

const client = require('twilio')(accountSid, authToken);

client.messages
  .create({
    to: '+14158675309',
    from: '+14258675310',
    body: 'Ahoy!',
  })
  .then(() => {
    // Access details about the last request
    console.log(client.lastRequest.method);
    console.log(client.lastRequest.url);
    console.log(client.lastRequest.auth);
    console.log(client.lastRequest.params);
    console.log(client.lastRequest.headers);
    console.log(client.lastRequest.data);

    // Access details about the last response
    console.log(client.httpClient.lastResponse.statusCode);
    console.log(client.httpClient.lastResponse.body);
  });
```

### Handle exceptions

If the Twilio API returns a 400 or a 500 level HTTP response, `twilio-node` will throw an error including relevant information, which you can then `catch`:

```js
client.messages
  .create({
    body: 'Hello from Node',
    to: '+12345678901',
    from: '+12345678901',
  })
  .then((message) => console.log(message))
  .catch((error) => {
    // You can implement your fallback code here
    console.log(error);
  });
```

or with `async/await`:

```js
try {
  const message = await client.messages.create({
    body: 'Hello from Node',
    to: '+12345678901',
    from: '+12345678901',
  });
  console.log(message);
} catch (error) {
  // You can implement your fallback code here
  console.error(error);
}
```

If you are using callbacks, error information will be included in the `error` parameter of the callback.

400-level errors are [normal during API operation](https://www.twilio.com/docs/api/rest/request#get-responses) ("Invalid number", "Cannot deliver SMS to that number", for example) and should be handled appropriately.

### Use a custom HTTP Client

To use a custom HTTP client with this helper library, please see the [advanced example of how to do so](./advanced-examples/custom-http-client.md).

### Use webhook validation

See [example](examples/express.js) for a code sample for incoming Twilio request validation.

## Docker image

The `Dockerfile` present in this repository and its respective `twilio/twilio-node` Docker image are currently used by Twilio for testing purposes only.

## Getting help

If you need help installing or using the library, please check the [Twilio Support Help Center](https://support.twilio.com) first, and [file a support ticket](https://twilio.com/help/contact) if you don't find an answer to your question.

If you've instead found a bug in the library or would like new features added, go ahead and open issues or pull requests against this repo!

## Contributing

Bug fixes, docs, and library improvements are always welcome. Please refer to our [Contributing Guide](CONTRIBUTING.md) for detailed information on how you can contribute.

> ⚠️ Please be aware that a large share of the files are auto-generated by our backend tool. You are welcome to suggest changes and submit PRs illustrating the changes. However, we'll have to make the changes in the underlying tool. You can find more info about this in the [Contributing Guide](CONTRIBUTING.md).

If you're not familiar with the GitHub pull request/contribution process, [this is a nice tutorial](https://gun.io/blog/how-to-github-fork-branch-and-pull-request/).

### Get started

If you want to familiarize yourself with the project, you can start by [forking the repository](https://help.github.com/articles/fork-a-repo/) and [cloning it in your local development environment](https://help.github.com/articles/cloning-a-repository/). The project requires [Node.js](https://nodejs.org) to be installed on your machine.

After cloning the repository, install the dependencies by running the following command in the directory of your cloned repository:

```bash
npm install
```

You can run the existing tests to see if everything is okay by executing:

```bash
npm test
```

To run just one specific test file instead of the whole suite, provide a JavaScript regular expression that will match your spec file's name, like:

```bash
npm run test:javascript -- -m .\*client.\*
```

[apidocs]: https://www.twilio.com/docs/api
[libdocs]: https://twilio.github.io/twilio-node

[test-workflow-image]: https://github.com/twilio/twilio-node/actions/workflows/test-and-deploy.yml/badge.svg
[test-workflow-url]: https://github.com/twilio/twilio-node/actions/workflows/test-and-deploy.yml
[npm-downloads-image]: https://img.shields.io/npm/dm/twilio.svg
[npm-downloads-url]: https://npmcharts.com/compare/twilio?minimal=true
[npm-install-size-image]: https://badgen.net/packagephobia/install/twilio
[npm-install-size-url]: https://packagephobia.com/result?p=twilio
[npm-url]: https://npmjs.org/package/twilio
[npm-version-image]: https://img.shields.io/npm/v/twilio.svg
