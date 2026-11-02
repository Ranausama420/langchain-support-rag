# NimbusDesk Configuration

## Request timeout
Set NIMBUSDESK_TIMEOUT_SECONDS to change the request timeout. The default timeout is 30 seconds. Accepted values range from 5 to 120 seconds. Restart the worker after changing the environment variable.

## Proxy configuration
Set HTTPS_PROXY to the corporate proxy address before starting the worker. Set NO_PROXY to a comma-separated list of internal hostnames that should bypass the proxy. Do not store proxy passwords in source control.
