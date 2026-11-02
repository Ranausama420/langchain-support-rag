# NimbusDesk Troubleshooting

## NET-201 connection refused
NET-201 means the worker cannot connect to the NimbusDesk server. Verify NIMBUSDESK_SERVER_URL, confirm that the server is running, and check outbound firewall access to the configured server port. Review the proxy settings if a corporate proxy is required.

## RATE-429 rate limit
RATE-429 means the server's request limit has been exceeded. Wait for the number of seconds in the Retry-After response header before retrying. Use exponential backoff with jitter for subsequent retries. Do not retry continuously without a delay.
