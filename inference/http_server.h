// http_server.h -- Minimal HTTP/1.1 server for ANE inference API
// Handles GET /health and POST /v1/completions using raw POSIX sockets.
// No external dependencies.
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <time.h>

#define HTTP_MAX_REQUEST   65536
#define HTTP_MAX_RESPONSE  262144
#define HTTP_MAX_BODY      65536

// --- HTTP request parsing ---

typedef struct {
    char method[8];       // GET, POST, etc.
    char path[256];       // /v1/completions, /health, etc.
    char body[HTTP_MAX_BODY];
    int body_len;
    int content_length;
} HttpRequest;

static int http_parse_request(const char *raw, int raw_len, HttpRequest *req) {
    memset(req, 0, sizeof(HttpRequest));

    // Parse request line: METHOD PATH HTTP/1.1\r\n
    const char *p = raw;
    int i = 0;
    while (*p && *p != ' ' && i < 7) req->method[i++] = *p++;
    req->method[i] = '\0';
    if (*p == ' ') p++;

    i = 0;
    while (*p && *p != ' ' && *p != '?' && i < 255) req->path[i++] = *p++;
    req->path[i] = '\0';

    // Skip to end of request line
    while (*p && *p != '\n') p++;
    if (*p) p++;

    // Parse headers (only need Content-Length)
    req->content_length = 0;
    while (*p && !(*p == '\r' && *(p+1) == '\n') && *p != '\n') {
        if (strncasecmp(p, "Content-Length:", 15) == 0) {
            req->content_length = atoi(p + 15);
        }
        while (*p && *p != '\n') p++;
        if (*p) p++;
    }
    // Skip blank line
    if (*p == '\r') p++;
    if (*p == '\n') p++;

    // Copy body
    int remaining = raw_len - (int)(p - raw);
    req->body_len = remaining < HTTP_MAX_BODY - 1 ? remaining : HTTP_MAX_BODY - 1;
    if (req->body_len > 0) memcpy(req->body, p, req->body_len);
    req->body[req->body_len] = '\0';

    return 0;
}

// --- HTTP response sending ---

static void http_send(int fd, int status, const char *status_text,
                      const char *content_type, const char *body, int body_len) {
    char header[1024];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);

    write(fd, header, hlen);
    if (body_len > 0) write(fd, body, body_len);
}

static void http_send_json(int fd, int status, const char *json) {
    const char *status_text = "OK";
    if (status == 400) status_text = "Bad Request";
    else if (status == 404) status_text = "Not Found";
    else if (status == 503) status_text = "Service Unavailable";
    http_send(fd, status, status_text, "application/json", json, (int)strlen(json));
}

// --- Minimal JSON field extraction ---

static int http_json_get_string(const char *json, const char *key,
                                char *out, int max_out) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    if (*p != '"') return -1;
    p++;
    int n = 0;
    while (*p && *p != '"' && n < max_out - 1) {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case 'n': out[n++] = '\n'; break;
                case 't': out[n++] = '\t'; break;
                case '"': out[n++] = '"'; break;
                case '\\': out[n++] = '\\'; break;
                default: out[n++] = *p;
            }
        } else {
            out[n++] = *p;
        }
        p++;
    }
    out[n] = '\0';
    return n;
}

static int http_json_get_int(const char *json, const char *key, int default_val) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return default_val;
    p += strlen(search);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    if (*p == '-' || (*p >= '0' && *p <= '9'))
        return (int)strtol(p, NULL, 10);
    return default_val;
}

// --- TCP server ---

typedef void (*HttpHandler)(int client_fd, HttpRequest *req, void *ctx);

static int http_serve(int port, HttpHandler handler, void *ctx) {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return -1; }

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(srv); return -1;
    }
    if (listen(srv, 8) < 0) {
        perror("listen"); close(srv); return -1;
    }

    printf("HTTP server listening on http://127.0.0.1:%d\n", port);
    printf("  POST /v1/completions  {\"prompt\": \"...\", \"max_tokens\": 50}\n");
    printf("  GET  /health\n");
    printf("READY\n");
    fflush(stdout);

    while (1) {
        int client = accept(srv, NULL, NULL);
        if (client < 0) { perror("accept"); continue; }

        // Read full request (headers + body)
        char buf[HTTP_MAX_REQUEST];
        int total = 0;
        int headers_done = 0;
        int content_length = 0;
        int body_start = 0;

        while (total < HTTP_MAX_REQUEST - 1) {
            ssize_t n = read(client, buf + total, HTTP_MAX_REQUEST - 1 - total);
            if (n <= 0) break;
            total += n;
            buf[total] = '\0';

            if (!headers_done) {
                char *hend = strstr(buf, "\r\n\r\n");
                if (hend) {
                    headers_done = 1;
                    body_start = (int)(hend - buf) + 4;
                    // Extract Content-Length
                    char *cl = strcasestr(buf, "Content-Length:");
                    if (cl) content_length = atoi(cl + 15);
                }
            }

            if (headers_done) {
                int body_received = total - body_start;
                if (body_received >= content_length) break;
            }
        }

        HttpRequest req;
        http_parse_request(buf, total, &req);

        // Handle OPTIONS preflight
        if (strcmp(req.method, "OPTIONS") == 0) {
            http_send(client, 204, "No Content", "text/plain", "", 0);
            close(client);
            continue;
        }

        handler(client, &req, ctx);
        close(client);
    }

    return 0;
}
