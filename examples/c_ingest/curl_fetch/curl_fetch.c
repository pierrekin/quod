/* Minimal libcurl demo: fetch a URL and dump the response to stdout.
   Demonstrates: pointer-typed locals, opaque handle types, enum constants
   (CURLOPT_URL), variadic call, linker flag plumbing, void-returning
   externs (curl_easy_cleanup).

   Builds with `[link] libraries = ["curl"]`. */

/* libcurl ships type-checking macros over curl_easy_setopt that expand into
   non-standard GCC statement expressions. Disable them so the call lands as
   a plain function call our ingester can walk. */
#define CURL_DISABLE_TYPECHECK
#include <curl/curl.h>

int main(void) {
    CURL *handle = curl_easy_init();
    curl_easy_setopt(handle, CURLOPT_URL, "https://example.com");
    curl_easy_perform(handle);
    curl_easy_cleanup(handle);
    return 0;
}
