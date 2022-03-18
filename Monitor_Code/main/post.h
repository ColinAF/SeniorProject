#ifndef POST_H
#define POST_H

#include "esp_err.h"
#include "esp_http_client.h"

esp_err_t _http_event_handle(esp_http_client_event_t *evt);
static void http_post_task(void *pvParameters);

#endif 