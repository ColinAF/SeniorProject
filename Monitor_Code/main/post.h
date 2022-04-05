#ifndef POST_H
#define POST_H

#ifdef __cplusplus
extern "C"
{
#endif 

#include "button.h"
#include "camera.h" // probably won't need this any longer 
#include "esp_err.h"
#include "esp_http_client.h"


// Semaphore to block acess to the jpg 
extern SemaphoreHandle_t xFrameSemaphore;


esp_err_t _http_event_handle(esp_http_client_event_t *evt);
void http_post_task(void *pvParameters);

#ifdef __cplusplus
}
#endif 

#endif 