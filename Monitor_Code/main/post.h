#ifndef POST_H
#define POST_H

#ifdef __cplusplus
extern "C"
{
#endif 

#include "camera.h"
#include "esp_err.h"
#include "esp_http_client.h"


#include "freertos/semphr.h"

// Semaphore to block acess to curFrame
extern SemaphoreHandle_t xFrameSemaphore;

// Frame buffer 
extern camera_fb_t *curFrame; 


esp_err_t _http_event_handle(esp_http_client_event_t *evt);
void http_post_task(void *pvParameters);

#ifdef __cplusplus
}
#endif 

#endif 