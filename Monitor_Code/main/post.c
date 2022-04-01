// Http task...

#include "post.h"

#include "img_converters.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"



#include "esp_log.h"

static const char* TAG = "HTTP Client";


esp_err_t _http_event_handle(esp_http_client_event_t *evt)
{
    switch(evt->event_id) {
        case HTTP_EVENT_ERROR:
            ESP_LOGI(TAG, "HTTP_EVENT_ERROR");
            break;
        case HTTP_EVENT_ON_CONNECTED:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_CONNECTED");
            break;
        case HTTP_EVENT_HEADER_SENT:
            ESP_LOGI(TAG, "HTTP_EVENT_HEADER_SENT");
            break;
        case HTTP_EVENT_ON_HEADER:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_HEADER");
            printf("%.*s", evt->data_len, (char*)evt->data);
            break;
        case HTTP_EVENT_ON_DATA:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_DATA, len=%d", evt->data_len);
            if (!esp_http_client_is_chunked_response(evt->client)) {
                printf("%.*s", evt->data_len, (char*)evt->data);
            }

            break;
        case HTTP_EVENT_ON_FINISH:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_FINISH");
            break;
        case HTTP_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "HTTP_EVENT_DISCONNECTED");
            break;
    }
    return ESP_OK;
}

SemaphoreHandle_t xFrameSemaphore = NULL;
camera_fb_t *curFrame = NULL;

void http_post_task(void *pvParameters)
{


    

    while(1)
    {   
        // Move the image capture functionality to button or camera!!! 
        size_t bufLen; 
        uint8_t* jpegBuf; 

        //if(xQueueReceive(xQueueAIFrame, &(curFrame), portMAX_DELAY))
        //if(curFrame != NULL)
        if( xSemaphoreTake(xFrameSemaphore, 1000 / portTICK_PERIOD_MS ) == pdTRUE )
        {  

            /* Redundant
            ESP_LOGI(TAG, "Got fb!");
            ESP_LOGI(TAG, "Lenght: %d", curFrame->len);
            ESP_LOGI(TAG, "Width: %d", curFrame->width);
            ESP_LOGI(TAG, "Height: %d", curFrame->height);
            */

             // Might cause issues later... populate these vars
            frame2jpg(curFrame, 80, &jpegBuf, &bufLen);

            // I had this url wrong the whole time I think :( improve the way this is set!! 
            esp_http_client_config_t config = {
            .url = "http://10.0.0.118:80",
            .event_handler = _http_event_handle,
            };
            
            esp_http_client_handle_t client = esp_http_client_init(&config);

            esp_http_client_set_method(client, HTTP_METHOD_POST);
            esp_http_client_set_header(client, "Content-Type", "image/jpeg");
            esp_http_client_set_post_field(client, (const char*) jpegBuf, bufLen);
            esp_err_t  err = esp_http_client_perform(client);

            if (err == ESP_OK) 
            {
                ESP_LOGI(TAG, "HTTP POST Status = %d, content_length = %d",
                        esp_http_client_get_status_code(client),
                        esp_http_client_get_content_length(client));
            } 
            else 
            {
                ESP_LOGE(TAG, "HTTP POST request failed: %s", esp_err_to_name(err));
            }

            esp_http_client_cleanup(client);
            
            /*
            for(int countdown = 10; countdown >= 0; countdown--) {
                ESP_LOGI(TAG, "%d... ", countdown);
                vTaskDelay(1000 / portTICK_PERIOD_MS);
            }
            */

            // Reset the frame buffer 
            //curFrame = NULL; 

            ESP_LOGI(TAG, "Starting again!");

            /*
            // Unblock the curFrame resource
            if( xSemaphoreGive(xFrameSemaphore) != pdTRUE )
            {
                ESP_LOGE(TAG, "Semaphore Give Error!");

            }
            */


       } 
       /*
       else // Failsafe 
       {
            ESP_LOGI(TAG, "Waiting for Button Press");
            vTaskDelay(10000 / portTICK_PERIOD_MS);

            // Unblock the curFrame resource
            if( xSemaphoreGive(xFrameSemaphore) != pdTRUE )
            {
                ESP_LOGE(TAG, "Semaphore Give Error!");

            }  
       } 
       */


    }

}