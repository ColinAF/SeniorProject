// Http task...

#include "post.h"
#include "camera.h"

#include "img_converters.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

static const char* TAG = "HTTP Client";

esp_err_t _http_event_handle(esp_http_client_event_t *evt)
{
    switch(evt->event_id) 
    {
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
            if (!esp_http_client_is_chunked_response(evt->client)) 
            {
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

// Make the jpg send in chunks instead of all at once? 
void http_post_task(void *pvParameters)
{
    while(1)
    {   

        if( xSemaphoreTake(xFrameSemaphore, 1000 / portTICK_PERIOD_MS ) == pdTRUE )
        {  
        
            // I think this works great hardcoded for now
            // Will need to update the IP and such!  
            esp_http_client_config_t config = {
            .url = "http://192.168.137.1:8000/produce_detector/",
            //.url = "http://10.0.0.118",
            .event_handler = _http_event_handle,
            };
            
            esp_http_client_handle_t client = esp_http_client_init(&config);

            esp_http_client_set_method(client, HTTP_METHOD_POST);
            esp_http_client_set_header(client, "Content-Type", "image/jpeg");
            //esp_http_client_set_post_field(client, (const char*) jpgBuf, bufLen);
            esp_http_client_set_post_field(client, (const char*) (curFrame->buf), curFrame->len);
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

            free_mem();

        
            ESP_LOGI(TAG, "Starting again!");

       } 

    }

}

void http_post_task_timer(void *pvParameters)
{
    while(1)
    {   
        vTaskDelay( 5000 / portTICK_PERIOD_MS ); // Wait for 5 sec 

        curFrame = esp_camera_fb_get();
        // I think this works great hardcoded for now
        // Will need to update the IP and such!  
        esp_http_client_config_t config = {
        .url = "http://10.0.0.118:8000/produce_detector/",
        //.url = "http://10.0.0.118",
        .event_handler = _http_event_handle,
        };
        
        esp_http_client_handle_t client = esp_http_client_init(&config);

        esp_http_client_set_method(client, HTTP_METHOD_POST);
        esp_http_client_set_header(client, "Content-Type", "image/jpeg");
        //esp_http_client_set_post_field(client, (const char*) jpgBuf, bufLen);
        esp_http_client_set_post_field(client, (const char*) (curFrame->buf), curFrame->len);
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

        free_mem();

    
        ESP_LOGI(TAG, "Starting again!");


    }

}