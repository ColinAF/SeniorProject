/*
 * I know this code is a mess right now....
 * TODO
 * - Organize Code into sperate files (GPIO, HTTP, CAM)
 * - Remove redundant code 
 * - Optamize Preformance
 * - Fix "failed to get frame bug"
*/

//#include "who_motion_detection.hpp"

#include "driver/gpio.h"
#include "esp_log.h"

// HTTP Stuff 
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "protocol_examples_common.h"

#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include "lwip/netdb.h"
#include "lwip/dns.h"


#include "esp_http_client.h"
#include "camera.h"
#include "img_converters.h"

// Compile the project in data collection mode
#define DATA_COLLECTION_MODE 1

#define GPIO_INPUT_IO_0 GPIO_NUM_15
#define GPIO_INPUT_PIN_SEL  (1ULL<<GPIO_INPUT_IO_0)
#define ESP_INTR_FLAG_DEFAULT 0

static QueueHandle_t gpio_evt_queue = NULL;
static QueueHandle_t xQueueAIFrame = NULL;

static const char* TAG = "HTTP Client";

static void IRAM_ATTR gpio_isr_handler(void* arg)
{
    uint32_t gpio_num = (uint32_t) arg;
    xQueueSendFromISR(gpio_evt_queue, &gpio_num, NULL);
}

static void gpio_task_example(void* arg)
{
    gpio_num_t io_num;
    for(;;) {
        if(xQueueReceive(gpio_evt_queue, &io_num, portMAX_DELAY)) {
            printf("GPIO[%d] intr, val: %d\n", io_num, gpio_get_level(io_num));

        }
    }
}

void init_gpio( void )
{
    
    ESP_LOGI("GPIO", "Init GPIO");

    //zero-initialize the config structure.
    gpio_config_t io_conf = {};
    //disable interrupt
    io_conf.intr_type = GPIO_INTR_POSEDGE;
    //set as input mode
    io_conf.mode = GPIO_MODE_INPUT;
    //bit mask of the pins that you want to set,e.g.GPIO18/19
    io_conf.pin_bit_mask = GPIO_INPUT_PIN_SEL;
    //disable pull-down mode
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    //disable pull-up mode
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE;

    //configure GPIO with the given settings
    gpio_config(&io_conf);

    //create a queue to handle gpio event from isr
    gpio_evt_queue = xQueueCreate(10, sizeof(gpio_num_t));
    //start gpio task
    xTaskCreate(gpio_task_example, "gpio_task_example", 2048, NULL, 10, NULL);

    //install gpio isr service
    gpio_install_isr_service(ESP_INTR_FLAG_DEFAULT);

    //hook isr handler for specific gpio pin
    gpio_isr_handler_add(GPIO_INPUT_IO_0, gpio_isr_handler, (void*) GPIO_INPUT_IO_0);

    return; 
}

// Http task...
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

static void http_post_task(void *pvParameters)
{
    camera_fb_t *curFrame = NULL; 
    size_t bufLen; 
    uint8_t* jpegBuf; 
        
    while(1)
    {   

        if(xQueueReceive(xQueueAIFrame, &(curFrame), portMAX_DELAY))
        {  
            ESP_LOGI(TAG, "Got fb!");
            ESP_LOGI(TAG, "Lenght: %d", curFrame->len);
            ESP_LOGI(TAG, "Width: %d", curFrame->width);
            ESP_LOGI(TAG, "Height: %d", curFrame->height);

            frame2jpg(curFrame, 80, &jpegBuf, &bufLen);
        }
               
        esp_http_client_config_t config = {
        .url = "http://10.0.0.111:80",
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
        
        for(int countdown = 10; countdown >= 0; countdown--) {
            ESP_LOGI(TAG, "%d... ", countdown);
            vTaskDelay(1000 / portTICK_PERIOD_MS);
        }

        ESP_LOGI(TAG, "Starting again!");

    }

}

// Http task...
extern "C" void app_main()
{

    #ifdef DATA_COLLECTION_MODE

        //init_gpio();

        ESP_ERROR_CHECK(nvs_flash_init());
        ESP_ERROR_CHECK(esp_netif_init());
        ESP_ERROR_CHECK(esp_event_loop_create_default());

        // Apparently I do still need this :(
        /* This helper function configures Wi-Fi or Ethernet, as selected in menuconfig.
            * Read "Establishing Wi-Fi or Ethernet Connection" section in
            * examples/protocols/README.md for more information about this function.
        */
        ESP_ERROR_CHECK(example_connect());

        xQueueAIFrame = xQueueCreate(2, sizeof(camera_fb_t *));
        register_camera(PIXFORMAT_RGB565, FRAMESIZE_QVGA, 2, xQueueAIFrame);

        xTaskCreatePinnedToCore(&http_post_task, "http_post_task", 4096, NULL, 5, NULL, 1);
        
    #endif
    
    /*
        xQueueHttpFrame = xQueueCreate(2, sizeof(camera_fb_t *));
        register_motion_detection(xQueueAIFrame, NULL, NULL, xQueueHttpFrame);
        register_httpd(xQueueHttpFrame, NULL, true);  
    */
}

