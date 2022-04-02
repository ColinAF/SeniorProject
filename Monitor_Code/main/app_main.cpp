/*
 * TODO
 * - Fix Bad CMake executable error 
 * - Doxygen Docs 
 * - Replace example connect wifi stuff
 * - Get rid of example connect and use proper wifi station code 
 * - Add button loop!  
 * - Make Wifi config more friendly! 
*/

//#include "wifi.h"

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_event.h"
#include "nvs_flash.h"

#include "esp_log.h"

#include "button.h"
#include "post.h"

#include "esp_wifi.h"
#include "protocol_examples_common.h"

#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include "lwip/netdb.h"
#include "lwip/dns.h"

// Compile the project in data collection mode (I could probably make this change more things to shorten compile time)
#define DATA_COLLECTION_MODE 

#ifndef DATA_COLLECTION_MODE 

    #include "who_motion_detection.hpp"
    #include "who_camera.h"
    
    static QueueHandle_t xQueueAIFrame = NULL;
    static QueueHandle_t xQueueHttpFrame = NULL;

#endif

static const char* TAG = "APP_MAIN";

extern "C" void app_main()
{
    

    #ifdef DATA_COLLECTION_MODE

        init_camera(PIXFORMAT_RGB565, FRAMESIZE_VGA, 2);
        ESP_LOGI(TAG, "Init Cam");

        /* Stuff for proper wifi station
        //Initialize NVS
        esp_err_t ret = nvs_flash_init();
        if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
        }
        ESP_ERROR_CHECK(ret);

        ESP_LOGI(TAG, "ESP_WIFI_MODE_STA");
        wifi_init_sta();
        */  
        
        ESP_ERROR_CHECK(nvs_flash_init());
        ESP_ERROR_CHECK(esp_netif_init());
        ESP_ERROR_CHECK(esp_event_loop_create_default());

        // Apparently I do still need this :(
        /* This helper function configures Wi-Fi or Ethernet, as selected in menuconfig.
            * Read "Establishing Wi-Fi or Ethernet Connection" section in
            * examples/protocols/README.md for more information about this function.
        */
        ESP_ERROR_CHECK(example_connect());
        ESP_LOGI(TAG, "Init Wifi");
        
        // Press button to capture an image
        init_button(); 
        ESP_LOGI(TAG, "Init Button");

        xTaskCreatePinnedToCore(&http_post_task, "http_post_task", 4096, NULL, 5, NULL, 1);
        
    #else 

        // Very Very broken!
        xQueueAIFrame = xQueueCreate(2, sizeof(camera_fb_t *));
        xQueueHttpFrame = xQueueCreate(2, sizeof(camera_fb_t *));

        register_camera(PIXFORMAT_RGB565, FRAMESIZE_QVGA, 2, xQueueAIFrame);
        register_motion_detection(xQueueAIFrame, NULL, NULL, xQueueHttpFrame);


    #endif
    

}

