/*
 * I know this code is a mess right now....
 * TODO
 * - Allow VS code compilation (Works Sorta)
 * - Remove redundant header files (Removed Some)
 * - Add include guards (Done)
 * - Fix include errors (Mostly Fixed) : still have that weird esp-who issue
 * - Fix Bad CMake executable error 
 * - Look into precompiled headers?
 * - Remove redundant code 
 * - Doxygen Docs 
 * - Fix "failed to get frame bug"
 * - Replace example connect wifi stuff
 * 
*/

//#include "who_motion_detection.hpp"

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

#include "esp_log.h"

// Compile the project in data collection mode (I could probably make this change more things to shorten compile time)
#define DATA_COLLECTION_MODE 

static const char* TAG = "APP_MAIN";

extern "C" void app_main()
{

    #ifdef DATA_COLLECTION_MODE

        //init_gpio();
        
        // Wifi should probably go in post? 
        ESP_ERROR_CHECK(nvs_flash_init());
        ESP_ERROR_CHECK(esp_netif_init());
        ESP_ERROR_CHECK(esp_event_loop_create_default());

        // Apparently I do still need this :(
        /* This helper function configures Wi-Fi or Ethernet, as selected in menuconfig.
            * Read "Establishing Wi-Fi or Ethernet Connection" section in
            * examples/protocols/README.md for more information about this function.
        */
        ESP_ERROR_CHECK(example_connect());

        //xTaskCreatePinnedToCore(&http_post_task, "http_post_task", 4096, NULL, 5, NULL, 1);
        
    #endif
    
    /*
        xQueueHttpFrame = xQueueCreate(2, sizeof(camera_fb_t *));
        register_motion_detection(xQueueAIFrame, NULL, NULL, xQueueHttpFrame);
        register_httpd(xQueueHttpFrame, NULL, true);  
    */

}

