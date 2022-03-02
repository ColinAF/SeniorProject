#include "who_camera.h"
#include "who_motion_detection.hpp"
#include "app_wifi.h"
#include "app_httpd.hpp"
#include "app_mdns.h"

#include "driver/gpio.h"
#include "esp_log.h"

#define GPIO_INPUT_IO_0 GPIO_NUM_15
#define GPIO_INPUT_PIN_SEL  (1ULL<<GPIO_INPUT_IO_0)
#define ESP_INTR_FLAG_DEFAULT 0

static QueueHandle_t gpio_evt_queue = NULL;

static QueueHandle_t xQueueAIFrame = NULL;
static QueueHandle_t xQueueHttpFrame = NULL;


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

extern "C" void app_main()
{
    app_wifi_main();
    //app_mdns_main();
    init_gpio();
    /*
    xQueueAIFrame = xQueueCreate(2, sizeof(camera_fb_t *));
    xQueueHttpFrame = xQueueCreate(2, sizeof(camera_fb_t *));

    register_camera(PIXFORMAT_RGB565, FRAMESIZE_QVGA, 2, xQueueAIFrame);
    */

    //register_motion_detection(xQueueAIFrame, NULL, NULL, xQueueHttpFrame);
    //register_httpd(xQueueHttpFrame, NULL, true);
}

