// Button code.. Update this text later plz...

#include "button.h"
#include "camera.h"
#include "post.h"

#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_event.h"

#include "freertos/semphr.h"

#define GPIO_INPUT_IO_0 GPIO_NUM_15
#define GPIO_INPUT_PIN_SEL  (1ULL<<GPIO_INPUT_IO_0)
#define ESP_INTR_FLAG_DEFAULT 0

static const char* TAG = "BUTTON";

static QueueHandle_t gpio_evt_queue = NULL;

void init_button( void )
{
    
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
    //gpio_install_isr_service(ESP_INTR_FLAG_DEFAULT);

    //hook isr handler for specific gpio pin
    gpio_isr_handler_add(GPIO_INPUT_IO_0, gpio_isr_handler, (void*) GPIO_INPUT_IO_0);

    // Create a semaphore to make operations to curFrame atomic 
    // Potentailly put this in a helper function! 
    xFrameSemaphore = xSemaphoreCreateBinary();

    if( xFrameSemaphore == NULL )
    {
        ESP_LOGE(TAG, "Failed to create semaphore.");

    }
    else
    {
        ESP_LOGI(TAG, "Semaphore created!");
        xSemaphoreTake(xFrameSemaphore, 1000 / portTICK_PERIOD_MS ); 

    }
    // Create a semaphore to make operations to curFrame atomic


    return; 
}

static void IRAM_ATTR gpio_isr_handler(void* arg)
{
    
    uint32_t gpio_num = (uint32_t) arg;
    xQueueSendFromISR(gpio_evt_queue, &gpio_num, NULL);

}

// Semaphore can't handel nested calls? 
static void gpio_task_example(void* arg)
{

    gpio_num_t io_num;
    for(;;) {
        if(xQueueReceive(gpio_evt_queue, &io_num, portMAX_DELAY)) {
            
            printf("GPIO[%d] intr, val: %d\n", io_num, gpio_get_level(io_num));

            // Loop until we actually get the frame most likely!!!
            curFrame = esp_camera_fb_get();
            if(curFrame)
            {
                ESP_LOGI(TAG, "Got fb!");
                ESP_LOGI(TAG, "Lenght: %d", curFrame->len);
                ESP_LOGI(TAG, "Width: %d", curFrame->width);
                ESP_LOGI(TAG, "Height: %d", curFrame->height);

                esp_camera_fb_return(curFrame);

                // Unblock the curFrame resource
                if( xSemaphoreGive(xFrameSemaphore) != pdTRUE )
                {
                    ESP_LOGE(TAG, "Semaphore Give Error!");

                }
            }   
            // Loop until we actually get the frame most likely!!!    
                 
        }
    }
}
