
#ifndef BUTTON_H
#define BUTTON_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "freertos/FreeRTOS.h"

void init_gpio( void );

static void gpio_task_example(void* arg);

static void IRAM_ATTR gpio_isr_handler(void* arg);

#ifdef __cplusplus
}
#endif 


#endif