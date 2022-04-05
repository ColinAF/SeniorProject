
#ifndef BUTTON_H
#define BUTTON_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "camera.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

extern SemaphoreHandle_t xFrameSemaphore;
extern size_t bufLen; 
extern uint8_t* jpgBuf; 
extern camera_fb_t *curFrame;

void init_button();
static void IRAM_ATTR gpio_isr_handler(void* arg);

static void gpio_task_example(void* arg);

void free_mem();

#ifdef __cplusplus
}
#endif 

#endif