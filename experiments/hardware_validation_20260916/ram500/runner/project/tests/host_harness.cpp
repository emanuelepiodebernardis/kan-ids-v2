#include "ram_host_stub.h"
#include <iostream>
unsigned host_calls=0;
uint16_t host_next_row=0;
bool host_corrupt_reference=false;
unsigned host_control_task_calls=0,host_control_probe_hwm_calls=0,host_heap_malloc_calls=0,host_heap_free_calls=0,host_heap_sample_calls=0;
bool host_control_task_active=false,host_task_create_fail=false,host_task_timeout=false,host_heap_malloc_fail=false,host_hwm_probe_fail=false;
size_t host_heap_allocation=0;
HostSerial Serial;
#include "../src/main.cpp"
int main() {
 setup(); ram_info();
#ifdef HW500_C3
 if(Serial.output.str().find("HELLO protocol=kanids-ram500-v2")==std::string::npos) return 2;
#else
 if(Serial.output.str().find("HELLO protocol=kanids-ram500-v1")==std::string::npos) return 2;
#endif
 if(Serial.output.str().find("INFO_DONE state=idle")==std::string::npos) return 3;
#ifdef HW500_C3
 Serial.input="RUN 0123456789abcdef\n"; loop();
#else
 ram_run();
#endif
 if(ram_state!=2 || host_calls!=1500 || host_next_row!=0 || ram_mismatches!=0) return 4;
 if(ram_checksum!=3*ram_checksum_per500 || ram_checksum_per500==0) return 5;
 if(Serial.output.str().find("DONE correct=1 reset_required=1")==std::string::npos) return 6;
 const auto calls=host_calls;
#ifdef HW500_C3
 Serial.input="RUN 0123456789abcdef\n";Serial.position=0;loop();
 if(ram_state!=2 || calls!=host_calls) return 7;
#else
 ram_run();
 if(ram_state!=3 || calls!=host_calls) return 7;
#endif
 std::cout<<"HOST_RAM_PASS model=" RAM_MODEL " checked="<<host_calls<<" checksum="<<ram_checksum<<" hardware_measurement=0\n";
 return 0;
}
