/* Deterministic software transport/state tests, NOT hardware RAM validation. */
#include "ram_host_stub.h"
#include <iostream>
#include <cstring>
unsigned host_calls=0;
uint16_t host_next_row=0;
bool host_corrupt_reference=false;
unsigned host_control_task_calls=0,host_control_probe_hwm_calls=0,host_heap_malloc_calls=0,host_heap_free_calls=0,host_heap_sample_calls=0;
bool host_control_task_active=false,host_task_create_fail=false,host_task_timeout=false,host_heap_malloc_fail=false,host_hwm_probe_fail=false;
size_t host_heap_allocation=0;
HostSerial Serial;
#include "../src/main.cpp"
static const char *token="0123456789abcdef";
static void require(bool good,const char *message) {if(!good)throw std::runtime_error(message);}
static std::string command(const std::string &s) {
 Serial.output.str("");Serial.output.clear();Serial.input=s+"\n";Serial.position=0;loop();
 return Serial.output.str();
}
static std::string baseline_bytes() {
 std::string bytes;
 #define KEEP(v) bytes.append(reinterpret_cast<const char *>(&(v)),sizeof(v))
 KEEP(ram_checksum);KEEP(ram_mismatches);KEEP(ram_checksum_per500);KEEP(ram_baseline_complete);
 KEEP(ram_baseline_correct);KEEP(ram_baseline_ram_valid);KEEP(ram_heap_start_info);KEEP(ram_heap_finish_info);
 KEEP(ram_heap_sample_min);KEEP(ram_loop_hwm_before);KEEP(ram_loop_hwm_after);KEEP(ram_loop_hwm_min);KEEP(ram_loop_reserved);
 KEEP(ram_run_token);
 #undef KEEP
 return bytes;
}
static std::string control_bytes() {
 std::string bytes;
 #define KEEP(v) bytes.append(reinterpret_cast<const char *>(&(v)),sizeof(v))
 KEEP(ram_control_complete);KEEP(ram_control_error);KEEP(ram_control_stack_before);KEEP(ram_control_stack_after);
 KEEP(ram_control_heap_before);KEEP(ram_control_heap_during);KEEP(ram_control_heap_after);KEEP(ram_control_stack_ok);KEEP(ram_control_heap_ok);
 #undef KEEP
 return bytes;
}
static void accepted_result(const std::string &s) {
 require(s.find("\nREPORT_BEGIN run_token=0123456789abcdef\nBEGIN rows=500 passes=3 scope=load_predict_reference_check\nRAM_RESULT board=c3 model=")==0,"result framing");
 require(s.find(" correct=1 ram_valid=1 ")!=std::string::npos,"result validity");
 require(s.find("\nDONE correct=1 reset_required=1 run_token=0123456789abcdef\n")!=std::string::npos,"done token");
 require(host_calls==1500,"workload must execute exactly once");
}
static void accepted_control(const std::string &s) {
 require(s.find("\nCONTROL_BEGIN run_token=0123456789abcdef\nCONTROL stack_pass=1 heap_pass=1 correct=1 ")==0,"control framing");
 require(s.find("baseline_already_saved=1 run_token=0123456789abcdef\n")!=std::string::npos,"control token");
 require(host_control_task_calls==1 && host_heap_malloc_calls==1 && host_heap_free_calls==1,"control only once");
}
int main(int argc,char **argv) {
 try {
  setup();const std::string scenario=argc>1?argv[1]:"replay";
  if(scenario=="invalid") {
   const char *bad[]={"RUN","CONTROL","CHECK","RUN 0123456789abcdeF","RUN 0123456789abcdeg","RUN 0123456789abcde","RUN 0123456789abcdef0","RUN 0123456789abcdef trailing","RESULT 0123456789abcdef","CHECK 0123456789abcdef"};
   for(auto c:bad) {const auto reply=command(c);require(reply.find("ERROR reason=")!=std::string::npos,"invalid command accepted");require(ram_state==0 && host_calls==0 && !ram_baseline_complete,"invalid query mutated state");}
   accepted_result(command(std::string("RUN ")+token).substr(std::string("RUN_BEGIN run_token=").size()+16+1));
  } else {
   if(scenario=="disconnect")Serial.connected=false;
   if(scenario=="partial")Serial.write_budget=137;
   if(scenario=="short")Serial.write_limit=16;
   auto first=command(std::string("RUN ")+token);
   require(host_calls==1500 && ram_state==2 && ram_baseline_complete,"baseline not frozen before report");
   const auto snapshot=baseline_bytes();const auto samples=host_heap_sample_calls;
   if(scenario=="disconnect")require(Serial.write_calls==0,"write attempted while disconnected");
   if(scenario=="partial")require(first.find("\nDONE ")==std::string::npos,"partial transport simulation ineffective");
   Serial.connected=true;Serial.write_budget=std::numeric_limits<size_t>::max();
   const auto complete=command(std::string("RESULT ")+token);accepted_result(complete);
   require(Serial.max_request<=32,"oversized CDC write");require(Serial.tx_timeout==100,"CDC timeout not bounded");
   for(auto query:{"RESULT fedcba9876543210","CHECK fedcba9876543210","RUN 0123456789abcdef","RUN fedcba9876543210","CHECK","CONTROL","RESULT 0123456789abcdeF"}) {
    require(command(query).find("ERROR reason=")!=std::string::npos,"bad postbaseline query accepted");
    require(ram_state==2 && baseline_bytes()==snapshot && host_calls==1500,"bad query destroyed baseline");
   }
   require(command(std::string("RESULT ")+token)==complete,"result replay changed payload");
   require(host_heap_sample_calls==samples && baseline_bytes()==snapshot,"result replay resampled baseline");
   if(scenario=="task_create_fail")host_task_create_fail=true;
   if(scenario=="task_timeout")host_task_timeout=true;
   if(scenario=="heap_control_fail")host_heap_malloc_fail=true;
   if(scenario=="stack_control_fail")host_hwm_probe_fail=true;
   if(scenario=="control_partial")Serial.write_budget=100;
   const auto first_control=command(std::string("CHECK ")+token);
   require(ram_control_complete,"control must freeze before report");
   const auto control_snapshot=control_bytes();const auto after_control_samples=host_heap_sample_calls;
   Serial.write_budget=std::numeric_limits<size_t>::max();
   const auto control=command(std::string("CHECK ")+token);
   if(scenario=="task_create_fail" || scenario=="task_timeout") {
    require(control.find(scenario=="task_create_fail"?"ERROR reason=control_task_create_failed":"ERROR reason=control_task_timeout")!=std::string::npos,"failed control not retained");
    require(host_control_task_calls==1 && host_heap_malloc_calls==0,"failed control repeated");
   } else if(scenario=="heap_control_fail" || scenario=="stack_control_fail") {
    require(control.find(scenario=="heap_control_fail"?"CONTROL stack_pass=1 heap_pass=0 correct=0":"CONTROL stack_pass=0 heap_pass=1 correct=0")!=std::string::npos,"failed positive control was hidden");
    require(host_control_task_calls==1 && host_heap_malloc_calls==1 && host_heap_free_calls==1,"failed positive control repeated");
   } else accepted_control(control);
   if(scenario=="control_partial")require(first_control!=control,"partial control simulation ineffective");
   require(command(std::string("CHECK ")+token)==control,"control replay changed payload");
   require(control_bytes()==control_snapshot && host_heap_sample_calls==after_control_samples,"control replay repeated sampling");
   require(command(std::string("RESULT ")+token)==complete,"postcontrol result differs from baseline");
   require(baseline_bytes()==snapshot && host_calls==1500,"control changed baseline or predictions");
   if(scenario=="emit")std::cout<<complete<<control;
  }
  std::cout<<"HOST_C3_PROTOCOL_PASS model=" RAM_MODEL " scenario="<<scenario<<" checked="<<host_calls<<" hardware_measurement=0\n";
  return 0;
 } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
