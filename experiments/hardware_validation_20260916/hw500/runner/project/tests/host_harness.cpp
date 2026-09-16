#include "hw500_host_stub.h"
#include <cassert>
#include <iostream>
uint64_t host_clock=0;
uint32_t host_prediction_us=10000, host_observed_calls=0;
uint16_t host_expected_row=0;
unsigned host_forbidden_io=0;
bool host_busy=false, host_row_order_enabled=true;
bool host_wdt_initialized=true,host_idle_subscribed=true,host_loop_subscribed=false;
bool host_wdt_delete_failure=false,host_wdt_restore_failure=false;
std::vector<std::pair<uint64_t,int>> host_led_edges;
HostSerial Serial;
HostESP ESP;
#include "../src/main.cpp"
static void command(const std::string &text) { Serial.input+=text+"\n"; loop(); }
static size_t occurrences(const std::string &s,const std::string &p) {
  size_t n=0,pos=0; while((pos=s.find(p,pos))!=std::string::npos){++n;pos+=p.size();}return n;
}
int main(int argc,char **argv) {
  setup();
  if(argc==2 && std::string(argv[1])=="--protocol-fixture") {
    const char *commands[]={"INFO","LEDTEST","CONFIRM_LED","ARM","RUN","INFO","ARM","RUN"};
    const char *stages[]={"info","ledtest","ledconfirm","arm_01","run_01","info_after","arm_02","run_02"};
    for(unsigned i=0;i<8;++i){size_t before=Serial.output.str().size();command(commands[i]);std::cout<<"===FIXTURE stage="<<stages[i]<<"===\n"<<Serial.output.str().substr(before);}
    return pilot_state==DONE?0:1;
  }
  assert(Serial.output.str().empty());
  command("INFO");assert(Serial.output.str().find("protocol=kanids-hw500-v1")!=std::string::npos);
  command("RUN");assert(pilot_state==IDLE);
  command("CONFIRM_LED");assert(!led_confirmed);
  command(std::string("RUN\0garbage",11));assert(pilot_state==IDLE);
  command(std::string(28,'A'));assert(pilot_state==IDLE);
  command("INFO\r");assert(Serial.output.str().find("INFO_DONE state=idle")!=std::string::npos);
  command("LEDTEST");assert(host_led_edges.size()==7);
  assert(host_led_edges[2].first-host_led_edges[1].first==2000000);
  assert(host_led_edges[3].first-host_led_edges[2].first==2000000);
  assert(host_led_edges[4].first-host_led_edges[3].first==4000000);
  command("CONFIRM_LED");assert(led_confirmed);
  uint32_t expected_sum=0; for(uint16_t i=0;i<500;++i)expected_sum+=HB_RD8(HB_EXPECTED[i]);
  command("ARM");assert(pilot_state==READY && checksum_per500==expected_sum);
  assert(calibration_count==500 && calibration_us==5000000 && selected_count==12000);
  command("ARM");assert(pilot_state==READY);
  /* Exercise Mega micros wrap during sequence and C3 beyond 32-bit time. */
  host_clock=(1ULL<<32)-10000000ULL;
  command("RUN");assert(pilot_state==DONE && host_idle_subscribed);
  assert(event_count==18);
  uint64_t active_begin=0,active_end=0;
  for(uint8_t i=0;i<event_count;++i){
    if(i)assert(events[i].relative_us>=events[i-1].relative_us);
    if(events[i].kind==ACTIVE_BEGIN)active_begin=events[i].relative_us;
    if(events[i].kind==ACTIVE_END)active_end=events[i].relative_us;
  }
  assert(active_end-active_begin==120000000 && active_begin==22000000);
  assert(observed_checksum==24*expected_sum);
  command("RUN");assert(pilot_state==DONE);
  command("ARM");assert(pilot_state==READY);command("RUN");assert(pilot_state==DONE);
  assert(occurrences(Serial.output.str(),"WAIT_DONE reset_required=0")==2);
  uint32_t out=0;
  assert(choose_count(500,100000,&out)&&out==600000);
  assert(!choose_count(500,99999,&out));assert(!choose_count(500,30000001,&out));
  assert(!choose_count(501,100000,&out));assert(!choose_count(CAL_MAX,100000,&out));
  assert(choose_count(500,30000000,&out)&&out==2000);
  host_prediction_us=100;command("ARM");
  assert(calibration_count==1000 && calibration_us==100000 && selected_count==1200000);
  pilot_state=DONE;host_prediction_us=10000;
#ifdef HW500_C3
  WdtGuard a;assert(a.begin()&&!host_idle_subscribed);assert(a.end()&&host_idle_subscribed);
  host_loop_subscribed=true;WdtGuard b;assert(!b.begin());host_loop_subscribed=false;
  host_wdt_delete_failure=true;WdtGuard c;assert(!c.begin());host_wdt_delete_failure=false;
  host_wdt_initialized=false;WdtGuard d;assert(d.begin()&&d.end());host_wdt_initialized=true;
  host_idle_subscribed=false;WdtGuard e;assert(e.begin()&&e.end());assert(!host_idle_subscribed);host_idle_subscribed=true;
  WdtGuard f;assert(f.begin());host_wdt_restore_failure=true;assert(!f.end());host_wdt_restore_failure=false;assert(f.end());
#endif
  const uint32_t calls=host_observed_calls;
  host_busy=true;assert(pilot_active_batch(1000)==2*expected_sum);host_busy=false;
  assert(host_observed_calls-calls==1000 && host_forbidden_io==0);
  /* Test checksum mismatch fail-closed via deliberately invalid expected sum. */
  command("ARM");checksum_per500+=1;command("RUN");assert(pilot_state==FAILED);
  command("ARM");assert(pilot_state==FAILED);
  std::cout<<"HOST_FIRMWARE_PASS variant="<<PILOT_VARIANT<<" model_bytes="<<HB_MODEL_BYTES<<" expected_per500="<<expected_sum
    <<" checks=all500_predictions,row_order,quiet_batch,protocol,rearm,markers,timer_wrap,calibration,count_limits,watchdog,checksum_failclosed\n";
}
