// Verify a 2-input fp16 elementwise-Add .rknn with fp16-SAFE bounded inputs,
// so arbitrarily large N can be checked (x[i]=i overflows fp16 above 2048).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "rknn_api.h"
static void chk(int r,const char*w){if(r<0){fprintf(stderr,"%s failed %d\n",w,r);exit(1);}}
static uint16_t f2h(float f){uint32_t x;memcpy(&x,&f,4);uint32_t s=(x>>16)&0x8000;int e=int((x>>23)&0xff)-127+15;uint32_t m=x&0x7fffff;if(e<=0)return s;if(e>=31)return s|0x7c00;return uint16_t(s|(uint32_t(e)<<10)|(m>>13));}
int main(int argc,char**argv){
  const char*model=argc>1?argv[1]:"/tmp/g1.rknn";
  rknn_context ctx=0; chk(rknn_init(&ctx,(void*)model,0,0,nullptr),"init");
  rknn_input_output_num io={}; chk(rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io)),"ionum");
  rknn_tensor_attr a={}; a.index=0; chk(rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&a,sizeof(a)),"attr");
  uint32_t n=a.n_elems;
  std::vector<uint16_t> xh(n),yh(n); std::vector<float> xf(n),yf(n);
  for(uint32_t i=0;i<n;i++){xf[i]=float(i%64)*0.5f; yf[i]=float(i%32)*0.25f; xh[i]=f2h(xf[i]); yh[i]=f2h(yf[i]);}
  std::vector<rknn_input> in(io.n_input); memset(in.data(),0,in.size()*sizeof(rknn_input));
  for(uint32_t i=0;i<io.n_input;i++){in[i].index=i;in[i].type=RKNN_TENSOR_FLOAT16;in[i].fmt=a.fmt;in[i].buf=(i==0)?(void*)xh.data():(void*)yh.data();in[i].size=n*2;}
  chk(rknn_inputs_set(ctx,io.n_input,in.data()),"set");
  chk(rknn_run(ctx,nullptr),"run");
  rknn_output o={}; o.index=0; o.want_float=1; chk(rknn_outputs_get(ctx,1,&o,nullptr),"get");
  float*r=(float*)o.buf; int bad=0; uint32_t firstbad=0;
  for(uint32_t i=0;i<n;i++){float e=xf[i]+yf[i]; if(fabs(r[i]-e)>1e-2f){if(!bad)firstbad=i; bad++;}}
  printf("N=%u mismatches=%d %s\n",n,bad,bad?"FAIL":"PASS");
  if(bad) printf("  first bad @%u: got %.3f want %.3f\n",firstbad,r[firstbad],xf[firstbad]+yf[firstbad]);
  rknn_outputs_release(ctx,1,&o); rknn_destroy(ctx); return bad?1:0;
}
