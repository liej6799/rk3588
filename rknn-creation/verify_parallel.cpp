// Verify a parallel And+Add model: inputs a,b (bool) x,y(float); out1=a&b, out2=x+y
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "rknn_api.h"

static uint16_t f2h(float f){uint32_t x;memcpy(&x,&f,4);uint32_t s=(x>>16)&0x8000;int e=int((x>>23)&0xff)-127+15;uint32_t m=x&0x7fffff;if(e<=0)return s;if(e>=31)return s|0x7c00;return s|(uint32_t(e)<<10)|(m>>13);}
static float h2f(uint16_t h){uint32_t s=(uint32_t(h)&0x8000)<<16,e=(h>>10)&0x1f,m=h&0x3ff,o;if(e==0){if(m==0)o=s;else{e=1;while(!(m&0x400)){m<<=1;e--;}m&=0x3ff;o=s|((e+112)<<23)|(m<<13);}}else if(e==31)o=s|0x7f800000|(m<<13);else o=s|((e+112)<<23)|(m<<13);float f;memcpy(&f,&o,4);return f;}

int main(int argc,char**argv){
  if(argc<2){fprintf(stderr,"usage: %s model.rknn\n",argv[0]);return 2;}
  FILE*fp=fopen(argv[1],"rb");fseek(fp,0,SEEK_END);long sz=ftell(fp);fseek(fp,0,SEEK_SET);
  void*model=malloc(sz);fread(model,1,sz,fp);fclose(fp);
  rknn_context ctx=0;
  int ret=rknn_init(&ctx,model,sz,0,NULL);
  printf("rknn_init: %d\n",ret);if(ret<0)return 1;
  rknn_input_output_num io;memset(&io,0,sizeof(io));
  rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io));
  printf("inputs=%u outputs=%u\n",io.n_input,io.n_output);
  for(uint32_t i=0;i<io.n_input;i++){rknn_tensor_attr a;memset(&a,0,sizeof(a));a.index=i;
    rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&a,sizeof(a));
    printf("  in[%u] name=%s n_elems=%u type=%d fmt=%d\n",i,a.name,a.n_elems,a.type,a.fmt);}
  for(uint32_t i=0;i<io.n_output;i++){rknn_tensor_attr a;memset(&a,0,sizeof(a));a.index=i;
    rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&a,sizeof(a));
    printf("  out[%u] name=%s n_elems=%u type=%d fmt=%d\n",i,a.name,a.n_elems,a.type,a.fmt);}

  // inputs: a,b bool[4]; x,y float[4]
  int8_t a[4]={1,0,1,1}, b[4]={1,1,0,1};
  float xf[4]={1.5f,2.0f,0.25f,3.0f}, yf[4]={0.5f,1.0f,4.0f,2.0f};
  uint16_t xh[4],yh[4]; for(int i=0;i<4;i++){xh[i]=f2h(xf[i]);yh[i]=f2h(yf[i]);}

  rknn_input ins[4];memset(ins,0,sizeof(ins));
  ins[0].index=0;ins[0].type=RKNN_TENSOR_INT8;ins[0].size=4;ins[0].fmt=RKNN_TENSOR_NCHW;ins[0].buf=a;ins[0].pass_through=1;
  ins[1].index=1;ins[1].type=RKNN_TENSOR_INT8;ins[1].size=4;ins[1].fmt=RKNN_TENSOR_NCHW;ins[1].buf=b;ins[1].pass_through=1;
  ins[2].index=2;ins[2].type=RKNN_TENSOR_FLOAT16;ins[2].size=8;ins[2].fmt=RKNN_TENSOR_NCHW;ins[2].buf=xh;
  ins[3].index=3;ins[3].type=RKNN_TENSOR_FLOAT16;ins[3].size=8;ins[3].fmt=RKNN_TENSOR_NCHW;ins[3].buf=yh;
  ret=rknn_inputs_set(ctx,4,ins);printf("inputs_set: %d\n",ret);if(ret<0)return 1;
  ret=rknn_run(ctx,NULL);printf("run: %d\n",ret);if(ret<0)return 1;

  rknn_output outs[2];memset(outs,0,sizeof(outs));
  outs[0].index=0;outs[0].want_float=0;  // bool out
  outs[1].index=1;outs[1].want_float=1;  // float out
  ret=rknn_outputs_get(ctx,2,outs,NULL);printf("outputs_get: %d\n",ret);if(ret<0)return 1;

  // Determine which output is bool vs float by size
  int bad=0;
  // out1 = a&b ; out2 = x+y. Identify ordering via output attr name not trivial here;
  // assume index order matches model (out1 then out2).
  int8_t*o1=(int8_t*)outs[0].buf; float*o2=(float*)outs[1].buf;
  printf("AND out:"); for(int i=0;i<4;i++) printf(" %d", o1[i]); printf("  (want");
  for(int i=0;i<4;i++) printf(" %d", a[i]&&b[i]); printf(")\n");
  printf("ADD out:"); for(int i=0;i<4;i++) printf(" %.3f", o2[i]); printf("  (want");
  for(int i=0;i<4;i++) printf(" %.3f", xf[i]+yf[i]); printf(")\n");
  for(int i=0;i<4;i++){ if((o1[i]!=0)!=((a[i]&&b[i])!=0)) bad++; if(o2[i]!=o2[i]||__builtin_fabsf(o2[i]-(xf[i]+yf[i]))>0.1f) bad++; }
  printf("%s mismatches=%d\n", bad?"FAIL":"PASS", bad);
  rknn_outputs_release(ctx,2,outs);rknn_destroy(ctx);
  return bad?1:0;
}
