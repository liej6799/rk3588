// verify_mixed.cpp - Verify mixed CPU(And)+NPU(Add) topologies
// Handles: And→Cast→Add, Add→Greater→And, And→Cast→Add→Greater→And,
//          parallel And+Add, parallel 2×And+Add
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
  for(uint32_t i=0;i<io.n_input;i++){
    rknn_tensor_attr a;memset(&a,0,sizeof(a));a.index=i;
    rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&a,sizeof(a));
    printf("  in[%u] name=%s n_elems=%u type=%d fmt=%d\n",i,a.name,a.n_elems,a.type,a.fmt);
  }
  for(uint32_t i=0;i<io.n_output;i++){
    rknn_tensor_attr a;memset(&a,0,sizeof(a));a.index=i;
    rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&a,sizeof(a));
    printf("  out[%u] name=%s n_elems=%u type=%d fmt=%d\n",i,a.name,a.n_elems,a.type,a.fmt);
  }

  // Identify topology by input/output count and names
  rknn_tensor_attr a0;memset(&a0,0,sizeof(a0));a0.index=0;
  rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&a0,sizeof(a0));
  rknn_tensor_attr o0;memset(&o0,0,sizeof(o0));o0.index=0;
  rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&o0,sizeof(o0));

  int bad=0;

  // ── And→Cast→Add→Greater→And chain (4 in: a,b,x,c; 1 out: bool) ──
  if(io.n_input==4 && io.n_output==1){
    printf("[And→Cast→Add→Greater→And chain]\n");
    int8_t a[4]={1,0,1,0},b[4]={1,1,0,0},c[4]={1,1,1,1};
    float xf[4]={0.5f,0.5f,0.5f,0.5f};
    uint16_t xh[4];for(int i=0;i<4;i++)xh[i]=f2h(xf[i]);
    rknn_input ins[4];memset(ins,0,sizeof(ins));
    ins[0].index=0;ins[0].type=RKNN_TENSOR_INT8;ins[0].size=4;ins[0].fmt=RKNN_TENSOR_NCHW;ins[0].buf=a;ins[0].pass_through=1;
    ins[1].index=1;ins[1].type=RKNN_TENSOR_INT8;ins[1].size=4;ins[1].fmt=RKNN_TENSOR_NCHW;ins[1].buf=b;ins[1].pass_through=1;
    ins[2].index=2;ins[2].type=RKNN_TENSOR_FLOAT16;ins[2].size=8;ins[2].fmt=RKNN_TENSOR_NCHW;ins[2].buf=xh;
    ins[3].index=3;ins[3].type=RKNN_TENSOR_INT8;ins[3].size=4;ins[3].fmt=RKNN_TENSOR_NCHW;ins[3].buf=c;ins[3].pass_through=1;
    ret=rknn_inputs_set(ctx,4,ins);printf("inputs_set: %d\n",ret);
    ret=rknn_run(ctx,NULL);printf("run: %d\n",ret);
    rknn_output out;memset(&out,0,sizeof(out));out.want_float=0;
    rknn_outputs_get(ctx,1,&out,NULL);printf("outputs_get: %d size=%d\n",ret,out.size);
    int8_t*od=(int8_t*)out.buf;
    // t1=a&b=[1,0,0,0]; t1f=[1,0,0,0]; t2=t1f+x=[1.5,0.5,0.5,0.5]; >0.5=[1,0,0,0]; &c=[1,0,0,0]
    int8_t exp[4]={1,0,0,0};
    printf("out:");for(int i=0;i<4;i++)printf(" %d",od[i]);printf("  (want");
    for(int i=0;i<4;i++)printf(" %d",exp[i]);printf(")\n");
    for(int i=0;i<4;i++)if((od[i]!=0)!=(exp[i]!=0))bad++;
    rknn_outputs_release(ctx,1,&out);
  }
  // ── And→Cast→Add chain (3 in: a,b,e; 1 out: float) ──
  else if(io.n_input==3 && io.n_output==1 && o0.type==RKNN_TENSOR_FLOAT16){
    printf("[And→Cast→Add chain]\n");
    int8_t a[4]={1,0,1,0},b[4]={1,1,0,0};
    float ef[4]={0.5f,0.5f,0.5f,0.5f};
    uint16_t eh[4];for(int i=0;i<4;i++)eh[i]=f2h(ef[i]);
    rknn_input ins[3];memset(ins,0,sizeof(ins));
    ins[0].index=0;ins[0].type=RKNN_TENSOR_INT8;ins[0].size=4;ins[0].fmt=RKNN_TENSOR_NCHW;ins[0].buf=a;ins[0].pass_through=1;
    ins[1].index=1;ins[1].type=RKNN_TENSOR_INT8;ins[1].size=4;ins[1].fmt=RKNN_TENSOR_NCHW;ins[1].buf=b;ins[1].pass_through=1;
    ins[2].index=2;ins[2].type=RKNN_TENSOR_FLOAT16;ins[2].size=8;ins[2].fmt=RKNN_TENSOR_NCHW;ins[2].buf=eh;
    ret=rknn_inputs_set(ctx,3,ins);printf("inputs_set: %d\n",ret);
    ret=rknn_run(ctx,NULL);printf("run: %d\n",ret);
    rknn_output out;memset(&out,0,sizeof(out));out.want_float=1;
    rknn_outputs_get(ctx,1,&out,NULL);printf("outputs_get: %d size=%d\n",ret,out.size);
    float*od=(float*)out.buf;
    // t=a&b=[1,0,0,0]; tf=[1,0,0,0]; out=tf+e=[1.5,0.5,0.5,0.5]
    float exp[4]={1.5f,0.5f,0.5f,0.5f};
    printf("out:");for(int i=0;i<4;i++)printf(" %.3f",od[i]);printf("  (want");
    for(int i=0;i<4;i++)printf(" %.3f",exp[i]);printf(")\n");
    for(int i=0;i<4;i++)if(od[i]!=od[i]||__builtin_fabsf(od[i]-exp[i])>0.1f)bad++;
    rknn_outputs_release(ctx,1,&out);
  }
  // ── Add→Greater→And chain (3 in: x,y,c; 1 out: bool) ──
  else if(io.n_input==3 && io.n_output==1 && o0.type==RKNN_TENSOR_BOOL){
    printf("[Add→Greater→And chain]\n");
    float xf[4]={1.5f,2.0f,0.25f,3.0f},yf[4]={0.5f,1.0f,4.0f,2.0f};
    uint16_t xh[4],yh[4];for(int i=0;i<4;i++){xh[i]=f2h(xf[i]);yh[i]=f2h(yf[i]);}
    int8_t c[4]={1,0,1,0};
    rknn_input ins[3];memset(ins,0,sizeof(ins));
    ins[0].index=0;ins[0].type=RKNN_TENSOR_FLOAT16;ins[0].size=8;ins[0].fmt=RKNN_TENSOR_NCHW;ins[0].buf=xh;
    ins[1].index=1;ins[1].type=RKNN_TENSOR_FLOAT16;ins[1].size=8;ins[1].fmt=RKNN_TENSOR_NCHW;ins[1].buf=yh;
    ins[2].index=2;ins[2].type=RKNN_TENSOR_INT8;ins[2].size=4;ins[2].fmt=RKNN_TENSOR_NCHW;ins[2].buf=c;ins[2].pass_through=1;
    ret=rknn_inputs_set(ctx,3,ins);printf("inputs_set: %d\n",ret);
    ret=rknn_run(ctx,NULL);printf("run: %d\n",ret);
    rknn_output out;memset(&out,0,sizeof(out));out.want_float=0;
    rknn_outputs_get(ctx,1,&out,NULL);printf("outputs_get: %d size=%d\n",ret,out.size);
    int8_t*od=(int8_t*)out.buf;
    // s=x+y=[2,3,4.25,5]; >0=[1,1,1,1]; &c=[1,0,1,0]
    int8_t exp[4]={1,0,1,0};
    printf("out:");for(int i=0;i<4;i++)printf(" %d",od[i]);printf("  (want");
    for(int i=0;i<4;i++)printf(" %d",exp[i]);printf(")\n");
    for(int i=0;i<4;i++)if((od[i]!=0)!=(exp[i]!=0))bad++;
    rknn_outputs_release(ctx,1,&out);
  }
  // ── Parallel 2×And + 1×Add (6 in, 3 out) ──
  else if(io.n_input==6 && io.n_output==3){
    printf("[Parallel 2×And + 1×Add]\n");
    int8_t a[4]={1,0,1,1},b[4]={1,1,0,1},c[4]={0,1,1,0},d[4]={1,1,0,1};
    float xf[4]={1.5f,2.0f,0.25f,3.0f},yf[4]={0.5f,1.0f,4.0f,2.0f};
    uint16_t xh[4],yh[4];for(int i=0;i<4;i++){xh[i]=f2h(xf[i]);yh[i]=f2h(yf[i]);}
    rknn_input ins[6];memset(ins,0,sizeof(ins));
    ins[0].index=0;ins[0].type=RKNN_TENSOR_INT8;ins[0].size=4;ins[0].fmt=RKNN_TENSOR_NCHW;ins[0].buf=a;ins[0].pass_through=1;
    ins[1].index=1;ins[1].type=RKNN_TENSOR_INT8;ins[1].size=4;ins[1].fmt=RKNN_TENSOR_NCHW;ins[1].buf=b;ins[1].pass_through=1;
    ins[2].index=2;ins[2].type=RKNN_TENSOR_INT8;ins[2].size=4;ins[2].fmt=RKNN_TENSOR_NCHW;ins[2].buf=c;ins[2].pass_through=1;
    ins[3].index=3;ins[3].type=RKNN_TENSOR_INT8;ins[3].size=4;ins[3].fmt=RKNN_TENSOR_NCHW;ins[3].buf=d;ins[3].pass_through=1;
    ins[4].index=4;ins[4].type=RKNN_TENSOR_FLOAT16;ins[4].size=8;ins[4].fmt=RKNN_TENSOR_NCHW;ins[4].buf=xh;
    ins[5].index=5;ins[5].type=RKNN_TENSOR_FLOAT16;ins[5].size=8;ins[5].fmt=RKNN_TENSOR_NCHW;ins[5].buf=yh;
    ret=rknn_inputs_set(ctx,6,ins);printf("inputs_set: %d\n",ret);
    ret=rknn_run(ctx,NULL);printf("run: %d\n",ret);
    rknn_output outs[3];memset(outs,0,sizeof(outs));
    for(int i=0;i<3;i++){outs[i].index=i;outs[i].want_float=(i==2);}
    rknn_outputs_get(ctx,3,outs,NULL);printf("outputs_get: %d\n",ret);
    int8_t*o1=(int8_t*)outs[0].buf,*o2=(int8_t*)outs[1].buf;float*o3=(float*)outs[2].buf;
    printf("AND1:");for(int i=0;i<4;i++)printf(" %d",o1[i]);printf("  (want");
    for(int i=0;i<4;i++)printf(" %d",a[i]&&b[i]);printf(")\n");
    printf("AND2:");for(int i=0;i<4;i++)printf(" %d",o2[i]);printf("  (want");
    for(int i=0;i<4;i++)printf(" %d",c[i]&&d[i]);printf(")\n");
    printf("ADD :");for(int i=0;i<4;i++)printf(" %.3f",o3[i]);printf("  (want");
    for(int i=0;i<4;i++)printf(" %.3f",xf[i]+yf[i]);printf(")\n");
    for(int i=0;i<4;i++){
      if((o1[i]!=0)!=((a[i]&&b[i])!=0))bad++;
      if((o2[i]!=0)!=((c[i]&&d[i])!=0))bad++;
      if(o3[i]!=o3[i]||__builtin_fabsf(o3[i]-(xf[i]+yf[i]))>0.1f)bad++;
    }
    rknn_outputs_release(ctx,3,outs);
  }
  // ── Parallel And + Add (4 in, 2 out) — use verify_parallel for this ──
  else{
    printf("Unknown topology (in=%u out=%u) — use the dedicated verifier\n",io.n_input,io.n_output);
    bad=1;
  }

  printf("%s mismatches=%d\n",bad?"FAIL":"PASS",bad);
  rknn_destroy(ctx);free(model);
  return bad?1:0;
}
