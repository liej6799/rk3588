// verify_hybrid_n.cpp - Verify an N-input parallel CPU-logical + NPU-arith model.
//
// Usage: verify_hybrid_n MODEL.rknn N_CPU CPU_OP N_NPU NPU_OP
//   N_CPU bool inputs  -> out1 = a0 OP ... (And/Or/Xor)
//   N_NPU fp16 inputs  -> out2 = x0 OP ... (Add/Sub/Div)
// Inputs are the first N_CPU (int8 bool) then N_NPU (fp16), matching the toolkit
// graph order (a0..,x0..).  out1 is bool, out2 is float.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "rknn_api.h"

static uint16_t f2h(float f){uint32_t x;memcpy(&x,&f,4);uint32_t s=(x>>16)&0x8000;int e=int((x>>23)&0xff)-127+15;uint32_t m=x&0x7fffff;if(e<=0)return s;if(e>=31)return s|0x7c00;return s|(uint32_t(e)<<10)|(m>>13);}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr,"usage: %s MODEL N_CPU CPU_OP N_NPU NPU_OP\n",argv[0]); return 2; }
  int ncpu=atoi(argv[2]); const char* cpu=argv[3];
  int nnpu=atoi(argv[4]); const char* npu=argv[5];

  FILE* fp=fopen(argv[1],"rb"); fseek(fp,0,SEEK_END); long sz=ftell(fp); fseek(fp,0,SEEK_SET);
  void* model=malloc(sz); fread(model,1,sz,fp); fclose(fp);
  rknn_context ctx=0;
  int ret=rknn_init(&ctx,model,sz,0,NULL); printf("rknn_init: %d\n",ret); if(ret<0) return 1;
  rknn_input_output_num io; memset(&io,0,sizeof(io));
  rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io));
  printf("inputs=%u outputs=%u\n",io.n_input,io.n_output);

  const int N=4;
  // bool inputs: a_j[i] = (i>>j)&1 ; fp16 inputs: distinct float patterns
  int8_t bin[64][4]; uint16_t fin[64][4]; float ff[64][4];
  for(int j=0;j<ncpu;j++) for(int i=0;i<N;i++) bin[j][i]=(int8_t)((i>>j)&1);
  for(int j=0;j<nnpu;j++) for(int i=0;i<N;i++){ ff[j][i]=(float)(j+1)+0.5f*i; fin[j][i]=f2h(ff[j][i]); }

  rknn_input ins[64]; memset(ins,0,sizeof(ins));
  for(int j=0;j<ncpu;j++){ ins[j].index=j; ins[j].type=RKNN_TENSOR_INT8; ins[j].size=N; ins[j].fmt=RKNN_TENSOR_NCHW; ins[j].buf=bin[j]; ins[j].pass_through=1; }
  for(int j=0;j<nnpu;j++){ int k=ncpu+j; ins[k].index=k; ins[k].type=RKNN_TENSOR_FLOAT16; ins[k].size=N*2; ins[k].fmt=RKNN_TENSOR_NCHW; ins[k].buf=fin[j]; }
  ret=rknn_inputs_set(ctx,io.n_input,ins); printf("inputs_set: %d\n",ret); if(ret<0) return 1;
  ret=rknn_run(ctx,NULL); printf("run: %d\n",ret); if(ret<0) return 1;

  rknn_output outs[2]; memset(outs,0,sizeof(outs));
  outs[0].index=0; outs[0].want_float=0;
  outs[1].index=1; outs[1].want_float=1;
  ret=rknn_outputs_get(ctx,2,outs,NULL); printf("outputs_get: %d\n",ret); if(ret<0) return 1;
  int8_t* o1=(int8_t*)outs[0].buf; float* o2=(float*)outs[1].buf;

  int bad=0;
  // CPU expected
  printf("%s out:",cpu); for(int i=0;i<N;i++) printf(" %d",o1[i]?1:0); printf("  (want");
  for(int i=0;i<N;i++){ int acc=bin[0][i]&1; for(int j=1;j<ncpu;j++){int b=bin[j][i]&1; if(!strcmp(cpu,"And"))acc&=b; else if(!strcmp(cpu,"Or"))acc|=b; else acc^=b;} printf(" %d",acc); if((o1[i]?1:0)!=acc) bad++; }
  printf(")\n");
  // NPU expected
  printf("%s out:",npu); for(int i=0;i<N;i++) printf(" %.3f",o2[i]); printf("  (want");
  for(int i=0;i<N;i++){ float acc=ff[0][i]; for(int j=1;j<nnpu;j++){float v=ff[j][i]; if(!strcmp(npu,"Add"))acc+=v; else if(!strcmp(npu,"Sub"))acc-=v; else acc/=v;} printf(" %.3f",acc); if(o2[i]!=o2[i]||fabsf(o2[i]-acc)>0.15f) bad++; }
  printf(")\n");
  printf("%s mismatches=%d\n", bad?"FAIL":"PASS", bad);
  rknn_outputs_release(ctx,2,outs); rknn_destroy(ctx); free(model);
  return bad?1:0;
}
