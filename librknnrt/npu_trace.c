// LD_PRELOAD shim: decode RKNPU DRM ioctls + dump the rknpu_task[] array.
// Reads dmabuf-backed task buffer in-process (CPU read works where ptrace fails).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#define MEM_CREATE 0xc0306442u
#define MEM_MAP    0xc0106443u
#define MEM_SYNC   0xc0206445u
#define SUBMIT     0xc0686441u
#define ACTION     0xc0086440u

#define MAXH 64
static uint64_t obj_by_h[MAXH], dma_by_h[MAXH], va_by_h[MAXH];
static uint64_t sz_by_h[MAXH];
static int pending_h = -1;

static int (*real_ioctl)(int,unsigned long,...);
static void* (*real_mmap)(void*,size_t,int,int,int,off_t);

static void init(void){
  if(!real_ioctl) real_ioctl=dlsym(RTLD_NEXT,"ioctl");
  if(!real_mmap)  real_mmap =dlsym(RTLD_NEXT,"mmap");
}

static void* (*real_mmap64)(void*,size_t,int,int,int,off_t);
static void* track_mmap(void* r){
  if(pending_h>=0 && pending_h<MAXH){ va_by_h[pending_h]=(uint64_t)r; pending_h=-1; }
  return r;
}
void* mmap(void* a,size_t len,int prot,int flags,int fd,off_t off){
  init(); return track_mmap(real_mmap(a,len,prot,flags,fd,off));
}
void* mmap64(void* a,size_t len,int prot,int flags,int fd,off_t off){
  init();
  if(!real_mmap64) real_mmap64=dlsym(RTLD_NEXT,"mmap64");
  return track_mmap(real_mmap64(a,len,prot,flags,fd,off));
}

static const char* act_name(uint32_t a){
  switch(a){case 0:return"GET_HW_VERSION";case 1:return"GET_DRV_VERSION";case 6:return"ACT_RESET";
  case 18:return"GET_IOMMU_EN";case 20:return"POWER_ON";case 21:return"POWER_OFF";}
  return "ACT";
}

int ioctl(int fd,unsigned long req,...){
  init();
  va_list ap; va_start(ap,req); void* argp=va_arg(ap,void*); va_end(ap);
  uint32_t r=(uint32_t)req;

  if(r==MEM_MAP){ uint32_t h=*(uint32_t*)argp; int ret=real_ioctl(fd,req,argp);
    if(h<MAXH) pending_h=(int)h; return ret; }

  if(r==SUBMIT){
    uint8_t* s=(uint8_t*)argp;
    uint32_t flags=*(uint32_t*)(s+0), to=*(uint32_t*)(s+4);
    uint32_t ts=*(uint32_t*)(s+8), tn=*(uint32_t*)(s+12);
    uint64_t tobj=*(uint64_t*)(s+24), rcobj=*(uint64_t*)(s+32), tbase=*(uint64_t*)(s+40);
    uint32_t cm=*(uint32_t*)(s+56); int32_t ff=*(int32_t*)(s+60);
    fprintf(stderr,"[SUBMIT] flags=0x%x timeout=%u task_start=%u task_number=%u core_mask=0x%x fence_fd=%d\n",
            flags,to,ts,tn,cm,ff);
    fprintf(stderr,"         task_obj=0x%lx regcfg_obj=0x%lx task_base=0x%lx\n",
            (unsigned long)tobj,(unsigned long)rcobj,(unsigned long)tbase);
    for(int i=0;i<5;i++){uint32_t st=*(uint32_t*)(s+64+i*8),num=*(uint32_t*)(s+68+i*8);
      if(st||num) fprintf(stderr,"         subcore[%d]=(start=%u,number=%u)\n",i,st,num);}
    int h=-1; for(int i=0;i<MAXH;i++) if(obj_by_h[i]==tobj){h=i;break;}
    if(h>=0 && va_by_h[h]){
      uint8_t* tv=(uint8_t*)va_by_h[h];
      int nrec=(int)(sz_by_h[h]/40); if(nrec>16)nrec=16;
      fprintf(stderr,"         task buf handle=%d va=0x%lx dma=0x%lx logsz=%lu nrec=%d\n",
              h,(unsigned long)va_by_h[h],(unsigned long)dma_by_h[h],(unsigned long)sz_by_h[h],nrec);
      for(int i=0;i<nrec;i++){uint8_t* b=tv+i*40;
        uint32_t fl=*(uint32_t*)(b+0),op=*(uint32_t*)(b+4),en=*(uint32_t*)(b+8),im=*(uint32_t*)(b+12);
        uint32_t ic=*(uint32_t*)(b+16),amt=*(uint32_t*)(b+24),roff=*(uint32_t*)(b+28);
        uint64_t ra=*(uint64_t*)(b+32);
        fprintf(stderr,"         task[%d] flags=0x%x op_idx=%u enable_mask=0x%x int_mask=0x%x int_clear=0x%x regcfg_amount=%u regcfg_offset=%u regcmd_addr=0x%lx\n",
                i,fl,op,en,im,ic,amt,roff,(unsigned long)ra);
      }
    } else fprintf(stderr,"         (task buf va not resolved h=%d)\n",h);
    return real_ioctl(fd,req,argp);
  }

  int ret=real_ioctl(fd,req,argp);
  if(r==MEM_CREATE){ uint8_t* s=(uint8_t*)argp; uint32_t h=*(uint32_t*)(s+0);
    if(h<MAXH){ sz_by_h[h]=*(uint64_t*)(s+8); obj_by_h[h]=*(uint64_t*)(s+16); dma_by_h[h]=*(uint64_t*)(s+24);
      fprintf(stderr,"[MEM_CREATE] handle=%u size=%lu obj=0x%lx dma=0x%lx\n",
              h,(unsigned long)sz_by_h[h],(unsigned long)obj_by_h[h],(unsigned long)dma_by_h[h]); } }
  else if(r==MEM_SYNC){ uint8_t*s=(uint8_t*)argp; uint32_t fl=*(uint32_t*)(s+0); uint64_t obj=*(uint64_t*)(s+8);
    uint64_t off=*(uint64_t*)(s+16); uint64_t sz2=*(uint64_t*)(s+24);
    int h2=-1; for(int i=0;i<MAXH;i++) if(obj_by_h[i]==obj){h2=i;break;}
    const char*dir=fl==1?"TO_DEV":fl==2?"FROM_DEV":fl==3?"BIDIR":"?";
    fprintf(stderr,"[MEM_SYNC] %s handle=%d off=%lu size=%lu\n",dir,h2,
            (unsigned long)off,(unsigned long)sz2); }
  else if(r==ACTION){ uint32_t a=*(uint32_t*)argp; (void)a; }
  return ret;
}
