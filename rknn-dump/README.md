When running rknn-demo with loop it will call this

lsof /dev/dri/card1

COMMAND     PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
rknn_crea 76370 root  mem    CHR  226,1           183 /dev/dri/card1
rknn_crea 76370 root    3u   CHR  226,1      0t0  183 /dev/dri/card1


strace -f -e trace=ioctl -p 77393 -o ioctl.log

7607  ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0x41, 0x68), 0xfffff7b88600) = 0
7607  ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0x45, 0x20), 0xfffff7b885f0) = 0
7607  ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0x45, 0x20), 0xfffff7b885f0) = 0
7607  ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0x45, 0x20), 0xfffff7b885f0) = 0
7607  ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0x45, 0x20), 0xfffff7b885f0) = 0
7607  ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0x45, 0x20), 0xfffff7b885f0) = 0
7607  ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0x45, 0x20), 0xfffff7b88780) = 0
7607  ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0x45, 0x20), 0xfffff7b88780) = 0
7607  ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0x41, 0x68), 0xfffff7b88600) = 0


