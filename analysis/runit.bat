REM @echo off

REM FRD
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_frd_256x256_tri.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_frd_256x256_uni.bmp
REM c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_frd_256x256_uni.hdr
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_frd_1024x1024_uni.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_frd_1024x1024_tri.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_frd_256x256_tri.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_frd_256x256_uni.bmp

REM VNC
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_vnc_256x256_tri_f11_rnd10.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_vnc_256x256_uni_f11_rnd10.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_vnc_512x512_tri_f11_rnd10.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_vnc_512x512_uni_f11_rnd10.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_vnc_1024x1024_tri_f11_rnd10.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_vnc_1024x1024_uni_f11_rnd10.bmp

REM c:\programdata\anaconda3\python.exe analyse.py ..\textures\bluenoise_vnc_256x256_uni_f11_rnd10.hdr

c:\programdata\anaconda3\python.exe analyse.py ..\ref\output_256x256_uni.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\ref\bluenoise_256_rgba_tri.bmp
c:\programdata\anaconda3\python.exe analyse.py ..\ref\bluenoise_256_rgba_uni.bmp

