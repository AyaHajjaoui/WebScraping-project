# verify_running.py
import subprocess
import psutil

print("Checking for main.py...")
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmd = ' '.join(proc.info['cmdline'] or [])
        if 'main.py' in cmd:
            print(f"✅ main.py is RUNNING! (PID: {proc.info['pid']})")
            print(f"   Started at: {proc.create_time()}")
            break
    except:
        pass
else:
    print("❌ main.py is NOT running")