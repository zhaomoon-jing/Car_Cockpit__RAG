"""修复 app.py 并启动应用"""
import sys, os, shutil

sys.dont_write_bytecode = True
root = r'f:\car_cockpit_rag'
os.chdir(root)
sys.path.insert(0, root)

# 1. 修复 app.py
app_path = os.path.join(root, 'gradio_web', 'app.py')
with open(app_path, 'r', encoding='utf-8') as f:
    c = f.read()
if 'type="messages"' in c or "type='messages'" in c:
    c = c.replace('type="messages"', '').replace("type='messages'", '')
    with open(app_path, 'w', encoding='utf-8') as f:
        f.write(c)
    print("[FIX] app.py 已修复")
else:
    print("[OK] app.py 无需修复")

# 2. 清除缓存
for r2, dirs, _ in os.walk(root):
    for d in dirs:
        if d == '__pycache__':
            try: shutil.rmtree(os.path.join(r2, d))
            except: pass
print("[OK] 缓存已清除")

# 3. 启动
print("[WEB] 正在启动...")
from gradio_web.app import create_app
demo = create_app()
print("[OK] 应用创建成功!")
demo.launch(server_name="0.0.0.0", server_port=7866, share=False, show_error=True)
