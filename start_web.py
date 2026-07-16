"""诊断并启动 web 应用"""
import sys, os

# 禁止字节码缓存
sys.dont_write_bytecode = True

# 检查文件
app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gradio_web', 'app.py')
print(f"检查文件: {app_path}")
print(f"文件存在: {os.path.exists(app_path)}")

with open(app_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 检查是否有 type="messages"
if 'type="messages"' in content or "type='messages'" in content:
    print("!! 发现 type='messages'，正在修复...")
    content = content.replace(',\n                    type="messages"', '')
    content = content.replace(",\n                    type='messages'", '')
    content = content.replace('type="messages"', '')
    content = content.replace("type='messages'", '')
    with open(app_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("已修复!")
else:
    print("OK: 文件中没有 type='messages'")

# 显示 Chatbot 附近的代码
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Chatbot(' in line:
        print(f"\nChatbot 定义在行 {i+1}:")
        for j in range(max(0,i-1), min(len(lines), i+6)):
            print(f"  {j+1}: {lines[j]}")

# 清除所有 __pycache__
import shutil
for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
    for d in dirs:
        if d == '__pycache__':
            p = os.path.join(root, d)
            shutil.rmtree(p)
            print(f"已删除缓存: {p}")

# 启动应用
print("\n正在启动应用...")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gradio_web.app import create_app
demo = create_app()
print("应用创建成功! 启动服务器...")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False, show_error=True)
