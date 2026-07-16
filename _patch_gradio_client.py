"""修补 gradio_client utils.py 中的 bool schema bug"""
import os

path = r'D:\install\python10\lib\site-packages\gradio_client\utils.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复 get_type 函数 - 添加 bool 检查
old = '''def get_type(schema: dict):
    if "const" in schema:'''
new = '''def get_type(schema: dict):
    if isinstance(schema, bool):
        return "Any"
    if "const" in schema:'''

if old in content:
    content = content.replace(old, new)
    print("[FIX 1/2] get_type() 已修补")
else:
    print("[SKIP 1/2] get_type() 已修补过")

# 修复 additionalProperties 处理 - 添加 bool 检查
old2 = '''        if "additionalProperties" in schema:
            des += [
                f"str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}"
            ]'''
new2 = '''        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
            des += [
                f"str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}"
            ]'''

if old2 in content:
    content = content.replace(old2, new2)
    print("[FIX 2/2] additionalProperties 处理已修补")
else:
    print("[SKIP 2/2] additionalProperties 已修补过")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("修补完成!")
