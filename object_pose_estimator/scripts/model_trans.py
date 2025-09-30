import bpy
 
def convert_obj_to_usd(obj_path, usd_path):
    # 清除现有场景
    bpy.ops.wm.read_factory_settings(use_empty=True)
 
    # 导入OBJ文件
    bpy.ops.import_scene.obj(filepath=obj_path)
 
    # 导出为USD文件（假设有USD导出插件）
    bpy.ops.export_scene.usd(filepath=usd_path)  # 确保有支持USD的插件
 
    print(f"OBJ文件已成功导入，准备导出为USD文件：{usd_path}")

if __name__ == "__main__":
    # 示例路径
    obj_file_path = "/home/niloiv/models/006_mustard_bottle/textured.obj"
    usd_file_path = "/home/niloiv/models/006_mustard_bottle/textured.usd"

    # 运行 Blender 并执行脚本（这通常需要在 Blender 的 Python 环境中运行）
    # 如果你从命令行运行 Blender，你可以使用以下命令：
    # blender --background --python this_script.py

    convert_obj_to_usd(obj_file_path, usd_file_path)
    print(f"Converted {obj_file_path} to {usd_file_path}")
