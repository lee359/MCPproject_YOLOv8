"""
快速驗證 Claude Desktop 配置
"""
import json
import os
import subprocess

print("=" * 70)
print("Claude Desktop MCP Server 配置驗證")
print("=" * 70)

# 讀取配置檔案
config_path = r"C:\Users\user\AppData\Roaming\Claude\claude_desktop_config.json"

print(f"\n[1] 檢查配置檔案")
print("-" * 70)
if os.path.exists(config_path):
    print(f"✅ 配置檔案存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if "YOLOv8 Detection Server" in config.get("mcpServers", {}):
        server_config = config["mcpServers"]["YOLOv8 Detection Server"]
        print("✅ YOLOv8 Detection Server 配置存在")
        print(f"\n配置內容:")
        print(json.dumps(server_config, indent=2, ensure_ascii=False))
        
        # 檢查命令
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        
        print(f"\n[2] 檢查命令和路徑")
        print("-" * 70)
        
        # 檢查 Python 可執行檔
        if os.path.exists(command):
            print(f"✅ Python 可執行檔存在: {command}")
            
            # 測試 Python 版本
            try:
                result = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=5)
                print(f"✅ Python 版本: {result.stdout.strip()}")
            except Exception as e:
                print(f"❌ 無法執行 Python: {e}")
        else:
            print(f"❌ Python 可執行檔不存在: {command}")
        
        # 檢查腳本檔案
        if args:
            script_path = args[0]
            if os.path.exists(script_path):
                print(f"✅ MCP 腳本存在: {script_path}")
            else:
                print(f"❌ MCP 腳本不存在: {script_path}")
        
        print(f"\n[3] 測試 MCP Server 啟動")
        print("-" * 70)
        print("測試命令（3 秒超時）...")
        
        try:
            cmd = [command] + args
            print(f"完整命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3,
                cwd=os.path.dirname(script_path)
            )
            
            if result.returncode == 0:
                print("✅ Server 啟動成功（返回碼 0）")
            else:
                print(f"❌ Server 啟動失敗（返回碼 {result.returncode}）")
                if result.stderr:
                    print(f"錯誤輸出:\n{result.stderr}")
                    
        except subprocess.TimeoutExpired:
            print("✅ Server 已啟動並持續運行（超時是正常的）")
        except Exception as e:
            print(f"❌ 啟動測試失敗: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        print("❌ YOLOv8 Detection Server 配置不存在")
        print("\n可用的 servers:")
        for name in config.get("mcpServers", {}).keys():
            print(f"  - {name}")
else:
    print(f"❌ 配置檔案不存在: {config_path}")

print("\n" + "=" * 70)
print("驗證完成")
print("=" * 70)

print("\n下一步:")
print("1. 如果所有檢查都通過 (✅)，請完全重啟 Claude Desktop")
print("2. 在 Claude Desktop 中測試: '請列出可用的工具'")
print("3. 使用工具: 'check_stream_health' 測試串流連接")
