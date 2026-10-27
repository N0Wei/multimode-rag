#!/usr/bin/env python
# coding: utf-8
"""
网络连接诊断工具
帮助排查 Gradio 公网链接在移动端无法访问的问题
"""

import socket
import subprocess
import platform
import requests
from urllib.parse import urlparse


def get_local_ip():
    """获取本机局域网 IP"""
    try:
        # 创建一个 UDP 连接（不会真的发送数据）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return f"无法获取 (错误: {e})"


def check_port(host, port):
    """检查端口是否开放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False


def test_gradio_link(url):
    """测试 Gradio 链接是否可访问"""
    try:
        response = requests.get(url, timeout=5, verify=False)
        return response.status_code == 200
    except requests.exceptions.SSLError:
        return "SSL 错误"
    except requests.exceptions.ConnectionError:
        return "连接失败"
    except requests.exceptions.Timeout:
        return "超时"
    except Exception as e:
        return f"错误: {str(e)}"


def main():
    print("=" * 70)
    print("🔍 Gradio 网络连接诊断工具")
    print("=" * 70)
    
    # 1. 系统信息
    print("\n[1] 系统信息")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  Python 版本: {platform.python_version()}")
    
    # 2. 网络信息
    print("\n[2] 网络信息")
    local_ip = get_local_ip()
    print(f"  本机 IP: {local_ip}")
    print(f"  主机名: {socket.gethostname()}")
    
    # 3. 端口检查
    print("\n[3] 端口检查")
    port = 7860
    is_open = check_port('127.0.0.1', port)
    print(f"  端口 {port}: {'✅ 开放' if is_open else '❌ 关闭'}")
    
    if is_open:
        print(f"\n  📱 手机访问地址 (需在同一 WiFi):")
        print(f"     http://{local_ip}:{port}")
    
    # 4. 防火墙提示
    print("\n[4] 防火墙检查")
    if platform.system() == "Darwin":  # macOS
        print("  💡 如果手机无法访问，请检查 macOS 防火墙:")
        print("     系统偏好设置 → 安全性与隐私 → 防火墙")
        print("     确保 Python 允许传入连接")
    elif platform.system() == "Windows":
        print("  💡 检查 Windows 防火墙设置")
    
    # 5. 公网链接测试
    print("\n[5] 公网链接测试")
    print("  输入你的 Gradio 公网链接进行测试")
    print("  (例如: https://abc123.gradio.live)")
    print("  或按 Enter 跳过")
    
    gradio_url = input("\n  公网链接: ").strip()
    
    if gradio_url:
        print(f"\n  测试中... {gradio_url}")
        result = test_gradio_link(gradio_url)
        
        if result == True:
            print("  ✅ 链接可访问")
        elif result == "SSL 错误":
            print("  ⚠️  SSL 证书问题")
            print("     手机浏览器可能拒绝自签名证书")
            print("     建议：使用局域网地址或等待 Gradio 修复")
        elif result == "连接失败":
            print("  ❌ 无法连接")
            print("     可能原因：网络问题、链接已过期")
        else:
            print(f"  ❌ {result}")
    
    # 6. 建议方案
    print("\n" + "=" * 70)
    print("📋 建议解决方案")
    print("=" * 70)
    print("\n✅ 方案 1: 局域网访问 (推荐)")
    print(f"   1. 确保手机和电脑连接同一 WiFi")
    print(f"   2. 手机浏览器访问: http://{local_ip}:{port}")
    print(f"   3. 如果无法访问，检查防火墙设置")
    
    print("\n✅ 方案 2: 修改 Gradio 配置")
    print("   在 app_gradio.py 中:")
    print("   demo.launch(")
    print("       share=True,")
    print("       ssl_verify=False  # 添加此行")
    print("   )")
    
    print("\n✅ 方案 3: 使用不同的浏览器")
    print("   • Chrome (推荐)")
    print("   • Safari")
    print("   • Firefox")
    print("   某些浏览器对自签名证书更宽容")
    
    print("\n✅ 方案 4: 切换网络")
    print("   • 如果手机用移动数据，改用 WiFi")
    print("   • 某些运营商网络可能拦截不安全连接")
    
    print("\n" + "=" * 70)
    print("需要更多帮助？查看 GRADIO_README.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
