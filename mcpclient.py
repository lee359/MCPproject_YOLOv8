# server.py
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage
from ultralytics import YOLO

# 加載訓練好的模型
#model = YOLO("best.pt")

# 測試單張圖像
#results = model.predict(source="456.jpg", save=True, imgsz=640, data=None)
#print(results)

# Create an MCP server
mcp = FastMCP("Image Example")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# 添加這行代碼以確保服務器可以運行
if __name__ == "__main__":
   mcp.run()