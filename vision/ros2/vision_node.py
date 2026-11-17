#!/usr/bin/env python3
"""ROS 2 node: lets mission control (the behaviour tree) ask the vision system for an item.

NOTE: written for ROS 2 Humble/Jazzy with rclpy but NOT yet tested on a robot.
Uses only standard std_msgs/String with JSON inside, so no custom message package is needed.

Topics
  /vision/find_item    (std_msgs/String, IN)   product name, e.g. "egg"
  /vision/item_result  (std_msgs/String, OUT)  JSON from VisionService.find_item(), e.g.
      {"product": "egg", "status": "found", "found": true, "confidence": 0.91,
       "position_arm_m": [0.21, -0.03, 0.12], ...}
  /vision/status       (std_msgs/String, OUT)  "idle" or "busy: egg"

Run:
  source /opt/ros/humble/setup.bash
  python3 ros2/vision_node.py
Test from another terminal:
  ros2 topic echo /vision/item_result
  ros2 topic pub --once /vision/find_item std_msgs/String "{data: egg}"
"""
import json
import sys
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopvision.config import load_products  # noqa: E402
from shopvision.service import VisionService  # noqa: E402


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.declare_parameter("source", "")
        source = self.get_parameter("source").get_parameter_value().string_value or None
        self.vision = VisionService(source=source)
        self.products = set(load_products())
        self.busy = threading.Lock()
        self.result_pub = self.create_publisher(String, "/vision/item_result", 10)
        self.status_pub = self.create_publisher(String, "/vision/status", 10)
        self.create_subscription(String, "/vision/find_item", self.on_request, 10)
        self.create_timer(1.0, self.publish_status)
        self.current = None
        self.get_logger().info(f"Vision ready. Products: {sorted(self.products)}")

    def publish_status(self):
        self.status_pub.publish(String(data=f"busy: {self.current}" if self.current else "idle"))

    def on_request(self, msg):
        product = msg.data.strip()
        if product not in self.products:
            self.send({"product": product, "status": "unknown_product", "found": False})
            return
        if not self.busy.acquire(blocking=False):
            self.send({"product": product, "status": "busy", "found": False, "busy_with": self.current})
            return
        self.current = product
        threading.Thread(target=self.work, args=(product,), daemon=True).start()

    def work(self, product):
        try:
            r = self.vision.find_item(product)
        except Exception as err:  # never let the node die on one bad request
            r = {"product": product, "status": "error", "found": False, "error": str(err)}
        finally:
            self.current = None
            self.busy.release()
        self.send(r)

    def send(self, result):
        self.result_pub.publish(String(data=json.dumps(result)))
        self.get_logger().info(f"{result.get('product')}: {result.get('status')}")

    def destroy_node(self):
        self.vision.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
