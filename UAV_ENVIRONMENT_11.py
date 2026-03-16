import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import time
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== Constants for state management =====
ARRIVAL_THRESHOLD = 0.5  # Distance threshold for considering drone arrived at target
DISTANCE_CLOSE_THRESHOLD = 0.15  # Distance threshold for decision point detection

# ===== Fixed speed multiplier (U8: no longer controlled by PPO) =====
FIXED_SPEED_MULTIPLIER = 1.0  # Fixed speed multiplier for all drones


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# 枚举定义
class OrderStatus(Enum):
    PENDING = 0
    ACCEPTED = 1
    PREPARING = 2
    READY = 3
    ASSIGNED = 4
    PICKED_UP = 5
    DELIVERED = 6
    CANCELLED = 7


class WeatherType(Enum):
    SUNNY = 0
    RAINY = 1
    WINDY = 2
    EXTREME = 3


class OrderType(Enum):
    NORMAL = 0
    FREE_SIMPLE = 1
    FREE_COMPLEX = 2


class DroneStatus(Enum):
    IDLE = 0
    ASSIGNED = 1
    FLYING_TO_MERCHANT = 2
    WAITING_FOR_PICKUP = 3
    FLYING_TO_CUSTOMER = 4
    DELIVERING = 5
    RETURNING_TO_BASE = 6
    CHARGING = 7


# 天级时间管理系统
class DailyTimeSystem:
    """天级时间管理系统"""

    def __init__(self, start_hour=6, end_hour=22, steps_per_hour=4):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.steps_per_hour = steps_per_hour
        self.operating_hours = end_hour - start_hour
        self.steps_per_day = self.operating_hours * steps_per_hour
        self.minutes_per_step = 60.0 / steps_per_hour  # Pre-compute for efficiency

        # 时间状态
        self.current_step = 0
        self.step_in_day = 0  # Steps within current day for termination check
        self.current_hour = start_hour
        self.current_minute = 0
        self.day_number = 0

    def reset(self):
        """重置时间系统"""
        self.current_step = 0
        self.step_in_day = 0
        self.current_hour = self.start_hour
        self.current_minute = 0
        self.day_number = 0
        return self.get_time_state()

    def step(self):
        """前进一个时间步"""
        self.current_step += 1
        self.step_in_day += 1

        # Update hour and minute for display purposes
        # Use precise calculation to avoid drift
        minutes_from_start = self.step_in_day * self.minutes_per_step
        hours_from_start = int(minutes_from_start // 60)
        self.current_hour = self.start_hour + hours_from_start
        self.current_minute = int(minutes_from_start % 60)

        # Check if day has ended using step count (robust and precise)
        if self.step_in_day >= self.steps_per_day:
            self.day_number += 1
            return True  # 表示一天结束

        return False

    def get_time_state(self):
        """获取当前时间状态"""
        return {
            'step': self.current_step,
            'hour': self.current_hour,
            'minute': self.current_minute,
            'day': self.day_number,
            'progress': self.current_step / self.steps_per_day,
            'is_peak_hour': self._is_peak_hour(),
            'is_business_hours': self._is_business_hours()
        }

    def _is_peak_hour(self):
        """判断是否是高峰时段"""
        peak_hours = [(7, 9), (11, 13), (17, 19)]  # 早中晚高峰
        for start, end in peak_hours:
            if start <= self.current_hour < end:
                return True
        return False

    def _is_business_hours(self):
        """判断是否在营业时间内"""
        return self.start_hour <= self.current_hour < self.end_hour

    def get_day_progress(self):
        """获取当天进度"""
        return self.current_step / self.steps_per_day


# 统一状态管理器
class StateManager:
    """统一状态管理器"""

    def __init__(self, env):
        self.env = env
        # Bounded deque to prevent unbounded GC pressure over long training runs
        self.state_log = deque(maxlen=5000)
        # First-anomaly snapshot: printed once to help diagnose index/status skew
        self._first_anomaly_logged = False

    def update_order_status(self, order_id, new_status, reason=""):
        """统一更新订单状态"""
        if order_id not in self.env.orders:
            return False

        order = self.env.orders[order_id]
        old_status = order['status']
        order['status'] = new_status

        # Record ready_step when transitioning to READY for the first time
        if new_status == OrderStatus.READY and old_status != OrderStatus.READY:
            if order.get('ready_step') is None:
                order['ready_step'] = self.env.time_system.current_step

        # Maintain incremental ready-orders cache (avoids full active_orders scan)
        cache = self.env._ready_orders_cache
        if new_status == OrderStatus.READY:
            cache.add(order_id)
            # If coming from an active state, evict stale candidate entries so no
            # other drone accidentally re-selects an order already being released.
            if old_status in (OrderStatus.ASSIGNED, OrderStatus.PICKED_UP):
                self._evict_order_from_candidate_caches(order_id)
        elif old_status == OrderStatus.READY:
            # Order is leaving READY: evict from all drone candidate caches immediately
            # so that no other drone can select a stale READY entry this step.
            cache.discard(order_id)
            self._evict_order_from_candidate_caches(order_id)
        elif old_status in (OrderStatus.ASSIGNED, OrderStatus.PICKED_UP):
            # Order transitioning to DELIVERED, CANCELLED, etc.:
            # evict stale candidate/cache entries unconditionally.
            self._evict_order_from_candidate_caches(order_id)

        # 记录状态变更
        state_change = {
            'time': self.env.time_system.current_step,
            'order_id': order_id,
            'old_status': old_status,
            'new_status': new_status,
            'reason': reason,
            'drone_id': order.get('assigned_drone')
        }
        self.state_log.append(state_change)

        return True

    def _evict_order_from_candidate_caches(self, order_id: int):
        """Remove *order_id* from every per-drone candidate cache.

        Called whenever an order leaves the READY, ASSIGNED, or PICKED_UP state
        so that stale entries cannot be selected in subsequent decisions.
        """
        env = self.env
        # Signal that candidate mappings contain stale data.
        env._candidate_mappings_dirty = True

        # 1. _filtered_candidates_sets: discard from every drone's set
        for drone_set in env._filtered_candidates_sets.values():
            drone_set.discard(order_id)

        # 2. filtered_candidates (list version): remove from every drone's list
        for drone_id, cand_list in env.filtered_candidates.items():
            if order_id in cand_list:
                env.filtered_candidates[drone_id] = [
                    oid for oid in cand_list if oid != order_id
                ]

        # 3. drone_candidate_mappings: mark as invalid (keep slot to preserve list length)
        for drone_id, mapping in env.drone_candidate_mappings.items():
            env.drone_candidate_mappings[drone_id] = [
                (oid, False) if oid == order_id else (oid, valid)
                for oid, valid in mapping
            ]

    def update_drone_status(self, drone_id, new_status, target_location=None):
        """统一更新无人机状态"""
        if drone_id not in self.env.drones:
            return False

        drone = self.env.drones[drone_id]
        old_status = drone['status']
        drone['status'] = new_status

        if target_location is not None:
            drone['target_location'] = target_location
        else:
            # 若显式传 None，则移除目标避免“旧目标残留”
            drone.pop('target_location', None)

        # 记录状态变更
        state_change = {
            'time': self.env.time_system.current_step,
            'drone_id': drone_id,
            'old_status': old_status,
            'new_status': new_status,
            'target_location': target_location
        }
        self.state_log.append(state_change)

        return True

    @staticmethod
    def categorize_issues(issues: List[str]) -> Dict[str, int]:
        """
        Categorize consistency issues by type.

        Args:
            issues: List of issue strings

        Returns:
            Dictionary with counts per category: Route, TaskSel, Legacy, Other
        """
        categories = {'Route': 0, 'TaskSel': 0, 'Legacy': 0, 'Other': 0}

        for issue in issues:
            if '[Route]' in issue:
                categories['Route'] += 1
            elif '[TaskSel]' in issue:
                categories['TaskSel'] += 1
            elif '[Legacy]' in issue:
                categories['Legacy'] += 1
            else:
                categories['Other'] += 1

        return categories

    def get_state_consistency_check(self):
        """检查状态一致性 (Route-aware for Task B, Task-selection aware for U7)"""
        issues = []
        first_anomaly_order_id = None  # tracks the order_id of the first issue found

        # 检查订单与无人机状态一致性
        for order_id, order in self.env.orders.items():
            issues_before = len(issues)
            drone_id = order.get('assigned_drone')
            if drone_id is not None and drone_id >= 0:
                if drone_id not in self.env.drones:
                    issues.append(f"订单 {order_id} 分配的无人机 {drone_id} 不存在")
                    continue

                drone = self.env.drones[drone_id]
                planned_stops = drone.get('planned_stops', [])
                serving_order_id = drone.get('serving_order_id')

                # Route-aware check: when drone has planned_stops, use route logic
                if planned_stops and len(planned_stops) > 0:
                    # For route-plan mode, check if order is in the route
                    order_in_route = self._order_in_planned_stops(order_id, planned_stops)
                    order_in_cargo = order_id in drone.get('cargo', set())

                    if order['status'] == OrderStatus.ASSIGNED:
                        # ASSIGNED order should be in route or about to be picked up
                        if not order_in_route:
                            issues.append(f"[Route] 订单 {order_id} 已分配但不在无人机 {drone_id} 的路线中")

                    elif order['status'] == OrderStatus.PICKED_UP:
                        # PICKED_UP order consistency checks (route-aware)
                        has_delivery_stop = self._has_delivery_stop(order_id, planned_stops)

                        if len(planned_stops) > 0 and has_delivery_stop:
                            # Drone has active route with D stop for this order
                            if not order_in_cargo:
                                # This is a real inconsistency - order should be in cargo
                                issues.append(f"[Route] 订单 {order_id} 已取货但不在无人机 {drone_id} 的货物集合中")
                        elif len(planned_stops) > 0 and order_in_cargo and not has_delivery_stop:
                            # Order in cargo but no D stop - missing delivery stop
                            issues.append(f"[Route] 订单 {order_id} 已取货但缺少对应的 D stop")
                        # If planned_stops is empty or no D stop and not in cargo,
                        # drone is likely completing/resetting - this is OK

                # Task-selection mode check: when drone has serving_order_id but no planned_stops
                elif serving_order_id is not None:
                    # Validate serving order consistency
                    if order_id == serving_order_id:
                        # This is the order being served - validate status vs drone status
                        if order['status'] == OrderStatus.ASSIGNED:
                            # Should be going to merchant
                            if drone['status'] not in [DroneStatus.FLYING_TO_MERCHANT, DroneStatus.WAITING_FOR_PICKUP]:
                                issues.append(
                                    f"[TaskSel] 订单 {order_id} 为 serving_order (ASSIGNED) 但无人机 {drone_id} 状态不匹配: {drone['status']}"
                                )
                        elif order['status'] == OrderStatus.PICKED_UP:
                            # Should be in cargo and going to customer
                            if order_id not in drone.get('cargo', set()):
                                issues.append(
                                    f"[TaskSel] 订单 {order_id} 为 serving_order (PICKED_UP) 但不在无人机 {drone_id} 的货物中"
                                )
                            if drone['status'] not in [DroneStatus.FLYING_TO_CUSTOMER, DroneStatus.DELIVERING]:
                                issues.append(
                                    f"[TaskSel] 订单 {order_id} 为 serving_order (PICKED_UP) 但无人机 {drone_id} 状态不匹配: {drone['status']}"
                                )
                    else:
                        # This is NOT the serving order, but assigned to this drone
                        # In task-selection mode, it's OK to have other assigned orders that are not being served yet
                        # Only check cargo invariants
                        if order['status'] == OrderStatus.PICKED_UP:
                            if order_id not in drone.get('cargo', set()):
                                issues.append(
                                    f"[TaskSel] 订单 {order_id} 已取货但不在无人机 {drone_id} 的货物中"
                                )

                else:
                    # Legacy mode: original consistency check (no planned_stops and no serving_order_id)
                    # Allow more flexibility to avoid false positives
                    if order['status'] == OrderStatus.ASSIGNED:
                        # Only flag if drone is in a clearly wrong state (IDLE/CHARGING with assigned orders is OK)
                        if drone['status'] in [DroneStatus.RETURNING_TO_BASE]:
                            issues.append(
                                f"[Legacy] 订单 {order_id} 已分配但无人机 {drone_id} 正在返航: {drone['status']}")

                    elif order['status'] == OrderStatus.PICKED_UP:
                        # PICKED_UP orders should be in cargo (check cargo invariant)
                        if order_id not in drone.get('cargo', set()):
                            issues.append(f"[Legacy] 订单 {order_id} 已取货但不在无人机 {drone_id} 的货物中")
                        # Allow FLYING_TO_MERCHANT for batch orders (picking up another order)
                        if drone['status'] not in [DroneStatus.FLYING_TO_CUSTOMER, DroneStatus.DELIVERING,
                                                   DroneStatus.FLYING_TO_MERCHANT]:
                            if not ('batch_orders' in drone and order_id in drone['batch_orders']):
                                issues.append(
                                    f"[Legacy] 订单 {order_id} 已取货但无人机 {drone_id} 状态不匹配: {drone['status']}")

            # Track the first order that triggered any new issues
            if first_anomaly_order_id is None and len(issues) > issues_before:
                first_anomaly_order_id = order_id

        # First-anomaly snapshot: log detailed index state on the first issue ever found
        if issues and first_anomaly_order_id is not None and not self._first_anomaly_logged:
            self._log_first_anomaly_snapshot(first_anomaly_order_id, issues)

        return issues

    def _log_first_anomaly_snapshot(self, order_id: int, issues: list):
        """Print a one-time diagnostic snapshot the first time a consistency issue
        is detected.  Records the full index/cache membership so it is easy to
        tell whether 'status changed but index not updated' or 'double-transition'.
        """
        if self._first_anomaly_logged:
            return
        self._first_anomaly_logged = True

        env = self.env
        step = env.time_system.current_step
        order = env.orders.get(order_id, {})
        drone_id = order.get('assigned_drone', -1)
        drone = env.drones.get(drone_id, {}) if drone_id is not None and drone_id >= 0 else {}

        # Which drones' candidate mappings contain this order?
        in_candidate_drones = [
            d for d, mapping in env.drone_candidate_mappings.items()
            if any(oid == order_id for oid, _ in mapping)
        ]
        in_filtered_drones = [
            d for d, s in env._filtered_candidates_sets.items()
            if order_id in s
        ]

        print(f"\n{'='*70}")
        print(f"[ANOMALY SNAPSHOT] first consistency issue at step {step}")
        print(f"  order_id       : {order_id}")
        print(f"  order.status   : {order.get('status')}")
        print(f"  in active_orders: {order_id in env.active_orders}")
        print(f"  in _ready_orders_cache: {order_id in env._ready_orders_cache}")
        print(f"  assigned_drone : {drone_id}")
        print(f"  drone.status   : {drone.get('status')}")
        print(f"  drone.serving_order_id: {drone.get('serving_order_id')}")
        print(f"  drone.planned_stops len: {len(drone.get('planned_stops', []))}")
        print(f"  in drone_candidate_mappings of drones: {in_candidate_drones}")
        print(f"  in _filtered_candidates_sets of drones: {in_filtered_drones}")
        print(f"  issues:")
        for iss in issues:
            print(f"    - {iss}")
        print(f"{'='*70}\n")

    def _order_in_planned_stops(self, order_id, planned_stops):
        """Check if an order appears in the planned stops (D stop)"""
        for stop in planned_stops:
            if stop.get('type') == 'D' and stop.get('order_id') == order_id:
                return True
        return False

    def _has_delivery_stop(self, order_id, planned_stops):
        """Check if a delivery stop exists for the order"""
        return self._order_in_planned_stops(order_id, planned_stops)


# 路径规划可视化类
class PathVisualizer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        # deque(maxlen=100) per drone: O(1) append/eviction, bounded memory
        self.path_history = defaultdict(lambda: deque(maxlen=100))
        self.planned_paths = {}

    def update_path_history(self, drone_id, location):
        """更新无人机路径历史"""
        hist = self.path_history[drone_id]
        # 只记录位置变化
        if not hist or hist[-1] != location:
            hist.append(location)

    def update_planned_path(self, drone_id, current_loc, target_loc, route_preferences=None):
        """更新规划路径"""
        if route_preferences is not None:
            # 基于路径偏好生成规划路径
            planned_path = self._generate_path_from_preferences(current_loc, target_loc, route_preferences)
            self.planned_paths[drone_id] = planned_path

    def _generate_path_from_preferences(self, start, end, preferences):
        """基于路径偏好生成路径"""
        path = [start]
        current = start

        max_steps = 20  # 最大路径规划步数
        for step in range(max_steps):
            if self._distance(current, end) < 1.0:
                break

            # 计算方向向量
            dx = end[0] - current[0]
            dy = end[1] - current[1]

            # 标准化方向
            dist = max(0.1, math.sqrt(dx * dx + dy * dy))
            dx /= dist
            dy /= dist

            # 应用路径偏好
            next_x = current[0] + dx
            next_y = current[1] + dy

            # 确保在网格范围内
            next_x = max(0, min(self.grid_size - 1, next_x))
            next_y = max(0, min(self.grid_size - 1, next_y))

            # 检查路径偏好
            x_int, y_int = int(next_x), int(next_y)
            if 0 <= x_int < self.grid_size and 0 <= y_int < self.grid_size:
                # 确保 preference 是标量值
                preference_value = preferences[x_int, y_int]
                if hasattr(preference_value, '__iter__'):
                    preference_value = float(np.mean(preference_value))
                else:
                    preference_value = float(preference_value)

                # 高偏好区域更容易被选中
                if random.random() < preference_value:
                    current = (next_x, next_y)
                    path.append(current)
                else:
                    # 随机探索其他方向
                    current = (
                        current[0] + random.uniform(-0.5, 0.5),
                        current[1] + random.uniform(-0.5, 0.5)
                    )
                    current = (
                        max(0, min(self.grid_size - 1, current[0])),
                        max(0, min(self.grid_size - 1, current[1]))
                    )
                    path.append(current)

        path.append(end)  # 确保路径到达目标
        return path

    def _distance(self, loc1, loc2):
        """计算两点距离"""
        return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    def clear_paths(self):
        """清除所有路径"""
        self.path_history.clear()
        self.planned_paths.clear()


# 位置数据加载器类
class LocationDataLoader:
    def __init__(self, merchant_csv_path, user_csv_path, grid_size=10):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(0)
        self.merchant_locations = self._load_merchant_locations(merchant_csv_path)
        self.user_locations = self._load_user_locations(user_csv_path)

        # 计算经纬度范围用于归一化到网格坐标
        self._calculate_coordinate_range()
        # 计算基站位置
        self.base_locations = []

    def _load_merchant_locations(self, csv_path):
        """加载商家位置数据"""
        try:
            df = pd.read_csv(csv_path)
            locations = []
            for _, row in df.iterrows():
                location_str = str(row['location'])
                if ',' in location_str:
                    lon, lat = map(float, location_str.split(','))
                    locations.append({
                        'id': row['id'],
                        'name': row['name'],
                        'business_type': row['business_type'],
                        'longitude': lon,
                        'latitude': lat,
                        'address': row.get('address', ''),
                        'rating': row.get('rating', 4.0),
                        'cost': row.get('cost', 30.0)
                    })
            return locations
        except Exception as e:
            print(f"加载商家位置数据失败: {e}")
            return self._create_fallback_merchant_locations()

    def _load_user_locations(self, csv_path):
        """加载用户位置数据"""
        try:
            df = pd.read_csv(csv_path)
            locations = []
            for _, row in df.iterrows():
                locations.append({
                    'user_id': row['user_id'],
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'type': row.get('type', 'user')
                })
            return locations
        except Exception as e:
            print(f"加载用户位置数据失败: {e}")
            return self._create_fallback_user_locations()

    def _calculate_coordinate_range(self):
        """计算经纬度范围用于坐标归一化"""
        all_lats = []
        all_lons = []

        # 收集所有商家和用户的经纬度
        for merchant in self.merchant_locations:
            all_lats.append(merchant['latitude'])
            all_lons.append(merchant['longitude'])

        for user in self.user_locations:
            all_lats.append(user['latitude'])
            all_lons.append(user['longitude'])

        if all_lats and all_lons:
            self.min_lat = min(all_lats)
            self.max_lat = max(all_lats)
            self.min_lon = min(all_lons)
            self.max_lon = max(all_lons)

            # 计算缩放比例，保持纵横比
            lat_range = self.max_lat - self.min_lat
            lon_range = self.max_lon - self.min_lon

            # 使用较大的范围作为基准
            self.range_lat = lat_range if lat_range > 0 else 0.01
            self.range_lon = lon_range if lon_range > 0 else 0.01
        else:
            # 备用范围
            self.min_lat = 25.81
            self.max_lat = 25.82
            self.min_lon = 114.92
            self.max_lon = 114.93
            self.range_lat = 0.01
            self.range_lon = 0.01

    def _create_fallback_merchant_locations(self):
        """创建备用商家位置数据"""
        print("使用备用商家位置数据")
        locations = []
        for i in range(5):
            locations.append({
                'id': f'B{i}',
                'name': f'商家{i}',
                'business_type': '餐饮',
                'longitude': 114.92 + self.rng.uniform(-0.005, 0.005),
                'latitude': 25.815 + self.rng.uniform(-0.005, 0.005),
                'address': f'地址{i}',
                'rating': self.rng.uniform(3.5, 5.0),
                'cost': self.rng.uniform(20, 50)
            })
        return locations

    def _create_fallback_user_locations(self):
        """创建备用用户位置数据"""
        print("使用备用用户位置数据")
        locations = []
        for i in range(50):
            locations.append({
                'user_id': f'user_{i:04d}',
                'latitude': 25.815 + self.rng.uniform(-0.01, 0.01),
                'longitude': 114.92 + self.rng.uniform(-0.01, 0.01),
                'type': 'user'
            })
        return locations

    def convert_to_grid_coordinates(self, longitude, latitude):
        """将经纬度坐标转换为网格坐标"""
        # 归一化到 [0, 1] 范围
        norm_x = (longitude - self.min_lon) / self.range_lon
        norm_y = (latitude - self.min_lat) / self.range_lat

        # 缩放到网格大小
        grid_x = norm_x * (self.grid_size - 1)
        grid_y = norm_y * (self.grid_size - 1)

        return (grid_x, grid_y)

    def get_merchant_grid_locations(self):
        """获取商家在网格中的位置"""
        grid_locations = []
        for merchant in self.merchant_locations:
            grid_loc = self.convert_to_grid_coordinates(
                merchant['longitude'],
                merchant['latitude']
            )
            grid_locations.append({
                'id': merchant['id'],
                'name': merchant['name'],
                'grid_location': grid_loc,
                'original_location': (merchant['longitude'], merchant['latitude']),
                'business_type': merchant['business_type'],
                'rating': merchant['rating'],
                'cost': merchant['cost']
            })
        return grid_locations

    def get_random_user_grid_location(self):
        """随机获取一个用户在网格中的位置"""
        if not self.user_locations:
            return self._generate_random_grid_location()

        idx = int(self.rng.integers(0, len(self.user_locations)))
        user = self.user_locations[idx]
        grid_loc = self.convert_to_grid_coordinates(
            user['longitude'],
            user['latitude']
        )
        return grid_loc

    def _generate_random_grid_location(self):
        """生成随机网格位置（备用）"""
        return (
            self.rng.uniform(0, self.grid_size - 1),
            self.rng.uniform(0, self.grid_size - 1)
        )

    def find_optimal_base_locations(self, num_bases, method='kmeans'):
        """使用K-means、密度或随机寻找基站位置"""
        if method == 'kmeans':
            return self._kmeans_base_locations(num_bases)
        elif method == 'centroid':
            return self._centroid_base_locations(num_bases)
        else:
            return self._random_base_locations(num_bases)

    def _kmeans_base_locations(self, num_bases):
        """使用K-means聚类确定基站位置"""
        # 收集所有商家和用户的位置
        all_locations = []

        # 添加商家位置
        for merchant in self.merchant_locations:
            grid_loc = self.convert_to_grid_coordinates(
                merchant['longitude'], merchant['latitude']
            )
            all_locations.append(grid_loc)

        # 添加用户位置（采样一部分避免过多）- 修复：按索引采样
        user_sample_size = min(100, len(self.user_locations))
        if len(self.user_locations) > 0 and user_sample_size > 0:
            idx = self.rng.choice(np.arange(len(self.user_locations)), size=user_sample_size, replace=False)
            sampled_users = [self.user_locations[int(i)] for i in idx]
        else:
            sampled_users = []

        for user in sampled_users:
            grid_loc = self.convert_to_grid_coordinates(
                user['longitude'], user['latitude']
            )
            all_locations.append(grid_loc)

        if len(all_locations) < num_bases:
            # 如果位置数量不足，补充随机位置
            while len(all_locations) < num_bases:
                all_locations.append((
                    self.rng.uniform(0, self.grid_size - 1),
                    self.rng.uniform(0, self.grid_size - 1)
                ))

        # 转换为numpy数组
        locations_array = np.array(all_locations)

        # 使用K-means聚类
        kmeans = KMeans(n_clusters=num_bases, random_state=42, n_init=10)
        kmeans.fit(locations_array)

        # 获取聚类中心作为基站位置
        base_locations = kmeans.cluster_centers_

        # 确保在网格范围内
        base_locations = np.clip(base_locations, 0, self.grid_size - 1)

        print(f"K-means找到 {len(base_locations)} 个基站位置")
        return base_locations.tolist()

    def _centroid_base_locations(self, num_bases):
        """基于密度确定基站位置"""
        # 创建网格密度图
        density_map = np.zeros((self.grid_size, self.grid_size))

        # 统计每个网格点的位置密度
        for merchant in self.merchant_locations:
            grid_loc = self.convert_to_grid_coordinates(
                merchant['longitude'], merchant['latitude']
            )
            x, y = int(grid_loc[0]), int(grid_loc[1])
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                density_map[x, y] += 2  # 商家权重更高

        # 修复：按索引采样用户
        user_sample_size = min(200, len(self.user_locations))
        if len(self.user_locations) > 0 and user_sample_size > 0:
            idx = self.rng.choice(np.arange(len(self.user_locations)), size=user_sample_size, replace=False)
            sampled_users = [self.user_locations[int(i)] for i in idx]
        else:
            sampled_users = []

        for user in sampled_users:
            grid_loc = self.convert_to_grid_coordinates(
                user['longitude'], user['latitude']
            )
            x, y = int(grid_loc[0]), int(grid_loc[1])
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                density_map[x, y] += 1

        # 找到密度最高的num_bases个位置
        base_locations = []
        for _ in range(num_bases):
            max_idx = np.unravel_index(np.argmax(density_map), density_map.shape)
            base_locations.append((max_idx[0] + 0.5, max_idx[1] + 0.5))  # 使用网格中心

            # 将周围区域密度降低，避免基站太近
            x, y = max_idx
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        distance = np.sqrt(dx ** 2 + dy ** 2)
                        density_map[nx, ny] *= max(0, 1 - distance / 5)

        print(f"基于密度找到 {len(base_locations)} 个基站位置")
        return base_locations

    def _random_base_locations(self, num_bases):
        """随机生成基站位置（备用方法）"""
        base_locations = []
        for i in range(num_bases):
            base_locations.append((
                self.rng.uniform(0, self.grid_size - 1),
                self.rng.uniform(0, self.grid_size - 1)
            ))
        print(f"随机生成 {len(base_locations)} 个基站位置")
        return base_locations


# 帕累托优化器
class ParetoOptimizer:
    def __init__(self, num_objectives=3):
        self.num_objectives = num_objectives
        self.pareto_front = []
        self.solutions_history = []

    def update_pareto_front(self, solution):
        """更新帕累托前沿"""
        solution = np.array(solution)
        is_dominated = False

        # 创建前沿的副本用于迭代
        front_to_check = self.pareto_front.copy()

        # 检查是否被现有前沿中的解支配
        for front_solution in front_to_check:
            if self._dominates(front_solution, solution):
                is_dominated = True
                break

        # 移除被新解支配的旧解
        new_front = []
        for front_solution in self.pareto_front:
            if not self._dominates(solution, front_solution):
                new_front.append(front_solution)

        self.pareto_front = new_front

        # 如果未被支配，加入帕累托前沿
        if not is_dominated:
            self.pareto_front.append(solution)

        self.solutions_history.append(solution.copy())

    def _dominates(self, sol1, sol2):
        """检查sol1是否支配sol2"""
        # 对于最大化问题，sol1在所有目标上都不比sol2差，且至少在一个目标上严格更好
        all_better_or_equal = np.all(sol1 >= sol2)
        at_least_one_better = np.any(sol1 > sol2)
        return all_better_or_equal and at_least_one_better

    def get_pareto_front(self):
        """获取帕累托前沿"""
        return np.array(self.pareto_front)

    def calculate_hypervolume(self, reference_point):
        """计算超体积指标"""
        if len(self.pareto_front) == 0:
            return 0.0

        front = np.array(self.pareto_front)
        # 简化的超体积计算
        return np.sum([np.prod(solution) for solution in front])

    def get_diversity(self):
        """计算帕累托前沿的多样性"""
        if len(self.pareto_front) < 2:
            return 0.0

        front = np.array(self.pareto_front)
        distances = []
        for i in range(len(front)):
            for j in range(i + 1, len(front)):
                distances.append(np.linalg.norm(front[i] - front[j]))

        return np.mean(distances) if distances else 0.0


# 天气数据处理器
class WeatherDataProcessor:
    def __init__(self, csv_path):
        self.weather_df = self._load_weather_data(csv_path)
        self.data_length = len(self.weather_df)

    def _load_weather_data(self, csv_path):
        """加载天气数据"""
        try:
            df = pd.read_csv(csv_path)
            df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='coerce', utc=True)
            df = df.dropna(subset=['Formatted Date', 'Summary', 'Temperature (C)'])
            return df
        except Exception as e:
            print(f"加载天气数据失败: {e}")
            return self._create_fallback_weather_data()

    def _create_fallback_weather_data(self):
        """创建备用天气数据"""
        print("使用备用天气数据")
        _rng = np.random.default_rng(0)
        dates = pd.date_range(start='2006-01-01', end='2016-12-31', freq='h')
        data = {
            'Formatted Date': dates,
            'Summary': _rng.choice(['Clear', 'Partly Cloudy', 'Cloudy', 'Rain', 'Windy', 'Fog'], len(dates)),
            'Temperature (C)': _rng.normal(15, 10, len(dates)),
            'Humidity': _rng.uniform(0.3, 0.9, len(dates)),
            'Wind Speed (km/h)': _rng.exponential(10, len(dates)),
            'Visibility (km)': _rng.uniform(5, 20, len(dates)),
            'Pressure (millibars)': _rng.normal(1013, 10, len(dates)),
            'Precip Type': _rng.choice(['rain', 'snow', 'none'], len(dates), p=[0.2, 0.05, 0.75])
        }
        return pd.DataFrame(data)

    def get_weather_at_time(self, env_time):
        """根据环境时间获取天气数据"""
        index = int(env_time) % max(1, self.data_length)
        return self.weather_df.iloc[index]

    def map_to_weather_type(self, weather_summary):
        """将天气摘要映射到WeatherType枚举"""
        summary_lower = str(weather_summary).lower()
        extreme_keywords = ['storm', 'thunderstorm', 'heavy', 'blizzard', 'hurricane', 'tornado']
        if any(keyword in summary_lower for keyword in extreme_keywords):
            return WeatherType.EXTREME
        rain_keywords = ['rain', 'drizzle', 'shower', 'precip', 'wet']
        if any(keyword in summary_lower for keyword in rain_keywords):
            return WeatherType.RAINY
        wind_keywords = ['wind', 'breezy', 'gust']
        if any(keyword in summary_lower for keyword in wind_keywords):
            return WeatherType.WINDY
        sunny_keywords = ['clear', 'sunny', 'fair']
        if any(keyword in summary_lower for keyword in sunny_keywords):
            return WeatherType.SUNNY
        return WeatherType.SUNNY


# 订单数据处理器
class OrderDataProcessor:
    def __init__(self, excel_path, grid_size=10, merchant_ids=None, time_system: Optional[DailyTimeSystem] = None):
        self.grid_size = grid_size
        self.merchant_ids = merchant_ids if merchant_ids else []
        self.num_merchants = len(merchant_ids) if merchant_ids else 5
        self.rng = np.random.default_rng(0)
        self.order_df = self._load_order_data(excel_path)
        self.order_patterns = self._analyze_order_patterns()
        self.time_system = time_system  # 注入时间系统，供准备时间计算
        print(f"订单模式分析完成，商家数量: {self.num_merchants}")

    def _load_order_data(self, excel_path):
        """加载订单数据"""
        try:
            df = pd.read_excel(excel_path)
            required_columns = ['order_time', 'merchant_id', 'order_type', 'preparation_time', 'distance']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'order_time':
                        df[col] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
                    elif col == 'merchant_id':
                        if self.merchant_ids:
                            df[col] = self.rng.choice(self.merchant_ids, len(df))
                        else:
                            df[col] = self.rng.integers(0, self.num_merchants, len(df))
                    elif col == 'order_type':
                        df[col] = self.rng.choice([0, 1, 2], len(df), p=[0.8, 0.15, 0.05])
                    elif col == 'preparation_time':
                        df[col] = self.rng.integers(3, 10, len(df))
                    elif col == 'distance':
                        df[col] = self.rng.exponential(3, len(df))
            return df
        except Exception as e:
            print(f"加载订单数据失败: {e}")
            return self._create_fallback_order_data()

    def _create_fallback_order_data(self):
        """创建备用订单数据"""
        print("使用备用订单数据")
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='h')
        data = {
            'order_time': dates,
            'merchant_id': self.rng.choice(self.merchant_ids, len(dates)) if self.merchant_ids else self.rng.integers(
                0, self.num_merchants, len(dates)),
            'order_type': self.rng.choice([0, 1, 2], len(dates), p=[0.8, 0.15, 0.05]),
            'preparation_time': self.rng.integers(3, 10, len(dates)),
            'distance': self.rng.exponential(3, len(dates)),
            'is_peak': self.rng.choice([0, 1], len(dates), p=[0.7, 0.3])
        }
        return pd.DataFrame(data)

    def _analyze_order_patterns(self):
        """分析订单模式"""
        if self.order_df is None or len(self.order_df) == 0:
            return self._create_default_patterns()
        hourly_pattern = self._analyze_hourly_pattern()
        merchant_distribution = self._analyze_merchant_distribution()
        order_type_distribution = self._analyze_order_type_distribution()
        return {
            'hourly_pattern': hourly_pattern,
            'merchant_distribution': merchant_distribution,
            'order_type_distribution': order_type_distribution,
            'peak_hours': [11, 12, 17, 18, 19],
            'weekend_boost': 1.3
        }

    def _analyze_hourly_pattern(self):
        """分析小时订单模式"""
        if 'order_time' not in self.order_df.columns:
            return self._create_default_hourly_pattern()
        self.order_df['hour'] = self.order_df['order_time'].dt.hour
        hourly_counts = self.order_df['hour'].value_counts().sort_index()
        pattern = np.zeros(24)
        for hour, count in hourly_counts.items():
            pattern[hour] = count
        if pattern.sum() > 0:
            pattern = pattern / pattern.max()
        return pattern

    def _analyze_merchant_distribution(self):
        """分析商家分布"""
        if 'merchant_id' not in self.order_df.columns:
            return np.ones(self.num_merchants) / self.num_merchants

        merchant_counts = self.order_df['merchant_id'].value_counts()
        distribution = np.zeros(self.num_merchants)

        if self.merchant_ids:
            for i, merchant_id in enumerate(self.merchant_ids):
                if merchant_id in merchant_counts.index:
                    distribution[i] = merchant_counts[merchant_id]
        else:
            for merchant_id, count in merchant_counts.items():
                if 0 <= merchant_id < self.num_merchants:
                    distribution[merchant_id] = count

        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
        return distribution

    def _analyze_order_type_distribution(self):
        """分析订单类型分布"""
        if 'order_type' not in self.order_df.columns:
            return [0.8, 0.15, 0.05]
        type_counts = self.order_df['order_type'].value_counts()
        distribution = np.zeros(3)
        for order_type, count in type_counts.items():
            if 0 <= order_type <= 2:
                distribution[order_type] = count
        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
        return distribution

    def _create_default_patterns(self):
        """创建默认模式"""
        return {
            'hourly_pattern': self._create_default_hourly_pattern(),
            'merchant_distribution': np.ones(self.num_merchants) / self.num_merchants,
            'order_type_distribution': [0.8, 0.15, 0.05],
            'peak_hours': [11, 12, 17, 18, 19],
            'weekend_boost': 1.3
        }

    def _create_default_hourly_pattern(self):
        """创建默认小时模式"""
        pattern = np.array([0.1, 0.05, 0.02, 0.01, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                            0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2])
        return pattern / pattern.max()

    def get_order_probability(self, env_time, weather_type):
        """获取订单生成概率 - 高负载版本（移除负载抑制）"""
        hour = env_time % 24
        base_prob = self.order_patterns['hourly_pattern'][hour]
        base_prob = min(0.95, base_prob * 3.0)

        weather_impact = self._get_weather_impact(weather_type)
        time_impact = 2.5 if hour in self.order_patterns['peak_hours'] else 1.0
        day_of_week = (env_time // 24) % 7
        weekend_impact = self.order_patterns['weekend_boost'] if day_of_week >= 5 else 1.0

        final_prob = base_prob * weather_impact * time_impact * weekend_impact
        final_prob = min(0.95, max(0.2, final_prob))
        return final_prob

    def _get_weather_impact(self, weather_type):
        """调整天气影响 - 降低负面影响"""
        if weather_type == WeatherType.SUNNY:
            return 1.5
        elif weather_type == WeatherType.RAINY:
            return 1.2
        elif weather_type == WeatherType.WINDY:
            return 0.9
        elif weather_type == WeatherType.EXTREME:
            return 0.6
        else:
            return 1.0

    def generate_order_details(self, env_time, weather_type):
        """生成订单详细信息"""
        if self.merchant_ids:
            merchant_id = self.rng.choice(self.merchant_ids, p=self.order_patterns['merchant_distribution'])
        else:
            merchant_id = self.rng.choice(range(self.num_merchants), p=self.order_patterns['merchant_distribution'])

        order_type = self.rng.choice([0, 1, 2], p=self.order_patterns['order_type_distribution'])
        self.current_order_type = order_type

        if weather_type in [WeatherType.RAINY, WeatherType.EXTREME]:
            order_type_weights = [0.6, 0.2, 0.2]
            order_type = self.rng.choice([0, 1, 2], p=order_type_weights)

        preparation_time = self._generate_preparation_time(weather_type)
        urgency = self._generate_urgency(env_time % 24)

        if weather_type in [WeatherType.RAINY, WeatherType.EXTREME]:
            max_distance = self.grid_size // 2
        else:
            max_distance = self.grid_size

        return {
            'merchant_id': merchant_id,
            'order_type': order_type,
            'preparation_time': preparation_time,
            'urgency': urgency,
            'max_distance': max_distance
        }

    def _generate_preparation_time(self, weather_type):
        """生成更符合实际的准备时间（返回 step）"""
        base_time_minutes = 0

        if hasattr(self, 'current_order_type'):
            if self.current_order_type == 0:
                base_time_minutes = int(self.rng.integers(1, 6))
            elif self.current_order_type == 1:
                base_time_minutes = int(self.rng.integers(2, 6))
            else:
                base_time_minutes = int(self.rng.integers(10, 21))
        else:
            base_time_minutes = int(self.rng.integers(5, 16))

        if weather_type == WeatherType.EXTREME:
            base_time_minutes += int(self.rng.integers(3, 9))
        elif weather_type == WeatherType.RAINY:
            base_time_minutes += int(self.rng.integers(1, 4))

        if hasattr(self, 'time_system') and self.time_system is not None:
            minutes_per_step = 60 // self.time_system.steps_per_hour
        else:
            minutes_per_step = 15

        preparation_steps = max(1, int(math.ceil(base_time_minutes / max(minutes_per_step, 1))))
        return preparation_steps

    def _generate_urgency(self, hour):
        """生成紧急程度"""
        return 0.3 if hour in [11, 12, 17, 18, 19] else 0.1


# 三目标帕累托无人机配送环境 - 天级版本 - 高负载无限制
class ThreeObjectiveDroneDeliveryEnv(gym.Env):
    """三目标帕累托无人机餐食配送环境 - 高负载无限制版本"""

    metadata = {'render_modes': ['human', 'rgb_array', 'matplotlib']}

    def __init__(self,
                 grid_size=16,
                 num_drones=6,
                 max_orders=100,
                 steps_per_hour=4,
                 weather_csv_path='weather_dataset.csv',
                 order_excel_path='food_delivery_data.xlsx',
                 merchant_location_path='food_business_zhanggong_locations.csv',
                 user_location_path='user_zhanggong_locations.csv',
                 base_placement_method='kmeans',
                 drone_max_capacity=10,
                 operating_hours=(6, 22),
                 high_load_factor=2.0,
                 distance_reward_weight=1.0,
                 multi_objective_mode: str = "conditioned",
                 fixed_objective_weights=(0.5, 0.3, 0.2),
                 shaping_progress_k: float = 0.3,
                 shaping_distance_k: float = 0.05,
                 shaping_energy_k: float = 0.01,
                 heading_guidance_alpha: float = 0.5,
                 # ===== 新增：固定 num_bases 与 Top-K 商家观测 =====
                 num_bases: Optional[int] = None,
                 top_k_merchants: int = 100,
                 reward_output_mode: str = "zero",
                 enable_random_events: bool = False,  # 可选：评估时建议关掉随机事件
                 debug_state_warnings: bool = False,  # Task B: control state consistency warning output
                 delivery_sla_steps: int = 6,  # READY-based delivery SLA in steps (increased for better pickup time)
                 timeout_factor: float = 4.0,  # Multiplier for deadline calculation (increased for better pickup time)
                 # ===== U7: Task selection parameters =====
                 num_candidates: int = 20,  # K=20 candidate orders per drone for PPO selection
                 # ===== U8: Rule-based action space =====
                 rule_count: int = 5,  # Number of interpretable rules for discrete action space
                 # ===== U9: Candidate-based filtering parameters =====
                 candidate_fallback_enabled: bool = True,  # Allow fallback to full active_orders if candidates empty
                 candidate_update_interval: int = 1,  # Update candidates every N steps (1=every step, 0=only on reset)
                 # ===== Legacy fallback control =====
                 enable_legacy_fallback: bool = True,  # Enable legacy fallback behavior for backward compatibility
                 # ===== Diagnostics control =====
                 enable_diagnostics: bool = False,  # Enable detailed diagnostics for debugging
                 diagnostics_interval: int = 64,  # Print diagnostics every N steps
                 # ===== Energy consumption model parameters =====
                 energy_e0: float = 0.1,  # Base energy consumption per unit distance (battery_units/distance_unit)
                 energy_alpha: float = 0.5,  # Load coefficient for energy consumption
                 battery_return_threshold: float = 10.0,  # Low battery threshold for forced return (battery_units)
                 # ===== Order cutoff window =====
                 order_cutoff_steps: int = 0,  # Stop accepting orders this many steps before business end (0=disabled)
                 # ===== Sigmoid hazard cancellation =====
                 enable_sigmoid_hazard_cancellation: bool = True,
                 # Enable stochastic sigmoid-hazard order cancellation (replaces hard deadline).
                 # When True, each active READY/ASSIGNED order is cancelled probabilistically each step
                 # using h(w) = hazard_p_max * sigmoid(hazard_k * (w - hazard_midpoint_steps)).
                 hazard_p_max: float = 0.05,
                 # Maximum per-step cancellation probability (upper bound of hazard curve).
                 # A value of 0.05 means at most 5% chance of cancellation per step even at very long waits.
                 hazard_k: float = 0.3,
                 # Steepness of the sigmoid hazard curve. Higher values create a sharper transition
                 # around the midpoint. Default 0.3 gives a gradual ramp-up.
                 hazard_midpoint_steps: float = 12.0,
                 # Waiting-time midpoint (in steps) where the hazard reaches 50% of p_max.
                 # Default 12 steps (~3 hours at 4 steps/hour) is when risk begins rising noticeably.
                 ):
        super().__init__()

        # ====== 独立 RNG（由 reset(seed=...) 重置，隔离环境随机流）======
        self.np_random = np.random.default_rng(0)
        # 订单生成专用 RNG：与 np_random 完全隔离，保证不同策略下相同种子产生相同订单序列
        self.order_rng = np.random.default_rng(1)

        # ====== 固定基础参数（init 一次性确定）======

        self.grid_size = int(grid_size)
        self.num_drones = int(num_drones)
        self.max_obs_orders = int(max_orders)
        self.high_load_factor = float(high_load_factor)
        self.num_objectives = 3
        self.drone_max_capacity = int(drone_max_capacity)

        self.distance_reward_weight = float(distance_reward_weight)

        self.top_k_merchants = int(top_k_merchants)

        # ========== U7: Task selection parameters ==========
        self.num_candidates = int(num_candidates)  # K=20 candidates per drone
        # Store candidate mappings per drone: {drone_id: [(order_id, is_valid), ...]}
        self.drone_candidate_mappings = {}

        # ========== U8: Rule-based action space ==========
        self.rule_count = int(rule_count)  # R=5 interpretable rules
        # Rule usage statistics for diagnostics
        self.rule_usage_stats = defaultdict(int)

        # ========== U9: Candidate-based filtering ==========
        self.candidate_fallback_enabled = bool(candidate_fallback_enabled)
        self.candidate_update_interval = int(candidate_update_interval)
        # External candidate generator (can be set via set_candidate_generator)
        self.candidate_generator = None
        # Filtered candidate sets: {drone_id: [order_id, ...]}
        self.filtered_candidates = {}
        # Cached set version of filtered_candidates for O(1) membership tests
        self._filtered_candidates_sets: Dict[int, set] = {}

        # ========== 多目标训练方式 ==========
        self.multi_objective_mode = multi_objective_mode
        self.fixed_objective_weights = np.array(fixed_objective_weights, dtype=np.float32)
        self.fixed_objective_weights = self.fixed_objective_weights / (self.fixed_objective_weights.sum() + 1e-8)
        self.objective_weights = self.fixed_objective_weights.copy()
        self.reward_output_mode = str(reward_output_mode)
        self.enable_random_events = bool(enable_random_events)
        self.debug_state_warnings = bool(debug_state_warnings)  # Task B: debug flag
        self.delivery_sla_steps = int(delivery_sla_steps)  # READY-based delivery SLA
        self.timeout_factor = float(timeout_factor)  # Deadline multiplier，已经用超时取消模型替代固定
        self.episode_r_vec = np.zeros(self.num_objectives, dtype=np.float32)

        # ========== Legacy fallback control ==========
        self.enable_legacy_fallback = bool(enable_legacy_fallback)
        self.legacy_blocked_count = 0  # Counter for blocked legacy attempts
        self.legacy_blocked_reasons = defaultdict(int)  # Track reasons for blocking

        # ========== Diagnostics control ==========
        self.enable_diagnostics = bool(enable_diagnostics)
        self.diagnostics_interval = int(diagnostics_interval)
        self.action_applied_count = 0  # How many drones had targets updated from action this step

        # Cache decision points from before action execution for consistent statistics
        self._last_decision_points_mask = [False] * self.num_drones  # List[bool] - which drones were at decision points
        self._last_decision_points_count = 0  # int - count of drones at decision points

        # Reward component tracking for diagnostics
        self.last_step_reward_components = {
            'obj0_total': 0.0,
            'obj0_completed': 0.0,
            'obj0_cancelled': 0.0,
            'obj0_progress_shaping': 0.0,
            'obj1_total': 0.0,
            'obj1_energy_cost': 0.0,
            'obj1_distance_cost': 0.0,
            'obj2_total': 0.0,
            'obj2_on_time': 0.0,
            'obj2_cancelled': 0.0,
            'obj2_backlog': 0.0,
            'delta_energy': 0.0,
            'delta_distance': 0.0,
            'delta_completed': 0.0,
            'delta_cancelled': 0.0
        }

        # ========== Energy consumption model parameters ==========
        self.energy_e0 = float(energy_e0)  # Base energy per distance
        self.energy_alpha = float(energy_alpha)  # Load coefficient
        self.battery_return_threshold = float(battery_return_threshold)  # Low battery threshold

        # ========== Order cutoff window ==========
        self.order_cutoff_steps = int(order_cutoff_steps)  # Steps before business end to stop accepting orders

        # ========== Sigmoid hazard cancellation ==========
        self.enable_sigmoid_hazard_cancellation = bool(enable_sigmoid_hazard_cancellation)
        self.hazard_p_max = float(hazard_p_max)
        self.hazard_k = float(hazard_k)
        self.hazard_midpoint_steps = float(hazard_midpoint_steps)

        # ========== shaping 参数 ==========
        self.shaping_progress_k = float(shaping_progress_k)
        self.shaping_distance_k = float(shaping_distance_k)
        self.shaping_energy_k = float(shaping_energy_k)
        self.heading_guidance_alpha = float(np.clip(heading_guidance_alpha, 0.0, 1.0))
        self._prev_target_dist = np.zeros(self.num_drones, dtype=np.float32)
        # Track previous target location for each drone to detect target changes
        self._prev_target_loc = [None] * self.num_drones

        # ====== 时间系统（先建立，后续多个组件要用）======
        start_hour, end_hour = operating_hours
        self.time_system = DailyTimeSystem(
            start_hour=start_hour,
            end_hour=end_hour,
            steps_per_hour=steps_per_hour
        )

        # ======== 1) 先加载位置与商家列表（全量商家数）========
        self.location_loader = LocationDataLoader(
            merchant_location_path, user_location_path, self.grid_size
        )
        merchant_grid_locations = self.location_loader.get_merchant_grid_locations()
        self.total_merchants = len(merchant_grid_locations)
        self.merchant_grid_data = merchant_grid_locations
        self.merchant_ids = [m['id'] for m in merchant_grid_locations]

        # ======== 2) 固定 num_bases（init 只算一次）========
        if num_bases is None:
            self.num_bases = max(2, self.num_drones // 4)
        else:
            self.num_bases = max(1, int(num_bases))

        # ======== 3) 数据处理器（依赖 time_system / merchant_ids）========
        self.weather_processor = WeatherDataProcessor(weather_csv_path)
        self.order_processor = OrderDataProcessor(
            order_excel_path, self.grid_size, self.merchant_ids, time_system=self.time_system
        )

        # 优化器/可视化
        self.pareto_optimizer = ParetoOptimizer(self.num_objectives)
        self.path_visualizer = PathVisualizer(self.grid_size)

        # ======== 4) 初始化实体：locations/bases/merchants/drones/orders ========
        self._init_locations_fixed_bases(base_placement_method)
        self._init_bases()
        self._init_merchants()
        self._init_drones()
        self._init_orders()

        # ======== 5) 固定观测维度（obs_num_*）========
        self.obs_num_bases = int(self.num_bases)
        self.obs_num_merchants = int(min(self.top_k_merchants, len(self.merchants)))

        # 状态管理器（放在 spaces 前后都行；这里放前面便于后续 reset/生成时用）
        # Initialize _ready_orders_cache before StateManager so update_order_status can use it
        self._ready_orders_cache: set = set()
        # Dirty flag: set True whenever candidate caches are evicted; cleared by update_filtered_candidates.
        # Consumers (get_decision_drones, candidate selectors) can use this as a hint to refresh.
        self._candidate_mappings_dirty: bool = False
        self.state_manager = StateManager(self)

        # 定义 spaces：只使用固定 shape
        self._define_spaces()

        # 每日统计
        self.daily_stats = {
            'day_number': 0,
            'orders_generated': 0,
            'orders_completed': 0,
            'orders_cancelled': 0,
            'revenue': 0,
            'costs': 0,
            'energy_consumed': 0,
            'on_time_deliveries': 0,
            'total_flight_distance': 0.0,
            'optimal_flight_distance': 0.0,
            'forced_return_events': 0,  # Track forced returns due to low battery
            'total_waiting_time': 0,    # Cumulative (delivery_time - creation_time) for completed orders
        }

        # Debug tracking for on_time_deliveries (to detect decreases)
        self._prev_on_time_deliveries = 0
        self._on_time_decrease_warned = False

        # 性能指标
        self.metrics = {
            'total_orders': 0,
            'completed_orders': 0,
            'cancelled_orders': 0,
            'total_delivery_time': 0,
            'total_waiting_time': 0,    # Cumulative (delivery_time - creation_time) for completed orders
            'total_revenue': 0,
            'total_cost': 0,
            'energy_consumed': 0,
            'collisions_avoided': 0,
            'on_time_deliveries': 0,
            'total_flight_distance': 0.0,
            'optimal_flight_distance': 0.0,
            'weather_impact_stats': {
                'sunny_deliveries': 0,
                'rainy_deliveries': 0,
                'windy_deliveries': 0,
                'extreme_deliveries': 0
            },
            # Bounded deques to prevent unbounded growth of sample lists
            'assignment_slack_samples': deque(maxlen=2000),
            'ready_based_lateness_samples': deque(maxlen=2000),
        }

        # 事件/历史 - 使用有界 deque 防止长训练时内存无限增长
        self.event_queue = deque()
        self.order_history = deque(maxlen=1000)
        self.weather_history = deque(maxlen=200)
        self.pareto_history = deque(maxlen=200)
        self.weather = WeatherType.SUNNY

        # Running stats for order_history summary (O(1) in _get_info)
        self._order_hist_merchant_ids: set = set()
        self._order_hist_dist_sum: float = 0.0
        self._order_hist_dist_count: int = 0

        # Lightweight per-step timing accumulators (active only when enable_diagnostics=True)
        # Keys: 'candidate_update', 'event_processing', 'position_update', 'observation_build'
        self._perf_accum: Dict[str, float] = {}
        self._perf_steps: int = 0

        # Decision tracking for decentralized execution
        self.last_decision_info = {
            'drone_id': -1,
            'rule_id': -1,
            'success': False,
            'failure_reason': None,
        }
        self.weather_details = {}

        # 营业后履约
        self.allow_overtime_fulfillment = False
        self.overtime_max_steps = self.time_system.steps_per_hour * 4
        self.in_overtime = False
        self.overtime_steps = 0

        self._assigned_in_step = set()

        # 归一化上限
        self.max_queue_cap: float = 50.0
        self.max_eff_cap: float = 2.0

        # 订单ID计数器
        self.global_order_counter = 0
        self.current_obs_order_ids = []

    # ------------------ 初始化相关 ------------------

    def _init_locations_fixed_bases(self, base_method: str = "kmeans"):
        """固定 num_bases：只生成 num_bases 个 base 坐标；不再在这里修改 self.num_bases"""
        self.locations = {}

        base_locations = self.location_loader.find_optimal_base_locations(
            self.num_bases, method=base_method
        )

        for i, base_loc in enumerate(base_locations):
            self.locations[f'base_{i}'] = (float(base_loc[0]), float(base_loc[1]))

        for merchant_data in self.merchant_grid_data:
            merchant_id = merchant_data['id']
            self.locations[f'merchant_{merchant_id}'] = merchant_data['grid_location']

    def _init_bases(self):
        """初始化无人机基站"""
        self.bases = {}
        for i in range(self.num_bases):
            base_location = self.locations.get(f'base_{i}')
            if base_location is None:
                base_location = (
                    self.np_random.uniform(0, self.grid_size - 1),
                    self.np_random.uniform(0, self.grid_size - 1)
                )
                self.locations[f'base_{i}'] = base_location

            self.bases[i] = {
                'location': base_location,
                'capacity': max(1, self.num_drones // self.num_bases),
                'drones_available': [],
                'charging_stations': 5,
                'charging_queue': deque(),
                'coverage_radius': self.grid_size / 3
            }

    def _init_merchants(self):
        """初始化商家"""
        self.merchants = {}

        for merchant_data in self.merchant_grid_data:
            merchant_id = merchant_data['id']

            business_type = merchant_data['business_type']
            if '饮料' in business_type or '奶茶' in business_type or '咖啡' in business_type:
                preparation_efficiency = self.np_random.uniform(1.5, 2.0)
                base_prep_time = int(self.np_random.integers(2, 5))
            elif '快餐' in business_type or '小吃' in business_type:
                preparation_efficiency = self.np_random.uniform(1.2, 1.5)
                base_prep_time = int(self.np_random.integers(5, 9))
            else:
                preparation_efficiency = self.np_random.uniform(1.0, 1.3)
                base_prep_time = int(self.np_random.integers(8, 13))

            self.merchants[merchant_id] = {
                'location': merchant_data['grid_location'],
                'original_location': merchant_data['original_location'],
                'name': merchant_data['name'],
                'business_type': merchant_data['business_type'],
                'rating': merchant_data['rating'],
                'avg_cost': merchant_data['cost'],
                'queue': deque(),
                'base_preparation_time': base_prep_time,
                'efficiency': preparation_efficiency,
                'order_count': 0,
                'cancellation_rate': self.np_random.uniform(0.01, 0.03),
                'landing_zone': True
            }

    def _init_drones(self):
        """初始化无人机"""
        self.drones = {}
        for i in range(self.num_drones):
            base_id = i % self.num_bases
            base_location = self.bases[base_id]['location']

            self.drones[i] = {
                'location': base_location,
                'base': base_id,
                'status': DroneStatus.IDLE,
                'current_order': None,
                'orders_completed': 0,
                'speed': 3,
                'reliability': self.np_random.uniform(0.95, 0.99),
                'max_capacity': self.drone_max_capacity,
                'current_load': 0,
                'battery_level': 100.0,
                'max_battery': 100.0,
                'battery_consumption_rate': 0.2,
                'charging_rate': 10.0,
                'cancellation_rate': 0.005,
                'total_distance_today': 0.0,
                'planned_stops': deque(),
                # deque of stops: {'type':'P','merchant_id':mid} or {'type':'D','order_id':oid}
                'cargo': set(),  # picked-up orders (order_ids) not yet delivered
                'current_stop': None,
                'route_committed': False,
                'serving_order_id': None,  # U7 task-selection: currently executing order
            }
            self.bases[base_id]['drones_available'].append(i)

    def _init_orders(self):
        """初始化订单池 - 支持无限订单"""
        self.orders = {}
        self.active_orders = set()
        self.completed_orders = set()
        self.cancelled_orders = set()
        self.global_order_counter = 0

    def _define_spaces(self):
        """定义观察和动作空间（固定 shape）"""
        self.observation_space = spaces.Dict({
            'orders': spaces.Box(low=0, high=1, shape=(self.max_obs_orders, 10), dtype=np.float32),
            'drones': spaces.Box(low=0, high=1, shape=(self.num_drones, 8), dtype=np.float32),

            # Top-K merchants
            'merchants': spaces.Box(low=0, high=1, shape=(self.obs_num_merchants, 4), dtype=np.float32),

            # 固定 num_bases
            'bases': spaces.Box(low=0, high=1, shape=(self.obs_num_bases, 3), dtype=np.float32),

            # U7: Candidate orders per drone (K=20, F=12 features)
            'candidates': spaces.Box(low=0, high=1, shape=(self.num_drones, self.num_candidates, 12), dtype=np.float32),

            'weather': spaces.Discrete(len(WeatherType)),
            'weather_details': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            'time': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            'day_progress': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'resource_saturation': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'order_pattern': spaces.Box(low=0, high=1, shape=(24,), dtype=np.float32),
            'pareto_info': spaces.Box(low=0, high=1, shape=(self.num_objectives + 2,), dtype=np.float32),
            'objective_weights': spaces.Box(low=0, high=1, shape=(self.num_objectives,), dtype=np.float32),
        })

        # U8: PPO outputs discrete rule_id for each drone (MultiDiscrete action space)
        # Action shape: (num_drones,) where each element is rule_id in [0, R-1]
        # R=5 interpretable rules for order selection at decision points
        self.action_space = spaces.MultiDiscrete([self.rule_count] * self.num_drones)

    # ------------------ 时间单位统一：minutes <-> steps ------------------

    def _minutes_to_steps(self, minutes: int) -> int:
        minutes_per_step = 60 // self.time_system.steps_per_hour
        return int(math.ceil(float(minutes) / max(float(minutes_per_step), 1.0)))

    def _get_promised_delivery_steps(self, order: dict) -> int:
        """承诺送达时间（step）：prep_steps + 15分钟SLA(转step)"""
        prep_steps = int(order.get('preparation_time', 1))
        sla_steps = self._minutes_to_steps(15)
        return prep_steps + sla_steps

    # ------------------ READY-based deadline helpers ------------------

    def _get_delivery_sla_steps(self, order: dict) -> int:
        """Get delivery SLA in steps. Returns configured delivery_sla_steps."""
        return self.delivery_sla_steps

    def _get_delivery_deadline_step(self, order: dict) -> int:
        """
        Get READY-based delivery deadline step for an order.
        Uses ready_step as start time, falls back to creation_time if not available.
        If neither is available (unlikely), uses current_step as last resort.
        """
        ready_step = order.get('ready_step')
        if ready_step is None:
            # Fallback: use creation_time if ready_step not set yet
            # (e.g., for orders not yet READY or old orders before this feature)
            ready_step = order.get('creation_time')
            if ready_step is None:
                # Last resort: use current_step (should never happen in practice)
                ready_step = self.time_system.current_step

        delivery_sla = self._get_delivery_sla_steps(order)
        deadline_step = ready_step + int(round(delivery_sla * self.timeout_factor))
        return deadline_step

    # ------------------ Top-K merchants 观测选择 ------------------

    def _select_topk_merchants_for_observation(self) -> List[str]:
        """
        选择用于观测的商家ID列表（长度=obs_num_merchants）
        默认策略：按队列长度降序 Top-K（稳定、可解释、适合高负载）。
        """
        merchant_items = list(self.merchants.items())
        merchant_items.sort(key=lambda kv: len(kv[1].get('queue', [])), reverse=True)
        top_ids = [mid for mid, _ in merchant_items[:self.obs_num_merchants]]

        if len(top_ids) < self.obs_num_merchants:
            top_set = set(top_ids)
            rest = [mid for mid in sorted(self.merchants.keys(), key=lambda x: str(x)) if mid not in top_set]
            top_ids.extend(rest[:(self.obs_num_merchants - len(top_ids))])

        return top_ids

    # ------------------ U7: Candidate order selection for task selection ------------------

    def _build_candidate_list_for_drone(self, drone_id: int) -> List[Tuple[int, bool]]:
        """
        Build candidate list for a drone for PPO task selection.
        Returns list of (order_id, is_valid) tuples of length num_candidates (K=20).

        Priority:
        1. Orders in drone cargo (PICKED_UP) assigned to this drone
        2. Orders ASSIGNED to this drone but not yet picked up
        3. READY orders not assigned to any drone (available for selection)
        4. Padding with (-1, False) for invalid slots

        Returns:
            List of (order_id, is_valid) tuples, always length K
        """
        drone = self.drones[drone_id]
        candidates = []

        # 1. Orders already in cargo (PICKED_UP)
        cargo_orders = list(drone.get('cargo', set()))
        for oid in cargo_orders:
            if oid in self.orders:
                order = self.orders[oid]
                if order['status'] == OrderStatus.PICKED_UP:
                    candidates.append((oid, True))

        # 2. Orders assigned to this drone but not picked up
        assigned_orders = []
        for oid in self.active_orders:
            order = self.orders.get(oid)
            if order and order.get('assigned_drone') == drone_id:
                if order['status'] == OrderStatus.ASSIGNED and oid not in cargo_orders:
                    assigned_orders.append(oid)

        for oid in assigned_orders:
            if len(candidates) >= self.num_candidates:
                break
            candidates.append((oid, True))

        # 3. READY orders not yet assigned to any drone (available for PPO selection)
        # This is the KEY FIX - PPO can now select from pool of READY orders
        ready_orders = []
        for oid in self.active_orders:
            if len(candidates) >= self.num_candidates:
                break
            order = self.orders.get(oid)
            if order and order['status'] == OrderStatus.READY:
                # Check not already assigned
                if order.get('assigned_drone', -1) in (-1, None):
                    ready_orders.append(oid)

        # Sort READY orders by urgency and age for better prioritization
        def ready_order_priority(oid):
            order = self.orders[oid]
            urgent_score = 1000 if order.get('urgent', False) else 0
            age = self.time_system.current_step - order['creation_time']
            return -(urgent_score + age)  # Negative for descending sort

        ready_orders.sort(key=ready_order_priority)

        for oid in ready_orders:
            if len(candidates) >= self.num_candidates:
                break
            candidates.append((oid, True))

        # 4. Pad with invalid entries
        while len(candidates) < self.num_candidates:
            candidates.append((-1, False))

        # Ensure exactly K candidates
        return candidates[:self.num_candidates]

    def _encode_candidate(self, order_id: int, is_valid: bool) -> np.ndarray:
        """
        Encode a candidate order into features (12-dimensional).
        Features:
        0: validity (1.0 if valid, 0.0 if padding)
        1-5: order status one-hot (5 relevant statuses)
        6: order type (0-1 normalized)
        7: age (normalized by 50 steps)
        8: urgency (1.0 if urgent, 0.0 otherwise)
        9: deadline slack (normalized, 1.0=no urgency, 0.0=overdue)
        10: merchant location x (normalized)
        11: customer location y (normalized)
        """
        encoding = np.zeros(12, dtype=np.float32)

        if not is_valid or order_id < 0:
            return encoding  # All zeros for invalid

        if order_id not in self.orders:
            return encoding

        order = self.orders[order_id]

        # Feature 0: validity
        encoding[0] = 1.0

        # Features 1-5: status one-hot (ASSIGNED=4, PICKED_UP=5 are most relevant)
        status_val = order['status'].value
        if status_val < 5:
            encoding[1 + status_val] = 1.0

        # Feature 6: order type
        encoding[6] = order['order_type'].value / 2.0

        # Feature 7: age
        age = self.time_system.current_step - order['creation_time']
        encoding[7] = min(age / 50.0, 1.0)

        # Feature 8: urgency
        encoding[8] = 1.0 if order.get('urgent', False) else 0.0

        # Feature 9: deadline slack
        deadline_step = self._get_delivery_deadline_step(order)
        current_step = self.time_system.current_step
        slack = deadline_step - current_step
        # Normalize: positive slack = good (1.0), negative = overdue (0.0)
        encoding[9] = np.clip(slack / 50.0 + 0.5, 0.0, 1.0)

        # Features 10-11: merchant location (for pickup)
        merchant_id = order.get('merchant_id')
        if merchant_id and merchant_id in self.merchants:
            merchant = self.merchants[merchant_id]
            mloc = merchant['location']
            encoding[10] = mloc[0] / self.grid_size
            encoding[11] = mloc[1] / self.grid_size

        return encoding

    def _update_candidate_mappings(self):
        """Update candidate mappings for all drones.

        Optimized: single pass over active_orders collects assigned/ready sets,
        then builds per-drone candidate lists. Replaces the previous O(D*N) approach
        of calling _build_candidate_list_for_drone separately for each drone.
        """
        K = self.num_candidates
        current_step = self.time_system.current_step
        num_drones = self.num_drones

        # Single pass over active_orders: gather assigned-per-drone and ready pool
        assigned_by_drone: Dict[int, List[int]] = {d: [] for d in range(num_drones)}
        ready_pool: List[int] = []

        for oid in self.active_orders:
            order = self.orders.get(oid)
            if order is None:
                continue
            status = order['status']
            if status == OrderStatus.ASSIGNED:
                d = order.get('assigned_drone', -1)
                # Use -1 as the unassigned sentinel throughout
                if d is not None and d >= 0 and d < num_drones:
                    assigned_by_drone[d].append(oid)
            elif status == OrderStatus.READY:
                if order.get('assigned_drone', -1) in (-1, None):
                    ready_pool.append(oid)

        # Sort READY orders once (shared across all drones)
        def _ready_priority(oid):
            o = self.orders[oid]
            urgent_score = 1000 if o.get('urgent', False) else 0
            age = current_step - o['creation_time']
            return -(urgent_score + age)

        ready_pool.sort(key=_ready_priority)

        # Build per-drone candidate lists
        for drone_id in range(num_drones):
            drone = self.drones[drone_id]
            candidates: List[Tuple[int, bool]] = []
            seen_ids: set = set()

            # 1. Orders already in cargo (PICKED_UP)
            for oid in drone.get('cargo', set()):
                order = self.orders.get(oid)
                if order and order['status'] == OrderStatus.PICKED_UP:
                    candidates.append((oid, True))
                    seen_ids.add(oid)

            # 2. Orders ASSIGNED to this drone (not yet picked up)
            for oid in assigned_by_drone[drone_id]:
                if len(candidates) >= K:
                    break
                if oid not in seen_ids:
                    candidates.append((oid, True))
                    seen_ids.add(oid)

            # 3. READY orders available for PPO selection
            for oid in ready_pool:
                if len(candidates) >= K:
                    break
                if oid not in seen_ids:
                    candidates.append((oid, True))
                    seen_ids.add(oid)

            # 4. Pad with invalid entries
            while len(candidates) < K:
                candidates.append((-1, False))

            self.drone_candidate_mappings[drone_id] = candidates[:K]

    # ------------------ reset / step ------------------

    def reset(self, seed=None, options=None):
        """重置环境 - 开始新的一天"""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            # Derive order_rng from the same seed but an independent stream so that
            # policy/MOPSO code (which uses np_random) cannot shift order generation.
            self.order_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        else:
            # No explicit seed: advance order_rng from the current np_random state.
            # _ORDER_RNG_SEED_BOUND is 2^31 (max value for a 32-bit signed-integer seed).
            _ORDER_RNG_SEED_BOUND = 2 ** 31
            self.order_rng = np.random.default_rng(int(self.np_random.integers(0, _ORDER_RNG_SEED_BOUND)))

        # 将环境 RNG 传播给依赖子对象
        # location_loader 使用 np_random（供 _handle_random_events 等模拟函数）
        self.location_loader.rng = self.np_random
        # order_processor 使用专用 order_rng，与策略随机流隔离
        self.order_processor.rng = self.order_rng

        self.time_system.reset()
        self.daily_stats['day_number'] = self.time_system.day_number

        # 重置订单系统
        self.orders = {}
        self.active_orders = set()
        self.completed_orders = set()
        self.cancelled_orders = set()
        self.global_order_counter = 0

        # Reset incremental caches that depend on order state
        self._ready_orders_cache = set()
        self._candidate_mappings_dirty = False
        self._filtered_candidates_sets = {}
        # Reset running order-history stats
        self._order_hist_merchant_ids = set()
        self._order_hist_dist_sum = 0.0
        self._order_hist_dist_count = 0
        # Reset per-step profiling accumulators
        self._perf_accum = {}
        self._perf_steps = 0

        self._reset_drones_and_bases()
        self._reset_daily_stats()

        # Reset debug tracking for on_time_deliveries
        self._prev_on_time_deliveries = 0
        self._on_time_decrease_warned = False

        self.last_stats = {
            'completed': 0,
            'energy': 0,
            'on_time': 0,
            'cancelled': 0,
            'distance': 0.0,
        }

        self.path_visualizer.clear_paths()

        self._update_weather_from_dataset()
        self._generate_morning_orders()

        # 重置营业后履约状态
        self.in_overtime = False
        self.overtime_steps = 0
        self._assigned_in_step = set()
        self._end_of_day_printed = False
        # ====== 多目标权重：每个 episode 一个偏好 ======
        if self.multi_objective_mode == "conditioned":
            w = self.np_random.dirichlet(alpha=np.ones(self.num_objectives)).astype(np.float32)
            self.objective_weights = w
        else:
            self.objective_weights = self.fixed_objective_weights.copy()

        # 初始化 shaping 距离缓存
        for d in range(self.num_drones):
            self._prev_target_dist[d] = self._get_dist_to_target(d)
            # Reset target location cache to current target (or None if no target)
            drone = self.drones[d]
            self._prev_target_loc[d] = drone.get('target_location', None)
        self.episode_r_vec[:] = 0.0

        # U7: Initialize candidate mappings
        self._update_candidate_mappings()

        # U9: Initialize filtered candidates from external generator
        self.update_filtered_candidates()

        # Reset legacy fallback counter and reasons
        self.legacy_blocked_count = 0
        self.legacy_blocked_reasons.clear()
        self.action_applied_count = 0

        # Initialize decision point cache for diagnostics
        self._last_decision_points_mask = [False] * self.num_drones
        self._last_decision_points_count = 0

        obs = self._get_observation()
        info = self._get_info()

        # Store last observation for decentralized executor
        self.last_obs = obs

        print(f"=== 开始 ===")
        print(f"营业时间: {self.time_system.start_hour}:00 - {self.time_system.end_hour}:00")
        print(f"今日天气: {self.weather_details.get('summary', 'Unknown')}")
        print(f"无人机数量: {self.num_drones}, 订单观测窗口: {self.max_obs_orders}, 商家TopK: {self.obs_num_merchants}")

        return obs, info

    def step(self, action):
        """执行一步环境动作"""
        self._assigned_in_step = set()
        self._last_route_heading = action

        _profile = self.enable_diagnostics
        _t0 = time.perf_counter() if _profile else 0.0

        day_ended = self.time_system.step()
        time_state = self.time_system.get_time_state()

        # 每小时更新一次天气：使用 step 索引
        if time_state['minute'] == 0:
            self._update_weather_from_dataset()

        # U7: Update candidate mappings before processing action
        _t1 = time.perf_counter() if _profile else 0.0
        self._update_candidate_mappings()

        # U9: Update filtered candidates based on interval
        _t2 = time.perf_counter() if _profile else 0.0
        if self.candidate_update_interval > 0:
            if self.time_system.current_step % self.candidate_update_interval == 0:
                self.update_filtered_candidates()

        # Cache decision points BEFORE processing action (for consistent diagnostics)
        self._last_decision_points_mask = [
            self._is_at_decision_point(drone_id) for drone_id in range(self.num_drones)
        ]
        self._last_decision_points_count = sum(self._last_decision_points_mask)

        # 处理动作（heading 不直接给奖励，奖励来自系统指标+shaping）
        r_vec = self._process_action(action)

        # dense shaping (no longer uses heading since we removed it)
        shaping_vec = self._calculate_shaping_reward(action)
        r_vec = r_vec + shaping_vec
        self.episode_r_vec = self.episode_r_vec + r_vec.astype(np.float32)

        # Update total reward components for diagnostics
        self.last_step_reward_components['obj0_total'] = float(r_vec[0])
        self.last_step_reward_components['obj1_total'] = float(r_vec[1])
        self.last_step_reward_components['obj2_total'] = float(r_vec[2])

        # 动态事件 (includes position update and merchant preparation)
        _t3 = time.perf_counter() if _profile else 0.0
        self._process_events()

        # 随机 sigmoid hazard 取消
        self._apply_sigmoid_hazard_cancellations()

        # 清理过期分配
        self._cleanup_stale_assignments()

        # 强制状态同步
        self._force_state_synchronization()

        # 更新系统状态（扩展点）
        self._update_system_state()

        # Debug guard: Check if on_time_deliveries ever decreases
        if self.debug_state_warnings:
            current_on_time = self.daily_stats.get('on_time_deliveries', 0)
            if current_on_time < self._prev_on_time_deliveries and not self._on_time_decrease_warned:
                print(
                    f"[WARNING] on_time_deliveries decreased from {self._prev_on_time_deliveries} to {current_on_time} at step {self.time_system.current_step}")
                self._on_time_decrease_warned = True
            self._prev_on_time_deliveries = current_on_time

        # 生成新订单（高负载）
        self._generate_new_orders()

        # 监控无人机状态（可扩展）
        if self.time_system.current_step % 4 == 0:
            self._monitor_drone_status()

        # 更新帕累托前沿
        self.pareto_optimizer.update_pareto_front(r_vec)

        # Collect profiling stats
        if _profile:
            _t4 = time.perf_counter()
            _pa = self._perf_accum
            _pa['candidate_update'] = _pa.get('candidate_update', 0.0) + (_t2 - _t1)
            _pa['event_processing'] = _pa.get('event_processing', 0.0) + (_t4 - _t3)
            self._perf_steps += 1

        # Print diagnostics if enabled
        if self.enable_diagnostics and self.time_system.current_step % self.diagnostics_interval == 0:
            self._print_diagnostics()

        # ---- 原逻辑：termination + final bonus ----
        terminated = self._check_termination(day_ended)
        truncated = False

        if terminated:

            # Ensure end-of-day summary is printed exactly once per episode
            if not getattr(self, "_end_of_day_printed", False):
                self._handle_end_of_day()
                self._end_of_day_printed = True

            final_bonus = self._calculate_daily_final_bonus()
            r_vec = r_vec + final_bonus
            self.in_overtime = False
            self.overtime_steps = 0

        # ---- 新增：累计 episode 向量回报（用最终 r_vec）----
        self.episode_r_vec = self.episode_r_vec + r_vec.astype(np.float32)

        # ---- 标量 reward 输出模式（路线1推荐用 "zero"）----
        if self.reward_output_mode == "scalar":
            scalar_reward = float(np.dot(self.objective_weights, r_vec))
        elif self.reward_output_mode == "obj0":
            scalar_reward = float(r_vec[0])
        elif self.reward_output_mode == "zero":
            scalar_reward = 0.0
        else:
            raise ValueError(f"Unknown reward_output_mode={self.reward_output_mode}")

        obs = self._get_observation()

        info = self._get_info()
        info['r_vec'] = r_vec.copy()
        info['episode_r_vec'] = self.episode_r_vec.copy()  # <- 新增
        info['objective_weights'] = self.objective_weights.copy()
        info['scalar_reward'] = scalar_reward
        info['UAV_episode'] = {
            'r': scalar_reward,
            'l': self.time_system.current_step,
            'day_number': self.time_system.day_number,
            'daily_stats': self.daily_stats.copy(),
            'r_vec': self.episode_r_vec.copy(),  # <- 新增：整集向量回报
        }

        # Store last observation for decentralized executor
        self.last_obs = obs

        return obs, scalar_reward, terminated, truncated, info

    # ------------------ reset helpers ------------------

    def _reset_drones_and_bases(self):
        """重置无人机和基站状态"""
        for base_id, base in self.bases.items():
            base['drones_available'] = []
            base['charging_queue'] = deque()

        for i in range(self.num_drones):
            base_id = i % self.num_bases
            base_location = self.bases[base_id]['location']

            self.drones[i] = {
                'location': base_location,
                'base': base_id,
                'status': DroneStatus.IDLE,
                'current_order': None,
                'orders_completed': 0,
                'speed': self.np_random.uniform(3.0, 5.0),
                'reliability': self.np_random.uniform(0.95, 0.99),
                'max_capacity': self.drone_max_capacity,
                'current_load': 0,
                'battery_level': 100.0,
                'max_battery': 100.0,
                'battery_consumption_rate': self.np_random.uniform(0.2, 0.4),
                'charging_rate': self.np_random.uniform(10.0, 15.0),
                'cancellation_rate': self.np_random.uniform(0.005, 0.015),
                'total_distance_today': 0.0,
            }
            self.drones[i]['planned_stops'] = deque()
            self.drones[i]['cargo'] = set()
            self.drones[i]['current_stop'] = None
            self.drones[i]['route_committed'] = False
            self.drones[i]['serving_order_id'] = None  # U7 task-selection state
            keys_to_remove = ['task_start_location', 'task_start_step', 'accumulated_distance',
                              'optimal_distance', 'last_location', 'target_location',
                              'batch_orders', 'current_batch_index', 'current_delivery_index',
                              'waiting_start_time', 'route_preferences',
                              'current_task_distance', 'task_optimal_distance', 'trip_started',
                              'trip_actual_distance', 'trip_optimal_distance']
            for key in keys_to_remove:
                self.drones[i].pop(key, None)

            self.bases[base_id]['drones_available'].append(i)

    def _reset_daily_stats(self):
        """重置每日统计"""
        self.daily_stats.update({
            'orders_generated': 0,
            'orders_completed': 0,
            'orders_cancelled': 0,
            'revenue': 0,
            'costs': 0,
            'energy_consumed': 0,
            'on_time_deliveries': 0,
            'total_flight_distance': 0.0,
            'optimal_flight_distance': 0.0,
            'total_waiting_time': 0,
        })

    def _generate_morning_orders(self):
        """生成早晨初始订单"""
        morning_prob = 0.5
        for _ in range(5):
            if self.np_random.random() < morning_prob:
                self._generate_single_order()

    # ------------------ trip distance helpers ------------------

    # ------------------ 强制状态同步/修复 ------------------

    def _reset_order_to_ready(self, order_id, reason=""):
        """将订单重置为READY状态，允许重新分配（走 StateManager）"""
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        old_drone = order.get('assigned_drone', -1)

        self.state_manager.update_order_status(order_id, OrderStatus.READY, reason=f"reset_to_ready:{reason}")
        order['assigned_drone'] = -1

        if old_drone is not None and old_drone >= 0 and old_drone in self.drones:
            self.drones[old_drone]['current_load'] = max(0, self.drones[old_drone]['current_load'] - 1)
            if 'batch_orders' in self.drones[old_drone]:
                if order_id in self.drones[old_drone]['batch_orders']:
                    self.drones[old_drone]['batch_orders'].remove(order_id)

    def _force_complete_order(self, order_id, drone_id):
        """强制完成订单（用于异常状态恢复）（走 StateManager）"""
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        if order['status'] == OrderStatus.DELIVERED:
            return

        self.state_manager.update_order_status(order_id, OrderStatus.DELIVERED, reason="force_complete")
        order['delivery_time'] = self.time_system.current_step
        order['assigned_drone'] = -1

        if drone_id in self.drones:
            self.drones[drone_id]['orders_completed'] += 1
            self.drones[drone_id]['current_load'] = max(0, self.drones[drone_id]['current_load'] - 1)

        self.metrics['completed_orders'] += 1
        self.daily_stats['orders_completed'] += 1

        # Track waiting time
        waiting_time = order['delivery_time'] - order['creation_time']
        self.metrics['total_waiting_time'] += waiting_time
        self.daily_stats['total_waiting_time'] = (
            self.daily_stats.get('total_waiting_time', 0)) + waiting_time

        # Check if delivery was on-time using helper method
        if self._is_order_on_time(order):
            self.metrics['on_time_deliveries'] += 1
            self.daily_stats['on_time_deliveries'] += 1
        self.active_orders.discard(order_id)
        self.completed_orders.add(order_id)

    def _force_state_synchronization(self):
        """强制状态同步：确保订单与无人机状态一致，包括货物不变性"""
        drone_real_orders = {d_id: set() for d_id in self.drones}

        for drone_id, drone in self.drones.items():
            if 'batch_orders' in drone and drone['batch_orders']:
                for order_id in drone['batch_orders']:
                    if order_id in self.orders:
                        order = self.orders[order_id]
                        if order['status'] in [OrderStatus.ASSIGNED, OrderStatus.PICKED_UP]:
                            drone_real_orders[drone_id].add(order_id)
            # --- ADD: cargo-based "real orders" (picked up but not delivered) ---
            for oid in drone.get('cargo', set()):
                if oid in self.orders:
                    order = self.orders[oid]
                    if order['status'] == OrderStatus.PICKED_UP:
                        drone_real_orders[drone_id].add(oid)

        # ========== Cargo Invariant Validation & Repair (Task B) ==========
        # Invariant 1: If order status is PICKED_UP and assigned_drone==d, order must be in drone['cargo']
        # Invariant 2: If order is in drone['cargo'], order status must be PICKED_UP and assigned_drone==d

        # Check all orders for invariant 1
        for order_id, order in list(self.orders.items()):
            if order['status'] == OrderStatus.PICKED_UP:
                drone_id = order.get('assigned_drone', -1)
                if drone_id >= 0 and drone_id in self.drones:
                    drone = self.drones[drone_id]
                    # Skip cargo repair for inactive drones: an IDLE/CHARGING drone
                    # should not hold cargo.  The main loop below will resolve the
                    # inconsistent PICKED_UP order instead of re-adding it to cargo
                    # (which would then prevent the main loop from fixing it).
                    if drone['status'] in (DroneStatus.IDLE, DroneStatus.CHARGING):
                        continue
                    if order_id not in drone.get('cargo', set()):
                        # Repair: add to cargo
                        if 'cargo' not in drone:
                            drone['cargo'] = set()
                        drone['cargo'].add(order_id)
                        if self.debug_state_warnings:
                            print(f"[Repair] 订单 {order_id} (PICKED_UP) 添加到无人机 {drone_id} 的货物中")

        # Check all drones for invariant 2
        for drone_id, drone in self.drones.items():
            cargo = drone.get('cargo', set())
            invalid_cargo = []
            for order_id in list(cargo):
                if order_id not in self.orders:
                    # Order doesn't exist - remove from cargo
                    invalid_cargo.append(order_id)
                    if self.debug_state_warnings:
                        print(f"[Repair] 订单 {order_id} 不存在，从无人机 {drone_id} 货物中移除")
                    continue

                order = self.orders[order_id]
                if order['status'] not in [OrderStatus.PICKED_UP]:
                    # Order status is not PICKED_UP - remove from cargo
                    invalid_cargo.append(order_id)
                    if self.debug_state_warnings:
                        print(f"[Repair] 订单 {order_id} 状态为 {order['status']}，从无人机 {drone_id} 货物中移除")
                    continue

                if order.get('assigned_drone', -1) != drone_id:
                    # Order not assigned to this drone - remove from cargo
                    invalid_cargo.append(order_id)
                    if self.debug_state_warnings:
                        print(f"[Repair] 订单 {order_id} 未分配给无人机 {drone_id}，从货物中移除")
                    continue

            # Remove invalid cargo items
            for order_id in invalid_cargo:
                cargo.discard(order_id)

        for order_id in list(self.active_orders):
            if order_id not in self.orders:
                self.active_orders.discard(order_id)
                continue

            order = self.orders[order_id]
            if order['status'] not in [OrderStatus.ASSIGNED, OrderStatus.PICKED_UP]:
                continue

            drone_id = order.get('assigned_drone', -1)

            if drone_id is None or drone_id < 0 or drone_id not in self.drones:
                self._reset_order_to_ready(order_id, "invalid_drone_id")
                continue

            drone = self.drones[drone_id]

            if drone['status'] in [DroneStatus.IDLE, DroneStatus.CHARGING]:
                # An inactive drone must not own any ASSIGNED or PICKED_UP orders.
                # Remove the drone_real_orders guard here: even if the order slipped
                # into cargo (e.g. via the Invariant-1 repair before this loop ran),
                # we still want to resolve it unconditionally.
                if order['status'] == OrderStatus.PICKED_UP:
                    self._force_complete_order(order_id, drone_id)
                else:
                    self._reset_order_to_ready(order_id, "drone_idle")
                continue

            if drone['status'] == DroneStatus.RETURNING_TO_BASE:
                if order['status'] == OrderStatus.ASSIGNED:
                    self._reset_order_to_ready(order_id, "drone_returning")
                elif order['status'] == OrderStatus.PICKED_UP:
                    self._force_complete_order(order_id, drone_id)
                continue

            if drone['status'] == DroneStatus.FLYING_TO_CUSTOMER:
                # Route-aware fix: Only auto-pickup if NOT in route-plan mode
                # In route-plan mode, orders are picked up explicitly at P stops
                # Note: route_committed is set to True in apply_route_plan and cleared to False
                # in _safe_reset_drone and when drone returns to base (RETURNING_TO_BASE arrival handler)
                if not drone.get('route_committed', False):
                    # Legacy mode: auto-pickup when flying to customer
                    # Don't auto-pickup in task-selection mode (serving_order_id present)
                    # Only execute if legacy fallback is enabled
                    if drone.get('serving_order_id') is None:
                        if self.enable_legacy_fallback:
                            if order['status'] == OrderStatus.ASSIGNED:
                                self.state_manager.update_order_status(order_id, OrderStatus.PICKED_UP,
                                                                       reason="sync_assigned_to_picked_up")
                                if 'pickup_time' not in order:
                                    order['pickup_time'] = self.time_system.current_step
                        else:
                            # Legacy auto-pickup blocked
                            if order['status'] == OrderStatus.ASSIGNED:
                                self.legacy_blocked_count += 1
                                if self.debug_state_warnings:
                                    print(f"[Legacy Blocked] Auto-pickup for order {order_id} on drone {drone_id} - "
                                          f"total_blocked={self.legacy_blocked_count}")

        for drone_id, drone in self.drones.items():
            if 'batch_orders' in drone:
                valid_batch = []
                for order_id in drone['batch_orders']:
                    if order_id in self.orders:
                        order = self.orders[order_id]
                        if (order['status'] in [OrderStatus.ASSIGNED, OrderStatus.PICKED_UP] and
                                order.get('assigned_drone') == drone_id):
                            valid_batch.append(order_id)

                if valid_batch:
                    drone['batch_orders'] = valid_batch
                else:
                    self._clear_drone_batch_state(drone)

        # Single-pass load recalculation: O(N+D) instead of O(D*N)
        load_counts = {d_id: 0 for d_id in self.drones}
        for order_id in self.active_orders:
            order = self.orders.get(order_id)
            if order is None:
                continue
            if order['status'] in (OrderStatus.ASSIGNED, OrderStatus.PICKED_UP):
                d = order.get('assigned_drone')
                if d is not None and d >= 0 and d in load_counts:
                    load_counts[d] += 1
        for drone_id, drone in self.drones.items():
            drone['current_load'] = load_counts[drone_id]

    def _clear_drone_batch_state(self, drone):
        keys_to_remove = ['batch_orders', 'current_batch_index', 'current_delivery_index', 'waiting_start_time']
        for key in keys_to_remove:
            if key in drone:
                del drone[key]

    # ------------------ overtime termination ------------------

    def _busy_drones_exist(self) -> bool:
        for d in self.drones.values():
            if d['status'] not in [DroneStatus.IDLE, DroneStatus.CHARGING]:
                return True
            if d.get('current_load', 0) > 0:
                return True
        return False

    def _overtime_done(self) -> bool:
        return len(self.active_orders) == 0 and not self._busy_drones_exist()

    def _check_termination(self, day_ended):
        if not self.in_overtime:
            if day_ended:
                if self.allow_overtime_fulfillment and (len(self.active_orders) > 0 or self._busy_drones_exist()):
                    self.in_overtime = True
                    self.overtime_steps = 0
                    return False
                else:
                    return True
            return False
        else:
            self.overtime_steps += 1
            if self._overtime_done():
                return True
            elif self.overtime_steps >= self.overtime_max_steps:
                self._handle_end_of_day()
                self._end_of_day_printed = True
                return True
            return False

    # ------------------ rewards ------------------

    def _is_at_decision_point(self, drone_id: int) -> bool:
        """
        Check if drone is at a decision point where PPO can change target.

        KEY FIX: Allow action whenever drone needs a serving_order, not just at restrictive conditions.
        Decision points:
        1. IDLE - drone has no work
        2. No serving_order_id - drone needs to select work
        3. Just arrived at target (close to merchant/customer) - can select next work
        """
        drone = self.drones[drone_id]
        status = drone['status']

        # Always a decision point when IDLE
        if status == DroneStatus.IDLE:
            return True

        # KEY FIX: Allow action whenever drone doesn't have a valid serving_order_id
        # This ensures drones can get work assigned even while flying
        serving_order_id = drone.get('serving_order_id')
        if serving_order_id is None:
            return True

        # Check if serving_order is still valid
        if serving_order_id not in self.orders:
            return True

        order = self.orders[serving_order_id]
        # If serving_order is cancelled or delivered, need new work
        if order['status'] in [OrderStatus.CANCELLED, OrderStatus.DELIVERED]:
            return True

        # Decision point if we just completed a pickup or delivery (arrived at target)
        if 'target_location' in drone:
            dist_to_target = self._get_dist_to_target(drone_id)

            # If very close to target and in appropriate status
            if dist_to_target < DISTANCE_CLOSE_THRESHOLD:
                if status in [DroneStatus.FLYING_TO_MERCHANT, DroneStatus.WAITING_FOR_PICKUP]:
                    # At merchant - decision point after pickup
                    return True
                elif status in [DroneStatus.FLYING_TO_CUSTOMER, DroneStatus.DELIVERING]:
                    # At customer - decision point after delivery
                    return True

        return False

    # ------------------ U8: Rule-based order selection ------------------

    def _select_order_by_rule(self, drone_id: int, rule_id: int) -> Optional[int]:
        """
        Select an order for a drone based on the specified rule ID.

        Rules (R=5):
        0: CARGO_FIRST - Prioritize delivering picked-up orders
        1: ASSIGNED_EDF - Earliest deadline first from assigned orders
        2: READY_EDF - Earliest deadline first from ready orders
        3: NEAREST_PICKUP - Closest pickup location
        4: SLACK_PER_DISTANCE - Maximize slack/distance ratio

        Returns:
            order_id if a valid order is selected, None otherwise
        """
        if rule_id == 0:
            return self._rule_cargo_first(drone_id)
        elif rule_id == 1:
            return self._rule_assigned_edf(drone_id)
        elif rule_id == 2:
            return self._rule_ready_edf(drone_id)
        elif rule_id == 3:
            return self._rule_nearest_pickup(drone_id)
        elif rule_id == 4:
            return self._rule_slack_per_distance(drone_id)
        else:
            return None

    def _rule_cargo_first(self, drone_id: int) -> Optional[int]:
        """Rule 0: CARGO_FIRST - Prioritize delivering picked-up orders in cargo."""
        drone = self.drones[drone_id]
        cargo = drone.get('cargo', set())

        if not cargo:
            # No cargo - fall back to assigned orders, then ready orders
            result = self._rule_assigned_edf(drone_id)
            if result is not None:
                return result
            return self._rule_ready_edf(drone_id)

        # Select from cargo: prefer earliest deadline
        best_order_id = None
        best_deadline = float('inf')

        for order_id in cargo:
            if order_id not in self.orders:
                continue
            order = self.orders[order_id]
            if order['status'] != OrderStatus.PICKED_UP:
                continue

            deadline = self._get_delivery_deadline_step(order)
            if deadline < best_deadline:
                best_deadline = deadline
                best_order_id = order_id

        return best_order_id

    def _rule_assigned_edf(self, drone_id: int) -> Optional[int]:
        """Rule 1: ASSIGNED_EDF - Earliest deadline first from assigned orders."""
        best_order_id = None
        best_deadline = float('inf')

        # U9: Apply candidate filtering
        candidate_constrained_orders = self._get_candidate_constrained_orders(
            drone_id, list(self.active_orders)
        )

        for order_id in candidate_constrained_orders:
            if order_id not in self.orders:
                continue
            order = self.orders[order_id]

            # Only consider ASSIGNED orders for this drone
            if order['status'] != OrderStatus.ASSIGNED:
                continue
            if order.get('assigned_drone') != drone_id:
                continue

            deadline = self._get_delivery_deadline_step(order)
            if deadline < best_deadline:
                best_deadline = deadline
                best_order_id = order_id

        return best_order_id

    def _rule_ready_edf(self, drone_id: int) -> Optional[int]:
        """Rule 2: READY_EDF - Earliest deadline first from ready unassigned orders."""
        drone = self.drones[drone_id]

        # Check capacity
        if drone['current_load'] >= drone['max_capacity']:
            return None

        best_order_id = None
        best_deadline = float('inf')

        # U9: Apply candidate filtering
        candidate_constrained_orders = self._get_candidate_constrained_orders(
            drone_id, list(self.active_orders)
        )

        for order_id in candidate_constrained_orders:
            if order_id not in self.orders:
                continue
            order = self.orders[order_id]

            # Only consider READY unassigned orders
            if order['status'] != OrderStatus.READY:
                continue
            if order.get('assigned_drone', -1) not in (-1, None):
                continue

            deadline = self._get_delivery_deadline_step(order)
            if deadline < best_deadline:
                best_deadline = deadline
                best_order_id = order_id

        return best_order_id

    def _rule_nearest_pickup(self, drone_id: int) -> Optional[int]:
        """Rule 3: NEAREST_PICKUP - Closest pickup location from available orders."""
        drone = self.drones[drone_id]
        drone_loc = drone['location']

        best_order_id = None
        best_distance = float('inf')

        # U9: Apply candidate filtering
        candidate_constrained_orders = self._get_candidate_constrained_orders(
            drone_id, list(self.active_orders)
        )

        # Consider both ASSIGNED orders and READY orders
        for order_id in candidate_constrained_orders:
            if order_id not in self.orders:
                continue
            order = self.orders[order_id]

            # For ASSIGNED orders: only this drone's assignments
            if order['status'] == OrderStatus.ASSIGNED:
                if order.get('assigned_drone') != drone_id:
                    continue
                merchant_loc = order['merchant_location']
                distance = self._calculate_euclidean_distance(drone_loc, merchant_loc)
                if distance < best_distance:
                    best_distance = distance
                    best_order_id = order_id

            # For READY orders: any unassigned order (if capacity allows)
            elif order['status'] == OrderStatus.READY:
                if order.get('assigned_drone', -1) not in (-1, None):
                    continue
                if drone['current_load'] >= drone['max_capacity']:
                    continue
                merchant_loc = order['merchant_location']
                distance = self._calculate_euclidean_distance(drone_loc, merchant_loc)
                if distance < best_distance:
                    best_distance = distance
                    best_order_id = order_id

        return best_order_id

    def _rule_slack_per_distance(self, drone_id: int) -> Optional[int]:
        """Rule 4: SLACK_PER_DISTANCE - Maximize slack/distance ratio."""
        drone = self.drones[drone_id]
        drone_loc = drone['location']
        current_step = self.time_system.current_step

        best_order_id = None
        best_score = -float('inf')

        # U9: Apply candidate filtering
        candidate_constrained_orders = self._get_candidate_constrained_orders(
            drone_id, list(self.active_orders)
        )

        # Consider both ASSIGNED and READY orders
        for order_id in candidate_constrained_orders:
            if order_id not in self.orders:
                continue
            order = self.orders[order_id]

            # Check if order is valid candidate
            is_valid = False
            if order['status'] == OrderStatus.ASSIGNED and order.get('assigned_drone') == drone_id:
                is_valid = True
            elif order['status'] == OrderStatus.READY and order.get('assigned_drone', -1) in (-1, None):
                if drone['current_load'] < drone['max_capacity']:
                    is_valid = True

            if not is_valid:
                continue

            # Calculate slack and distance
            deadline = self._get_delivery_deadline_step(order)
            slack = deadline - current_step
            merchant_loc = order['merchant_location']
            distance = self._calculate_euclidean_distance(drone_loc, merchant_loc)

            # Score: slack / (distance + epsilon)
            # Higher score = better (more slack per unit distance)
            epsilon = 0.1
            score = slack / (distance + epsilon)

            if score > best_score:
                best_score = score
                best_order_id = order_id

        return best_order_id

    def _process_action(self, action):
        """
        U8: Process rule-based discrete actions.
        Action shape: (num_drones,) where:
          action[d]: rule_id in [0, R-1] for selecting orders at decision points
        Speed is controlled by a fixed multiplier (default 1.0).

        Strict applied counting: Only count action as applied when:
        - Drone was at decision point BEFORE this action (cached)
        - Rule selected a valid order
        - Actual state change occurred (assignment, target update, status change)
        """
        self._last_route_heading = action

        # Reset action applied counter and rule usage stats for diagnostics
        self.action_applied_count = 0
        step_rule_usage = defaultdict(int)

        # Process each drone's action
        for drone_id in range(self.num_drones):
            drone = self.drones[drone_id]

            # Store fixed speed multiplier (used in movement)
            drone['ppo_speed_multiplier'] = FIXED_SPEED_MULTIPLIER

            # STRICT: Only process rule if drone was at decision point BEFORE action
            # Use cached decision points, not real-time check
            if not self._last_decision_points_mask[drone_id]:
                continue

            # Extract rule_id from action
            rule_id = int(action[drone_id])

            # Track rule usage
            step_rule_usage[rule_id] += 1
            self.rule_usage_stats[rule_id] += 1

            # Apply rule to select an order
            order_id = self._select_order_by_rule(drone_id, rule_id)

            # If no order selected by rule, skip (not applied)
            if order_id is None or order_id not in self.orders:
                continue

            order = self.orders[order_id]

            # Track state before changes to detect if action actually applied
            state_changed = False

            # Store previous state
            prev_serving_order_id = drone.get('serving_order_id')
            prev_target_location = drone.get('target_location')
            prev_status = drone['status']
            prev_load = drone['current_load']

            # Handle READY orders - assign them first
            if order['status'] == OrderStatus.READY:
                # Rule selected a READY order -> try to assign it to this drone
                if order.get('assigned_drone', -1) in (-1, None):
                    if drone['current_load'] < drone['max_capacity']:
                        prev_order_status = order['status']

                        # Assign the order using the standard assignment mechanism
                        self._process_single_assignment(drone_id, order_id, allow_busy=True)

                        # Check if assignment actually happened (state changed)
                        # State change indicators for READY->ASSIGNED:
                        # 1. Order status changed to ASSIGNED
                        # 2. Order's assigned_drone is now this drone
                        # 3. Drone's current_load increased
                        new_order_status = order['status']
                        new_assigned_drone = order.get('assigned_drone', -1)
                        new_load = drone['current_load']

                        # State changed if order is now assigned to this drone and either:
                        # - Load increased (order added to drone's capacity)
                        # - Status changed from READY to ASSIGNED
                        if (new_order_status == OrderStatus.ASSIGNED and
                                new_assigned_drone == drone_id and
                                new_load > prev_load):
                            state_changed = True

            # Always set serving_order_id to track which order drone is executing
            # This maintains correct drone state regardless of whether it's counted as "applied"
            drone['serving_order_id'] = order_id

            # Determine target based on order status (now handling ASSIGNED and PICKED_UP)
            if order['status'] == OrderStatus.PICKED_UP:
                # Order in cargo -> deliver to customer
                customer_loc = order.get('customer_location')
                if customer_loc:
                    drone['target_location'] = customer_loc
                    self.state_manager.update_drone_status(
                        drone_id, DroneStatus.FLYING_TO_CUSTOMER, target_location=customer_loc
                    )
                    # Check if this represents a state change (if not already counted)
                    if not state_changed:
                        new_target = drone.get('target_location')
                        new_status = drone['status']
                        if (prev_target_location != new_target or
                                prev_status != new_status or
                                prev_serving_order_id != order_id):
                            state_changed = True

            elif order['status'] == OrderStatus.ASSIGNED:
                # Order assigned but not picked up -> go to merchant
                merchant_id = order.get('merchant_id')
                if merchant_id and merchant_id in self.merchants:
                    merchant_loc = self.merchants[merchant_id]['location']
                    drone['target_location'] = merchant_loc
                    drone['current_merchant_id'] = merchant_id
                    self.state_manager.update_drone_status(
                        drone_id, DroneStatus.FLYING_TO_MERCHANT, target_location=merchant_loc
                    )
                    # Check if this represents a state change (if not already counted)
                    if not state_changed:
                        new_target = drone.get('target_location')
                        new_status = drone['status']
                        if (prev_target_location != new_target or
                                prev_status != new_status or
                                prev_serving_order_id != order_id):
                            state_changed = True

            # STRICT: Only count as applied if actual state change occurred
            if state_changed:
                self.action_applied_count += 1

        # Calculate rewards based on system metrics
        r_vec = self._calculate_three_objective_rewards()
        return r_vec

    def _get_dist_to_target(self, drone_id: int) -> float:
        drone = self.drones[drone_id]
        if 'target_location' not in drone:
            return 0.0
        cx, cy = drone['location']
        tx, ty = drone['target_location']
        return float(math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2))

    def _calculate_shaping_reward(self, action):
        r = np.zeros(self.num_objectives, dtype=np.float32)
        progress_shaping = 0.0

        for d in range(self.num_drones):
            drone = self.drones[d]
            # Skip if drone is not in flying/returning status
            if drone['status'] not in [DroneStatus.FLYING_TO_MERCHANT, DroneStatus.FLYING_TO_CUSTOMER,
                                       DroneStatus.RETURNING_TO_BASE]:
                continue

            # Get current target location
            current_target = drone.get('target_location', None)

            # If no target, reset cache and skip
            if current_target is None:
                self._prev_target_loc[d] = None
                self._prev_target_dist[d] = self._get_dist_to_target(d)
                continue

            # Check if target changed (handles None on first step after reset)
            prev_target = self._prev_target_loc[d]
            target_changed = (prev_target is None or prev_target != current_target)

            # If target changed, update cache and skip progress shaping for this step
            if target_changed:
                self._prev_target_loc[d] = current_target
                self._prev_target_dist[d] = self._get_dist_to_target(d)
                continue

            # Target is the same, calculate normal progress shaping
            new_dist = self._get_dist_to_target(d)
            prev_dist = float(self._prev_target_dist[d])

            progress = (prev_dist - new_dist)
            progress_contribution = self.shaping_progress_k * float(progress)
            r[0] += progress_contribution
            progress_shaping += progress_contribution

            # REMOVED: Speed-based and battery_consumption_rate-based shaping to avoid
            # duplicate cost penalties and unit inconsistency with the actual energy model.
            # Cost penalties are already applied in _calculate_three_objective_rewards
            # using actual delta_distance and delta_energy (step increments).
            #
            # Previous code (removed to fix reward scaling inconsistency):
            # speed = float(drone['speed']) * float(self._get_weather_speed_factor())
            # r[1] -= self.shaping_distance_k * speed
            # battery_consumption = float(drone["battery_consumption_rate"]) * float(self._get_weather_battery_factor())
            # r[1] -= self.shaping_energy_k * battery_consumption

            # Update prev_dist for next step
            self._prev_target_dist[d] = new_dist

        # Track progress shaping for diagnostics
        self.last_step_reward_components['obj0_progress_shaping'] = float(progress_shaping)

        return r

    def _calculate_three_objective_rewards(self):
        """
        三目标语义单一：
        - obj0: 吞吐/效率（完成数、距离效率结算、取消惩罚等）
        - obj1: -成本（距离、能耗等纯负成本）
        - obj2: 服务质量（准时、取消、积压）
        """
        rewards = np.zeros(self.num_objectives, dtype=np.float32)

        current_completed = self.daily_stats['orders_completed']
        current_energy = self.daily_stats['energy_consumed']
        current_on_time = self.daily_stats['on_time_deliveries']
        current_cancelled = self.daily_stats['orders_cancelled']
        current_distance = self.daily_stats['total_flight_distance']

        delta_completed = current_completed - self.last_stats['completed']
        delta_energy = current_energy - self.last_stats['energy']
        delta_on_time = current_on_time - self.last_stats['on_time']
        delta_cancelled = current_cancelled - self.last_stats['cancelled']
        delta_distance = current_distance - self.last_stats['distance']

        self.last_stats['completed'] = current_completed
        self.last_stats['energy'] = current_energy
        self.last_stats['on_time'] = current_on_time
        self.last_stats['cancelled'] = current_cancelled
        self.last_stats['distance'] = current_distance

        # obj0：吞吐/效率
        completed_reward = float(delta_completed) * 2.0
        cancelled_penalty_obj0 = float(delta_cancelled) * 2.0
        rewards[0] += completed_reward
        rewards[0] -= cancelled_penalty_obj0

        # obj1：-成本（纯负成本）
        energy_cost = float(delta_energy) * 0.01
        distance_cost = float(delta_distance) * 0.001
        rewards[1] -= energy_cost
        rewards[1] -= distance_cost

        # obj2：服务质量
        on_time_reward = float(delta_on_time) * 1.5
        cancelled_penalty_obj2 = float(delta_cancelled) * 1.0
        backlog = len(self.active_orders)
        backlog_penalty = float(backlog) * 0.05
        rewards[2] += on_time_reward
        rewards[2] -= cancelled_penalty_obj2
        rewards[2] -= backlog_penalty

        # Track reward components for diagnostics
        self.last_step_reward_components.update({
            'obj0_completed': completed_reward,
            'obj0_cancelled': -cancelled_penalty_obj0,
            'obj1_energy_cost': -energy_cost,
            'obj1_distance_cost': -distance_cost,
            'obj2_on_time': on_time_reward,
            'obj2_cancelled': -cancelled_penalty_obj2,
            'obj2_backlog': -backlog_penalty,
            'delta_energy': float(delta_energy),
            'delta_distance': float(delta_distance),
            'delta_completed': float(delta_completed),
            'delta_cancelled': float(delta_cancelled)
        })

        return rewards

    def _calculate_daily_final_bonus(self):
        bonus = np.zeros(self.num_objectives, dtype=np.float32)

        daily_completion_rate = self.daily_stats['orders_completed'] / max(1, self.daily_stats['orders_generated'])
        bonus[0] += daily_completion_rate * 3.0
        bonus[2] += daily_completion_rate * 2.0

        if self.daily_stats['orders_completed'] > 0:
            daily_on_time_rate = self.daily_stats['on_time_deliveries'] / self.daily_stats['orders_completed']
            bonus[2] += daily_on_time_rate * 1.5

        if self.daily_stats['orders_completed'] > 0:
            energy_per_order = self.daily_stats['energy_consumed'] / self.daily_stats['orders_completed']
            energy_efficiency = max(0, 1 - energy_per_order / 30.0)
            bonus[1] += energy_efficiency * 1.2

        if len(self.active_orders) > 0:
            penalty = len(self.active_orders) * 0.1
            bonus[0] -= penalty
            bonus[2] -= penalty

        return bonus

    # ------------------ MPSO 分配相关：保留接口但不从 action 中学习 ------------------

    def apply_route_plan(self,
                         drone_id: int,
                         planned_stops: List[dict],
                         commit_orders: Optional[List[int]] = None,
                         allow_busy: bool = True) -> bool:
        """
        Apply a cross-merchant interleaved route plan.
        planned_stops:
            [{'type':'P','merchant_id':mid}, {'type':'D','order_id':oid}, ...]
        Constraints (per your requirement):
            - planned_stops must not include orders that are not READY at dispatch time.
            - Pickup is executed only when arriving at that merchant stop.
            - Delivery is executed only when arriving at that delivery stop.

        Commit semantics:
            commit_orders are moved READY -> ASSIGNED and assigned to this drone.
            Orders are NOT marked PICKED_UP at commit time.

        Returns:
            bool: True if route was successfully applied, False otherwise.
        """
        if drone_id not in self.drones:
            return False
        drone = self.drones[drone_id]

        if (not allow_busy) and drone['status'] != DroneStatus.IDLE:
            return False

        # Derive commit_orders if not provided
        if commit_orders is None:
            commit_orders = []
            for st in planned_stops:
                if st.get('type') == 'D' and 'order_id' in st:
                    commit_orders.append(int(st['order_id']))
            # unique, keep order
            commit_orders = list(dict.fromkeys(commit_orders))

        # 1) Commit orders: only READY orders are allowed
        committed = []
        for oid in commit_orders:
            o = self.orders.get(oid)
            if o is None:
                continue

            # Enforced by requirement: only READY can be in planned_stops/commit list
            if o['status'] != OrderStatus.READY:
                continue
            if o.get('assigned_drone', -1) not in (-1, None):
                continue
            if drone['current_load'] >= drone['max_capacity']:
                break

            self.state_manager.update_order_status(oid, OrderStatus.ASSIGNED, reason=f"route_committed_{drone_id}")
            o['assigned_drone'] = drone_id
            drone['current_load'] += 1
            committed.append(oid)

            # Record READY-based assignment slack for diagnostics
            deadline_step = self._get_delivery_deadline_step(o)
            assignment_slack = deadline_step - self.time_system.current_step
            self.metrics['assignment_slack_samples'].append(assignment_slack)

        if not committed:
            return False

        # Task C: Validate that orders in planned_stops are in committed set
        # and filter out D stops for non-committed orders
        committed_set = set(committed)
        filtered_stops = []
        for stop in planned_stops:
            if stop.get('type') == 'D':
                oid = stop.get('order_id')
                if oid is not None and oid not in committed_set:
                    if self.debug_state_warnings:
                        print(f"[Warning] Drone {drone_id} skipping D stop for uncommitted order {oid}")
                    continue  # Skip this D stop
            filtered_stops.append(stop)

        # If no stops remain after filtering, don't install route
        if not filtered_stops:
            if self.debug_state_warnings:
                print(f"[Warning] Drone {drone_id} route empty after filtering - no valid D stops")
            return False

        # 2) Install route plan with filtered stops
        drone['planned_stops'] = deque(filtered_stops)
        drone['cargo'] = set()  # picked-up set starts empty
        drone['current_stop'] = None
        drone['route_committed'] = True

        # Clear legacy batch state to avoid conflicts with route-plan mode
        self._clear_drone_batch_state(drone)

        # 3) Start execution: set target to the first stop
        self._set_next_target_from_plan(drone_id, drone)

        return True

    def append_route_plan(self,
                          drone_id: int,
                          planned_stops: List[dict],
                          commit_orders: Optional[List[int]] = None) -> bool:
        """
        Append additional stops to a drone's existing route plan without interrupting current execution.

        This method allows assigning additional orders to drones that have remaining capacity
        but are currently busy executing a route. Unlike apply_route_plan which replaces
        the entire route, this method safely extends the existing route.

        Args:
            drone_id: ID of the drone to extend
            planned_stops: List of stops to append [{'type':'P','merchant_id':mid}, {'type':'D','order_id':oid}, ...]
            commit_orders: Optional list of order IDs to commit (if None, derived from planned_stops)

        Returns:
            bool: True if route was successfully extended, False otherwise

        Constraints:
            - Only READY orders can be committed
            - Drone must have remaining capacity (current_load < max_capacity)
            - Does not reset existing cargo or planned_stops
            - Does not interrupt current target if drone is already executing a route
        """
        if drone_id not in self.drones:
            return False
        drone = self.drones[drone_id]

        # Check if drone has remaining capacity
        if drone['current_load'] >= drone['max_capacity']:
            return False

        # Derive commit_orders if not provided
        if commit_orders is None:
            commit_orders = []
            for st in planned_stops:
                if st.get('type') == 'D' and 'order_id' in st:
                    commit_orders.append(int(st['order_id']))
            # unique, keep order
            commit_orders = list(dict.fromkeys(commit_orders))

        # 1) Commit orders: only READY orders are allowed
        committed = []
        for oid in commit_orders:
            o = self.orders.get(oid)
            if o is None:
                continue

            # Only READY can be in planned_stops/commit list
            if o['status'] != OrderStatus.READY:
                continue
            if o.get('assigned_drone', -1) not in (-1, None):
                continue
            if drone['current_load'] >= drone['max_capacity']:
                break

            self.state_manager.update_order_status(oid, OrderStatus.ASSIGNED, reason=f"route_append_{drone_id}")
            o['assigned_drone'] = drone_id
            drone['current_load'] += 1
            committed.append(oid)

            # Record READY-based assignment slack for diagnostics
            deadline_step = self._get_delivery_deadline_step(o)
            assignment_slack = deadline_step - self.time_system.current_step
            self.metrics['assignment_slack_samples'].append(assignment_slack)

        if not committed:
            return False

        # Validate and filter stops
        committed_set = set(committed)
        filtered_stops = []
        for stop in planned_stops:
            if stop.get('type') == 'D':
                oid = stop.get('order_id')
                if oid is not None and oid not in committed_set:
                    if self.debug_state_warnings:
                        print(f"[Warning] Drone {drone_id} append skipping D stop for uncommitted order {oid}")
                    continue
            filtered_stops.append(stop)

        if not filtered_stops:
            if self.debug_state_warnings:
                print(f"[Warning] Drone {drone_id} append route empty after filtering - no valid stops")
            return False

        # 2) Append stops to existing route (don't replace!)
        # Track count of stops before appending to determine if we should start execution
        existing_stops_count = 0

        # Initialize planned_stops if it doesn't exist
        if 'planned_stops' not in drone or drone['planned_stops'] is None:
            drone['planned_stops'] = deque()
        else:
            existing_stops_count = len(drone['planned_stops'])

        # Append new stops to the end of existing route
        for stop in filtered_stops:
            drone['planned_stops'].append(stop)

        # Initialize cargo if needed (don't reset!)
        if 'cargo' not in drone or drone['cargo'] is None:
            drone['cargo'] = set()

        # Mark route as committed
        drone['route_committed'] = True

        # 3) Start execution only if this is the first route assignment
        # If drone already has planned_stops and is executing, don't interrupt
        # Only set next target if drone is idle or had no stops before appending
        if drone['status'] == DroneStatus.IDLE or existing_stops_count == 0:
            # Drone was idle or we just added the first stops - start execution
            self._set_next_target_from_plan(drone_id, drone)

        return True

    def _process_batch_assignment(self, drone_id, order_ids):
        """环境内部执行批量订单分配：给 MPSO 调用"""
        if not order_ids:
            return
        drone = self.drones[drone_id]

        can_take = max(0, drone['max_capacity'] - drone['current_load'])
        if can_take <= 0:
            return

        cand = []
        for oid in order_ids:
            o = self.orders.get(oid)
            if o is None or o['status'] != OrderStatus.READY:
                continue
            if o.get('assigned_drone', -1) not in (-1, None):
                continue
            cand.append(oid)
            if len(cand) >= can_take:
                break
        if not cand:
            return

        actually_assigned = []
        for oid in cand:
            before = drone['current_load']
            self._process_single_assignment(drone_id, oid, allow_busy=True)
            if drone['current_load'] > before:
                actually_assigned.append(oid)

        if not actually_assigned:
            return

        # KEY FIX: Set serving_order_id to first order in batch
        drone['serving_order_id'] = actually_assigned[0]

        first_order = self.orders[actually_assigned[0]]
        self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_MERCHANT,
                                               first_order['merchant_location'])

        if 'batch_orders' not in drone:
            drone['batch_orders'] = []
        drone['batch_orders'].extend(actually_assigned)
        if 'current_batch_index' not in drone:
            drone['current_batch_index'] = 0

    def _process_single_assignment(self, drone_id, order_id, allow_busy=False):
        """处理单个订单分配（记录任务起点 + 走 StateManager）"""
        if order_id not in self.orders or drone_id not in self.drones:
            return

        order = self.orders[order_id]
        drone = self.drones[drone_id]

        if order['status'] != OrderStatus.READY:
            return
        if order.get('assigned_drone', -1) not in (-1, None):
            return
        if not allow_busy and drone['status'] != DroneStatus.IDLE:
            return
        if drone['current_load'] >= drone['max_capacity']:
            return

        self.state_manager.update_order_status(order_id, OrderStatus.ASSIGNED, reason=f"assigned_to_drone_{drone_id}")
        order['assigned_drone'] = drone_id
        drone['current_load'] += 1

        # Record READY-based assignment slack for diagnostics
        deadline_step = self._get_delivery_deadline_step(order)
        assignment_slack = deadline_step - self.time_system.current_step
        self.metrics['assignment_slack_samples'].append(assignment_slack)

        target_merchant_loc = order['merchant_location']

        if drone['status'] in [DroneStatus.IDLE, DroneStatus.RETURNING_TO_BASE, DroneStatus.CHARGING]:
            # KEY FIX: Set serving_order_id when assigning to IDLE/RETURNING/CHARGING drone
            drone['serving_order_id'] = order_id

            self._start_new_task(drone_id, drone, target_merchant_loc)
            self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_MERCHANT, target_merchant_loc)

        elif drone['status'] == DroneStatus.FLYING_TO_MERCHANT:
            # Already flying to merchant, just add to load (batch order scenario)
            # Keep existing serving_order_id
            pass

        elif drone['status'] == DroneStatus.FLYING_TO_CUSTOMER:
            # Redirect to new merchant for additional pickup
            # Keep serving_order_id as the order in cargo (being delivered)
            # This new assignment is for after delivery

            self._start_new_task(drone_id, drone, target_merchant_loc)
            self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_MERCHANT, target_merchant_loc)

    # ------------------ 事件处理与移动 ------------------

    def _process_events(self):
        self._update_merchant_preparation()
        self._update_drone_positions()
        if self.enable_random_events:
            self._handle_random_events()

        # Task B: State consistency checking with categorized debug logging
        consistency_issues = self.state_manager.get_state_consistency_check()

        if self.debug_state_warnings:
            # Detailed mode: print all issues with categorization
            if consistency_issues:
                # Categorize issues using helper method
                category_lists = {
                    'Route': [],
                    'TaskSel': [],
                    'Legacy': [],
                    'Other': []
                }

                for issue in consistency_issues:
                    if '[Route]' in issue:
                        category_lists['Route'].append(issue)
                    elif '[TaskSel]' in issue:
                        category_lists['TaskSel'].append(issue)
                    elif '[Legacy]' in issue:
                        category_lists['Legacy'].append(issue)
                    else:
                        category_lists['Other'].append(issue)

                print(f"\n=== 状态一致性警告 (Step {self.time_system.current_step}) ===")
                for category, issues in category_lists.items():
                    if issues:
                        print(f"\n{category} 问题 ({len(issues)}):")
                        for issue in issues:
                            print(f"  - {issue}")
                print("=" * 60)
        else:
            # Periodic summary mode: print categorized count every 64 steps
            if self.time_system.current_step % 64 == 0:
                if consistency_issues:
                    # Categorize and count using helper method
                    counts = self.state_manager.categorize_issues(consistency_issues)

                    # Print summary
                    category_str = ", ".join([f"{k}={v}" for k, v in counts.items() if v > 0])
                    print(
                        f"[Step {self.time_system.current_step}] 状态一致性问题计数: {len(consistency_issues)} ({category_str})")
                else:
                    # No issues - print success message
                    print(f"[Step {self.time_system.current_step}] 状态一致性检查: ✓ 无问题")

    def _update_merchant_preparation(self):
        for merchant_id, merchant in self.merchants.items():
            ready_orders = []
            for order_id in list(merchant['queue']):
                if order_id not in self.orders:
                    continue

                order = self.orders[order_id]
                if order['status'] == OrderStatus.ACCEPTED:
                    minutes_per_step = 60 // self.time_system.steps_per_hour

                    time_elapsed_minutes = (self.time_system.current_step - order['creation_time']) * minutes_per_step
                    preparation_required_minutes = order['preparation_time'] * minutes_per_step

                    merchant_efficiency = merchant.get('efficiency', 1.0)
                    adjusted_preparation_time = preparation_required_minutes / merchant_efficiency

                    if time_elapsed_minutes >= adjusted_preparation_time:
                        self.state_manager.update_order_status(
                            order_id, OrderStatus.READY, reason="preparation_complete"
                        )
                        ready_orders.append(order_id)

            for order_id in ready_orders:
                if order_id in merchant['queue']:
                    merchant['queue'].remove(order_id)

    def _sync_drone_status_with_route(self):
        """
        Synchronize drone status with route plan (Task A).
        Called at the start of each step to ensure drone status matches planned_stops.
        """
        for drone_id, drone in self.drones.items():
            planned_stops = drone.get('planned_stops')
            if not planned_stops or len(planned_stops) == 0:
                # No planned stops - allow IDLE or RETURNING
                continue

            # Get first stop
            stop = planned_stops[0]
            stop_type = stop.get('type')

            if stop_type == 'P':
                # Pickup stop - should be FLYING_TO_MERCHANT
                expected_status = DroneStatus.FLYING_TO_MERCHANT
                loc = self._stop_to_location(stop)
                if loc and drone['status'] != expected_status:
                    self.state_manager.update_drone_status(drone_id, expected_status, target_location=loc)
                    drone['current_merchant_id'] = stop.get('merchant_id')
                # Ensure target_location matches stop location
                if loc and drone.get('target_location') != loc:
                    drone['target_location'] = loc

            elif stop_type == 'D':
                # Delivery stop - should be FLYING_TO_CUSTOMER
                expected_status = DroneStatus.FLYING_TO_CUSTOMER
                loc = self._stop_to_location(stop)
                if loc and drone['status'] != expected_status:
                    self.state_manager.update_drone_status(drone_id, expected_status, target_location=loc)
                    drone['current_order_id'] = stop.get('order_id')
                # Ensure target_location matches stop location
                if loc and drone.get('target_location') != loc:
                    drone['target_location'] = loc

    def _update_drone_positions(self):
        # Sync drone status with route plan at the start of position update
        self._sync_drone_status_with_route()

        for drone_id, drone in self.drones.items():
            self.path_visualizer.update_path_history(drone_id, drone["location"])

            if drone["status"] in [
                DroneStatus.FLYING_TO_MERCHANT,
                DroneStatus.FLYING_TO_CUSTOMER,
                DroneStatus.RETURNING_TO_BASE,
            ]:
                if "target_location" not in drone:
                    self._reset_drone_to_base(drone_id, drone)
                    continue

                cx, cy = drone["location"]
                tx, ty = drone["target_location"]

                to_target_dx = tx - cx
                to_target_dy = ty - cy
                dist_to_target = float(math.sqrt(to_target_dx * to_target_dx + to_target_dy * to_target_dy))

                ARRIVAL_THRESHOLD = 0.5
                if dist_to_target <= ARRIVAL_THRESHOLD:
                    drone["location"] = (tx, ty)
                    self._handle_drone_arrival(drone_id, drone)
                    continue

                speed = float(drone["speed"]) * float(self._get_weather_speed_factor())
                if speed <= 1e-6:
                    continue

                # U8: Use fixed speed multiplier (stored in drone during action processing)
                # Default to 1.0 if not set yet
                speed_multiplier = drone.get('ppo_speed_multiplier', 1.0)

                # Move directly towards target (no heading guidance, since we removed that feature)
                tgt_hx = to_target_dx / max(dist_to_target, 1e-6)
                tgt_hy = to_target_dy / max(dist_to_target, 1e-6)

                # Apply speed multiplier
                step_len = min(speed * speed_multiplier, dist_to_target)
                nx = float(np.clip(cx + tgt_hx * step_len, 0, self.grid_size - 1))
                ny = float(np.clip(cy + tgt_hy * step_len, 0, self.grid_size - 1))

                if "last_location" in drone:
                    step_distance = float(
                        math.sqrt((nx - drone["last_location"][0]) ** 2 + (ny - drone["last_location"][1]) ** 2)
                    )
                else:
                    step_distance = float(math.sqrt((nx - cx) ** 2 + (ny - cy) ** 2))

                drone["last_location"] = (nx, ny)

                drone["total_distance_today"] = float(drone.get("total_distance_today", 0.0)) + step_distance
                self.daily_stats["total_flight_distance"] = float(
                    self.daily_stats.get("total_flight_distance", 0.0)) + step_distance
                self.metrics["total_flight_distance"] = float(
                    self.metrics.get("total_flight_distance", 0.0)) + step_distance

                if drone.get("trip_started", False):
                    drone["trip_actual_distance"] = float(drone.get("trip_actual_distance", 0.0)) + step_distance

                drone["location"] = (nx, ny)

                new_dist = float(math.sqrt((tx - nx) ** 2 + (ty - ny) ** 2))
                if new_dist <= ARRIVAL_THRESHOLD:
                    drone["location"] = (tx, ty)
                    self._handle_drone_arrival(drone_id, drone)

                # Energy consumption model: E_step = e0 * d * (1 + alpha * load / max_capacity) * weather_battery_factor
                # Only consume energy if the drone actually moved (step_distance > 0)
                if step_distance > 0:
                    current_load = float(drone.get("current_load", 0))
                    max_capacity = float(drone.get("max_capacity", self.drone_max_capacity))
                    weather_battery_factor = float(self._get_weather_battery_factor())

                    # Calculate energy consumption based on distance and load
                    # Ensure max_capacity is at least 1.0 to prevent division issues
                    load_factor = 1.0 + self.energy_alpha * (current_load / max(1.0, max_capacity))
                    energy_consumption = self.energy_e0 * step_distance * load_factor * weather_battery_factor

                    # Update battery level (clip to 0)
                    drone["battery_level"] = max(0.0, float(drone["battery_level"]) - energy_consumption)

                    # Accumulate energy consumed
                    self.metrics["energy_consumed"] = float(
                        self.metrics.get("energy_consumed", 0.0)) + energy_consumption
                    self.daily_stats["energy_consumed"] = float(
                        self.daily_stats.get("energy_consumed", 0.0)) + energy_consumption

                # Check for low battery and force return to base
                if drone["battery_level"] <= self.battery_return_threshold:
                    if drone["status"] != DroneStatus.RETURNING_TO_BASE:
                        self._force_return_due_to_low_battery(drone_id, drone)

            elif drone["status"] == DroneStatus.WAITING_FOR_PICKUP:
                self._handle_waiting_pickup(drone_id, drone)

            elif drone["status"] == DroneStatus.DELIVERING:
                self._handle_delivering(drone_id, drone)

            elif drone["status"] == DroneStatus.CHARGING:
                self._handle_charging(drone_id, drone)

    # ------------------ 任务/到达处理（统一结算出口）------------------
    def _stop_to_location(self, stop: dict):
        """Map a planned stop to a concrete target location."""
        stype = stop.get('type', None)
        if stype == 'P':
            mid = stop.get('merchant_id', None)
            if mid is None or mid not in self.merchants:
                return None
            return self.merchants[mid]['location']
        if stype == 'D':
            oid = stop.get('order_id', None)
            if oid is None or oid not in self.orders:
                return None
            return self.orders[oid]['customer_location']
        return None

    def _set_next_target_from_plan(self, drone_id: int, drone: dict) -> None:
        """
        Pop invalid stops until a valid target is found; then set drone target/status.
        If no stop remains, reset/return-to-base.
        Task C: Add validation for stop/order consistency.
        """
        while drone.get('planned_stops') and len(drone['planned_stops']) > 0:
            stop = drone['planned_stops'][0]
            loc = self._stop_to_location(stop)
            if loc is None:
                if self.debug_state_warnings:
                    print(f"[Drone {drone_id}] 无效 stop (location not found): {stop}")
                drone['planned_stops'].popleft()
                continue

            # Task C: Validate D stops reference valid orders
            if stop.get('type') == 'D':
                oid = stop.get('order_id')
                if oid is not None and oid in self.orders:
                    o = self.orders[oid]
                    # Check if order is valid for delivery
                    if o.get('assigned_drone') != drone_id:
                        if self.debug_state_warnings:
                            print(f"[Drone {drone_id}] D stop {oid} 不属于该无人机，跳过")
                        drone['planned_stops'].popleft()
                        continue
                    if o['status'] in [OrderStatus.CANCELLED, OrderStatus.DELIVERED]:
                        if self.debug_state_warnings:
                            print(f"[Drone {drone_id}] D stop {oid} 订单已取消或已送达，跳过")
                        drone['planned_stops'].popleft()
                        continue

            drone['current_stop'] = stop
            self._start_new_task(drone_id, drone, loc)

            if stop.get('type') == 'P':
                self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_MERCHANT, target_location=loc)
            else:
                self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_CUSTOMER, target_location=loc)
            return

        # No stops left: end this route cleanly
        self._safe_reset_drone(drone_id, drone)

    def _execute_pickup_stop(self, drone_id: int, drone: dict, stop: dict) -> None:
        """
        Arrive at merchant mid: pick up only orders assigned to this drone AND belonging to this merchant.
        READY-only planning: orders should already be ASSIGNED here, but we still check status strictly.
        """
        mid = stop.get('merchant_id', None)
        if mid is None:
            return

        for oid in list(self.active_orders):
            o = self.orders.get(oid)
            if o is None:
                continue
            if o.get('assigned_drone', -1) != drone_id:
                continue
            if o.get('merchant_id', None) != mid:
                continue
            if o['status'] != OrderStatus.ASSIGNED:
                continue

            self.state_manager.update_order_status(oid, OrderStatus.PICKED_UP, reason=f"pickup_at_merchant_{mid}")
            o['pickup_time'] = self.time_system.current_step
            drone['cargo'].add(oid)

    def _execute_delivery_stop(self, drone_id: int, drone: dict, stop: dict) -> None:
        """Arrive at customer: deliver exactly the specified order (must be PICKED_UP)."""
        oid = stop.get('order_id', None)
        if oid is None or oid not in self.orders:
            return

        o = self.orders[oid]
        if o.get('assigned_drone', -1) != drone_id:
            return

        # Strict legality for READY-only planning: must have been picked up at its merchant stop
        if o['status'] != OrderStatus.PICKED_UP:
            return

        self._complete_order_delivery(oid, drone_id)

        if oid in drone.get('cargo', set()):
            drone['cargo'].remove(oid)

    def _safe_reset_drone(self, drone_id, drone):
        """唯一出口：清理关联订单 + 返航"""

        for order_id, order in list(self.orders.items()):
            if order.get('assigned_drone') == drone_id:
                if order['status'] == OrderStatus.PICKED_UP:
                    self._force_complete_order(order_id, drone_id)
                elif order['status'] == OrderStatus.ASSIGNED:
                    self._reset_order_to_ready(order_id, "drone_reset")

        self._clear_drone_batch_state(drone)
        # --- ADD: clear planned route data ---
        drone['planned_stops'] = deque()
        drone['cargo'] = set()
        drone['current_stop'] = None
        drone['route_committed'] = False
        drone['serving_order_id'] = None  # Clear task-selection state
        # 返航（返航不计入整趟，因为 trip 字段已清理）
        base_loc = self.bases[drone['base']]['location']
        self.state_manager.update_drone_status(drone_id, DroneStatus.RETURNING_TO_BASE, target_location=base_loc)
        drone['current_load'] = 0

    def _handle_batch_pickup(self, drone_id, drone):
        if 'batch_orders' not in drone or not drone['batch_orders']:
            self._safe_reset_drone(drone_id, drone)
            return

        picked_count = 0
        for order_id in drone['batch_orders']:
            order = self.orders.get(order_id)
            if order and order['status'] == OrderStatus.ASSIGNED:
                self.state_manager.update_order_status(order_id, OrderStatus.PICKED_UP, reason="batch_pickup")
                order['pickup_time'] = self.time_system.current_step
                picked_count += 1

        if picked_count == 0:
            self._safe_reset_drone(drone_id, drone)
            return

        self._start_batch_delivery(drone_id, drone)

    def _start_batch_delivery(self, drone_id, drone):
        if 'batch_orders' not in drone or not drone['batch_orders']:
            self._safe_reset_drone(drone_id, drone)
            return

        drone['current_delivery_index'] = 0

        for i, order_id in enumerate(drone['batch_orders']):
            order = self.orders.get(order_id)
            if order and order['status'] == OrderStatus.PICKED_UP:
                drone['current_delivery_index'] = i

                self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_CUSTOMER,
                                                       target_location=order['customer_location'])
                return

        self._safe_reset_drone(drone_id, drone)

    def _handle_batch_delivery(self, drone_id, drone):
        if 'batch_orders' not in drone or not drone['batch_orders']:
            self._safe_reset_drone(drone_id, drone)
            return

        current_index = drone.get('current_delivery_index', 0)

        if current_index < len(drone['batch_orders']):
            order_id = drone['batch_orders'][current_index]
            order = self.orders.get(order_id)
            if order and order['status'] == OrderStatus.PICKED_UP:
                self._complete_order_delivery(order_id, drone_id)

        current_index += 1

        while current_index < len(drone['batch_orders']):
            order_id = drone['batch_orders'][current_index]
            order = self.orders.get(order_id)

            if order and order['status'] == OrderStatus.PICKED_UP:
                drone['current_delivery_index'] = current_index
                self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_CUSTOMER,
                                                       target_location=order['customer_location'])
                return

            current_index += 1

        self._safe_reset_drone(drone_id, drone)

    def _handle_drone_arrival(self, drone_id, drone):
        """
        Arrival handler:
          - If planned_stops is present: execute interleaved multi-merchant pickup/delivery.
          - Else if serving_order_id is present: task-selection mode logic.
          - Else: fallback to legacy single/batch logic.
        """
        # -------- New route-plan logic (priority) --------
        if drone.get('planned_stops') and len(drone['planned_stops']) > 0:
            stop = drone['planned_stops'][0]
            stype = stop.get('type', None)

            if stype == 'P':
                self._execute_pickup_stop(drone_id, drone, stop)
                drone['planned_stops'].popleft()
                return self._set_next_target_from_plan(drone_id, drone)

            if stype == 'D':
                self._execute_delivery_stop(drone_id, drone, stop)
                drone['planned_stops'].popleft()
                return self._set_next_target_from_plan(drone_id, drone)

            # Unknown stop type -> drop and continue
            drone['planned_stops'].popleft()
            return self._set_next_target_from_plan(drone_id, drone)

        # -------- Task-selection mode logic (U7) --------
        serving_order_id = drone.get('serving_order_id')
        if serving_order_id is not None and serving_order_id in self.orders:
            order = self.orders[serving_order_id]

            if drone['status'] == DroneStatus.FLYING_TO_MERCHANT:
                # Arrived at merchant - perform pickup
                if order['status'] == OrderStatus.ASSIGNED and order.get('assigned_drone') == drone_id:
                    merchant_id = order.get('merchant_id')
                    # Verify we're at the right merchant
                    if merchant_id and merchant_id in self.merchants:
                        merchant_loc = self.merchants[merchant_id]['location']
                        if self._calculate_euclidean_distance(drone['location'], merchant_loc) < ARRIVAL_THRESHOLD:
                            # Perform pickup
                            self.state_manager.update_order_status(
                                serving_order_id, OrderStatus.PICKED_UP,
                                reason=f"task_selection_pickup_at_merchant_{merchant_id}"
                            )
                            order['pickup_time'] = self.time_system.current_step
                            drone['cargo'].add(serving_order_id)

                            # Set target to customer and transition status
                            customer_loc = order.get('customer_location')
                            if customer_loc:
                                self.state_manager.update_drone_status(
                                    drone_id, DroneStatus.FLYING_TO_CUSTOMER, target_location=customer_loc
                                )
                                return

                # If we couldn't perform pickup, release the ASSIGNED order back to READY
                # so it can be re-assigned to another drone.
                if order.get('assigned_drone') == drone_id and order['status'] == OrderStatus.ASSIGNED:
                    self._reset_order_to_ready(serving_order_id, "task_selection_pickup_failed")
                drone['serving_order_id'] = None
                self.state_manager.update_drone_status(drone_id, DroneStatus.IDLE, target_location=None)

            elif drone['status'] == DroneStatus.FLYING_TO_CUSTOMER:
                # Arrived at customer - perform delivery
                if order['status'] == OrderStatus.PICKED_UP and order.get('assigned_drone') == drone_id:
                    customer_loc = order.get('customer_location')
                    # Verify we're at the right customer location
                    if customer_loc and self._calculate_euclidean_distance(drone['location'],
                                                                           customer_loc) < ARRIVAL_THRESHOLD:
                        # Perform delivery
                        self._complete_order_delivery(serving_order_id, drone_id)

                        # Remove from cargo
                        if serving_order_id in drone['cargo']:
                            drone['cargo'].remove(serving_order_id)

                        # Clear serving order and go idle (PPO will decide next action)
                        drone['serving_order_id'] = None
                        self.state_manager.update_drone_status(drone_id, DroneStatus.IDLE, target_location=None)
                        return

                # If we couldn't deliver, release the PICKED_UP order back to READY
                # so it can be re-routed to another drone.
                if order.get('assigned_drone') == drone_id and order['status'] == OrderStatus.PICKED_UP:
                    drone.get('cargo', set()).discard(serving_order_id)
                    self._reset_order_to_ready(serving_order_id, "task_selection_delivery_failed")
                drone['serving_order_id'] = None
                self.state_manager.update_drone_status(drone_id, DroneStatus.IDLE, target_location=None)

            return

        # -------- Legacy logic (batch orders) --------
        # Only execute if legacy fallback is enabled
        if not self.enable_legacy_fallback:
            # Legacy fallback is disabled - track blocked attempt with reason
            self.legacy_blocked_count += 1

            # Determine reason for blocking
            if serving_order_id is not None:
                reason = "has_serving_order_id"
            elif drone.get('planned_stops') and len(drone['planned_stops']) > 0:
                reason = "has_planned_stops"
            else:
                reason = f"status_{drone['status'].name}"

            self.legacy_blocked_reasons[reason] += 1

            if self.debug_state_warnings:
                print(f"[Legacy Blocked] Drone {drone_id} arrival at {drone['status']} - "
                      f"serving_order_id={serving_order_id}, reason={reason}, total_blocked={self.legacy_blocked_count}")
            # Without legacy fallback, drone should remain in current state or go idle
            # If drone has no serving_order_id and no planned route, it should be idle
            if serving_order_id is None and (not drone.get('planned_stops') or len(drone['planned_stops']) == 0):
                if drone['status'] in [DroneStatus.FLYING_TO_MERCHANT, DroneStatus.FLYING_TO_CUSTOMER]:
                    # Reset to idle - PPO/dispatcher will decide next action
                    self._safe_reset_drone(drone_id, drone)
            return

        # Legacy fallback enabled - execute original legacy logic
        if drone['status'] == DroneStatus.FLYING_TO_MERCHANT:
            if 'batch_orders' in drone and drone['batch_orders']:
                self._handle_batch_pickup(drone_id, drone)
            else:
                # Legacy single-order mode - only use if not in task-selection mode
                if serving_order_id is None:
                    assigned_order = self._get_drone_assigned_order(drone_id)
                    if assigned_order and assigned_order['status'] == OrderStatus.ASSIGNED:
                        oid = assigned_order['id']
                        self.state_manager.update_order_status(oid, OrderStatus.PICKED_UP,
                                                               reason="arrived_merchant_pickup")
                        assigned_order['pickup_time'] = self.time_system.current_step

                        self._start_new_task(drone_id, drone, assigned_order['customer_location'])
                        self.state_manager.update_drone_status(
                            drone_id, DroneStatus.FLYING_TO_CUSTOMER,
                            target_location=assigned_order['customer_location']
                        )
                    else:
                        self._safe_reset_drone(drone_id, drone)
                else:
                    self._safe_reset_drone(drone_id, drone)

        elif drone['status'] == DroneStatus.FLYING_TO_CUSTOMER:
            if 'batch_orders' in drone and drone['batch_orders']:
                self._handle_batch_delivery(drone_id, drone)
            else:
                # Legacy single-order mode - only use if not in task-selection mode
                if serving_order_id is None:
                    assigned_order = self._get_drone_assigned_order(drone_id)
                    if assigned_order and assigned_order['status'] == OrderStatus.PICKED_UP:
                        self._complete_order_delivery(assigned_order['id'], drone_id)
                self._safe_reset_drone(drone_id, drone)

        elif drone['status'] == DroneStatus.RETURNING_TO_BASE:
            # Defensive: resolve any PICKED_UP/ASSIGNED orders that are still linked
            # to this drone before we clear its state.  This guards against cases
            # where the send-to-base path did not fully clean up (e.g. task-selection
            # edge paths, direct status transitions, etc.).
            for oid in list(self.active_orders):
                if oid not in self.orders:
                    continue
                ord_ = self.orders[oid]
                if ord_.get('assigned_drone') == drone_id:
                    if ord_['status'] == OrderStatus.PICKED_UP:
                        self._force_complete_order(oid, drone_id)
                    elif ord_['status'] == OrderStatus.ASSIGNED:
                        self._reset_order_to_ready(oid, "returning_base_arrival")

            if drone['battery_level'] < 80:
                self.state_manager.update_drone_status(drone_id, DroneStatus.CHARGING, target_location=None)
            else:
                self.state_manager.update_drone_status(drone_id, DroneStatus.IDLE, target_location=None)

            drone['current_load'] = 0
            self._clear_drone_batch_state(drone)
            self._clear_task_data(drone_id, drone)

            # --- ADD: clear route-plan state ---
            drone['planned_stops'] = deque()
            drone['cargo'] = set()
            drone['current_stop'] = None
            drone['route_committed'] = False
            drone['serving_order_id'] = None  # Clear task-selection state

    # ------------------ 单段任务统计（保持原逻辑）------------------

    def _start_new_task(self, drone_id, drone, target_location):
        drone['task_start_location'] = drone['location']
        drone['task_start_step'] = self.time_system.current_step
        drone['current_task_distance'] = 0.0

        optimal_distance = math.sqrt(
            (target_location[0] - drone['location'][0]) ** 2 +
            (target_location[1] - drone['location'][1]) ** 2
        )
        drone['task_optimal_distance'] = optimal_distance

        self.daily_stats['optimal_flight_distance'] = self.daily_stats.get('optimal_flight_distance',
                                                                           0.0) + optimal_distance
        self.metrics['optimal_flight_distance'] = self.metrics.get('optimal_flight_distance', 0.0) + optimal_distance

        drone['target_location'] = target_location

    def _clear_task_data(self, drone_id, drone):
        keys_to_remove = [
            'task_start_location',
            'task_start_step',
            'current_task_distance',
            'task_optimal_distance',
            'last_location'
        ]
        for key in keys_to_remove:
            drone.pop(key, None)

    # ------------------ 订单完成/取消（统一走 StateManager）------------------

    def _calculate_delivery_lateness(self, order: dict) -> float:
        """
        Calculate delivery lateness for an order.
        Returns: delivery_lateness = delivery_time - (ready_step + sla_steps)
        Positive values indicate late delivery, negative/zero indicate on-time.
        """
        ready_step = order.get('ready_step')
        if ready_step is None:
            ready_step = order['creation_time']

        return order['delivery_time'] - ready_step - self._get_delivery_sla_steps(order)

    def _is_order_on_time(self, order: dict) -> bool:
        """
        Check if an order was delivered on-time.
        Returns True if delivery_lateness <= 0, False otherwise.
        """
        return self._calculate_delivery_lateness(order) <= 0

    def _complete_order_delivery(self, order_id, drone_id):
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        if order['status'] == OrderStatus.DELIVERED:
            return

        self.state_manager.update_order_status(order_id, OrderStatus.DELIVERED, reason=f"delivered_by_drone_{drone_id}")
        order['delivery_time'] = self.time_system.current_step
        order['assigned_drone'] = -1

        if drone_id in self.drones:
            drone = self.drones[drone_id]
            drone['orders_completed'] += 1
            drone['current_load'] = max(0, drone['current_load'] - 1)
            if 'batch_orders' in drone and order_id in drone['batch_orders']:
                drone['batch_orders'].remove(order_id)

        delivery_duration = order['delivery_time'] - order['creation_time']
        self.metrics['total_delivery_time'] += delivery_duration
        self.metrics['total_waiting_time'] += delivery_duration
        self.metrics['completed_orders'] += 1
        self.daily_stats['orders_completed'] += 1
        self.daily_stats['total_waiting_time'] = (
            self.daily_stats.get('total_waiting_time', 0)) + delivery_duration

        # Calculate delivery lateness for diagnostics using helper method
        delivery_lateness = self._calculate_delivery_lateness(order)

        # Record lateness for diagnostics
        self.metrics['ready_based_lateness_samples'].append(delivery_lateness)

        # Check if delivery was on-time using helper method
        if self._is_order_on_time(order):
            self.metrics['on_time_deliveries'] += 1
            self.daily_stats['on_time_deliveries'] += 1

        weather_key = f"{self.weather.name.lower()}_deliveries"
        if weather_key in self.metrics['weather_impact_stats']:
            self.metrics['weather_impact_stats'][weather_key] += 1

        self.active_orders.discard(order_id)
        self.completed_orders.add(order_id)

    def _cancel_order(self, order_id, reason):
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        if order['status'] == OrderStatus.CANCELLED:
            return

        self.state_manager.update_order_status(order_id, OrderStatus.CANCELLED, reason=reason)
        order['cancellation_reason'] = reason
        order['cancellation_time'] = self.time_system.current_step

        self.active_orders.discard(order_id)
        self.cancelled_orders.add(order_id)
        self.metrics['cancelled_orders'] += 1
        self.daily_stats['orders_cancelled'] += 1

        drone_id = order.get('assigned_drone', -1)
        if drone_id is not None and 0 <= drone_id < self.num_drones:
            drone = self.drones[drone_id]
            drone['current_load'] = max(0, drone['current_load'] - 1)
            if drone['current_load'] == 0:
                base_loc = self.bases[drone['base']]['location']
                self.state_manager.update_drone_status(drone_id, DroneStatus.RETURNING_TO_BASE,
                                                       target_location=base_loc)

    # ------------------ drone helper states ------------------

    def _get_drone_assigned_order(self, drone_id):
        for order_id in self.active_orders:
            order = self.orders[order_id]
            if order.get('assigned_drone') == drone_id:
                return order
        return None

    def _reset_drone_to_base(self, drone_id, drone):
        for order_id, order in list(self.orders.items()):
            if order.get('assigned_drone') == drone_id:
                if order['status'] == OrderStatus.ASSIGNED:
                    self._reset_order_to_ready(order_id, "reset_drone_to_base")
                elif order['status'] == OrderStatus.PICKED_UP:
                    self._complete_order_delivery(order_id, drone_id)

        if 'batch_orders' in drone:
            for batch_order_id in drone['batch_orders']:
                if batch_order_id in self.orders:
                    batch_order = self.orders[batch_order_id]
                    if batch_order['status'] == OrderStatus.ASSIGNED:
                        self._reset_order_to_ready(batch_order_id, "reset_drone_to_base_batch")
            del drone['batch_orders']

        drone.pop('current_batch_index', None)
        drone.pop('current_delivery_index', None)

        base_loc = self.bases[drone['base']]['location']
        self.state_manager.update_drone_status(drone_id, DroneStatus.RETURNING_TO_BASE, target_location=base_loc)
        drone['current_load'] = 0

    def _force_return_due_to_low_battery(self, drone_id: int, drone: dict):
        """
        Force drone to return to base due to low battery.
        Releases all undelivered orders (ASSIGNED and PICKED_UP) back to READY state.
        """
        # Track forced return event
        self.daily_stats["forced_return_events"] = int(self.daily_stats.get("forced_return_events", 0)) + 1

        # Release all orders assigned to this drone
        # First collect order IDs to modify (more efficient than list copy of all active orders)
        orders_to_release = [
            order_id for order_id in self.active_orders
            if order_id in self.orders and self.orders[order_id].get('assigned_drone') == drone_id
        ]

        for order_id in orders_to_release:
            order = self.orders[order_id]

            # Release ASSIGNED orders (not yet picked up)
            if order['status'] == OrderStatus.ASSIGNED:
                self.state_manager.update_order_status(
                    order_id, OrderStatus.READY, reason="force_return_low_battery"
                )
                order['assigned_drone'] = None
                # Clear assignment timestamp if present
                if 'assigned_time' in order:
                    del order['assigned_time']

            # Release PICKED_UP orders (in cargo but not delivered)
            # In paper-level simplification, allow "delivery failure recovery"
            elif order['status'] == OrderStatus.PICKED_UP:
                self.state_manager.update_order_status(
                    order_id, OrderStatus.READY, reason="force_return_low_battery_pickup"
                )
                order['assigned_drone'] = None
                # Clear pickup timestamp
                if 'pickup_time' in order:
                    del order['pickup_time']
                # Remove from cargo
                if order_id in drone.get('cargo', set()):
                    drone['cargo'].discard(order_id)

        # Clear drone route/task state
        drone['planned_stops'] = deque()
        drone['cargo'] = set()
        drone['current_stop'] = None
        drone['route_committed'] = False
        drone['serving_order_id'] = None
        drone['current_load'] = 0

        # Clear batch state if exists
        if 'batch_orders' in drone:
            del drone['batch_orders']
        drone.pop('current_batch_index', None)
        drone.pop('current_delivery_index', None)

        # Set drone to return to base
        base_loc = self.bases[drone['base']]['location']
        self.state_manager.update_drone_status(drone_id, DroneStatus.RETURNING_TO_BASE, target_location=base_loc)

    def _handle_waiting_pickup(self, drone_id, drone):
        assigned_order = self._get_drone_assigned_order(drone_id)

        if assigned_order:
            if assigned_order['status'] == OrderStatus.READY:
                self.state_manager.update_order_status(
                    assigned_order['id'], OrderStatus.ASSIGNED, reason="waiting_to_assigned"
                )
                assigned_order['assigned_drone'] = drone_id

            if assigned_order['status'] == OrderStatus.ASSIGNED:
                self.state_manager.update_order_status(
                    assigned_order['id'], OrderStatus.PICKED_UP, reason="pickup_complete"
                )
                assigned_order['pickup_time'] = self.time_system.current_step

                self.state_manager.update_drone_status(
                    drone_id, DroneStatus.FLYING_TO_CUSTOMER, target_location=assigned_order['customer_location']
                )
                return

        if 'waiting_start_time' not in drone:
            drone['waiting_start_time'] = self.time_system.current_step

        waiting_duration = self.time_system.current_step - drone['waiting_start_time']
        if waiting_duration > 10:
            if assigned_order:
                self._cancel_order(assigned_order['id'], "waiting_timeout")
            self._reset_drone_to_base(drone_id, drone)

    def _handle_delivering(self, drone_id, drone):
        assigned_order = self._get_drone_assigned_order(drone_id)
        if assigned_order and assigned_order['status'] == OrderStatus.PICKED_UP:
            self._complete_order_delivery(assigned_order['id'], drone_id)
        self._reset_drone_to_base(drone_id, drone)

    def _handle_charging(self, drone_id, drone):
        if drone['battery_level'] < 95:
            drone['battery_level'] = min(
                drone['max_battery'],
                drone['battery_level'] + drone['charging_rate']
            )
        else:
            self.state_manager.update_drone_status(drone_id, DroneStatus.IDLE, target_location=None)

    # ------------------ random events ------------------

    def _handle_random_events(self):
        cancellation_factor = self._get_weather_cancellation_factor()

        for order_id in list(self.active_orders):
            order = self.orders[order_id]

            if (order['status'] in [OrderStatus.PENDING, OrderStatus.ACCEPTED] and
                    self.np_random.random() < 0.02 * cancellation_factor):
                self._cancel_order(order_id, "user_cancellation")

            elif (order['status'] == OrderStatus.ACCEPTED and
                  self.np_random.random() < self.merchants[order['merchant_id']]['cancellation_rate']):
                self._cancel_order(order_id, "merchant_cancellation")

            elif (order['status'] == OrderStatus.ASSIGNED and
                  self.np_random.random() < self.drones[order['assigned_drone']]['cancellation_rate'] * cancellation_factor):
                self._cancel_order(order_id, "drone_cancellation")

            elif (order['status'] in [OrderStatus.ACCEPTED, OrderStatus.ASSIGNED] and
                  self.np_random.random() < 0.01):
                self._change_order_address(order_id)

    def _get_weather_cancellation_factor(self):
        if self.weather == WeatherType.SUNNY:
            return 1.0
        elif self.weather == WeatherType.RAINY:
            return 1.5
        elif self.weather == WeatherType.WINDY:
            return 1.3
        else:
            return 2.0

    def _change_order_address(self, order_id):
        order = self.orders[order_id]
        new_customer_location = self.location_loader.get_random_user_grid_location()
        order['customer_location'] = new_customer_location

        if order.get('assigned_drone', -1) >= 0 and order['status'] == OrderStatus.ASSIGNED:
            drone_id = order['assigned_drone']
            drone = self.drones[drone_id]
            if drone['status'] == DroneStatus.FLYING_TO_CUSTOMER:
                drone['target_location'] = new_customer_location

    # ------------------  weather ------------------

    def _update_weather_from_dataset(self):
        """用 step 做索引（全局）"""
        try:
            current_weather = self.weather_processor.get_weather_at_time(self.time_system.current_step)
            self.weather = self.weather_processor.map_to_weather_type(current_weather['Summary'])

            self.weather_details = {
                'summary': current_weather.get('Summary', 'Unknown'),
                'temperature': current_weather.get('Temperature (C)', 15),
                'humidity': current_weather.get('Humidity', 0.5),
                'wind_speed': current_weather.get('Wind Speed (km/h)', 10),
                'visibility': current_weather.get('Visibility (km)', 10),
                'pressure': current_weather.get('Pressure (millibars)', 1013),
                'precip_type': current_weather.get('Precip Type', 'none')
            }
        except Exception as e:
            print(f"更新天气数据失败: {e}，使用默认天气")
            self.weather = WeatherType.SUNNY
            self.weather_details = {
                'summary': 'Sunny',
                'temperature': 20,
                'humidity': 0.5,
                'wind_speed': 5,
                'visibility': 15,
                'pressure': 1013,
                'precip_type': 'none'
            }

        self.weather_history.append({
            'time': self.time_system.current_step,
            'weather': self.weather,
            'details': self.weather_details.copy()
        })

    def _get_weather_speed_factor(self):
        if self.weather == WeatherType.SUNNY:
            return 1.0
        elif self.weather == WeatherType.RAINY:
            return 0.7
        elif self.weather == WeatherType.WINDY:
            return 0.6
        else:
            return 0.3

    def _get_weather_battery_factor(self):
        if self.weather == WeatherType.SUNNY:
            return 1.0
        elif self.weather == WeatherType.RAINY:
            return 1.3
        elif self.weather == WeatherType.WINDY:
            return 1.6
        else:
            return 2.0

    # ------------------ order generation ------------------

    def _get_business_end_step(self) -> int:
        """Return the step number at which business hours end for the current day.

        current_step starts at 0 and business hours span steps [0, steps_per_day).
        Therefore business ends at step steps_per_day.
        """
        return self.time_system.steps_per_day

    def _generate_new_orders(self):
        time_state = self.time_system.get_time_state()
        if not time_state['is_business_hours']:
            return

        if self.order_cutoff_steps > 0:
            business_end_step = self._get_business_end_step()
            if self.time_system.current_step >= business_end_step - self.order_cutoff_steps:
                return

        order_prob = self.order_processor.get_order_probability(
            env_time=self.time_system.current_step,
            weather_type=self.weather
        )

        if self.order_rng.random() < order_prob:
            if order_prob > 0.8:
                base_batch = int(self.order_rng.integers(3, 7))
            elif order_prob > 0.5:
                base_batch = int(self.order_rng.integers(2, 5))
            else:
                base_batch = int(self.order_rng.integers(1, 3))

            batch_size = int(base_batch * self.high_load_factor)

            if time_state['is_peak_hour']:
                batch_size += int(self.order_rng.integers(1, 4))

            for _ in range(batch_size):
                self._generate_single_order()

    def _generate_single_order(self):
        try:
            order_id = self.global_order_counter
            self.global_order_counter += 1

            env_time = self.time_system.day_number * 24 + self.time_system.current_hour
            order_details = self.order_processor.generate_order_details(env_time, self.weather)

            if order_details['merchant_id'] not in self.merchant_ids:
                order_details['merchant_id'] = self.merchant_ids[int(self.order_rng.integers(0, len(self.merchant_ids)))]

            self._generate_order_with_details(order_details, order_id)

        except Exception as e:
            print(f"生成订单时出错: {e}")

    def _generate_order_with_details(self, order_details, order_id):
        try:
            merchant_id = order_details['merchant_id']
            if merchant_id not in self.merchants:
                merchant_id = self.merchant_ids[int(self.order_rng.integers(0, len(self.merchant_ids)))]

            merchant_loc = self.merchants[merchant_id]['location']
            max_distance = order_details['max_distance']
            customer_loc = self._generate_customer_location(merchant_loc, max_distance)

            if self.order_rng.random() < 0.3:
                customer_loc = self._generate_distant_location(merchant_loc)

            weather_summary = self.weather_details.get('summary', 'Unknown')
            preparation_time = int(order_details.get('preparation_time', int(self.order_rng.integers(2, 7))))

            order = {
                'id': order_id,
                'order_type': OrderType(order_details['order_type']),
                'merchant_id': merchant_id,
                'merchant_location': merchant_loc,
                'customer_location': customer_loc,
                'status': OrderStatus.PENDING,
                'creation_time': self.time_system.current_step,
                'creation_step': self.time_system.current_step,  # explicit step-coordinate field for SC/GC stats
                'assigned_drone': -1,
                'preparation_time': preparation_time,  # step
                'urgent': self.order_rng.random() < order_details['urgency'],
                'weather_conditions': weather_summary
            }

            self.orders[order_id] = order
            self.active_orders.add(order_id)
            self.metrics['total_orders'] += 1
            self.daily_stats['orders_generated'] += 1

            self.merchants[merchant_id]['queue'].append(order_id)

            # PENDING -> ACCEPTED
            self.state_manager.update_order_status(order_id, OrderStatus.ACCEPTED, reason="order_created_and_accepted")

            dist = self._calculate_euclidean_distance(merchant_loc, customer_loc)
            self.order_history.append({
                'time': self.time_system.current_step,
                'weather': weather_summary,
                'order_type': order_details['order_type'],
                'distance': dist,
                'merchant_id': merchant_id
            })
            # Update O(1) running stats (avoids per-step O(N) scan in _get_info)
            self._order_hist_merchant_ids.add(merchant_id)
            self._order_hist_dist_sum += dist
            self._order_hist_dist_count += 1

        except Exception as e:
            print(f"生成订单详细时出错: {e}")

    def _generate_distant_location(self, merchant_loc):
        merchant_x, merchant_y = merchant_loc
        edges = ['top', 'bottom', 'left', 'right']
        edge = edges[int(self.order_rng.integers(0, len(edges)))]
        if edge == 'top':
            return self.order_rng.uniform(0, self.grid_size - 1), self.grid_size - 1
        elif edge == 'bottom':
            return self.order_rng.uniform(0, self.grid_size - 1), 0
        elif edge == 'left':
            return 0, self.order_rng.uniform(0, self.grid_size - 1)
        else:
            return self.grid_size - 1, self.order_rng.uniform(0, self.grid_size - 1)

    def _generate_customer_location(self, merchant_loc, max_distance):
        loc_loader = self.location_loader
        for _ in range(10):
            # Use order_rng directly to stay within the isolated order-generation stream
            if loc_loader.user_locations:
                idx = int(self.order_rng.integers(0, len(loc_loader.user_locations)))
                user = loc_loader.user_locations[idx]
                customer_loc = loc_loader.convert_to_grid_coordinates(
                    user['longitude'], user['latitude']
                )
            else:
                customer_loc = (
                    self.order_rng.uniform(0, self.grid_size - 1),
                    self.order_rng.uniform(0, self.grid_size - 1)
                )
            distance = self._calculate_euclidean_distance(merchant_loc, customer_loc)
            if distance <= max_distance:
                return customer_loc

        merchant_x, merchant_y = merchant_loc
        return (
            max(0, min(self.grid_size - 1, merchant_x + int(self.order_rng.integers(-2, 3)))),
            max(0, min(self.grid_size - 1, merchant_y + int(self.order_rng.integers(-2, 3))))
        )

    def _calculate_euclidean_distance(self, loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    # ------------------ stale assignment cleanup ------------------

    def _get_order_cancel_probability(self, order: dict, current_step: int) -> float:
        """Compute per-step cancellation probability using a sigmoid hazard model.

        The hazard function is:
            h(w) = p_max * sigmoid(k * (w - w0))

        where:
            w   = elapsed waiting time in steps since the order became READY
                  (falls back to creation_time if ready_step is unavailable)
            w0  = hazard_midpoint_steps  (midpoint where risk reaches 50% of p_max)
            k   = hazard_k               (slope of the sigmoid)
            p_max = hazard_p_max         (maximum per-step cancellation probability)

        Args:
            order: Order dict from self.orders.
            current_step: Current simulation step.

        Returns:
            Float in [0, p_max]: probability of cancellation this step.
        """
        start_step = order.get('ready_step')
        if start_step is None:
            start_step = order.get('creation_time', current_step)
        waiting_time = max(0, current_step - start_step)
        raw_exponent = self.hazard_k * (waiting_time - self.hazard_midpoint_steps)
        # sigmoid = 1 / (1 + exp(-exponent)), clipped for numerical safety
        clipped_exponent = float(np.clip(raw_exponent, -50.0, 50.0))
        sigmoid_val = 1.0 / (1.0 + math.exp(-clipped_exponent))
        return self.hazard_p_max * sigmoid_val

    def _apply_sigmoid_hazard_cancellations(self):
        """Stochastically cancel active orders based on sigmoid hazard probabilities.

        For each active order in READY or ASSIGNED status (not PICKED_UP), sample
        a cancellation event this step with probability given by
        _get_order_cancel_probability().  PICKED_UP orders are not cancelled here
        because the drone is already en route to the customer.
        """
        if not self.enable_sigmoid_hazard_cancellation:
            return

        current_step = self.time_system.current_step
        for order_id in list(self.active_orders):
            if order_id not in self.orders:
                continue
            order = self.orders[order_id]
            if order['status'] not in (OrderStatus.READY, OrderStatus.ASSIGNED):
                continue
            prob = self._get_order_cancel_probability(order, current_step)
            if self.np_random.random() < prob:
                self._cancel_order(order_id, "stochastic_timeout")

    def _cleanup_stale_assignments(self):
        current_step = self.time_system.current_step
        stale_threshold = 50

        # Iterate active_orders directly (avoids scanning completed/cancelled orders)
        for order_id in list(self.active_orders):
            order = self.orders.get(order_id)
            if order is None:
                continue

            # Apply sigmoid hazard cancellations (replaces hard deadline for READY/ASSIGNED).
            # PICKED_UP orders are not hard-cancelled here either; the hazard method above
            # already skips them.  If sigmoid hazard is disabled, fall back to the old
            # deadline-based cancellation for READY and ASSIGNED only.
            if not self.enable_sigmoid_hazard_cancellation:
                if order['status'] in [OrderStatus.READY, OrderStatus.ASSIGNED]:
                    deadline_step = self._get_delivery_deadline_step(order)
                    if current_step > deadline_step:
                        self._cancel_order(order_id, "ready_based_timeout")
                        continue

            if order['status'] == OrderStatus.ASSIGNED:
                drone_id = order.get('assigned_drone', -1)

                if drone_id < 0 or drone_id not in self.drones:
                    self._reset_order_to_ready(order_id, "stale_invalid_drone")
                    continue

                drone = self.drones[drone_id]

                if drone['status'] in [DroneStatus.IDLE, DroneStatus.CHARGING]:
                    age = current_step - order['creation_time']
                    if age > stale_threshold:
                        self._reset_order_to_ready(order_id, "stale_age")

    # ------------------ end-of-day ------------------

    def _handle_end_of_day(self):
        print(f"\n=== 结束 ===")
        print(f"今日统计:")
        print(f"  生成订单: {self.daily_stats['orders_generated']}")
        print(f"  完成订单: {self.daily_stats['orders_completed']}")
        print(f"  完成率: {self.daily_stats['orders_completed'] / self.daily_stats['orders_generated']:.2%}")
        print(f"  取消订单: {self.daily_stats['orders_cancelled']}")
        if self.daily_stats['orders_completed'] > 0:
            avg_wait = self.daily_stats['total_waiting_time'] / self.daily_stats['orders_completed']
            print(f"  平均等待时间: {avg_wait:.2f} steps")
        #print(f"  准时交付: {self.daily_stats['on_time_deliveries']}")

        # Report legacy blocked count if any
        if self.legacy_blocked_count > 0:
            print(f"  Legacy fallback blocked: {self.legacy_blocked_count} times")
        elif not self.enable_legacy_fallback:
            print(f"  Legacy fallback: DISABLED (0 attempts blocked)")

        unfinished_orders = list(self.active_orders)
        for order_id in unfinished_orders:
            self._cancel_order(order_id, "end_of_day")

        for drone_id, drone in self.drones.items():
            if drone['status'] not in [DroneStatus.IDLE, DroneStatus.CHARGING]:
                base_loc = self.bases[drone['base']]['location']
                self.state_manager.update_drone_status(drone_id, DroneStatus.RETURNING_TO_BASE,
                                                       target_location=base_loc)

        print(f"  未完成订单: {len(unfinished_orders)} 已取消")
        print("=== 当日营业结束 ===\n")

    # ------------------ observation encoding ------------------

    def _get_observation(self):
        order_obs = np.zeros((self.max_obs_orders, 10), dtype=np.float32)

        active_ids = list(self.active_orders)

        def sort_key(oid):
            o = self.orders[oid]
            return (not o.get('urgent', False), o['creation_time'])

        active_ids.sort(key=sort_key)
        self.current_obs_order_ids = active_ids[:self.max_obs_orders]

        for i, order_id in enumerate(self.current_obs_order_ids):
            order = self.orders[order_id]
            order_obs[i] = self._encode_order(order)

        drone_obs = np.zeros((self.num_drones, 8), dtype=np.float32)
        for drone_id, drone in self.drones.items():
            drone_obs[drone_id] = self._encode_drone(drone)

        # Top-K merchants
        merchant_obs = np.zeros((self.obs_num_merchants, 4), dtype=np.float32)
        obs_merchant_ids = self._select_topk_merchants_for_observation()
        for i, merchant_id in enumerate(obs_merchant_ids):
            merchant = self.merchants[merchant_id]
            merchant_obs[i] = self._encode_merchant(merchant)

        q_cap = max(getattr(self, "max_queue_cap", 50.0), 1e-6)
        e_cap = max(getattr(self, "max_eff_cap", 2.0), 1e-6)
        merchant_obs[:, 0] = np.clip(merchant_obs[:, 0] / q_cap, 0.0, 1.0)
        merchant_obs[:, 1] = np.clip(merchant_obs[:, 1] / e_cap, 0.0, 1.0)
        merchant_obs[:, 2] = np.clip(merchant_obs[:, 2], 0.0, 1.0)
        merchant_obs[:, 3] = (merchant_obs[:, 3] > 0).astype(np.float32)

        base_obs = np.zeros((self.obs_num_bases, 3), dtype=np.float32)
        for base_id in range(self.obs_num_bases):
            base = self.bases[base_id]
            base_obs[base_id] = self._encode_base(base)

        weather_details = np.zeros(5, dtype=np.float32)
        if self.weather_details:
            temp = self.weather_details.get('temperature', 15)
            weather_details[0] = (temp + 10) / 50
            humidity = self.weather_details.get('humidity', 0.5)
            weather_details[1] = humidity
            wind_speed = self.weather_details.get('wind_speed', 10)
            weather_details[2] = wind_speed / 100
            visibility = self.weather_details.get('visibility', 10)
            weather_details[3] = visibility / 20
            pressure = self.weather_details.get('pressure', 1013)
            weather_details[4] = (pressure - 980) / 60
        weather_details = np.clip(weather_details, 0.0, 1.0).astype(np.float32)

        time_state = self.time_system.get_time_state()
        time_obs = np.array([
            time_state['hour'] / 24.0,
            time_state['minute'] / 60.0,
            time_state['progress'],
            1.0 if time_state['is_peak_hour'] else 0.0,
            1.0 if time_state['is_business_hours'] else 0.0
        ], dtype=np.float32)

        order_pattern = np.asarray(self.order_processor.order_patterns['hourly_pattern'], dtype=np.float32)
        order_pattern = np.clip(order_pattern, 0.0, 1.0)

        pareto_info = np.zeros(self.num_objectives + 2, dtype=np.float32)
        if len(self.pareto_optimizer.pareto_front) > 0:
            pareto_front = self.pareto_optimizer.get_pareto_front()
            pareto_mean = np.mean(pareto_front, axis=0)
            pareto_info[:self.num_objectives] = pareto_mean
            reference_point = np.ones(self.num_objectives, dtype=np.float32) * 0.5
            pareto_info[self.num_objectives] = self.pareto_optimizer.calculate_hypervolume(reference_point)
            pareto_info[self.num_objectives + 1] = self.pareto_optimizer.get_diversity()
        pareto_info = np.nan_to_num(pareto_info, nan=0.0, posinf=1.0, neginf=0.0)
        pareto_info = np.clip(pareto_info, 0.0, 1.0).astype(np.float32)

        order_obs = np.clip(order_obs, 0.0, 1.0).astype(np.float32)
        drone_obs = np.clip(drone_obs, 0.0, 1.0).astype(np.float32)
        base_obs = np.clip(base_obs, 0.0, 1.0).astype(np.float32)

        # U7: Build candidate observations
        candidates_obs = np.zeros((self.num_drones, self.num_candidates, 12), dtype=np.float32)
        for drone_id in range(self.num_drones):
            if drone_id not in self.drone_candidate_mappings:
                # No candidates yet, use empty
                continue
            candidate_list = self.drone_candidate_mappings[drone_id]
            for i, (order_id, is_valid) in enumerate(candidate_list):
                candidates_obs[drone_id, i] = self._encode_candidate(order_id, is_valid)
        candidates_obs = np.clip(candidates_obs, 0.0, 1.0).astype(np.float32)

        obs_dict = {
            'orders': order_obs,
            'drones': drone_obs,
            'merchants': merchant_obs,
            'bases': base_obs,
            'candidates': candidates_obs,
            'weather': int(self.weather.value),
            'weather_details': weather_details,
            'time': time_obs,
            'day_progress': np.array([time_state['progress']], dtype=np.float32),
            'resource_saturation': np.array([self._calculate_resource_saturation()], dtype=np.float32),
            'order_pattern': order_pattern,
            'pareto_info': pareto_info,
            'objective_weights': self.objective_weights.copy()
        }

        # Self-check in debug mode: Verify all observation_space keys are present
        if self.debug_state_warnings:
            obs_space_keys = set(self.observation_space.spaces.keys())
            obs_keys = set(obs_dict.keys())
            missing_keys = obs_space_keys - obs_keys
            if missing_keys:
                raise KeyError(
                    f"_get_observation() missing required keys: {sorted(missing_keys)}. "
                    f"observation_space defines: {sorted(obs_space_keys)}, "
                    f"but observation returned: {sorted(obs_keys)}"
                )

        return obs_dict

    # ---- constant: dimension of rule-based compact state ----
    RULE_BASED_STATE_DIM = 20

    def _get_rule_based_state_for_drone(self, drone_id: int) -> np.ndarray:
        """
        Build the rule-discriminant compact state for a single drone (20-dimensional).

        This compact state is designed for the lower-layer PPO to discriminate between
        the 5 rule actions:
          Rule 0: CARGO_FIRST       – prioritise delivering cargo already on board
          Rule 1: ASSIGNED_EDF      – earliest-deadline assigned order
          Rule 2: READY_EDF         – earliest-deadline ready (unassigned) order
          Rule 3: NEAREST_PICKUP    – closest pickup merchant
          Rule 4: SLACK_PER_DISTANCE– highest slack / distance ratio

        It replaces the former high-dimensional global dict observation (orders,
        merchants, bases, pareto_info, objective_weights, etc.) with a
        low-dimensional, interpretable, rule-relevant feature vector.

        State layout (20 features, all normalised to [0, 1]):
          A. Drone own state (indices 0-4)
             0  u_status          – drone status / 7
             1  u_battery_ratio   – battery_level / max_battery
             2  u_load_ratio      – current_load / max_capacity
             3  u_cargo_ratio     – |cargo| / max_capacity
             4  u_dist_to_target  – distance to current target / max_grid_dist
          B. Candidate task structure (indices 5-9)
             5  cand_cargo_count_ratio    – PICKED_UP fraction among valid candidates
             6  cand_assigned_count_ratio – ASSIGNED fraction
             7  cand_ready_count_ratio    – READY fraction
             8  cand_urgent_ratio         – urgent fraction
             9  cand_min_slack            – tightest deadline slack (normalised)
          C. Rule-discriminant features (indices 10-15)
             10 cargo_best_slack         – best slack among on-board cargo
             11 assigned_best_slack      – best slack among assigned orders
             12 ready_best_slack         – best slack among ready orders
             13 nearest_pickup_dist      – nearest merchant distance (normalised)
             14 best_slack_per_distance  – best slack/dist ratio (normalised)
             15 ready_urgent_ratio       – urgent fraction among READY candidates
          D. Global correction (indices 16-19)
             16 sys_resource_saturation  – busy-drone ratio
             17 sys_backlog_ratio        – active orders / max_obs_orders
             18 env_weather_severity     – weather enum value / 3
             19 env_is_peak_hour         – 1.0 if peak hour else 0.0

        Returns:
            np.ndarray of shape (20,), dtype=np.float32, values in [0, 1]
        """
        state = np.zeros(self.RULE_BASED_STATE_DIM, dtype=np.float32)

        if drone_id not in self.drones:
            return state

        drone = self.drones[drone_id]
        current_step = self.time_system.current_step
        # Largest possible distance on the grid (corner to corner)
        max_dist = math.sqrt(2) * self.grid_size + 1e-6
        # Normalisation constant for deadline slack (steps)
        slack_norm = 50.0

        # ---- A. Drone own state ----
        # DroneStatus has values 0-7 (IDLE=0 … CHARGING=7); divide by 7 to normalise
        state[0] = drone['status'].value / 7.0
        state[1] = drone['battery_level'] / max(1.0, drone['max_battery'])
        state[2] = drone['current_load'] / max(1, drone['max_capacity'])
        cargo = drone.get('cargo', set())
        state[3] = len(cargo) / max(1, drone['max_capacity'])
        dist_to_target = self._get_dist_to_target(drone_id)
        state[4] = min(dist_to_target / max_dist, 1.0)

        # ---- B. Candidate task structure state ----
        candidate_list = self.drone_candidate_mappings.get(drone_id, [])
        valid_orders = []
        for oid, is_valid in candidate_list:
            if is_valid and oid >= 0 and oid in self.orders:
                valid_orders.append(self.orders[oid])
        num_valid = max(len(valid_orders), 1)

        cargo_count = sum(1 for o in valid_orders if o['status'] == OrderStatus.PICKED_UP)
        assigned_count = sum(1 for o in valid_orders if o['status'] == OrderStatus.ASSIGNED)
        ready_count = sum(1 for o in valid_orders if o['status'] == OrderStatus.READY)
        urgent_count = sum(1 for o in valid_orders if o.get('urgent', False))

        state[5] = cargo_count / num_valid
        state[6] = assigned_count / num_valid
        state[7] = ready_count / num_valid
        state[8] = urgent_count / num_valid

        if valid_orders:
            slacks = [self._get_delivery_deadline_step(o) - current_step for o in valid_orders]
            min_slack = min(slacks)
            state[9] = float(np.clip(min_slack / slack_norm + 0.5, 0.0, 1.0))
        else:
            state[9] = 1.0  # No urgency when no candidates

        # ---- C. Rule-discriminant features ----
        drone_loc = drone['location']

        # Cargo orders currently on board (PICKED_UP)
        cargo_orders = [self.orders[oid] for oid in cargo
                        if oid in self.orders and
                        self.orders[oid]['status'] == OrderStatus.PICKED_UP]
        if cargo_orders:
            best_cargo_slack = max(
                self._get_delivery_deadline_step(o) - current_step for o in cargo_orders
            )
            state[10] = float(np.clip(best_cargo_slack / slack_norm + 0.5, 0.0, 1.0))
        else:
            state[10] = 1.0  # Neutral when no cargo

        # Assigned orders for this drone among valid candidates
        assigned_orders = [o for o in valid_orders
                           if o['status'] == OrderStatus.ASSIGNED and
                           o.get('assigned_drone') == drone_id]
        if assigned_orders:
            best_assigned_slack = max(
                self._get_delivery_deadline_step(o) - current_step for o in assigned_orders
            )
            state[11] = float(np.clip(best_assigned_slack / slack_norm + 0.5, 0.0, 1.0))
        else:
            state[11] = 1.0

        # Ready unassigned orders among valid candidates
        ready_orders = [o for o in valid_orders
                        if o['status'] == OrderStatus.READY and
                        o.get('assigned_drone', -1) in (-1, None)]
        if ready_orders:
            best_ready_slack = max(
                self._get_delivery_deadline_step(o) - current_step for o in ready_orders
            )
            state[12] = float(np.clip(best_ready_slack / slack_norm + 0.5, 0.0, 1.0))
        else:
            state[12] = 1.0

        # Nearest pickup distance (min merchant distance for ASSIGNED/READY candidates)
        pickup_dists = []
        for o in valid_orders:
            if o['status'] in (OrderStatus.ASSIGNED, OrderStatus.READY):
                ml = o.get('merchant_location')
                if ml:
                    pickup_dists.append(
                        self._calculate_euclidean_distance(drone_loc, ml)
                    )
        state[13] = (min(pickup_dists) / max_dist) if pickup_dists else 1.0

        # Best slack-per-distance ratio across all valid candidates
        best_spd = 0.0
        for o in valid_orders:
            ml = None
            if o['status'] in (OrderStatus.ASSIGNED, OrderStatus.READY):
                ml = o.get('merchant_location')
            elif o['status'] == OrderStatus.PICKED_UP:
                ml = o.get('customer_location')
            if ml:
                dist = self._calculate_euclidean_distance(drone_loc, ml)
                slack = self._get_delivery_deadline_step(o) - current_step
                # Small epsilon avoids division by zero when drone is at the location
                spd = slack / (dist + 0.1)
                if spd > best_spd:
                    best_spd = spd
        # Typical slack/dist range: slack ≤ ~50 steps, dist ≥ 0.1  → max ≈ 500
        state[14] = float(np.clip(best_spd / 500.0, 0.0, 1.0))

        # Urgent fraction within READY candidates
        if ready_orders:
            urgent_ready = sum(1 for o in ready_orders if o.get('urgent', False))
            state[15] = urgent_ready / len(ready_orders)
        else:
            state[15] = 0.0

        # ---- D. Global correction state ----
        state[16] = self._calculate_resource_saturation()
        state[17] = min(len(self.active_orders) / max(self.max_obs_orders, 1), 1.0)
        state[18] = self.weather.value / 3.0
        time_state = self.time_system.get_time_state()
        state[19] = 1.0 if time_state['is_peak_hour'] else 0.0

        return np.clip(state, 0.0, 1.0)

    def _encode_order(self, order):
        encoding = np.zeros(10, dtype=np.float32)
        encoding[order['status'].value] = 1.0
        encoding[6] = order['order_type'].value / 2.0
        encoding[7] = (self.time_system.current_step - order['creation_time']) / 50.0

        assigned_drone = order.get('assigned_drone', -1)
        if assigned_drone is None:
            assigned_drone = -1
        encoding[8] = assigned_drone / max(1, self.num_drones)

        encoding[9] = 1.0 if order.get('urgent', False) else 0.0
        return encoding

    def _encode_drone(self, drone):
        """
        Encode a drone into an 8-dimensional feature vector.

        Features:
          0: status (normalized by max status value 7)
          1: location x (normalized by grid_size)
          2: location y (normalized by grid_size)
          3: target location x (normalized by grid_size; 0 if no target)
          4: target location y (normalized by grid_size; 0 if no target)
          5: cargo ratio = len(cargo) / max_capacity, clipped to [0, 1]
             (0.0 = empty cargo / pickup phase; >0 = has cargo / delivery phase)
          6: current load ratio = current_load / max_capacity
          7: battery level ratio = battery_level / max_battery
        """
        encoding = np.zeros(8, dtype=np.float32)

        encoding[0] = drone['status'].value / 7.0
        encoding[1] = drone['location'][0] / self.grid_size
        encoding[2] = drone['location'][1] / self.grid_size

        if 'target_location' in drone:
            encoding[3] = drone['target_location'][0] / self.grid_size
            encoding[4] = drone['target_location'][1] / self.grid_size

        cargo = drone.get('cargo', set())
        max_capacity = drone['max_capacity']
        encoding[5] = min(len(cargo) / max(1, max_capacity), 1.0)

        encoding[6] = drone['current_load'] / max(1, max_capacity)
        encoding[7] = drone['battery_level'] / max(1.0, drone['max_battery'])
        return encoding

    def _encode_merchant(self, merchant):
        encoding = np.zeros(4, dtype=np.float32)
        encoding[0] = len(merchant['queue'])  # 先用原始值，后面统一归一化
        encoding[1] = merchant['efficiency']
        encoding[2] = merchant['cancellation_rate']
        encoding[3] = 1.0 if merchant['landing_zone'] else 0.0
        return encoding

    def _encode_base(self, base):
        encoding = np.zeros(3, dtype=np.float32)
        encoding[0] = len(base['drones_available']) / max(1, base['capacity'])
        encoding[1] = base['charging_stations'] / 5.0
        encoding[2] = len(base['charging_queue']) / 5.0
        return encoding

    # ------------------ info/report ------------------

    def _calculate_resource_saturation(self):
        busy_drones = sum(1 for d in self.drones.values()
                          if d['status'] not in [DroneStatus.IDLE, DroneStatus.CHARGING])
        return busy_drones / max(1, self.num_drones)

    def _monitor_drone_status(self):
        pass

    def _update_system_state(self):
        pass

    def _print_diagnostics(self):
        """Print detailed diagnostics for debugging PPO+MOPSO training."""
        print(f"\n=== Diagnostics (Step {self.time_system.current_step}) ===")

        # Battery and energy metrics
        battery_levels = [drone['battery_level'] for drone in self.drones.values()]
        avg_battery = np.mean(battery_levels) if battery_levels else 0.0
        min_battery = np.min(battery_levels) if battery_levels else 0.0
        print(f"  Battery: avg={avg_battery:.1f}, min={min_battery:.1f}")
        print(f"  Energy consumed today: {self.daily_stats.get('energy_consumed', 0.0):.2f}")
        print(f"  Forced return events: {self.daily_stats.get('forced_return_events', 0)}")

        # Count drones by state
        drones_with_serving_order = 0
        drones_with_valid_candidates = 0
        drones_with_assigned_candidates = 0
        drones_with_cargo_candidates = 0
        drones_at_decision_points = 0

        for drone_id in range(self.num_drones):
            drone = self.drones[drone_id]

            # Check serving order
            if drone.get('serving_order_id') is not None:
                drones_with_serving_order += 1

            # Check decision point
            if self._is_at_decision_point(drone_id):
                drones_at_decision_points += 1

            # Check candidates
            if drone_id in self.drone_candidate_mappings:
                candidate_list = self.drone_candidate_mappings[drone_id]
                has_valid = any(is_valid for _, is_valid in candidate_list)
                if has_valid:
                    drones_with_valid_candidates += 1

                # Count assigned vs cargo candidates
                for order_id, is_valid in candidate_list:
                    if is_valid and order_id >= 0 and order_id in self.orders:
                        order = self.orders[order_id]
                        if order['status'] == OrderStatus.PICKED_UP:
                            drones_with_cargo_candidates += 1
                            break  # Count drone only once
                for order_id, is_valid in candidate_list:
                    if is_valid and order_id >= 0 and order_id in self.orders:
                        order = self.orders[order_id]
                        if order['status'] == OrderStatus.ASSIGNED:
                            drones_with_assigned_candidates += 1
                            break  # Count drone only once

        print(f"  Drones with serving_order_id: {drones_with_serving_order}/{self.num_drones}")
        # Use cached decision points count (from before action was processed)
        print(f"  Drones at decision points: {self._last_decision_points_count}/{self.num_drones}")
        print(f"  Drones with ≥1 valid candidate: {drones_with_valid_candidates}/{self.num_drones}")
        print(f"  Drones with cargo candidates: {drones_with_cargo_candidates}/{self.num_drones}")
        print(f"  Drones with assigned candidates: {drones_with_assigned_candidates}/{self.num_drones}")
        print(f"  Actions applied this step: {self.action_applied_count}")

        # Rule usage statistics (U8: interpretable rule-based control)
        if self.rule_usage_stats:
            print(f"\n  --- Rule Usage Statistics (Cumulative) ---")
            rule_names = ["CARGO_FIRST", "ASSIGNED_EDF", "READY_EDF", "NEAREST_PICKUP", "SLACK_PER_DISTANCE"]
            total_rule_uses = sum(self.rule_usage_stats.values())
            for rule_id in range(self.rule_count):
                count = self.rule_usage_stats.get(rule_id, 0)
                pct = (count / total_rule_uses * 100) if total_rule_uses > 0 else 0.0
                rule_name = rule_names[rule_id] if rule_id < len(rule_names) else f"Rule{rule_id}"
                print(f"    Rule {rule_id} ({rule_name}): {count} ({pct:.1f}%)")

        # Legacy blocking info
        if not self.enable_legacy_fallback:
            print(f"  Legacy fallback: DISABLED")
            print(f"  Legacy blocked count: {self.legacy_blocked_count}")
            if self.legacy_blocked_reasons:
                print(f"  Legacy blocked reasons: {dict(self.legacy_blocked_reasons)}")
        else:
            print(f"  Legacy fallback: ENABLED")

        # Order stats - use incremental cache (always maintained, even when empty)
        ready_count = len(self._ready_orders_cache)
        assigned_count = sum(1 for oid in self.active_orders if self.orders[oid]['status'] == OrderStatus.ASSIGNED)
        picked_up_count = sum(1 for oid in self.active_orders if self.orders[oid]['status'] == OrderStatus.PICKED_UP)

        print(f"  Orders: READY={ready_count}, ASSIGNED={assigned_count}, PICKED_UP={picked_up_count}")
        print(
            f"  Orders completed: {self.daily_stats['orders_completed']}, cancelled: {self.daily_stats['orders_cancelled']}")

        # Per-step performance summary (D: lightweight profiling)
        if self._perf_steps > 0:
            n = self._perf_steps
            pa = self._perf_accum
            print(f"\n  --- Avg Per-Step Timings ({n} steps) ---")
            for key in ('candidate_update', 'event_processing'):
                avg_ms = pa.get(key, 0.0) / n * 1000.0
                print(f"    {key}: {avg_ms:.2f} ms/step")

        # Reward component breakdown (new diagnostic)
        print(f"\n  --- Reward Components (Last Step) ---")
        rc = self.last_step_reward_components
        print(f"  Obj0 (Throughput/Efficiency): {rc['obj0_total']:+.4f}")
        print(f"    - Completed bonus: {rc['obj0_completed']:+.4f} (delta={rc['delta_completed']:.0f})")
        print(f"    - Cancelled penalty: {rc['obj0_cancelled']:+.4f} (delta={rc['delta_cancelled']:.0f})")
        print(f"    - Progress shaping: {rc['obj0_progress_shaping']:+.4f}")
        print(f"  Obj1 (Cost): {rc['obj1_total']:+.4f}")
        print(f"    - Energy cost: {rc['obj1_energy_cost']:+.4f} (delta_energy={rc['delta_energy']:.2f})")
        print(f"    - Distance cost: {rc['obj1_distance_cost']:+.4f} (delta_distance={rc['delta_distance']:.2f})")
        print(f"  Obj2 (Service Quality): {rc['obj2_total']:+.4f}")
        print(f"    - On-time reward: {rc['obj2_on_time']:+.4f}")
        print(f"    - Cancelled penalty: {rc['obj2_cancelled']:+.4f}")
        print(f"    - Backlog penalty: {rc['obj2_backlog']:+.4f}")
        print("=" * 60)

    def _get_info(self):
        info = {
            'metrics': self.metrics.copy(),
            'daily_stats': self.daily_stats.copy(),
            'resource_saturation': self._calculate_resource_saturation(),
            'weather_impact_stats': self.metrics['weather_impact_stats'].copy(),
            'current_weather': self.weather_details.copy(),
            'pareto_front_size': len(self.pareto_optimizer.pareto_front),
            'pareto_hypervolume': self.pareto_optimizer.calculate_hypervolume(np.ones(self.num_objectives) * 0.5),
            'pareto_diversity': self.pareto_optimizer.get_diversity(),
            'order_history_summary': {
                # O(1): use running stats updated when orders are appended
                'total_orders': self._order_hist_dist_count,
                'unique_merchants': len(self._order_hist_merchant_ids),
                'avg_distance': (self._order_hist_dist_sum / self._order_hist_dist_count
                                 if self._order_hist_dist_count > 0 else 0)
            },
            'time_state': self.time_system.get_time_state(),
            'backlog_size': len(self.active_orders),
            'legacy_blocked_count': self.legacy_blocked_count,
            'legacy_fallback_enabled': self.enable_legacy_fallback,
            # Add diagnostics statistics (consistent with strict counting)
            'drones_at_decision_points': self._last_decision_points_count,
            'actions_applied_this_step': self.action_applied_count,
            # Add last decision info for decentralized execution tracking
            'last_decision_drone_id': self.last_decision_info['drone_id'],
            'last_decision_rule_id': self.last_decision_info['rule_id'],
            'last_decision_success': self.last_decision_info['success'],
            'last_decision_failure_reason': self.last_decision_info['failure_reason'],
        }

        if self.daily_stats['orders_completed'] > 0:
            info['avg_delivery_time'] = self.metrics['total_delivery_time'] / self.daily_stats['orders_completed']
            info['energy_efficiency'] = self.daily_stats['energy_consumed'] / self.daily_stats['orders_completed']
            info['on_time_rate'] = self.daily_stats['on_time_deliveries'] / self.daily_stats['orders_completed']
            info['avg_distance_per_order'] = self.daily_stats['total_flight_distance'] / self.daily_stats[
                'orders_completed']
            info['avg_waiting_time'] = self.daily_stats['total_waiting_time'] / self.daily_stats['orders_completed']
        else:
            info['avg_delivery_time'] = 0
            info['energy_efficiency'] = 0
            info['on_time_rate'] = 0
            info['avg_distance_per_order'] = 0
            info['avg_waiting_time'] = 0

        return info

    def get_daily_report(self):
        time_state = self.time_system.get_time_state()

        report = {
            'day_number': self.time_system.day_number,
            'current_time': f"{time_state['hour']:02d}:{time_state['minute']:02d}",
            'weather': self.weather_details.get('summary', 'Unknown'),
            'order_stats': {
                'generated': self.daily_stats['orders_generated'],
                'completed': self.daily_stats['orders_completed'],
                'cancelled': self.daily_stats['orders_cancelled'],
                'active': len(self.active_orders),
                'completion_rate': self.daily_stats['orders_completed'] / max(1, self.daily_stats['orders_generated'])
            },
            'performance_metrics': {
                'on_time_rate': self.daily_stats['on_time_deliveries'] / max(1, self.daily_stats['orders_completed']),
                'energy_efficiency': self.daily_stats['energy_consumed'] / max(1, self.daily_stats['orders_completed']),
                'resource_utilization': self._calculate_resource_saturation(),
                'total_flight_distance': self.daily_stats['total_flight_distance'],
                'optimal_flight_distance': self.daily_stats['optimal_flight_distance'],
                'avg_waiting_time': (
                    self.daily_stats['total_waiting_time'] / self.daily_stats['orders_completed']
                    if self.daily_stats['orders_completed'] > 0 else 0
                ),
            },
            'drone_status': self._get_drone_status_summary()
        }

        return report

    def _get_drone_status_summary(self):
        status_count = {
            'idle': 0,
            'assigned': 0,
            'flying_to_merchant': 0,
            'waiting_for_pickup': 0,
            'flying_to_customer': 0,
            'delivering': 0,
            'returning_to_base': 0,
            'charging': 0
        }

        for drone in self.drones.values():
            status = drone['status']
            if status == DroneStatus.IDLE:
                status_count['idle'] += 1
            elif status == DroneStatus.ASSIGNED:
                status_count['assigned'] += 1
            elif status == DroneStatus.FLYING_TO_MERCHANT:
                status_count['flying_to_merchant'] += 1
            elif status == DroneStatus.WAITING_FOR_PICKUP:
                status_count['waiting_for_pickup'] += 1
            elif status == DroneStatus.FLYING_TO_CUSTOMER:
                status_count['flying_to_customer'] += 1
            elif status == DroneStatus.DELIVERING:
                status_count['delivering'] += 1
            elif status == DroneStatus.RETURNING_TO_BASE:
                status_count['returning_to_base'] += 1
            elif status == DroneStatus.CHARGING:
                status_count['charging'] += 1

        return status_count

    # ------------------ Snapshot interfaces for MOPSO ------------------

    def get_ready_orders_snapshot(self, limit: int = 200) -> List[dict]:
        """
        Get snapshot of READY orders for MOPSO scheduling.
        Returns list of order dicts with essential fields.

        Uses incremental _ready_orders_cache to avoid full active_orders scan.
        """
        ready_orders = []
        # _ready_orders_cache is always maintained by StateManager; use it directly
        for oid in self._ready_orders_cache:
            if oid not in self.orders:
                continue
            order = self.orders[oid]
            if order['status'] != OrderStatus.READY:
                continue
            if order.get('assigned_drone', -1) not in (-1, None):
                continue

            # Create snapshot with essential fields
            snapshot = {
                'order_id': oid,
                'merchant_id': order['merchant_id'],
                'merchant_location': order['merchant_location'],
                'customer_location': order['customer_location'],
                'creation_time': order['creation_time'],
                'deadline_step': self._get_delivery_deadline_step(order),
                'urgent': order.get('urgent', False),
                'distance': order.get('distance', 0.0),
            }
            ready_orders.append(snapshot)

            if len(ready_orders) >= limit:
                break

        return ready_orders

    def get_drones_snapshot(self) -> List[dict]:
        """
        Get snapshot of all drones for MOPSO scheduling.
        Returns list of drone dicts with essential fields.
        """
        drones_snapshot = []
        for drone_id, drone in self.drones.items():
            snapshot = {
                'drone_id': drone_id,
                'location': drone['location'],
                'base': drone['base'],
                'status': drone['status'],
                'battery_level': drone['battery_level'],
                'current_load': drone['current_load'],
                'max_capacity': drone['max_capacity'],
                'speed': drone['speed'],
                'battery_consumption_rate': drone['battery_consumption_rate'],
                'has_route': drone.get('route_committed', False),
                'can_accept_more': drone['current_load'] < drone['max_capacity'],
            }
            drones_snapshot.append(snapshot)

        return drones_snapshot

    def get_merchants_snapshot(self) -> Dict[str, dict]:
        """
        Get snapshot of all merchants for MOPSO scheduling.
        Returns dict mapping merchant_id to merchant info.
        """
        merchants_snapshot = {}
        for merchant_id, merchant in self.merchants.items():
            snapshot = {
                'merchant_id': merchant_id,
                'location': merchant['location'],
                'queue_length': len(merchant.get('queue', [])),
                'cancellation_rate': merchant.get('cancellation_rate', 0.01),
                'landing_zone': merchant.get('landing_zone', True),
            }
            merchants_snapshot[merchant_id] = snapshot

        return merchants_snapshot

    def get_route_plan_constraints(self) -> dict:
        """
        Get constraints for route plan generation.
        Returns dict with constraint parameters.
        """
        constraints = {
            'grid_size': self.grid_size,
            'current_step': self.time_system.current_step,
            'max_capacity_per_drone': self.drone_max_capacity,
            'weather_speed_factor': self._get_weather_speed_factor(),
            'weather_battery_factor': self._get_weather_battery_factor(),
            # Energy model parameters for dispatcher consistency
            'base_energy_per_distance': self.energy_e0,
            'energy_alpha': self.energy_alpha,
            'battery_return_threshold': self.battery_return_threshold,
            'battery_scale': 100.0,  # Battery level range: 0-100
        }
        return constraints

    # ================ U9: Candidate-based filtering support ================

    def set_candidate_generator(self, generator):
        """
        Set external candidate generator for upper-layer candidate generation.

        Args:
            generator: Instance of CandidateGenerator or compatible class
                      Must implement generate_candidates(env) -> Dict[int, List[int]]
        """
        self.candidate_generator = generator

    def update_filtered_candidates(self):
        """
        Update filtered_candidates using the external candidate generator.
        If no generator is set, filtered_candidates remains empty (fallback to active_orders).

        This should be called:
        - On reset
        - Every candidate_update_interval steps (if > 0)
        - When decision queue is empty (in wrapper context)
        """
        if self.candidate_generator is None:
            self.filtered_candidates = {}
            self._filtered_candidates_sets = {}
            return

        # Generate candidates using the external generator
        self.filtered_candidates = self.candidate_generator.generate_candidates(self)
        # Build cached set version for O(1) membership tests in _get_candidate_constrained_orders
        self._filtered_candidates_sets = {
            drone_id: set(order_list)
            for drone_id, order_list in self.filtered_candidates.items()
        }
        # Caches are now fresh; clear the dirty flag.
        self._candidate_mappings_dirty = False

    def get_filtered_candidates_for_drone(self, drone_id: int) -> List[int]:
        """
        Get filtered candidate order IDs for a specific drone.

        Args:
            drone_id: The drone ID

        Returns:
            List of order IDs that are candidates for this drone
            Returns empty list if no candidates are set
        """
        return self.filtered_candidates.get(drone_id, [])

    def _get_candidate_constrained_orders(self, drone_id: int, order_ids: List[int]) -> List[int]:
        """
        Filter order_ids to only include those in the drone's candidate set.

        Args:
            drone_id: The drone ID
            order_ids: List of order IDs to filter

        Returns:
            Filtered list of order IDs that are in candidates[drone_id] ∩ order_ids
            If candidate_fallback_enabled and no candidates, returns original order_ids
        """
        # Use cached set for O(1) membership test (rebuilt by update_filtered_candidates)
        candidate_set = self._filtered_candidates_sets.get(drone_id)

        # If no candidates and fallback is enabled, return all orders
        if not candidate_set:
            if self.candidate_fallback_enabled:
                return order_ids
            return []

        return [oid for oid in order_ids if oid in candidate_set and self._is_candidate_selectable(oid)]

    def _is_candidate_selectable(self, order_id: int) -> bool:
        """Final dirty check: ensure an order in the candidate set is still selectable.

        Guards against stale candidate entries when the order was cancelled,
        delivered, or assigned to another drone between candidate refreshes.
        A READY order must also be genuinely unassigned.
        """
        order = self.orders.get(order_id)
        if order is None:
            return False
        status = order['status']
        if status == OrderStatus.READY:
            # Must be genuinely unassigned (no drone has claimed it yet)
            return order.get('assigned_drone', -1) in (-1, None)
        # ASSIGNED and PICKED_UP orders remain selectable only when their drone
        # is actively working (not IDLE/CHARGING which indicates a stale reference).
        if status in (OrderStatus.ASSIGNED, OrderStatus.PICKED_UP):
            drone_id = order.get('assigned_drone', -1)
            if drone_id < 0 or drone_id not in self.drones:
                return False
            return self.drones[drone_id]['status'] not in (DroneStatus.IDLE, DroneStatus.CHARGING)
        return False

    # ================ U9: Event-driven single UAV decision support ================

    def get_decision_drones(self) -> List[int]:
        """
        Get list of drone IDs that are currently at decision points.

        Returns:
            List of drone IDs at decision points
        """
        decision_drones = []
        for drone_id in range(self.num_drones):
            if self._is_at_decision_point(drone_id):
                decision_drones.append(drone_id)
        return decision_drones

    def apply_rule_to_drone(self, drone_id: int, rule_id: int) -> bool:
        """
        Apply a rule to a specific drone for order selection and assignment.

        This method is used by the EventDrivenSingleUAVWrapper to apply
        a single rule action to one drone at a time.

        Args:
            drone_id: The drone to apply the rule to
            rule_id: The rule ID (0-4) to apply

        Returns:
            True if the rule was successfully applied and changed state, False otherwise

        Side effects:
            Updates self.last_decision_info with decision details and failure reason
        """
        # Reset decision info
        self.last_decision_info = {
            'drone_id': drone_id,
            'rule_id': rule_id,
            'success': False,
            'failure_reason': None,
            'order_id': None,
        }

        if drone_id < 0 or drone_id >= self.num_drones:
            self.last_decision_info['failure_reason'] = 'invalid_drone_id'
            return False

        drone = self.drones[drone_id]

        # Check if drone is at decision point
        if not self._is_at_decision_point(drone_id):
            self.last_decision_info['failure_reason'] = 'not_at_decision_point'
            return False

        # Apply rule to select an order
        order_id = self._select_order_by_rule(drone_id, rule_id)

        # If no order selected, rule didn't apply
        if order_id is None or order_id not in self.orders:
            self.last_decision_info['failure_reason'] = 'no_order_selected'
            return False

        # Record selected order_id
        self.last_decision_info['order_id'] = order_id

        order = self.orders[order_id]

        # Track state before changes
        state_changed = False
        prev_serving_order_id = drone.get('serving_order_id')
        prev_target_location = drone.get('target_location')
        prev_status = drone['status']
        prev_load = drone['current_load']

        # Handle READY orders - assign them first
        if order['status'] == OrderStatus.READY:
            if order.get('assigned_drone', -1) in (-1, None):
                if drone['current_load'] < drone['max_capacity']:
                    # Assign the order
                    self._process_single_assignment(drone_id, order_id, allow_busy=True)

                    # Check if assignment happened
                    new_order_status = order['status']
                    new_assigned_drone = order.get('assigned_drone', -1)
                    new_load = drone['current_load']

                    if (new_order_status == OrderStatus.ASSIGNED and
                            new_assigned_drone == drone_id and
                            new_load > prev_load):
                        state_changed = True
                    else:
                        self.last_decision_info['failure_reason'] = 'assignment_rejected'
                else:
                    self.last_decision_info['failure_reason'] = 'drone_at_capacity'
            else:
                self.last_decision_info['failure_reason'] = 'order_already_assigned'
        else:
            # Order not READY - might be ASSIGNED or PICKED_UP by this drone
            if order.get('assigned_drone', -1) != drone_id:
                self.last_decision_info['failure_reason'] = 'order_not_ready_or_not_mine'

        # Set serving_order_id and target
        drone['serving_order_id'] = order_id

        # Determine target based on order status
        if order['status'] == OrderStatus.PICKED_UP:
            # Deliver to customer
            customer_loc = order.get('customer_location')
            if customer_loc:
                drone['target_location'] = customer_loc
                self.state_manager.update_drone_status(
                    drone_id, DroneStatus.FLYING_TO_CUSTOMER, target_location=customer_loc
                )
                if not state_changed:
                    new_target = drone.get('target_location')
                    new_status = drone['status']
                    if (prev_target_location != new_target or
                            prev_status != new_status or
                            prev_serving_order_id != order_id):
                        state_changed = True

        elif order['status'] == OrderStatus.ASSIGNED:
            # Go to merchant
            merchant_id = order.get('merchant_id')
            if merchant_id and merchant_id in self.merchants:
                merchant_loc = self.merchants[merchant_id]['location']
                drone['target_location'] = merchant_loc
                drone['current_merchant_id'] = merchant_id
                self.state_manager.update_drone_status(
                    drone_id, DroneStatus.FLYING_TO_MERCHANT, target_location=merchant_loc
                )
                if not state_changed:
                    new_target = drone.get('target_location')
                    new_status = drone['status']
                    if (prev_target_location != new_target or
                            prev_status != new_status or
                            prev_serving_order_id != order_id):
                        state_changed = True

        # Update decision info
        self.last_decision_info['success'] = state_changed
        if state_changed and self.last_decision_info['failure_reason'] is None:
            self.last_decision_info['failure_reason'] = None  # Success, no failure

        return state_changed

    def apply_rule_to_drone_with_info(self, drone_id: int, rule_id: int) -> Tuple[bool, Dict]:
        """
        Apply a rule to a specific drone, returning both success flag and detailed info.

        This is a convenience wrapper around apply_rule_to_drone that also returns
        the decision info dict, avoiding the need to separately read last_decision_info.

        Args:
            drone_id: The drone to apply the rule to
            rule_id: The rule ID (0-4) to apply

        Returns:
            Tuple of (success: bool, info: dict) where info contains:
                - drone_id: int
                - rule_id: int
                - success: bool
                - failure_reason: str or None (None on success)
                - order_id: int or None (selected order id, if any)
        """
        success = self.apply_rule_to_drone(drone_id, rule_id)
        info = dict(self.last_decision_info)
        return success, info