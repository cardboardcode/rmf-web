# pragma: no cover

import asyncio
import base64
import hashlib
import logging
from datetime import datetime
from typing import Any, cast

import rclpy
import rclpy.client
import rclpy.node
import rclpy.qos
from fastapi import HTTPException
from rclpy.subscription import Subscription
from rmf_building_map_msgs.msg import AffineImage as RmfAffineImage
from rmf_building_map_msgs.msg import BuildingMap as RmfBuildingMap
from rmf_building_map_msgs.msg import Level as RmfLevel
from rmf_dispenser_msgs.msg import DispenserState as RmfDispenserState
from rmf_door_msgs.msg import DoorMode as RmfDoorMode
from rmf_door_msgs.msg import DoorRequest as RmfDoorRequest
from rmf_door_msgs.msg import DoorState as RmfDoorState
from rmf_fleet_msgs.msg import BeaconState as RmfBeaconState
from rmf_fleet_msgs.msg import DeliveryAlert as RmfDeliveryAlert
from rmf_fleet_msgs.msg import DeliveryAlertAction as RmfDeliveryAlertAction
from rmf_fleet_msgs.msg import DeliveryAlertCategory as RmfDeliveryAlertCategory
from rmf_fleet_msgs.msg import DeliveryAlertTier as RmfDeliveryAlertTier
from rmf_fleet_msgs.msg import MutexGroupManualRelease as RmfMutexGroupManualRelease
from rmf_ingestor_msgs.msg import IngestorState as RmfIngestorState
from rmf_lift_msgs.msg import LiftRequest as RmfLiftRequest
from rmf_lift_msgs.msg import LiftState as RmfLiftState
from rmf_task_msgs.msg import Alert as RmfAlert
from rmf_task_msgs.msg import AlertResponse as RmfAlertResponse
from rosidl_runtime_py.convert import message_to_ordereddict
from std_msgs.msg import Bool as BoolMsg
from tortoise.exceptions import IntegrityError, ProgrammingError

from api_server.exceptions import AlreadyExistsError, InvalidInputError, NotFoundError
from api_server.fast_io.singleton_dep import singleton_dep
from api_server.models.user import User
from api_server.repositories.alerts import AlertRepository
from api_server.repositories.cached_files import get_cached_file_repo
from api_server.repositories.rmf import RmfRepository
from api_server.rmf_io.events import (
    AlertEvents,
    RmfEvents,
    get_alert_events,
    get_rmf_events,
)
from api_server.ros import get_ros_node

from .models import (
    AlertParameter,
    AlertRequest,
    BeaconState,
    BuildingMap,
    DeliveryAlert,
    DispenserState,
    DoorState,
    FireAlarmTriggerState,
    IngestorState,
    LiftState,
)
from .repositories import CachedFilesRepository


class RmfGateway:
    def __init__(
        self,
        cached_files: CachedFilesRepository,
        ros_node: rclpy.node.Node,
        alert_events: AlertEvents,
        alert_repo: AlertRepository,
        rmf_events: RmfEvents,
        rmf_repo: RmfRepository,
        loop: asyncio.AbstractEventLoop,
        *,
        logger: logging.Logger | None = None,
    ):
        self._cached_files = cached_files
        self._ros_node = ros_node
        self._alert_events = alert_events
        self._alert_repo = alert_repo
        self._rmf_events = rmf_events
        self._rmf_repo = rmf_repo
        self._loop = loop
        self._logger = logger or logging.getLogger()

        # ----------------------------
        # lifecycle state (IMPORTANT)
        # ----------------------------
        self._tasks: set[asyncio.Task] = set()
        self._subscriptions: list[Subscription] = []
        self._closing = False

        # ----------------------------
        # publishers
        # ----------------------------
        self._door_req = self._ros_node.create_publisher(
            RmfDoorRequest, "adapter_door_requests", 10
        )

        transient_qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._adapter_lift_req = self._ros_node.create_publisher(
            RmfLiftRequest, "adapter_lift_requests", transient_qos
        )

        self._delivery_alert_response = self._ros_node.create_publisher(
            RmfDeliveryAlert,
            "delivery_alert_response",
            rclpy.qos.QoSProfile(
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._mutex_group_release = self._ros_node.create_publisher(
            RmfMutexGroupManualRelease,
            "mutex_group_manual_release",
            rclpy.qos.QoSProfile(
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._fire_alarm_trigger = self._ros_node.create_publisher(
            BoolMsg,
            "fire_alarm_trigger",
            rclpy.qos.QoSProfile(
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._alert_response = self._ros_node.create_publisher(
            RmfAlertResponse,
            "alert_response",
            rclpy.qos.QoSProfile(
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._subscribe_all()

    # =========================================================
    # lifecycle helpers
    # =========================================================

    def _spawn(self, coro):
        if self._closing:
            return None

        task = self._loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def stop(self):
        self._closing = True

        # stop subscriptions first
        for sub in self._subscriptions:
            try:
                sub.destroy()
            except Exception:
                pass
        self._subscriptions.clear()

        # cancel async tasks
        for task in list(self._tasks):
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def __aexit__(self, *exc):
        await self.stop()

    # =========================================================
    # core utilities
    # =========================================================

    async def call_service(self, client: rclpy.client.Client, req, timeout=1) -> Any:
        fut = client.call_async(req)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError as e:
            raise HTTPException(503, "ros service call timed out") from e

    def _process_building_map(self, rmf_building_map: RmfBuildingMap) -> BuildingMap:
        processed_map = message_to_ordereddict(rmf_building_map)

        for i, level in enumerate(rmf_building_map.levels):
            for j, image in enumerate(level.images):
                image = cast(RmfAffineImage, image)

                sha1 = hashlib.sha1()
                sha1.update(image.data)
                fingerprint = base64.b32encode(sha1.digest()).lower().decode()

                relpath = (
                    f"{rmf_building_map.name}/"
                    f"{level.name}-{image.name}.{fingerprint}.{image.encoding}"
                )

                urlpath = self._cached_files.add_file(image.data, relpath)
                processed_map["levels"][i]["images"][j]["data"] = urlpath

        return BuildingMap(**processed_map)

    # =========================================================
    # subscriptions
    # =========================================================

    def _subscribe_all(self):

        def handle_door_state(msg):
            async def save(state: DoorState):
                if self._closing:
                    return
                await self._rmf_repo.save_door_state(state)
                self._rmf_events.door_states.on_next(state)

            self._spawn(save(DoorState.model_validate(msg)))

        self._subscriptions.append(
            self._ros_node.create_subscription(
                RmfDoorState, "door_states", handle_door_state, 100
            )
        )

        def handle_lift_state(msg):
            async def save(state: LiftState):
                if self._closing:
                    return
                await self._rmf_repo.save_lift_state(state)
                self._rmf_events.lift_states.on_next(state)

            dic = message_to_ordereddict(msg)
            self._spawn(save(LiftState(**dic)))

        self._subscriptions.append(
            self._ros_node.create_subscription(
                RmfLiftState, "lift_states", handle_lift_state, 10
            )
        )

        def handle_dispenser_state(msg):
            async def save(state: DispenserState):
                if self._closing:
                    return
                await self._rmf_repo.save_dispenser_state(state)
                self._rmf_events.dispenser_states.on_next(state)

            self._spawn(save(DispenserState.model_validate(msg)))

        self._subscriptions.append(
            self._ros_node.create_subscription(
                RmfDispenserState, "dispenser_states", handle_dispenser_state, 10
            )
        )

        def handle_ingestor_state(msg):
            async def save(state: IngestorState):
                if self._closing:
                    return
                await self._rmf_repo.save_ingestor_state(state)
                self._rmf_events.ingestor_states.on_next(state)

            self._spawn(save(IngestorState.model_validate(msg)))

        self._subscriptions.append(
            self._ros_node.create_subscription(
                RmfIngestorState, "ingestor_states", handle_ingestor_state, 10
            )
        )

        def handle_building_map(msg):
            async def save(bm: BuildingMap):
                if self._closing:
                    return
                await self._rmf_repo.save_building_map(bm)
                self._rmf_events.building_map.on_next(bm)

            bm = self._process_building_map(cast(RmfBuildingMap, msg))
            self._spawn(save(bm))

        self._subscriptions.append(
            self._ros_node.create_subscription(
                RmfBuildingMap,
                "map",
                handle_building_map,
                rclpy.qos.QoSProfile(
                    history=rclpy.qos.HistoryPolicy.KEEP_ALL,
                    depth=1,
                    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
        )

        def handle_beacon_state(msg):
            async def save(state: BeaconState):
                if self._closing:
                    return
                await self._rmf_repo.save_beacon_state(state)
                self._rmf_events.beacons.on_next(state)

            msg = cast(RmfBeaconState, msg)
            state = BeaconState(
                id=msg.id,
                online=msg.online,
                category=msg.category,
                activated=msg.activated,
                level=msg.level,
            )

            self._spawn(save(state))

        self._subscriptions.append(
            self._ros_node.create_subscription(
                RmfBeaconState, "beacon_state", handle_beacon_state, 100
            )
        )

        # alerts
        def convert_alert(msg):
            alert = cast(RmfAlert, msg)

            tier = AlertRequest.Tier.Info
            if alert.tier == RmfAlert.TIER_WARNING:
                tier = AlertRequest.Tier.Warning
            elif alert.tier == RmfAlert.TIER_ERROR:
                tier = AlertRequest.Tier.Error

            return AlertRequest(
                id=alert.id,
                unix_millis_alert_time=round(datetime.now().timestamp() * 1000),
                title=alert.title,
                subtitle=alert.subtitle,
                message=alert.message,
                display=alert.display,
                tier=tier,
                responses_available=list(alert.responses_available),
                alert_parameters=[
                    AlertParameter(name=p.name, value=p.value)
                    for p in alert.alert_parameters
                ],
                task_id=alert.task_id or None,
            )

        def handle_alert(alert: AlertRequest):
            async def create_alert(a: AlertRequest):
                if self._closing:
                    return
                try:
                    created = await self._alert_repo.create_new_alert(a)
                except Exception as e:
                    self._logger.error("%s", e)
                    return

                self._alert_events.alert_requests.on_next(created)

            self._spawn(create_alert(alert))

        self._subscriptions.append(
            self._ros_node.create_subscription(
                RmfAlert,
                "alert",
                lambda msg: handle_alert(convert_alert(msg)),
                rclpy.qos.QoSProfile(
                    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                    depth=10,
                    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
        )

        def handle_alert_response(msg):
            async def create_response():
                if self._closing:
                    return
                try:
                    created = await self._alert_repo.create_response(
                        msg.id, msg.response
                    )
                except Exception as e:
                    self._logger.error("%s", e)
                    return

                self._alert_events.alert_responses.on_next(created)

            self._spawn(create_response())

        self._subscriptions.append(
            self._ros_node.create_subscription(
                RmfAlertResponse,
                "alert_response",
                lambda msg: handle_alert_response(msg),
                rclpy.qos.QoSProfile(
                    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                    depth=10,
                    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
        )

        def handle_fire_alarm_trigger(msg):
            msg = cast(BoolMsg, msg)

            state = FireAlarmTriggerState(
                unix_millis_time=round(datetime.now().timestamp() * 1000),
                trigger=msg.data,
            )

            self._rmf_events.fire_alarm_trigger.on_next(state)

        self._subscriptions.append(
            self._ros_node.create_subscription(
                BoolMsg,
                "fire_alarm_trigger",
                handle_fire_alarm_trigger,
                10,
            )
        )

    # =========================================================
    # public API
    # =========================================================

    def request_door(self, door_name: str, mode: int) -> None:
        msg = RmfDoorRequest(
            door_name=door_name,
            request_time=self._ros_node.get_clock().now().to_msg(),
            requester_id=self._ros_node.get_name(),
            requested_mode=RmfDoorMode(value=mode),
        )
        self._door_req.publish(msg)

    def request_lift(self, lift_name, destination, request_type, door_mode, additional):
        msg = RmfLiftRequest(
            lift_name=lift_name,
            request_time=self._ros_node.get_clock().now().to_msg(),
            session_id=self._ros_node.get_name(),
            request_type=request_type,
            destination_floor=destination,
            door_state=door_mode,
        )

        self._adapter_lift_req.publish(msg)

        for sid in additional:
            msg.session_id = sid
            self._adapter_lift_req.publish(msg)

    def respond_to_delivery_alert(self, alert_id, category, tier, task_id, action, message):
        msg = RmfDeliveryAlert()
        msg.id = alert_id
        msg.category = RmfDeliveryAlertCategory(value=category)
        msg.tier = RmfDeliveryAlertTier(value=tier)
        msg.task_id = task_id
        msg.action = RmfDeliveryAlertAction(value=action)
        msg.message = message
        self._delivery_alert_response.publish(msg)

    def respond_to_alert(self, alert_id: str, response: str):
        msg = RmfAlertResponse()
        msg.id = alert_id
        msg.response = response
        self._alert_response.publish(msg)

    def manual_release_mutex_groups(self, mutex_groups, fleet, robot):
        msg = RmfMutexGroupManualRelease()
        msg.release_mutex_groups = mutex_groups
        msg.fleet = fleet
        msg.robot = robot
        self._mutex_group_release.publish(msg)

    def reset_fire_alarm_trigger(self):
        msg = BoolMsg()
        msg.data = False
        self._fire_alarm_trigger.publish(msg)


@singleton_dep
def get_rmf_gateway():
    return RmfGateway(
        get_cached_file_repo(),
        get_ros_node(),
        get_alert_events(),
        AlertRepository(),
        get_rmf_events(),
        RmfRepository(User.get_system_user()),
        asyncio.get_event_loop(),
    )