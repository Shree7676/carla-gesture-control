import carla
from app.expression import ExpressionDetector
import logging
# import pygame


class Hero(object):
    def __init__(self):
        self.world = None
        self.actor = None
        self.control = None
        self.expression_detector = ExpressionDetector()
        self.logger = logging.getLogger(__name__)

    def start(self, world):
        self.world = world
        self.actor = self.world.spawn_hero(blueprint_filter=world.args.filter)
        self.actor.set_autopilot(False, world.args.tm_port)
        self.logger.info(f"Hero vehicle spawned: {self.actor.id}")

    def tick(self, clock):
        self.expression_detector.update()
        ctrl = carla.VehicleControl()

        if self.expression_detector.is_mouth_open():  # Brake
            self.logger.info("Gesture detected: Mouth open - Braking")
            ctrl.brake = 0.9
        else:
            if self.expression_detector.is_palm_open(): # Go
                self.logger.info("Gesture detected: Palm open - Moving forward")
                ctrl.reverse = False
                ctrl.throttle = 0.7
            if self.expression_detector.is_index_finger_raised(): # Reverse
                self.logger.info("Gesture detected: Index finger - Reversing")
                ctrl.reverse = True
                ctrl.throttle = 0.5
            if self.expression_detector.is_head_tilted_left(): # Steer left
                self.logger.info("Gesture detected: Head tilt left - Steering left")
                ctrl.steer = 0.5
            if self.expression_detector.is_head_tilted_right():  # Steer right
                self.logger.info("Gesture detected: Head tilt right - Steering right")
                ctrl.steer = -0.5

        self.actor.apply_control(ctrl)

    def destroy(self):
        """Destroy the hero actor when class instance is destroyed"""
        if self.actor is not None:
            self.actor.destroy()
            self.logger.info("Hero vehicle destroyed")
