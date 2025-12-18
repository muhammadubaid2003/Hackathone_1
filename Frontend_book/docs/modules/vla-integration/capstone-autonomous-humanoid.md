---
id: capstone-autonomous-humanoid
title: "Capstone: The Autonomous Humanoid"
sidebar_label: "Capstone: The Autonomous Humanoid"
sidebar_position: 3
---

# Capstone: The Autonomous Humanoid

## Introduction to the Complete Autonomous System

The autonomous humanoid represents the convergence of multiple sophisticated technologies: vision systems for perceiving the environment, large language models for understanding and reasoning about goals, and action systems for executing physical tasks. This capstone chapter integrates all components covered in the previous chapters to create a complete end-to-end autonomous humanoid system.

### System Architecture Overview

The complete autonomous humanoid system integrates the following components:

1. **Voice Command Interface**: Processing natural language commands
2. **Cognitive Planning**: Translating goals into action sequences
3. **Navigation System**: Moving the humanoid through environments
4. **Perception System**: Understanding the environment and objects
5. **Manipulation System**: Interacting with objects and environment
6. **Action Execution**: Coordinating all subsystems for task completion

### Design Philosophy

The autonomous humanoid follows these key design principles:

- **Modularity**: Components operate independently but communicate through standard interfaces
- **Robustness**: Failsafe mechanisms and error recovery procedures
- **Adaptability**: Ability to adjust behavior based on context and feedback
- **Safety**: Built-in safety checks and human oversight capabilities
- **Scalability**: Designed to accommodate future enhancements and capabilities

## End-to-End System Architecture

### System Flow

The complete system follows this flow:

```
Voice Command → Speech Recognition → Natural Language Processing → Cognitive Planning →
Task Execution → Navigation → Perception → Manipulation → Action Completion
```

### High-Level Architecture

```python
# Example: High-level autonomous humanoid system architecture
import asyncio
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, GoalResponse
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
import queue

class SystemState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTING = "executing"
    RECOVERING = "recovering"
    SAFETY_STOP = "safety_stop"

@dataclass
class SystemContext:
    """Represents the current state of the system"""
    state: SystemState
    robot_position: Dict[str, float]
    battery_level: float
    available_objects: List[str]
    environmental_conditions: Dict[str, Any]
    safety_constraints: List[str]
    recent_interactions: List[Dict[str, Any]]

class AutonomousHumanoid(Node):
    def __init__(self):
        super().__init__('autonomous_humanoid')

        # Initialize all subsystems
        self.voice_processor = WhisperVoiceProcessor(model_size="base", device="cuda")
        self.goal_parser = LLMGoalParser(model_name="gpt-3.5-turbo")
        self.task_decomposer = TaskDecomposer(model_name="gpt-3.5-turbo")
        self.context_planner = ContextAwarePlanner(model_name="gpt-3.5-turbo")
        self.plan_validator = PlanValidator()
        self.ros2_planner = LLMROS2Planner()

        # Publishers and subscribers
        self.voice_command_sub = self.create_subscription(String, '/voice_commands', self.voice_command_callback, 10)
        self.state_publisher = self.create_publisher(String, '/system_state', 10)
        self.feedback_publisher = self.create_publisher(String, '/system_feedback', 10)

        # Action clients for complex tasks
        self.nav_action_client = ActionClient(self, NavigateAction, 'navigate_to_pose')
        self.manip_action_client = ActionClient(self, ManipulateAction, 'manipulation_server')

        # System queues
        self.command_queue = queue.Queue()
        self.state_queue = queue.Queue()

        # System state
        self.current_state = SystemState.IDLE
        self.active_plan = None
        self.current_context = SystemContext(
            state=SystemState.IDLE,
            robot_position={"x": 0.0, "y": 0.0, "z": 0.0},
            battery_level=100.0,
            available_objects=[],
            environmental_conditions={},
            safety_constraints=[],
            recent_interactions=[]
        )

        # Async processing
        self.processing_task = None

        # Start system monitoring
        self.system_monitor = self.create_timer(1.0, self.monitor_system_state)

        self.get_logger().info("Autonomous Humanoid System initialized")

    def voice_command_callback(self, msg: String):
        """Handle incoming voice commands"""
        command_text = msg.data
        self.get_logger().info(f"Received voice command: {command_text}")

        # Add to processing queue
        self.command_queue.put(command_text)

        # Start processing if not already running
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self.process_command_queue())

    async def process_command_queue(self):
        """Process commands from the queue asynchronously"""
        while not self.command_queue.empty():
            command = self.command_queue.get_nowait()

            try:
                # Process the command end-to-end
                await self.process_command_end_to_end(command)
            except Exception as e:
                self.get_logger().error(f"Error processing command {command}: {e}")
                self.handle_error(e, command)

    async def process_command_end_to_end(self, command: str):
        """Process a command from voice to action completion"""
        self.get_logger().info(f"Processing command end-to-end: {command}")

        # Update system state
        self.update_state(SystemState.PROCESSING)

        try:
            # 1. Get current context
            context = await self.get_current_context()

            # 2. Parse the natural language goal
            self.get_logger().info("Parsing natural language goal")
            parsed_goal = self.goal_parser.parse_goal(command)

            # 3. Decompose task
            self.get_logger().info("Decomposing task into subtasks")
            subtasks = self.task_decomposer.decompose_task(parsed_goal)

            # 4. Create context-aware plan
            self.get_logger().info("Creating context-aware plan")
            plan = self.context_planner.plan_with_context(command, context)

            # 5. Validate plan
            self.get_logger().info("Validating plan for safety and feasibility")
            is_valid, safety_violations, feasibility_issues = self.plan_validator.validate_plan(
                plan, context.get('robot_capabilities', {})
            )

            if not is_valid:
                self.get_logger().warn(f"Plan validation issues: {safety_violations + feasibility_issues}")
                # Implement recovery or request clarification

            # 6. Execute plan
            self.get_logger().info("Executing plan")
            self.update_state(SystemState.EXECUTING)
            await self.execute_plan(plan)

            # 7. Complete successfully
            self.get_logger().info("Command execution completed successfully")
            self.publish_feedback(f"Successfully completed: {command}")
            self.update_state(SystemState.IDLE)

        except Exception as e:
            self.get_logger().error(f"Error in end-to-end processing: {e}")
            self.handle_error(e, command)

    async def get_current_context(self) -> Dict[str, Any]:
        """Get current system context"""
        # In a real system, this would query multiple sources
        # - Robot state (position, battery, etc.)
        # - Environment sensors
        # - Object detection systems
        # - Navigation maps

        context = {
            "environment": "home_environment",
            "available_objects": self.current_context.available_objects,
            "robot_state": {
                "position": self.current_context.robot_position,
                "battery": self.current_context.battery_level,
                "payload": 0.0  # Current payload weight
            },
            "robot_capabilities": {
                "navigation": {"max_speed": 0.5, "precision": "centimeter"},
                "manipulation": {"max_payload": 2.0, "precision": "millimeter"},
                "perception": {"range": 5.0, "resolution": "high"}
            },
            "time_of_day": self.get_current_time_of_day(),
            "safety_constraints": self.current_context.safety_constraints,
            "recent_interactions": self.current_context.recent_interactions
        }

        return context

    async def execute_plan(self, plan: Dict[str, Any]):
        """Execute a planned sequence of actions"""
        self.active_plan = plan

        for step in plan['plan']:
            if self.current_state == SystemState.SAFETY_STOP:
                self.get_logger().warn("Safety stop engaged, stopping execution")
                break

            await self.execute_plan_step(step)

    async def execute_plan_step(self, step: Dict[str, Any]):
        """Execute a single step of the plan"""
        action_type = step.get('action', '').lower()
        parameters = step.get('parameters', {})

        self.get_logger().info(f"Executing step: {action_type} with parameters: {parameters}")

        try:
            if action_type == 'navigate':
                await self.execute_navigation(parameters)
            elif action_type == 'manipulate':
                await self.execute_manipulation(parameters)
            elif action_type == 'perceive':
                await self.execute_perception(parameters)
            elif action_type == 'communicate':
                await self.execute_communication(parameters)
            else:
                self.get_logger().warn(f"Unknown action type: {action_type}")

        except Exception as e:
            self.get_logger().error(f"Error executing step {action_type}: {e}")
            self.handle_error(e, f"{action_type} with params {parameters}")

    async def execute_navigation(self, params: Dict[str, Any]):
        """Execute navigation action"""
        # Create navigation goal
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'

        goal.pose.position.x = params.get('x', 0.0)
        goal.pose.position.y = params.get('y', 0.0)
        goal.pose.position.z = params.get('z', 0.0)

        # Set orientation (simplified - would need proper quaternion calculation)
        goal.pose.orientation.w = 1.0

        # Publish navigation command
        self.nav_publisher.publish(goal)

        self.get_logger().info(f"Navigating to: ({goal.pose.position.x}, {goal.pose.position.y})")

    async def execute_manipulation(self, params: Dict[str, Any]):
        """Execute manipulation action"""
        command = String()
        command.data = json.dumps({
            'action': params.get('action', 'unknown'),
            'object': params.get('target_object', 'unknown'),
            'position': params.get('position', [0, 0, 0]),
            'gripper_position': params.get('gripper_position', 'open')
        })

        self.manipulation_publisher.publish(command)
        self.get_logger().info(f"Manipulation command: {command.data}")

    async def execute_perception(self, params: Dict[str, Any]):
        """Execute perception action"""
        request = String()
        request.data = json.dumps({
            'task': params.get('task', 'detect'),
            'target': params.get('target_object', 'any'),
            'location': params.get('location', 'current')
        })

        self.perception_publisher.publish(request)
        self.get_logger().info(f"Perception request: {request.data}")

    async def execute_communication(self, params: Dict[str, Any]):
        """Execute communication action"""
        command = String()
        command.data = params.get('text', 'Hello')

        self.communication_publisher.publish(command)
        self.get_logger().info(f"Communication: {command.data}")

    def update_state(self, new_state: SystemState):
        """Update the system state"""
        old_state = self.current_state
        self.current_state = new_state

        # Log state changes
        self.get_logger().info(f"State changed from {old_state.value} to {new_state.value}")

        # Publish state update
        state_msg = String()
        state_msg.data = json.dumps({
            "old_state": old_state.value,
            "new_state": new_state.value,
            "timestamp": self.get_clock().now().to_msg().sec
        })
        self.state_publisher.publish(state_msg)

    def publish_feedback(self, message: str):
        """Publish system feedback"""
        feedback_msg = String()
        feedback_msg.data = json.dumps({
            "message": message,
            "timestamp": self.get_clock().now().to_msg().sec,
            "state": self.current_state.value
        })
        self.feedback_publisher.publish(feedback_msg)

    def get_current_time_of_day(self) -> str:
        """Get current time of day"""
        import datetime
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def monitor_system_state(self):
        """Monitor system state for safety and performance"""
        # Check battery level
        if self.current_context.battery_level < 10:
            self.get_logger().warn("Battery level critically low!")
            self.update_state(SystemState.SAFETY_STOP)

        # Check for safety constraints
        # In a real system, this would check sensor data, proximity alerts, etc.

        # Update state publisher periodically
        self.publish_system_state()

    def handle_error(self, error: Exception, context: str = ""):
        """Handle errors in the system"""
        self.get_logger().error(f"Error in context '{context}': {error}")
        self.update_state(SystemState.RECOVERING)

        # Publish error information
        error_msg = String()
        error_msg.data = json.dumps({
            "error": str(error),
            "context": context,
            "type": type(error).__name__,
            "timestamp": self.get_clock().now().to_msg().sec
        })
        self.feedback_publisher.publish(error_msg)

        # Attempt recovery based on error type
        self.attempt_recovery(error)

    def attempt_recovery(self, error: Exception):
        """Attempt to recover from errors"""
        error_type = type(error).__name__

        if error_type == "NavigationFailure":
            # Retry navigation with different parameters
            self.get_logger().info("Attempting navigation recovery")
        elif error_type == "ManipulationFailure":
            # Retry manipulation or find alternative approach
            self.get_logger().info("Attempting manipulation recovery")
        elif error_type == "SafetyViolation":
            # Stop execution and alert human operator
            self.update_state(SystemState.SAFETY_STOP)
            self.get_logger().warn("Safety violation - system stopped")
        else:
            # General error handling
            self.get_logger().info("Attempting general error recovery")

        # Return to idle state after recovery attempt
        self.update_state(SystemState.IDLE)

    def speak_response(self, text: str):
        """Simulate speaking response"""
        # In a real system, this would use TTS
        self.get_logger().info(f"Speaking: {text}")

    def publish_system_state(self):
        """Publish current system state"""
        state_msg = String()
        state_msg.data = json.dumps({
            "state": self.current_state.value,
            "robot_position": self.current_context.robot_position,
            "battery_level": self.current_context.battery_level,
            "timestamp": self.get_clock().now().to_msg().sec
        })
        self.state_publisher.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)

    humanoid = AutonomousHumanoid()

    try:
        rclpy.spin(humanoid)
    except KeyboardInterrupt:
        pass
    finally:
        humanoid.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Voice Command Processing Integration

### Complete Voice Pipeline

```python
# Example: Complete voice command processing integration
class VoiceCommandProcessor:
    def __init__(self, humanoid_system: AutonomousHumanoid):
        self.humanoid = humanoid_system
        self.voice_processor = WhisperVoiceProcessor(model_size="base", device="cuda")
        self.command_parser = CommandParser()
        self.pipeline_monitor = VoiceSystemMonitor()

    def process_voice_input(self, audio_data: bytes) -> Optional[str]:
        """
        Process voice input from audio data to structured command

        Args:
            audio_data: Raw audio data from microphones

        Returns:
            Structured command text or None if processing fails
        """
        start_time = time.time()

        try:
            # Preprocess audio data
            audio_array = self.preprocess_audio(audio_data)

            # Transcribe using Whisper
            transcription = self.voice_processor.transcribe_audio(audio_array)
            text = transcription.get("text", "").strip()

            if not text:
                return None

            self.humanoid.get_logger().info(f"Transcribed: {text}")

            # Parse command using our parser
            intent = self.command_parser.parse(text)

            if intent:
                # Validate intent
                is_valid, errors = self.humanoid.plan_validator.validate_intent(
                    intent, self.humanoid.current_context.__dict__
                )

                if is_valid:
                    self.pipeline_monitor.record_response_time(time.time() - start_time)
                    return str(intent)
                else:
                    self.humanoid.get_logger().warn(f"Invalid intent: {errors}")
                    return None
            else:
                self.humanoid.get_logger().warn(f"Could not parse command: {text}")
                return None

        except Exception as e:
            self.humanoid.get_logger().error(f"Error processing voice input: {e}")
            return None
        finally:
            self.pipeline_monitor.record_response_time(time.time() - start_time)

    def preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Preprocess audio data for Whisper"""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Normalize
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))

        return audio_array
```

## Planning and Task Execution Integration

### Coordinated Planning and Execution

```python
# Example: Coordinated planning and execution
class CoordinatedPlanner:
    def __init__(self, humanoid_system: AutonomousHumanoid):
        self.humanoid = humanoid_system
        self.llm_planner = LLMPlanningSystem()
        self.executor = ActionExecutor(humanoid_system)

    async def plan_and_execute(self, goal: str, context: Dict[str, Any]) -> PlanExecutionResult:
        """
        Plan and execute a goal with coordination between all subsystems

        Args:
            goal: Natural language goal to achieve
            context: Current context information

        Returns:
            Result of plan execution
        """
        # Create detailed plan using LLM
        self.humanoid.get_logger().info(f"Creating plan for goal: {goal}")

        plan_result = await self.llm_planner.create_and_execute_plan(goal, context)

        if plan_result.success:
            # Execute the plan with coordinated subsystems
            self.humanoid.get_logger().info("Executing coordinated plan")

            execution_result = await self.execute_coordinated_plan(plan_result, context)

            return execution_result
        else:
            return plan_result

    async def execute_coordinated_plan(self, plan_result: PlanExecutionResult, context: Dict[str, Any]) -> PlanExecutionResult:
        """Execute a plan coordinating all subsystems"""
        try:
            for step in plan_result.execution_log:  # Using the execution log as plan steps
                if isinstance(step, dict) and 'action' in step:
                    action_type = step['action']

                    # Coordinate with perception before navigation/manipulation
                    if action_type in ['navigate', 'manipulate']:
                        await self.coordinate_perception(action_type, step)

                    # Execute the action
                    await self.executor.execute_action(action_type, step.get('parameters', {}))

                    # Update system state after each action
                    await self.update_system_state_after_action(action_type, step)

            return PlanExecutionResult(
                status=PlanStatus.COMPLETED,
                success=True,
                message="Plan executed successfully with coordinated subsystems",
                execution_log=plan_result.execution_log
            )

        except Exception as e:
            self.humanoid.get_logger().error(f"Error in coordinated execution: {e}")
            return PlanExecutionResult(
                status=PlanStatus.FAILED,
                success=False,
                message=f"Coordinated execution failed: {str(e)}",
                execution_log=plan_result.execution_log,
                error_details={"exception": str(e), "type": type(e).__name__}
            )

    async def coordinate_perception(self, action_type: str, step: Dict[str, Any]):
        """Coordinate perception with planned actions"""
        # Before navigation, perceive the destination area
        if action_type == 'navigate':
            destination = step.get('parameters', {}).get('target_location')
            if destination:
                # Request perception system to scan destination area
                self.humanoid.get_logger().info(f"Scanning destination area: {destination}")
                await self.humanoid.execute_perception({
                    'task': 'scan_area',
                    'target': destination,
                    'context': 'navigation'
                })

        # Before manipulation, perceive the target object
        elif action_type == 'manipulate':
            target_obj = step.get('parameters', {}).get('target_object')
            if target_obj:
                # Request perception system to locate object
                self.humanoid.get_logger().info(f"Locating object: {target_obj}")
                await self.humanoid.execute_perception({
                    'task': 'locate_object',
                    'target': target_obj,
                    'context': 'manipulation'
                })

    async def update_system_state_after_action(self, action_type: str, step: Dict[str, Any]):
        """Update system state after action execution"""
        # Update robot position after navigation
        if action_type == 'navigate':
            new_position = step.get('parameters', {}).get('destination')
            if new_position:
                self.humanoid.current_context.robot_position = new_position

        # Update available objects after manipulation
        elif action_type == 'manipulate':
            target_obj = step.get('parameters', {}).get('target_object')
            if target_obj and target_obj in self.humanoid.current_context.available_objects:
                # Object may have been moved or manipulated
                pass  # Update object locations in system context

class ActionExecutor:
    def __init__(self, humanoid_system: AutonomousHumanoid):
        self.humanoid = humanoid_system

    async def execute_action(self, action_type: str, parameters: Dict[str, Any]):
        """Execute a specific action type"""
        if action_type == 'navigate':
            await self.humanoid.execute_navigation(parameters)
        elif action_type == 'manipulate':
            await self.humanoid.execute_manipulation(parameters)
        elif action_type == 'perceive':
            await self.humanoid.execute_perception(parameters)
        elif action_type == 'communicate':
            await self.humanoid.execute_communication(parameters)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
```

## Navigation System Integration

### Safe Navigation with Perception Integration

```python
# Example: Navigation system with perception integration
class SafeNavigationSystem:
    def __init__(self, humanoid_system: AutonomousHumanoid):
        self.humanoid = humanoid_system
        self.perception_client = PerceptionClient(humanoid_system)
        self.path_planner = ROS2PathPlanner(humanoid_system)

    async def navigate_with_perception(self, target_location: Dict[str, float], safety_constraints: List[str]) -> bool:
        """
        Navigate to target location with real-time perception integration

        Args:
            target_location: Destination coordinates
            safety_constraints: Safety constraints to observe

        Returns:
            True if navigation completed successfully, False otherwise
        """
        try:
            # Pre-navigate perception: scan path before moving
            path_scan = await self.perception_client.scan_path_to_target(target_location)

            # Plan safe path considering obstacles
            safe_path = await self.path_planner.plan_safe_path(target_location, path_scan)

            # Monitor environment during navigation
            navigation_task = asyncio.create_task(self.execute_navigation(safe_path, safety_constraints))
            monitoring_task = asyncio.create_task(self.monitor_environment_during_navigation())

            # Wait for navigation to complete
            await navigation_task

            # Cancel monitoring task
            monitoring_task.cancel()

            return True

        except Exception as e:
            self.humanoid.get_logger().error(f"Navigation failed: {e}")
            return False

    async def execute_navigation(self, path: List[Dict[str, float]], safety_constraints: List[str]):
        """Execute navigation along a planned path"""
        for waypoint in path:
            # Check safety constraints at each waypoint
            if not await self.check_safety_constraints(safety_constraints):
                raise Exception("Safety constraint violated during navigation")

            # Move to waypoint
            await self.move_to_waypoint(waypoint)

            # Verify position after movement
            await self.verify_position(waypoint)

    async def monitor_environment_during_navigation(self):
        """Monitor environment for dynamic obstacles during navigation"""
        while self.humanoid.current_state == SystemState.EXECUTING:
            try:
                # Scan for dynamic obstacles
                obstacles = await self.perception_client.scan_for_dynamic_obstacles()

                if obstacles:
                    # Check if obstacles interfere with navigation path
                    imminent_threat = self.check_if_obstacles_on_path(obstacles)

                    if imminent_threat:
                        self.humanoid.get_logger().warn("Dynamic obstacle detected on path, pausing navigation")

                        # Pause navigation and replan if necessary
                        await self.handle_dynamic_obstacle(obstacles)

                # Wait before next scan
                await asyncio.sleep(0.5)  # Scan every 500ms

            except Exception as e:
                self.humanoid.get_logger().error(f"Error in environment monitoring: {e}")
                break

    async def check_safety_constraints(self, constraints: List[str]) -> bool:
        """Check if safety constraints are satisfied"""
        for constraint in constraints:
            if constraint == "avoid_humans":
                humans_nearby = await self.perception_client.detect_humans_in_proximity()
                if humans_nearby:
                    return False
            elif constraint == "maintain_safe_distance":
                nearby_objects = await self.perception_client.get_objects_in_proximity(0.5)  # 50cm threshold
                if nearby_objects:
                    return False
            # Add more safety constraints as needed

        return True

    async def move_to_waypoint(self, waypoint: Dict[str, float]):
        """Move robot to a specific waypoint"""
        # In a real system, this would use ROS 2 navigation stack
        # For this example, we'll simulate
        import random
        movement_time = random.uniform(1.0, 3.0)
        await asyncio.sleep(movement_time)

    async def verify_position(self, expected_position: Dict[str, float]):
        """Verify that robot is at expected position"""
        # In a real system, this would use localization data
        # For this example, we'll assume successful
        pass

    def check_if_obstacles_on_path(self, obstacles: List[Dict[str, Any]]) -> bool:
        """Check if obstacles are on the current navigation path"""
        # Implement path-geometry intersection checking
        # For this example, return True if any obstacle is close to current path
        return len(obstacles) > 0

    async def handle_dynamic_obstacle(self, obstacles: List[Dict[str, Any]]):
        """Handle dynamic obstacle by pausing/replanning"""
        # Pause current navigation
        self.humanoid.get_logger().info("Pausing navigation for obstacle handling")

        # Decide whether to wait or replan based on obstacle characteristics
        for obstacle in obstacles:
            if obstacle['type'] == 'moving_human':
                # Wait for human to pass
                await asyncio.sleep(2.0)
            elif obstacle['type'] == 'stationary_object':
                # Replan around obstacle
                await self.replan_around_obstacle(obstacle)

    async def replan_around_obstacle(self, obstacle: Dict[str, Any]):
        """Replan navigation path around an obstacle"""
        self.humanoid.get_logger().info(f"Replanning around obstacle: {obstacle}")
        # Implement replanning logic
        pass

class PerceptionClient:
    def __init__(self, humanoid_system: AutonomousHumanoid):
        self.humanoid = humanoid_system

    async def scan_path_to_target(self, target_location: Dict[str, float]) -> Dict[str, Any]:
        """Scan the path to the target location"""
        # In a real system, this would activate perception sensors
        # For this example, we'll return mock data
        return {
            "obstacles": [],
            "free_space": True,
            "navigation_difficulty": "easy"
        }

    async def scan_for_dynamic_obstacles(self) -> List[Dict[str, Any]]:
        """Scan for dynamic obstacles in the environment"""
        # In a real system, this would use real-time perception data
        # For this example, return mock data
        import random
        if random.random() < 0.3:  # 30% chance of detecting an obstacle
            return [{
                "type": "moving_human",
                "position": {"x": 2.0, "y": 1.5, "z": 0.0},
                "velocity": {"x": 0.3, "y": 0.0, "z": 0.0},
                "timestamp": time.time()
            }]
        return []

    async def detect_humans_in_proximity(self) -> bool:
        """Detect humans in proximity to robot"""
        # In a real system, this would use person detection
        # For this example, return mock data
        return False

    async def get_objects_in_proximity(self, radius: float) -> List[Dict[str, Any]]:
        """Get objects within specified radius of robot"""
        # In a real system, this would use object detection and localization
        # For this example, return mock data
        return []
```

## Manipulation System Integration

### Perception-Guided Manipulation

```python
# Example: Perception-guided manipulation system
class PerceptionGuidedManipulation:
    def __init__(self, humanoid_system: AutonomousHumanoid):
        self.humanoid = humanoid_system
        self.perception_client = PerceptionClient(humanoid_system)
        self.manipulation_controller = ManipulationController(humanoid_system)

    async def execute_manipulation_with_perception(self, manipulation_goal: Dict[str, Any]) -> bool:
        """
        Execute manipulation with real-time perception feedback

        Args:
            manipulation_goal: Details of the manipulation to perform

        Returns:
            True if manipulation completed successfully, False otherwise
        """
        try:
            target_object = manipulation_goal.get('target_object')
            action = manipulation_goal.get('action')  # grasp, place, move, etc.

            if not target_object or not action:
                raise ValueError("Missing required manipulation parameters")

            # Locate the target object using perception
            object_pose = await self.locate_object(target_object)
            if not object_pose:
                self.humanoid.get_logger().error(f"Could not locate object: {target_object}")
                return False

            # Verify object properties and safety
            object_properties = await self.analyze_object_properties(target_object, object_pose)
            if not await self.verify_manipulation_safety(object_properties, action):
                self.humanoid.get_logger().warn("Safety check failed for manipulation")
                return False

            # Execute manipulation with continuous perception feedback
            success = await self.manipulation_controller.execute_manipulation_with_feedback(
                manipulation_goal, object_pose, self.perception_callback
            )

            if success:
                self.humanoid.get_logger().info(f"Manipulation completed: {action} {target_object}")
            else:
                self.humanoid.get_logger().error(f"Manipulation failed: {action} {target_object}")

            return success

        except Exception as e:
            self.humanoid.get_logger().error(f"Manipulation error: {e}")
            return False

    async def locate_object(self, target_object: str) -> Optional[Dict[str, Any]]:
        """Locate target object using perception system"""
        self.humanoid.get_logger().info(f"Locating object: {target_object}")

        # Use perception to detect and localize the object
        detection_result = await self.perception_client.locate_object(target_object)

        if detection_result and 'pose' in detection_result:
            return detection_result['pose']
        else:
            self.humanoid.get_logger().warn(f"Object {target_object} not found")
            return None

    async def analyze_object_properties(self, target_object: str, object_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze properties of target object"""
        self.humanoid.get_logger().info(f"Analyzing properties of {target_object}")

        # Get detailed information about the object
        properties = await self.perception_client.analyze_object(target_object, object_pose)

        return {
            "object_name": target_object,
            "pose": object_pose,
            "dimensions": properties.get('dimensions', {}),
            "weight": properties.get('weight', 0.0),
            "material": properties.get('material', 'unknown'),
            "fragility": properties.get('fragility', 'normal'),
            "grasp_points": properties.get('grasp_points', []),
            "orientation": properties.get('orientation', {})
        }

    async def verify_manipulation_safety(self, object_properties: Dict[str, Any], action: str) -> bool:
        """Verify that manipulation action is safe to perform"""
        # Check if object is too heavy for robot's manipulator
        max_payload = self.humanoid.current_context.robot_capabilities.get('manipulation', {}).get('max_payload', 2.0)
        object_weight = object_properties.get('weight', 0.0)

        if object_weight > max_payload:
            self.humanoid.get_logger().warn(f"Object too heavy: {object_weight}kg > {max_payload}kg")
            return False

        # Check if object is fragile and action is aggressive
        fragility = object_properties.get('fragility', 'normal')
        if fragility == 'fragile' and action in ['grasp_forcefully', 'drop']:
            self.humanoid.get_logger().warn(f"Fragile object should not be {action}ed")
            return False

        # Check if grasp points are accessible
        grasp_points = object_properties.get('grasp_points', [])
        if not grasp_points and action in ['grasp', 'lift']:
            self.humanoid.get_logger().warn("No grasp points identified for object")
            return False

        return True

    async def perception_callback(self, feedback: Dict[str, Any]):
        """Callback function for real-time perception feedback during manipulation"""
        # This would be called continuously during manipulation
        # to provide real-time feedback and adjust actions as needed
        pass
```

## Performance Monitoring and Optimization

### System Performance Tracking

```python
# Example: Performance monitoring for the autonomous humanoid system
import time
import statistics
from collections import defaultdict, deque
import psutil

class AutonomousHumanoidMonitor:
    def __init__(self):
        self.metrics = defaultdict(lambda: deque(maxlen=100))  # Keep last 100 values
        self.start_times = {}
        self.component_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.event_log = deque(maxlen=1000)  # Keep last 1000 events

    def start_component_timer(self, component_name: str):
        """Start timing a specific component"""
        self.start_times[component_name] = time.time()

    def end_component_timer(self, component_name: str):
        """End timing a specific component and record performance"""
        if component_name in self.start_times:
            elapsed = time.time() - self.start_times[component_name]
            self.metrics[f"{component_name}_time"].append(elapsed)
            self.component_times[component_name].append(elapsed)
            del self.start_times[component_name]

    def record_metric(self, metric_name: str, value: float):
        """Record a specific metric value"""
        self.metrics[metric_name].append(value)

    def log_event(self, event_type: str, message: str, severity: str = "info"):
        """Log system events"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "message": message,
            "severity": severity
        }
        self.event_log.append(event)

    def get_component_statistics(self, component_name: str) -> Dict[str, float]:
        """Get statistical summary for a specific component"""
        times = self.component_times.get(component_name, [])
        if not times:
            return {}

        return {
            "count": len(times),
            "average": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        return {
            "performance_metrics": {
                "avg_voice_processing_time": statistics.mean(self.metrics.get("voice_processing_time", [0])),
                "avg_plan_generation_time": statistics.mean(self.metrics.get("plan_generation_time", [0])),
                "avg_execution_time": statistics.mean(self.metrics.get("execution_time", [0])),
            },
            "system_resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
            },
            "error_counts": dict(self.error_counts),
            "recent_events": list(self.event_log)[-10:],  # Last 10 events
            "total_events": len(self.event_log),
        }

    def is_system_healthy(self) -> bool:
        """Check if system is operating within healthy parameters"""
        health = self.get_system_health()

        # Check if any performance metrics are degraded
        perf_metrics = health["performance_metrics"]
        for metric_name, value in perf_metrics.items():
            if "time" in metric_name and value > 5.0:  # More than 5 seconds for any operation
                return False

        # Check system resources
        resources = health["system_resources"]
        if resources["cpu_percent"] > 90 or resources["memory_percent"] > 90:
            return False

        return True

    def get_performance_recommendations(self) -> List[str]:
        """Get recommendations for improving system performance"""
        recommendations = []
        health = self.get_system_health()

        # Check for performance issues
        perf_metrics = health["performance_metrics"]
        for metric_name, value in perf_metrics.items():
            if "time" in metric_name and value > 2.0:  # More than 2 seconds
                recommendations.append(f"Optimize {metric_name.replace('_time', '')} - currently averaging {value:.2f}s")

        # Check for resource issues
        resources = health["system_resources"]
        if resources["cpu_percent"] > 80:
            recommendations.append("CPU usage is high - consider optimizing algorithms or adding more resources")
        if resources["memory_percent"] > 80:
            recommendations.append("Memory usage is high - check for memory leaks")

        # Check for error patterns
        if any(count > 10 for count in health["error_counts"].values()):
            recommendations.append("High error frequency detected - investigate error sources")

        return recommendations if recommendations else ["System is performing optimally"]
```

## Safety and Error Handling

### Comprehensive Safety Framework

```python
# Example: Safety and error handling framework
from enum import Enum
import asyncio

class SafetyLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CAUTION = "caution"
    DANGER = "danger"
    EMERGENCY_STOP = "emergency_stop"

class SafetyFramework:
    def __init__(self, humanoid_system: AutonomousHumanoid):
        self.humanoid = humanoid_system
        self.safety_level = SafetyLevel.NORMAL
        self.safety_constraints = []
        self.emergency_handlers = []
        self.safety_monitors = []

    def register_safety_constraint(self, constraint_func, priority: int = 1):
        """Register a safety constraint function"""
        self.safety_constraints.append({
            "func": constraint_func,
            "priority": priority,
            "last_check": 0
        })
        # Sort by priority (higher priority checked first)
        self.safety_constraints.sort(key=lambda x: x["priority"], reverse=True)

    def register_emergency_handler(self, handler_func):
        """Register an emergency handler"""
        self.emergency_handlers.append(handler_func)

    def register_safety_monitor(self, monitor_func):
        """Register a safety monitoring function"""
        self.safety_monitors.append(monitor_func)

    async def check_safety(self) -> Tuple[SafetyLevel, List[str]]:
        """Check all safety constraints and return current safety level and violations"""
        violations = []

        for constraint in self.safety_constraints:
            try:
                is_safe, reason = await constraint["func"](self.humanoid.current_context)
                if not is_safe:
                    violations.append(reason)
                    # Update safety level based on violation severity
                    if "emergency" in reason.lower():
                        self.safety_level = SafetyLevel.EMERGENCY_STOP
                    elif "danger" in reason.lower() or "collision" in reason.lower():
                        if self.safety_level.value < SafetyLevel.DANGER.value:
                            self.safety_level = SafetyLevel.DANGER
                    elif "warning" in reason.lower():
                        if self.safety_level.value < SafetyLevel.WARNING.value:
                            self.safety_level = SafetyLevel.WARNING
            except Exception as e:
                violations.append(f"Error checking safety constraint: {e}")
                if self.safety_level.value < SafetyLevel.WARNING.value:
                    self.safety_level = SafetyLevel.WARNING

        return self.safety_level, violations

    async def enforce_safety(self, action_type: str, parameters: Dict[str, Any]) -> bool:
        """Enforce safety before executing an action"""
        safety_level, violations = await self.check_safety()

        if safety_level == SafetyLevel.EMERGENCY_STOP:
            self.humanoid.get_logger().error("EMERGENCY STOP - All operations halted")
            self.trigger_emergency_stop()
            return False

        if safety_level == SafetyLevel.DANGER:
            self.humanoid.get_logger().warn(f"DANGER level safety violations: {violations}")
            # Check if action is safe to proceed with warnings
            if not await self.is_action_safe_with_warnings(action_type, parameters, violations):
                return False

        if safety_level == SafetyLevel.CAUTION:
            self.humanoid.get_logger().warn(f"CAUTION level safety issues: {violations}")
            # Log caution but allow proceeding with action

        return True

    def trigger_emergency_stop(self):
        """Trigger emergency stop procedures"""
        self.humanoid.get_logger().error("Triggering emergency stop procedures")

        # Stop all robot motion
        self.humanoid.stop_all_motion()

        # Execute registered emergency handlers
        for handler in self.emergency_handlers:
            try:
                handler(self.humanoid)
            except Exception as e:
                self.humanoid.get_logger().error(f"Error in emergency handler: {e}")

        # Update system state
        self.humanoid.update_state(SystemState.SAFETY_STOP)

    async def is_action_safe_with_warnings(self, action_type: str, parameters: Dict[str, Any], violations: List[str]) -> bool:
        """Check if an action is safe to proceed despite warnings"""
        # Some actions might be allowed even with warnings
        if action_type in ['communicate', 'perceive']:  # Low-risk actions
            return True

        # For risky actions, check specific violation types
        dangerous_violations = [v for v in violations if any(danger in v.lower() for danger in ['collision', 'human', 'obstacle'])]
        if dangerous_violations:
            self.humanoid.get_logger().warn(f"Action {action_type} blocked due to dangerous violations: {dangerous_violations}")
            return False

        # Allow action with caution
        self.humanoid.get_logger().warn(f"Proceeding with action {action_type} despite warnings")
        return True

    def stop_all_motion(self):
        """Stop all robot motion immediately"""
        # In a real system, this would send emergency stop commands to all controllers
        self.humanoid.get_logger().info("Stopping all robot motion")
        # Publish emergency stop messages to all motion topics
        # This is a simplified example - real implementation would depend on robot controller

# Example safety constraint functions
async def check_human_proximity(context: SystemContext) -> Tuple[bool, str]:
    """Check if humans are in proximity to robot"""
    # In a real system, this would check perception data
    # For this example, we'll use mock data
    import random

    if random.random() < 0.1:  # 10% chance of human detected nearby
        return False, "Human detected in proximity - CAUTION"

    return True, ""

async def check_collision_risk(context: SystemContext) -> Tuple[bool, str]:
    """Check for collision risk based on current state"""
    # Check if robot is in collision course with obstacles
    # This would check navigation plan against obstacle data
    return True, ""  # For this example, assume no collision risk

async def check_battery_level(context: SystemContext) -> Tuple[bool, str]:
    """Check if battery level is adequate"""
    if context.battery_level < 10:
        return False, "Battery level critically low - EMERGENCY STOP"
    elif context.battery_level < 20:
        return False, "Battery level low - DANGER"
    elif context.battery_level < 30:
        return False, "Battery level warning - CAUTION"

    return True, ""
```

## Integration Testing and Validation

### Comprehensive Testing Framework

```python
# Example: Integration testing for the autonomous humanoid system
import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

class AutonomousHumanoidTests(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.humanoid = Mock(spec=AutonomousHumanoid)
        self.humanoid.current_context = SystemContext(
            state=SystemState.IDLE,
            robot_position={"x": 0.0, "y": 0.0, "z": 0.0},
            battery_level=100.0,
            available_objects=["cup", "table"],
            environmental_conditions={},
            safety_constraints=[],
            recent_interactions=[]
        )
        self.humanoid.get_logger.return_value = Mock()

        # Mock ROS 2 components
        self.humanoid.voice_processor = Mock()
        self.humanoid.goal_parser = Mock()
        self.humanoid.task_decomposer = Mock()
        self.humanoid.context_planner = Mock()
        self.humanoid.plan_validator = Mock()

        self.test_system = AutonomousHumanoidMonitor()

    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_process_voice_command_success(self, mock_sleep):
        """Test successful processing of a voice command"""
        # Setup mocks
        self.humanoid.voice_processor.transcribe_audio.return_value = {"text": "move to the kitchen"}
        self.humanoid.goal_parser.parse_goal.return_value = {"action": "navigate", "target": "kitchen"}
        self.humanoid.task_decomposer.decompose_task.return_value = [{"action": "navigate", "target": "kitchen"}]
        self.humanoid.context_planner.plan_with_context.return_value = {"plan": [{"action": "navigate", "target": "kitchen"}]}
        self.humanoid.plan_validator.validate_plan.return_value = (True, [], [])

        # Test the function
        result = asyncio.run(self.humanoid.process_command_end_to_end("move to the kitchen"))

        # Assertions
        self.humanoid.voice_processor.transcribe_audio.assert_called_once()
        self.humanoid.goal_parser.parse_goal.assert_called_once()
        self.humanoid.task_decomposer.decompose_task.assert_called_once()
        self.humanoid.context_planner.plan_with_context.assert_called_once()
        self.humanoid.plan_validator.validate_plan.assert_called_once()

    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_process_voice_command_invalid_goal(self, mock_sleep):
        """Test handling of invalid goals"""
        # Setup mocks to simulate invalid goal
        self.humanoid.goal_parser.parse_goal.return_value = None

        # Test the function
        result = asyncio.run(self.humanoid.process_command_end_to_end("invalid command"))

        # The function should handle the error gracefully
        self.humanoid.get_logger().warn.assert_called()

    def test_safe_navigation_with_obstacles(self):
        """Test navigation with obstacle detection"""
        # Create a mock perception client
        mock_perception = Mock()
        mock_perception.scan_for_dynamic_obstacles.return_value = [
            {"type": "moving_human", "position": {"x": 2.0, "y": 1.5, "z": 0.0}}
        ]

        # Create safety framework
        safety_framework = SafetyFramework(self.humanoid)

        # Register safety constraint
        safety_framework.register_safety_constraint(
            lambda ctx: (False, "Dynamic obstacle detected") if True else (True, "")
        )

        # Test safety check
        result = asyncio.run(safety_framework.check_safety())
        safety_level, violations = result

        self.assertEqual(safety_level, SafetyLevel.DANGER)
        self.assertIn("Dynamic obstacle detected", violations)

    def test_sensor_simulation_accuracy(self):
        """Test accuracy of sensor simulations"""
        # Test LiDAR simulation
        lidar_sim = UnityLidarSimulation()

        # Set up a simple environment with known obstacles
        lidar_sim.horizontal_samples = 4  # Simplified for testing
        lidar_sim.min_range = 0.1
        lidar_sim.max_range = 10.0

        # Simulate a simple environment
        # In a real test, we'd have specific objects at known positions
        # and verify that the LiDAR detects them at the expected ranges

        # Mock the raycasting to return known values
        with patch.object(lidar_sim, 'Physics.Raycast', return_value=(True, Mock(distance=2.0))):
            # This would test the LiDAR simulation accuracy
            pass

    def test_imu_noise_characteristics(self):
        """Test that IMU simulation produces realistic noise characteristics"""
        imu_sim = UnityIMUSimulation()

        # Set noise parameters
        imu_sim.noise_std_dev = 0.01
        imu_sim.bias_drift = 0.001

        # Generate multiple readings
        readings = []
        for _ in range(100):
            reading = imu_sim.ApplyNoise(1.0, 0.0, 1.0)  # nominal value of 1.0
            readings.append(reading)

        # Check that readings have expected statistical properties
        mean_reading = sum(readings) / len(readings)
        std_dev = (sum((x - mean_reading) ** 2 for x in readings) / len(readings)) ** 0.5

        # The mean should be close to the nominal value (within noise bounds)
        self.assertAlmostEqual(mean_reading, 1.0, delta=0.1)

        # The standard deviation should be close to the expected noise level
        self.assertAlmostEqual(std_dev, 0.01, delta=0.01)

class AutonomousHumanoidIntegrationTests(unittest.TestCase):
    """Integration tests for the complete autonomous humanoid system"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.mock_rclpy = Mock()
        self.mock_node = Mock()
        self.mock_node.get_logger.return_value = Mock()

        # Create a more complete test system
        self.integration_test_system = {
            'voice_processor': Mock(),
            'llm_planner': Mock(),
            'navigation_system': Mock(),
            'perception_system': Mock(),
            'manipulation_system': Mock(),
            'safety_framework': Mock()
        }

    def test_complete_autonomous_cycle(self):
        """Test a complete autonomous cycle: voice command → planning → execution"""
        # This would test the full integration of all components
        # For brevity, we'll outline the test structure:

        # 1. Simulate voice command input
        # 2. Verify it gets processed by voice processor
        # 3. Verify LLM generates appropriate plan
        # 4. Verify plan gets validated by safety framework
        # 5. Verify plan gets executed by navigation/manipulation systems
        # 6. Verify feedback is provided to user
        pass

    def test_error_recovery_scenarios(self):
        """Test system behavior in error recovery scenarios"""
        # Test what happens when navigation fails
        # Test what happens when manipulation fails
        # Test what happens when perception fails
        # Verify safety systems engage appropriately
        pass

    def test_concurrent_operations(self):
        """Test handling of concurrent operations"""
        # Test multiple simultaneous commands
        # Test interruption of ongoing tasks
        # Verify proper task prioritization
        pass

# Performance tests
class AutonomousHumanoidPerformanceTests(unittest.TestCase):
    """Performance tests for the autonomous humanoid system"""

    def test_real_time_performance(self):
        """Test that system operates in real-time constraints"""
        import time

        # Test voice processing latency
        start_time = time.time()
        # Simulate voice processing
        time.sleep(0.05)  # Simulated processing time
        end_time = time.time()

        # Should process voice commands in under 100ms
        self.assertLess(end_time - start_time, 0.1)

    def test_plan_generation_speed(self):
        """Test that plan generation meets real-time requirements"""
        import time

        # Test plan generation time
        start_time = time.time()
        # Simulate plan generation
        time.sleep(0.2)  # Simulated planning time
        end_time = time.time()

        # Should generate plans in under 500ms for interactive applications
        self.assertLess(end_time - start_time, 0.5)

if __name__ == '__main__':
    unittest.main()
```

## Practical Implementation Examples

### Complete Autonomous Humanoid System

```python
# Example: Complete autonomous humanoid system implementation
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan, Image, Imu
import asyncio
import threading
import queue
import time

class CompleteAutonomousHumanoid(Node):
    """
    Complete implementation of an autonomous humanoid system
    integrating all components from the VLA module
    """

    def __init__(self):
        super().__init__('complete_autonomous_humanoid')

        # Initialize all components
        self.voice_processor = WhisperVoiceProcessor(model_size="base", device="cuda")
        self.llm_planner = LLMPlanningSystem()
        self.safety_framework = SafetyFramework(self)
        self.monitoring_system = AutonomousHumanoidMonitor()

        # Initialize perception systems
        self.lidar_subscriber = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.camera_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Initialize control publishers
        self.cmd_publisher = self.create_publisher(String, '/robot_commands', 10)
        self.nav_publisher = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)

        # Initialize action clients
        self.nav_client = ActionClient(self, NavigateAction, 'navigate_to_pose')
        self.manip_client = ActionClient(self, ManipulateAction, 'manipulation_server')

        # System state
        self.current_state = SystemState.IDLE
        self.perception_data = {
            'lidar': None,
            'camera': None,
            'imu': None
        }
        self.robot_context = SystemContext(
            state=SystemState.IDLE,
            robot_position={"x": 0.0, "y": 0.0, "z": 0.0},
            battery_level=100.0,
            available_objects=[],
            environmental_conditions={},
            safety_constraints=[],
            recent_interactions=[]
        )

        # Command processing
        self.command_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.processing_thread.start()

        # System monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_system)

        # Register safety constraints
        self.safety_framework.register_safety_constraint(check_human_proximity)
        self.safety_framework.register_safety_constraint(check_collision_risk)
        self.safety_framework.register_safety_constraint(check_battery_level)

        self.get_logger().info("Complete Autonomous Humanoid System initialized")

    def lidar_callback(self, msg: LaserScan):
        """Handle LiDAR data"""
        self.perception_data['lidar'] = msg
        self.update_robot_context()

    def camera_callback(self, msg: Image):
        """Handle camera data"""
        self.perception_data['camera'] = msg
        self.update_robot_context()

    def imu_callback(self, msg: Imu):
        """Handle IMU data"""
        self.perception_data['imu'] = msg
        self.update_robot_context()

    def update_robot_context(self):
        """Update robot context with latest perception data"""
        # This would update the robot's understanding of its environment
        # based on the latest sensor data
        pass

    def process_voice_command(self, command: str):
        """Process a voice command through the complete pipeline"""
        self.monitoring_system.start_component_timer("total_command_processing")

        try:
            # 1. Parse the command
            self.monitoring_system.start_component_timer("parsing")
            parsed_command = self.llm_planner.parse_command(command)
            self.monitoring_system.end_component_timer("parsing")

            # 2. Generate plan
            self.monitoring_system.start_component_timer("planning")
            context = self.get_current_context()
            plan = self.llm_planner.generate_plan(parsed_command, context)
            self.monitoring_system.end_component_timer("planning")

            # 3. Validate plan with safety framework
            self.monitoring_system.start_component_timer("validation")
            is_safe, violations = self.safety_framework.check_safety()
            if is_safe != SafetyLevel.NORMAL and is_safe != SafetyLevel.WARNING:
                self.get_logger().error(f"Plan violates safety: {violations}")
                self.publish_feedback(f"Cannot execute: {violations}")
                return False
            self.monitoring_system.end_component_timer("validation")

            # 4. Execute plan
            self.monitoring_system.start_component_timer("execution")
            success = asyncio.run(self.execute_plan_with_safety(plan))
            self.monitoring_system.end_component_timer("execution")

            # 5. Update context and log
            self.update_context_after_execution(plan, success)
            self.log_interaction(command, plan, success)

            return success

        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")
            self.publish_feedback(f"Error processing command: {e}")
            return False
        finally:
            self.monitoring_system.end_component_timer("total_command_processing")

    async def execute_plan_with_safety(self, plan: Dict[str, Any]) -> bool:
        """Execute a plan with continuous safety monitoring"""
        for step in plan.get('steps', []):
            # Check safety before each step
            can_proceed = await self.safety_framework.enforce_safety(
                step.get('action', ''),
                step.get('parameters', {})
            )

            if not can_proceed:
                self.get_logger().error(f"Safety check failed for step: {step}")
                return False

            # Execute the step
            success = await self.execute_plan_step(step)
            if not success:
                self.get_logger().error(f"Failed to execute step: {step}")
                return False

        return True

    async def execute_plan_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single plan step"""
        action_type = step.get('action', '').lower()

        if action_type == 'navigate':
            return await self.execute_navigation_step(step)
        elif action_type == 'manipulate':
            return await self.execute_manipulation_step(step)
        elif action_type == 'perceive':
            return await self.execute_perception_step(step)
        elif action_type == 'communicate':
            return await self.execute_communication_step(step)
        else:
            self.get_logger().warn(f"Unknown action type: {action_type}")
            return False

    async def execute_navigation_step(self, step: Dict[str, Any]) -> bool:
        """Execute navigation step"""
        try:
            target = step.get('target', {})
            x = target.get('x', 0.0)
            y = target.get('y', 0.0)
            z = target.get('z', 0.0)

            # Create and send navigation goal
            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'
            goal.pose.position.x = x
            goal.pose.position.y = y
            goal.pose.position.z = z
            goal.pose.orientation.w = 1.0  # No rotation

            self.nav_publisher.publish(goal)

            # Wait for navigation to complete (simplified)
            await asyncio.sleep(2.0)  # Simulate navigation time

            self.get_logger().info(f"Navigated to ({x}, {y})")
            return True

        except Exception as e:
            self.get_logger().error(f"Navigation error: {e}")
            return False

    async def execute_manipulation_step(self, step: Dict[str, Any]) -> bool:
        """Execute manipulation step"""
        try:
            # In a real system, this would interface with the manipulation stack
            target_object = step.get('target_object', 'unknown')
            action = step.get('manipulation_action', 'unknown')

            self.get_logger().info(f"Performing {action} on {target_object}")

            # Simulate manipulation time
            await asyncio.sleep(1.5)

            return True

        except Exception as e:
            self.get_logger().error(f"Manipulation error: {e}")
            return False

    async def execute_perception_step(self, step: Dict[str, Any]) -> bool:
        """Execute perception step"""
        try:
            task = step.get('perception_task', 'unknown')
            target = step.get('target', 'unknown')

            self.get_logger().info(f"Performing perception task: {task} on {target}")

            # Simulate perception time
            await asyncio.sleep(0.5)

            return True

        except Exception as e:
            self.get_logger().error(f"Perception error: {e}")
            return False

    async def execute_communication_step(self, step: Dict[str, Any]) -> bool:
        """Execute communication step"""
        try:
            text = step.get('text', '')

            if text:
                self.get_logger().info(f"Communicating: {text}")
                # In a real system, this would use TTS
                self.publish_feedback(text)

            return True

        except Exception as e:
            self.get_logger().error(f"Communication error: {e}")
            return False

    def process_commands(self):
        """Process commands from the queue in a separate thread"""
        while rclpy.ok():
            try:
                if not self.command_queue.empty():
                    command = self.command_queue.get_nowait()
                    asyncio.run(self.process_voice_command(command))
                else:
                    time.sleep(0.1)  # Sleep briefly to prevent busy waiting
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f"Error in command processing thread: {e}")

    def monitor_system(self):
        """Monitor system health and performance"""
        health = self.monitoring_system.get_system_health()

        if not self.monitoring_system.is_system_healthy():
            recommendations = self.monitoring_system.get_performance_recommendations()
            for rec in recommendations:
                self.get_logger().warn(f"Performance recommendation: {rec}")

        # Log system health periodically
        self.get_logger().debug(f"System health: {health['performance_metrics']}")

    def get_current_context(self) -> Dict[str, Any]:
        """Get current system context"""
        return {
            "environment": "known",
            "available_objects": self.robot_context.available_objects,
            "robot_state": {
                "position": self.robot_context.robot_position,
                "battery": self.robot_context.battery_level,
                "payload": 0.0
            },
            "robot_capabilities": {
                "navigation": {"max_speed": 0.5, "precision": "centimeter"},
                "manipulation": {"max_payload": 2.0, "precision": "millimeter"},
                "perception": {"range": 5.0, "resolution": "high"}
            },
            "time_of_day": "day",
            "safety_constraints": self.robot_context.safety_constraints,
            "recent_interactions": self.robot_context.recent_interactions,
            "perception_data": self.perception_data
        }

    def update_context_after_execution(self, plan: Dict[str, Any], success: bool):
        """Update context after plan execution"""
        # Update robot position based on navigation results
        # Update available objects based on manipulation results
        # Update system state
        pass

    def log_interaction(self, command: str, plan: Dict[str, Any], success: bool):
        """Log interaction for learning and improvement"""
        interaction = {
            "timestamp": time.time(),
            "command": command,
            "plan": plan,
            "success": success,
            "context": self.get_current_context()
        }

        # In a real system, this would be stored in a database for analysis
        self.get_logger().info(f"Interaction logged: {interaction['success']}")

    def publish_feedback(self, message: str):
        """Publish feedback to user"""
        feedback_msg = String()
        feedback_msg.data = message
        self.cmd_publisher.publish(feedback_msg)

def main(args=None):
    """Main function to run the complete autonomous humanoid system"""
    rclpy.init(args=args)

    humanoid = CompleteAutonomousHumanoid()

    try:
        rclpy.spin(humanoid)
    except KeyboardInterrupt:
        humanoid.get_logger().info("Shutting down autonomous humanoid system")
    finally:
        humanoid.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary and Cross-References

The autonomous humanoid system represents the culmination of all components covered in the Vision-Language-Action module. By integrating voice processing, cognitive planning, navigation, perception, and manipulation systems, we create a complete AI-driven robotic platform capable of understanding and executing complex natural language commands in real-world environments.

This capstone chapter brings together concepts from:
- [Chapter 1: Voice-to-Action Pipelines](./voice-to-action-pipelines.md) - where you learned about speech recognition and natural language processing
- [Chapter 2: Cognitive Planning with LLMs](./cognitive-planning-llms.md) - where you learned about translating goals into action sequences
- [Previous modules on ROS 2 and simulation](../ros2-nervous-system/introduction.md) - providing the communication and control infrastructure

The complete system demonstrates how vision, language, and action components work together to create truly autonomous humanoid robots that can understand human commands, plan appropriate responses, and execute complex tasks in dynamic environments.

## Learning Objectives

By completing this capstone module, you should now be able to:
- Integrate multiple AI and robotics components into a cohesive autonomous system
- Implement end-to-end voice-command-to-action pipelines for humanoid robots
- Design safety frameworks that protect both robots and humans during autonomous operation
- Create perception-action loops that enable adaptive robot behavior
- Implement monitoring and error recovery systems for robust autonomous operation
- Understand the complete pipeline from natural language understanding to physical action execution