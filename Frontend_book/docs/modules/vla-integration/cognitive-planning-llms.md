---
id: cognitive-planning-llms
title: Cognitive Planning with LLMs
sidebar_label: Cognitive Planning with LLMs
sidebar_position: 2
---

# Cognitive Planning with LLMs

## Introduction to Cognitive Planning in Robotics

Cognitive planning represents a paradigm shift in robotics, where high-level goals expressed in natural language are transformed into executable action sequences using advanced reasoning capabilities. Large Language Models (LLMs) have emerged as powerful tools for this transformation, bridging the gap between human intention and robotic execution.

### The Role of LLMs in Cognitive Robotics

Large Language Models bring several key capabilities to cognitive robotics:

- **Natural Language Understanding**: Interpreting complex, nuanced human instructions
- **Knowledge Integration**: Leveraging vast world knowledge for planning
- **Reasoning**: Breaking down complex tasks into simpler, executable steps
- **Adaptability**: Handling novel situations through generalization
- **Context Awareness**: Understanding situational context for appropriate responses

### From Natural Language Goals to Action Sequences

The cognitive planning process involves several key transformations:

1. **Goal Comprehension**: Understanding the human's intent from natural language
2. **Task Decomposition**: Breaking complex goals into simpler subtasks
3. **Constraint Analysis**: Identifying physical, temporal, and safety constraints
4. **Resource Allocation**: Determining required resources and capabilities
5. **Action Sequencing**: Ordering actions for optimal execution
6. **Plan Validation**: Verifying plan feasibility and safety

## Translating Natural Language Goals into Action Sequences

### Understanding Natural Language Goals

Natural language goals can vary significantly in complexity and specificity:

**Simple Goals**:
- "Bring me a cup of coffee"
- "Go to the kitchen"
- "Clean the table"

**Complex Goals**:
- "Set the dining table for four people with plates, utensils, and glasses, then announce dinner"
- "Organize the books on the shelf by height, then dust the surface"
- "Help the elderly person get dressed, then assist them to the living room"

### LLM-Based Goal Parsing

LLMs can be prompted to parse and understand natural language goals:

```python
# Example: LLM-based goal parsing
import openai
import json
from typing import Dict, List, Any

class LLMGoalParser:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    def parse_goal(self, natural_language_goal: str) -> Dict[str, Any]:
        """
        Parse a natural language goal into structured components

        Args:
            natural_language_goal: Natural language description of the goal

        Returns:
            Structured goal components
        """
        prompt = f"""
        Parse the following natural language goal into structured components:

        Goal: "{natural_language_goal}"

        Return a JSON object with the following structure:
        {{
            "original_goal": "Original goal text",
            "primary_task": "Main task to accomplish",
            "subtasks": [
                {{
                    "description": "Brief description of subtask",
                    "action_type": "Type of action (navigate, manipulate, perceive, etc.)",
                    "objects": ["list of relevant objects"],
                    "locations": ["list of relevant locations"],
                    "constraints": ["list of constraints for this subtask"]
                }}
            ],
            "required_resources": ["list of required robot capabilities"],
            "estimated_duration": "Estimated time in seconds",
            "priority_level": "low|medium|high|critical",
            "dependencies": ["list of tasks that must be completed first"],
            "success_criteria": ["list of conditions that indicate success"]
        }}

        Be specific about objects, locations, and constraints. Keep subtasks granular and actionable.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # Low temperature for consistency
            )

            # Extract JSON from response
            content = response.choices[0].message.content.strip()

            # Sometimes the LLM wraps JSON in code blocks
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
                content = content.rstrip("`").rstrip()  # Remove trailing ```

            parsed_goal = json.loads(content)
            return parsed_goal

        except Exception as e:
            print(f"Error parsing goal: {e}")
            return self._fallback_parse(natural_language_goal)

    def _fallback_parse(self, goal: str) -> Dict[str, Any]:
        """
        Fallback parsing if LLM fails
        """
        return {
            "original_goal": goal,
            "primary_task": "Unknown",
            "subtasks": [{"description": "Parse failed", "action_type": "unknown", "objects": [], "locations": [], "constraints": []}],
            "required_resources": [],
            "estimated_duration": 0,
            "priority_level": "medium",
            "dependencies": [],
            "success_criteria": []
        }

# Example usage
parser = LLMGoalParser()

test_goals = [
    "Bring me a cup of coffee",
    "Clean the living room table",
    "Set the dining table for four people with plates, utensils, and glasses, then announce dinner"
]

for goal in test_goals:
    parsed = parser.parse_goal(goal)
    print(f"Goal: {goal}")
    print(f"Parsed: {json.dumps(parsed, indent=2)}\n")
```

### Task Decomposition with LLMs

Breaking down complex goals into manageable subtasks is crucial for robotic execution:

```python
# Example: LLM-based task decomposition
class TaskDecomposer:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    def decompose_task(self, goal_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose a complex task into executable subtasks

        Args:
            goal_components: Parsed goal components from LLMGoalParser

        Returns:
            List of executable subtasks
        """
        prompt = f"""
        Decompose the following goal into executable subtasks for a humanoid robot:

        Original Goal: {goal_components['original_goal']}
        Primary Task: {goal_components['primary_task']}
        Required Resources: {goal_components['required_resources']}
        Constraints: {goal_components.get('constraints', [])}

        Return a list of subtasks, each with the following structure:
        [
            {{
                "id": "unique identifier for the subtask",
                "description": "Detailed description of what to do",
                "action_type": "navigation|manipulation|perception|communication|etc.",
                "parameters": {{
                    "target_object": "object to interact with",
                    "destination": "location to navigate to",
                    "gripper_position": "specific manipulation parameters",
                    "speech_text": "text to speak if communication task"
                }},
                "preconditions": ["conditions that must be true before executing"],
                "effects": ["changes that occur after execution"],
                "success_criteria": ["how to verify task completion"],
                "estimated_duration": "seconds to complete",
                "priority": 1-10 (higher number = higher priority)
            }}
        ]

        Make subtasks granular enough for a robot to execute but high-level enough to be meaningful.
        Consider the physical constraints and capabilities of a humanoid robot.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()

            if content.startswith("```json"):
                content = content[7:]
                content = content.rstrip("`").rstrip()

            subtasks = json.loads(content)
            return subtasks

        except Exception as e:
            print(f"Error decomposing task: {e}")
            return self._fallback_decompose(goal_components)

    def _fallback_decompose(self, goal_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fallback task decomposition
        """
        return [{
            "id": "fallback-1",
            "description": f"Execute {goal_components['primary_task']}",
            "action_type": "unknown",
            "parameters": {},
            "preconditions": [],
            "effects": [],
            "success_criteria": [],
            "estimated_duration": 60,
            "priority": 5
        }]

# Example usage
decomposer = TaskDecomposer()

# Example goal components (from previous parser)
example_goal = {
    "original_goal": "Set the dining table for four people with plates, utensils, and glasses, then announce dinner",
    "primary_task": "Set dining table and announce dinner",
    "required_resources": ["navigation", "manipulation", "communication"],
    "constraints": []
}

subtasks = decomposer.decompose_task(example_goal)
print("Decomposed subtasks:")
for i, task in enumerate(subtasks):
    print(f"{i+1}. {task['description']} ({task['action_type']})")
```

### Context-Aware Planning

LLMs can incorporate contextual information to make more informed planning decisions:

```python
# Example: Context-aware planning with LLMs
class ContextAwarePlanner:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.robot_capabilities = {
            "navigation": {"speed": 0.5, "precision": "centimeter"},
            "manipulation": {"reach": 1.2, "payload": 2.0, "precision": "millimeter"},
            "perception": {"range": 5.0, "resolution": "high"},
            "communication": {"languages": ["English", "Spanish"]}
        }
        self.environment_state = {}
        self.user_preferences = {}

    def plan_with_context(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan with awareness of current context

        Args:
            goal: Natural language goal
            context: Current context including environment, capabilities, etc.

        Returns:
            Detailed plan with context considerations
        """
        prompt = f"""
        Create a detailed plan for the following goal considering the provided context:

        GOAL: {goal}

        CONTEXT:
        - Current Environment: {context.get('environment', 'Unknown')}
        - Robot Capabilities: {self.robot_capabilities}
        - Available Objects: {context.get('available_objects', [])}
        - Current Robot State: {context.get('robot_state', {})}
        - User Preferences: {self.user_preferences}
        - Time of Day: {context.get('time_of_day', 'Unknown')}
        - Safety Constraints: {context.get('safety_constraints', [])}

        Return a JSON object with the following structure:
        {{
            "goal": "Original goal",
            "context_summary": "Brief summary of relevant context",
            "plan": [
                {{
                    "step": 1,
                    "action": "High-level action description",
                    "subtasks": [
                        {{
                            "description": "Specific subtask",
                            "type": "navigation|manipulation|perception|communication",
                            "parameters": {{}},
                            "expected_outcome": "What should happen after execution",
                            "context_awareness": "How this step considers the context"
                        }}
                    ],
                    "reasoning": "Why this step is needed in this context"
                }}
            ],
            "adaptations": "List of how the plan adapts to the context",
            "risks": "Potential risks given the context",
            "alternatives": "Alternative approaches if primary plan fails"
        }}

        Consider how the context affects the plan. For example, if it's nighttime,
        the robot might need to be quieter. If the user prefers Spanish,
        communication should be in Spanish. If objects are in unusual locations,
        the navigation plan should account for this.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            content = response.choices[0].message.content.strip()

            if content.startswith("```json"):
                content = content[7:]
                content = content.rstrip("`").rstrip()

            plan = json.loads(content)
            return plan

        except Exception as e:
            print(f"Error creating context-aware plan: {e}")
            return self._fallback_plan(goal, context)

    def _fallback_plan(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback plan when LLM fails
        """
        return {
            "goal": goal,
            "context_summary": "Context not considered due to error",
            "plan": [{"step": 1, "action": "Attempt to execute goal", "subtasks": [], "reasoning": "Fallback plan"}],
            "adaptations": [],
            "risks": [],
            "alternatives": []
        }

# Example usage
planner = ContextAwarePlanner()

context_example = {
    "environment": "Home environment with kitchen, living room, and bedrooms",
    "available_objects": ["plates", "utensils", "glasses", "coffee maker"],
    "robot_state": {"location": "kitchen", "battery": 85, "payload": 0.0},
    "time_of_day": "Evening",
    "safety_constraints": ["avoid stairs", "maintain safe distance from humans"]
}

plan = planner.plan_with_context("Serve dinner to family", context_example)
print("Context-aware plan:")
print(json.dumps(plan, indent=2))
```

## LLM-Driven Task Planning for ROS 2

### Integration with ROS 2 Action Architecture

LLMs can generate action sequences that map directly to ROS 2 action structures:

```python
# Example: LLM-to-ROS2 action mapping
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from typing import Dict, List, Any
import json

class LLMROS2Planner(Node):
    def __init__(self):
        super().__init__('llm_ros2_planner')

        # Publishers for different action types
        self.nav_publisher = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        self.manipulation_publisher = self.create_publisher(String, '/manipulation_command', 10)
        self.perception_publisher = self.create_publisher(String, '/perception_request', 10)
        self.communication_publisher = self.create_publisher(String, '/tts_command', 10)

        # Action clients for more complex tasks
        self.nav_action_client = ActionClient(self, MoveBaseAction, 'move_base')
        self.manipulation_action_client = ActionClient(self, ManipulationAction, 'manipulation')

        # LLM planner
        self.llm_planner = ContextAwarePlanner()

        self.get_logger().info('LLM-ROS2 Planner initialized')

    def execute_plan(self, natural_language_goal: str, context: Dict[str, Any]):
        """
        Execute a plan generated from natural language goal
        """
        # Generate plan using LLM
        plan = self.llm_planner.plan_with_context(natural_language_goal, context)

        # Execute plan step by step
        for step in plan['plan']:
            self.execute_step(step)

    def execute_step(self, step: Dict[str, Any]):
        """
        Execute a single step of the plan
        """
        for subtask in step['subtasks']:
            action_type = subtask['type']
            parameters = subtask['parameters']

            if action_type == 'navigation':
                self.execute_navigation(parameters)
            elif action_type == 'manipulation':
                self.execute_manipulation(parameters)
            elif action_type == 'perception':
                self.execute_perception(parameters)
            elif action_type == 'communication':
                self.execute_communication(parameters)
            else:
                self.get_logger().warn(f'Unknown action type: {action_type}')

    def execute_navigation(self, params: Dict[str, Any]):
        """
        Execute navigation action
        """
        # Create navigation goal
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'

        # Set position from parameters
        goal.pose.position.x = params.get('x', 0.0)
        goal.pose.position.y = params.get('y', 0.0)
        goal.pose.position.z = params.get('z', 0.0)

        # Set orientation (simplified - would need proper quaternion calculation)
        goal.pose.orientation.w = 1.0

        # Publish navigation command
        self.nav_publisher.publish(goal)

        self.get_logger().info(f'Navigating to: ({goal.pose.position.x}, {goal.pose.position.y})')

    def execute_manipulation(self, params: Dict[str, Any]):
        """
        Execute manipulation action
        """
        command = String()
        command.data = json.dumps({
            'action': params.get('action', 'unknown'),
            'object': params.get('target_object', 'unknown'),
            'position': params.get('position', [0, 0, 0]),
            'gripper_position': params.get('gripper_position', 'open')
        })

        self.manipulation_publisher.publish(command)
        self.get_logger().info(f'Manipulation command: {command.data}')

    def execute_perception(self, params: Dict[str, Any]):
        """
        Execute perception action
        """
        request = String()
        request.data = json.dumps({
            'task': params.get('task', 'detect'),
            'target': params.get('target_object', 'any'),
            'location': params.get('location', 'current')
        })

        self.perception_publisher.publish(request)
        self.get_logger().info(f'Perception request: {request.data}')

    def execute_communication(self, params: Dict[str, Any]):
        """
        Execute communication action
        """
        command = String()
        command.data = params.get('text', 'Hello')

        self.communication_publisher.publish(command)
        self.get_logger().info(f'Communication: {command.data}')

# Example usage in a ROS 2 node
def main(args=None):
    rclpy.init(args=args)

    planner = LLMROS2Planner()

    # Example context
    context = {
        "environment": "home kitchen",
        "available_objects": ["cup", "coffee", "table"],
        "robot_state": {"location": "counter", "battery": 90},
        "time_of_day": "morning",
        "safety_constraints": ["avoid hot surfaces"]
    }

    # Execute a natural language goal
    planner.execute_plan("Make coffee and bring it to the dining table", context)

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Prompt Engineering for Robotic Planning

Effective prompt engineering is crucial for getting reliable outputs from LLMs for robotic planning:

```python
# Example: Advanced prompt engineering for robotic planning
class PromptEngineer:
    def __init__(self):
        self.system_prompt = """
        You are an expert robotic planning assistant. Your role is to convert natural language goals into structured robotic plans.

        When creating plans, consider:
        1. Physical constraints of humanoid robots
        2. Safety requirements
        3. Efficiency of execution
        4. Feasibility of actions
        5. Sequential dependencies

        Always return structured JSON with clear, executable actions.
        """

    def create_planning_prompt(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create a structured prompt for LLM planning

        Args:
            goal: Natural language goal
            context: Context information

        Returns:
            Formatted prompt for LLM API
        """
        user_message = f"""
        CONTEXT:
        Environment: {context.get('environment', 'unknown')}
        Available Objects: {context.get('available_objects', [])}
        Robot Capabilities: {context.get('robot_capabilities', {})}
        Current State: {context.get('current_state', {})}
        Constraints: {context.get('constraints', [])}

        GOAL: {goal}

        INSTRUCTIONS:
        1. Analyze the goal and context
        2. Break down the goal into sequential, executable actions
        3. Consider physical constraints and safety
        4. Return a detailed plan in the specified JSON format
        5. Ensure each action is atomic and achievable

        OUTPUT FORMAT:
        {{
            "analysis": "Brief analysis of the goal and context",
            "plan": [
                {{
                    "step": 1,
                    "action": "action_type",
                    "description": "What to do",
                    "parameters": {{"key": "value"}},
                    "safety_check": "Brief safety consideration",
                    "success_criteria": "How to verify completion"
                }}
            ],
            "potential_issues": ["List of potential challenges"],
            "safety_considerations": ["List of safety aspects to consider"]
        }}

        BE SPECIFIC AND CONCISE. AVOID AMBIGUOUS LANGUAGE.
        """

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

    def create_verification_prompt(self, plan: Dict[str, Any], goal: str) -> str:
        """
        Create a prompt to verify a plan
        """
        return f"""
        Verify the following robotic plan for the goal: "{goal}"

        PLAN TO VERIFY:
        {json.dumps(plan, indent=2)}

        CHECK THE FOLLOWING:
        1. Does the plan achieve the stated goal?
        2. Are all actions physically feasible for a humanoid robot?
        3. Are there any safety concerns?
        4. Are the action sequences logical and complete?
        5. Are the success criteria clear and measurable?

        Return VERIFIED if the plan passes all checks, otherwise return ISSUES with specific problems.
        """

    def create_error_recovery_prompt(self, error: str, original_goal: str, context: Dict[str, Any]) -> str:
        """
        Create a prompt for error recovery planning
        """
        return f"""
        The robot encountered an error: "{error}"

        ORIGINAL GOAL: {original_goal}
        CURRENT CONTEXT: {context}

        SUGGEST RECOVERY ACTIONS:
        1. Assess the current situation
        2. Propose alternative approaches
        3. Consider safety implications
        4. Prioritize actions for recovery

        Return a recovery plan in the same format as the original plan.
        """
```

### Planning Validation and Safety Checks

Implementing validation mechanisms to ensure plans are safe and executable:

```python
# Example: Planning validation and safety checks
class PlanValidator:
    def __init__(self):
        self.safety_rules = [
            "avoid_collision_with_humans",
            "respect_personal_space",
            "operate_within_physical_limits",
            "maintain_balance",
            "avoid_damage_to_objects_or_environment"
        ]

        self.feasibility_rules = [
            "actions_must_be_atomic",
            "dependencies_must_be_resolved",
            "resources_must_be_available",
            "constraints_must_be_satisfiable"
        ]

    def validate_plan(self, plan: Dict[str, Any], robot_capabilities: Dict[str, Any]) -> tuple[bool, List[str], List[str]]:
        """
        Validate a plan for safety and feasibility

        Args:
            plan: Plan to validate
            robot_capabilities: Capabilities of the executing robot

        Returns:
            Tuple of (is_valid, safety_violations, feasibility_issues)
        """
        safety_violations = []
        feasibility_issues = []

        # Check safety rules
        for rule in self.safety_rules:
            violations = self._check_safety_rule(rule, plan, robot_capabilities)
            safety_violations.extend(violations)

        # Check feasibility rules
        for rule in self.feasibility_rules:
            issues = self._check_feasibility_rule(rule, plan, robot_capabilities)
            feasibility_issues.extend(issues)

        is_valid = len(safety_violations) == 0 and len(feasibility_issues) == 0
        return is_valid, safety_violations, feasibility_issues

    def _check_safety_rule(self, rule: str, plan: Dict[str, Any], robot_capabilities: Dict[str, Any]) -> List[str]:
        """
        Check a specific safety rule
        """
        violations = []

        if rule == "avoid_collision_with_humans":
            # Check if navigation actions consider human locations
            for step in plan.get('plan', []):
                if step.get('action') == 'navigation':
                    # In a real implementation, this would check against human location data
                    pass

        elif rule == "respect_personal_space":
            # Check if actions respect personal space boundaries
            for step in plan.get('plan', []):
                if step.get('action') == 'navigation' or step.get('action') == 'manipulation':
                    # Check if target is too close to human
                    pass

        elif rule == "operate_within_physical_limits":
            # Check if actions exceed robot's physical capabilities
            for step in plan.get('plan', []):
                params = step.get('parameters', {})

                # Check navigation limits
                if step.get('action') == 'navigation':
                    max_speed = robot_capabilities.get('navigation', {}).get('max_speed', 1.0)
                    requested_speed = params.get('speed', 0.5)
                    if requested_speed > max_speed:
                        violations.append(f"Navigation speed {requested_speed} exceeds robot limit of {max_speed}")

                # Check manipulation limits
                elif step.get('action') == 'manipulation':
                    payload = params.get('payload', 0.0)
                    max_payload = robot_capabilities.get('manipulation', {}).get('max_payload', 1.0)
                    if payload > max_payload:
                        violations.append(f"Payload {payload}kg exceeds robot limit of {max_payload}kg")

        elif rule == "maintain_balance":
            # Check if actions could compromise robot balance
            for step in plan.get('plan', []):
                if step.get('action') == 'manipulation':
                    # Check if object is too heavy or positioned to affect balance
                    payload = step.get('parameters', {}).get('payload', 0.0)
                    if payload > 0.8 * robot_capabilities.get('manipulation', {}).get('max_payload', 1.0):
                        violations.append("Heavy payload may affect robot balance")

        elif rule == "avoid_damage_to_objects_or_environment":
            # Check if actions might damage objects or environment
            for step in plan.get('plan', []):
                if step.get('action') == 'manipulation':
                    # Check for excessive force parameters
                    force = step.get('parameters', {}).get('force', 10.0)
                    max_force = robot_capabilities.get('manipulation', {}).get('max_force', 50.0)
                    if force > 0.9 * max_force:
                        violations.append("Excessive force may damage objects")

        return violations

    def _check_feasibility_rule(self, rule: str, plan: Dict[str, Any], robot_capabilities: Dict[str, Any]) -> List[str]:
        """
        Check a specific feasibility rule
        """
        issues = []

        if rule == "actions_must_be_atomic":
            # Check if all actions are atomic and executable
            for step in plan.get('plan', []):
                action_type = step.get('action')
                if action_type not in ['navigation', 'manipulation', 'perception', 'communication']:
                    issues.append(f"Unknown action type: {action_type}")

        elif rule == "dependencies_must_be_resolved":
            # Check if action dependencies are properly handled
            completed_actions = set()
            for step in plan.get('plan', []):
                deps = step.get('dependencies', [])
                for dep in deps:
                    if dep not in completed_actions:
                        issues.append(f"Dependency {dep} not satisfied before action execution")
                completed_actions.add(step.get('id', f"step_{step.get('step')}"))

        elif rule == "resources_must_be_available":
            # Check if required resources are available
            for step in plan.get('plan', []):
                required_resources = step.get('required_resources', [])
                for resource in required_resources:
                    if resource not in robot_capabilities:
                        issues.append(f"Required resource {resource} not available")

        elif rule == "constraints_must_be_satisfiable":
            # Check if all constraints can be satisfied
            for step in plan.get('plan', []):
                constraints = step.get('constraints', [])
                for constraint in constraints:
                    # Validate constraint format and feasibility
                    pass

        return issues

    def suggest_plan_modifications(self, plan: Dict[str, Any], violations: List[str], issues: List[str]) -> List[str]:
        """
        Suggest modifications to address violations and issues
        """
        suggestions = []

        for violation in violations:
            if "exceeds robot limit" in violation:
                # Suggest reducing the parameter
                suggestions.append(f"Reduce parameter to stay within robot limits: {violation}")
            elif "may affect balance" in violation:
                # Suggest alternative approach
                suggestions.append(f"Consider alternative approach to maintain balance: {violation}")
            elif "may damage objects" in violation:
                # Suggest reducing force
                suggestions.append(f"Reduce force/application to prevent damage: {violation}")

        for issue in issues:
            if "dependency" in issue:
                # Suggest reordering actions
                suggestions.append(f"Reorder actions to satisfy dependencies: {issue}")
            elif "resource" in issue:
                # Suggest alternative resources or capabilities
                suggestions.append(f"Ensure required resources are available: {issue}")

        return suggestions
```

## Practical Implementation Examples

### Complete LLM-Driven Planning System

```python
# Example: Complete LLM-driven planning system
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Any

class PlanStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PlanExecutionResult:
    status: PlanStatus
    success: bool
    message: str
    execution_log: List[Dict[str, Any]]
    error_details: Optional[Dict[str, Any]] = None

class LLMPlanningSystem:
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        self.llm_model = llm_model
        self.goal_parser = LLMGoalParser(llm_model)
        self.task_decomposer = TaskDecomposer(llm_model)
        self.context_planner = ContextAwarePlanner(llm_model)
        self.plan_validator = PlanValidator()
        self.ros2_planner = LLMROS2Planner()

        # Current plan tracking
        self.current_plan = None
        self.plan_status = PlanStatus.PENDING
        self.execution_log = []

    async def create_and_execute_plan(self, goal: str, context: Dict[str, Any]) -> PlanExecutionResult:
        """
        Create and execute a plan from natural language goal
        """
        try:
            self.execution_log = []
            self.plan_status = PlanStatus.PENDING

            # Step 1: Parse the goal
            self.log_execution("Parsing natural language goal")
            parsed_goal = self.goal_parser.parse_goal(goal)

            # Step 2: Decompose the task
            self.log_execution("Decomposing task into subtasks")
            subtasks = self.task_decomposer.decompose_task(parsed_goal)

            # Step 3: Create context-aware plan
            self.log_execution("Creating context-aware plan")
            plan = self.context_planner.plan_with_context(goal, context)

            # Step 4: Validate the plan
            self.log_execution("Validating plan for safety and feasibility")
            is_valid, safety_violations, feasibility_issues = self.plan_validator.validate_plan(
                plan, context.get('robot_capabilities', {})
            )

            if not is_valid:
                # Try to modify plan to address issues
                modifications = self.plan_validator.suggest_plan_modifications(
                    plan, safety_violations, feasibility_issues
                )

                if modifications:
                    self.log_execution(f"Suggesting plan modifications: {modifications}")
                    # In a real system, you might ask for user confirmation or automatically apply modifications
                    # For this example, we'll proceed with the original plan but log the issues
                    self.log_execution("Proceeding with plan despite identified issues")
                else:
                    return PlanExecutionResult(
                        status=PlanStatus.FAILED,
                        success=False,
                        message=f"Plan validation failed with issues: {safety_violations + feasibility_issues}",
                        execution_log=self.execution_log,
                        error_details={
                            "safety_violations": safety_violations,
                            "feasibility_issues": feasibility_issues
                        }
                    )

            # Step 5: Execute the plan
            self.log_execution("Starting plan execution")
            self.current_plan = plan
            self.plan_status = PlanStatus.EXECUTING

            # Execute each step
            for step in plan['plan']:
                if self.plan_status != PlanStatus.EXECUTING:
                    break  # Plan was cancelled

                await self.execute_plan_step(step, context)

            # Check final status
            if self.plan_status == PlanStatus.EXECUTING:
                self.plan_status = PlanStatus.COMPLETED
                return PlanExecutionResult(
                    status=PlanStatus.COMPLETED,
                    success=True,
                    message="Plan executed successfully",
                    execution_log=self.execution_log
                )
            else:
                return PlanExecutionResult(
                    status=self.plan_status,
                    success=False,
                    message="Plan execution stopped",
                    execution_log=self.execution_log
                )

        except Exception as e:
            self.plan_status = PlanStatus.FAILED
            self.log_execution(f"Error during plan execution: {str(e)}")
            return PlanExecutionResult(
                status=PlanStatus.FAILED,
                success=False,
                message=f"Plan execution failed: {str(e)}",
                execution_log=self.execution_log,
                error_details={"exception": str(e), "type": type(e).__name__}
            )

    async def execute_plan_step(self, step: Dict[str, Any], context: Dict[str, Any]):
        """
        Execute a single step of the plan
        """
        self.log_execution(f"Executing step: {step['action']}")

        try:
            # In a real implementation, this would interface with the ROS 2 system
            # For this example, we'll simulate execution

            # Simulate execution time
            import random
            execution_time = random.uniform(1.0, 5.0)
            await asyncio.sleep(execution_time)

            # Check for simulated errors
            if random.random() < 0.1:  # 10% chance of error
                raise Exception("Simulated execution error")

            self.log_execution(f"Completed step: {step['action']}")

        except Exception as e:
            self.log_execution(f"Error in step execution: {str(e)}")
            # In a real system, you would implement error recovery
            # For this example, we'll just log the error
            pass

    def log_execution(self, message: str):
        """
        Log execution events
        """
        import datetime
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": message,
            "status": self.plan_status.value
        }
        self.execution_log.append(log_entry)
        print(f"[{log_entry['timestamp']}] {message}")

    def cancel_current_plan(self):
        """
        Cancel the currently executing plan
        """
        if self.plan_status == PlanStatus.EXECUTING:
            self.plan_status = PlanStatus.CANCELLED
            self.log_execution("Plan cancelled by user")

    def get_plan_status(self) -> PlanStatus:
        """
        Get current plan status
        """
        return self.plan_status

# Example usage
async def main():
    # Initialize the planning system
    planning_system = LLMPlanningSystem()

    # Define a goal and context
    goal = "Clean the kitchen table and put the dishes in the dishwasher"

    context = {
        "environment": "home kitchen",
        "available_objects": ["dishes", "dishwasher", "cleaning_cloth", "table"],
        "robot_capabilities": {
            "navigation": {"max_speed": 0.5, "precision": "centimeter"},
            "manipulation": {"max_payload": 2.0, "max_force": 50.0}
        },
        "current_state": {"location": "entrance", "battery": 85},
        "constraints": ["avoid wet floor", "be quiet during night hours"]
    }

    # Execute the plan
    result = await planning_system.create_and_execute_plan(goal, context)

    print("\nPlan Execution Result:")
    print(f"Status: {result.status.value}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")

    if result.error_details:
        print(f"Errors: {result.error_details}")

# Run the example
# asyncio.run(main())
```

## Integration with ROS 2 Action Architecture

### Creating Custom Action Messages

For LLM-driven planning to work effectively with ROS 2, we need custom action messages:

```python
# Example: Custom action messages for LLM planning
# This would typically be in a package like 'vla_msgs/action/CognitivePlan.action'

# CognitivePlan.action
"""
# Request from user for cognitive planning
string natural_language_goal
string user_context

---
# Feedback during plan execution
string current_step
string status_message
float64 progress_percentage

---
# Result of the planning and execution
bool success
string message
string[] execution_log
string final_plan_json
"""

# ComplexTask.action
"""
# Define a complex task with multiple subtasks
string task_description
string priority  # low, medium, high, critical

# Subtasks to execute
CognitiveSubtask[] subtasks

# Constraints
string[] safety_constraints
string[] temporal_constraints
string[] resource_constraints

---
# Feedback during execution
string current_subtask
string status
float64 progress_percentage
string[] completed_tasks

---
# Final result
bool success
string message
string[] execution_log
builtin_interfaces/Time execution_time
"""

# CognitiveSubtask.msg
"""
string id
string description
string action_type  # navigation, manipulation, perception, communication
builtin_interfaces/KeyValue[] parameters
string[] preconditions
string[] effects
string[] success_criteria
float64 estimated_duration
int8 priority  # 1-10
"""
```

### Action Server Implementation

```python
# Example: Cognitive planning action server
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from vla_msgs.action import CognitivePlan
from std_msgs.msg import String
import json

class CognitivePlanningActionServer(Node):
    def __init__(self):
        super().__init__('cognitive_planning_action_server')

        # Initialize the action server
        self._action_server = ActionServer(
            self,
            CognitivePlan,
            'cognitive_plan',
            self.execute_callback
        )

        # Initialize the LLM planning system
        self.planning_system = LLMPlanningSystem()

        # Publishers for different aspects
        self.feedback_publisher = self.create_publisher(String, 'cognitive_plan_feedback', 10)

        self.get_logger().info('Cognitive Planning Action Server initialized')

    def execute_callback(self, goal_handle):
        """
        Execute the cognitive planning action
        """
        self.get_logger().info('Executing cognitive planning goal')

        # Extract goal information
        natural_language_goal = goal_handle.request.natural_language_goal
        user_context = json.loads(goal_handle.request.user_context) if goal_handle.request.user_context else {}

        # Prepare feedback
        feedback_msg = CognitivePlan.Feedback()

        try:
            # Execute the plan asynchronously
            result = self.planning_system.create_and_execute_plan(natural_language_goal, user_context)

            # Create result message
            result_msg = CognitivePlan.Result()
            result_msg.success = result.success
            result_msg.message = result.message
            result_msg.execution_log = result.execution_log
            result_msg.final_plan_json = json.dumps(result)  # This would be the actual plan

            if result.success:
                goal_handle.succeed()
                return result_msg
            else:
                goal_handle.abort()
                return result_msg

        except Exception as e:
            self.get_logger().error(f'Error executing cognitive plan: {e}')
            goal_handle.abort()

            result_msg = CognitivePlan.Result()
            result_msg.success = False
            result_msg.message = f'Execution failed: {str(e)}'
            result_msg.execution_log = [f"Error: {str(e)}"]
            result_msg.final_plan_json = ""

            return result_msg

def main(args=None):
    rclpy.init(args=args)

    action_server = CognitivePlanningActionServer()

    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting LLM-Driven Planning

### Common Issues and Solutions

#### 1. Hallucination in Planning
**Problem**: LLM generates plans that include impossible or fictional elements.
**Solutions**:
- Implement strict validation of LLM outputs
- Use constrained decoding to limit possibilities
- Cross-reference with known world state
- Implement fact-checking mechanisms

#### 2. Context Window Limitations
**Problem**: Complex goals exceed the LLM's context window.
**Solutions**:
- Implement hierarchical planning with summaries
- Use retrieval-augmented generation (RAG)
- Break complex tasks into manageable chunks
- Implement external memory systems

#### 3. Safety and Feasibility Issues
**Problem**: LLM generates unsafe or infeasible plans.
**Solutions**:
- Implement comprehensive validation layers
- Use safety-constrained prompting
- Add explicit safety rules to prompts
- Implement runtime safety monitoring

### Performance Monitoring

```python
# Example: Performance monitoring for LLM planning
import time
import statistics
from collections import defaultdict

class LLMPlanningMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}

    def start_metric(self, metric_name: str):
        """Start timing a metric"""
        self.start_times[metric_name] = time.time()

    def end_metric(self, metric_name: str):
        """End timing a metric"""
        if metric_name in self.start_times:
            elapsed = time.time() - self.start_times[metric_name]
            self.metrics[metric_name].append(elapsed)
            del self.start_times[metric_name]

    def record_metric(self, metric_name: str, value: float):
        """Record a specific metric value"""
        self.metrics[metric_name].append(value)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of metrics"""
        stats = {}
        for name, values in self.metrics.items():
            if values:
                stats[name] = {
                    'count': len(values),
                    'average': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
                }
        return stats

    def is_performance_degraded(self) -> bool:
        """Check if performance is degraded"""
        stats = self.get_statistics()

        # Check if planning time is too high (more than 5 seconds average)
        if 'planning_time' in stats and stats['planning_time']['average'] > 5.0:
            return True

        return False
```

## Summary and Cross-References

Cognitive planning with LLMs represents a significant advancement in robotic autonomy, enabling robots to understand and execute complex natural language goals. By combining the reasoning capabilities of large language models with the execution framework of ROS 2, we can create intelligent humanoid robots that respond to human instructions in sophisticated ways.

This chapter builds upon the voice-to-action pipelines from [Chapter 1: Voice-to-Action Pipelines](./voice-to-action-pipelines.md), where we learned to convert speech to structured commands. The plans generated here can be triggered by voice commands and executed by the robotic systems we'll discuss in the next chapter.

In the next chapter, we'll explore how to integrate all these components into a complete autonomous humanoid system.

[Continue to Chapter 3: Capstone: The Autonomous Humanoid](./capstone-autonomous-humanoid.md)

## Learning Objectives

By the end of this chapter, you should be able to:
- Translate natural language goals into structured action sequences using LLMs
- Implement LLM-driven task planning for ROS 2 systems
- Design validation and safety mechanisms for LLM-generated plans
- Integrate LLM planning with ROS 2 action architecture
- Troubleshoot common issues in LLM-driven planning systems