---
id: voice-to-action-pipelines
title: Voice-to-Action Pipelines
sidebar_label: Voice-to-Action Pipelines
sidebar_position: 1
---

# Voice-to-Action Pipelines

## Introduction to Voice-to-Action Systems

Voice-to-action pipelines represent a critical component in modern humanoid robotics, enabling natural human-robot interaction through spoken language. These systems transform human voice commands into structured robotic actions, creating an intuitive interface that bridges the gap between human communication and robotic execution.

### The Role of Voice in Humanoid Robotics

Voice interfaces provide several advantages for humanoid robots:

- **Natural Interaction**: Humans naturally communicate through speech
- **Accessibility**: Allows hands-free operation and interaction
- **Intuitive Control**: Complex tasks can be initiated with simple voice commands
- **Social Integration**: Enhances the humanoid's ability to interact in social settings
- **Cognitive Offloading**: Reduces the need for users to learn complex control interfaces

### Voice-to-Action Pipeline Architecture

A complete voice-to-action pipeline consists of several interconnected components:

1. **Audio Capture**: Microphone array or single microphone for voice input
2. **Speech Recognition**: Converting audio to text (ASR - Automatic Speech Recognition)
3. **Intent Classification**: Understanding the meaning behind the spoken words
4. **Action Mapping**: Converting intents into executable robotic actions
5. **Execution**: Performing the mapped actions through robot systems

## Speech Recognition with OpenAI Whisper

### Understanding OpenAI Whisper

OpenAI Whisper is a state-of-the-art automatic speech recognition (ASR) system that demonstrates remarkable performance across multiple languages and domains. For humanoid robotics applications, Whisper provides:

- **Multilingual Support**: Capable of recognizing speech in numerous languages
- **Robustness**: Performs well in various acoustic conditions
- **Accuracy**: High transcription accuracy across different accents and speaking styles
- **Open Source**: Available for research and commercial use
- **Real-time Capabilities**: Can be optimized for real-time processing

### Whisper Architecture and Capabilities

Whisper uses a transformer-based architecture with the following key features:

- **Encoder-Decoder Structure**: Processes audio through encoder, generates text with decoder
- **Multilingual Training**: Trained on 98 languages for broad applicability
- **Robust Preprocessing**: Handles various audio formats and quality levels
- **Flexible Output**: Can generate word-level timing information alongside transcriptions

### Implementing Whisper for Robotics

```python
# Example: Implementing Whisper for voice-to-action pipeline
import whisper
import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional

class WhisperVoiceProcessor:
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        """
        Initialize Whisper voice processor for robotics applications

        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run model on ('cpu', 'cuda', 'mps')
        """
        self.device = device
        self.model = whisper.load_model(model_size, device=device)

        # Configuration for robotics use
        self.language = "en"  # Can be dynamically detected
        self.task = "transcribe"  # Always transcribe for voice commands
        self.energy_threshold = 0.01  # Minimum audio energy to consider speech

        # Audio preprocessing parameters
        self.sample_rate = 16000
        self.chunk_size = 1024  # For streaming processing
        self.silence_duration = 0.5  # Duration of silence to trigger processing

        print(f"Whisper voice processor initialized on {device}")

    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Preprocess audio data for Whisper

        Args:
            audio_data: Raw audio data
            sample_rate: Sample rate of audio data

        Returns:
            Processed audio data ready for Whisper
        """
        # Resample if necessary
        if sample_rate != self.sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)

        # Normalize audio
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Pad or trim to minimum length (Whisper expects at least ~0.1 seconds)
        min_length = int(0.1 * self.sample_rate)  # 100ms minimum
        if len(audio_data) < min_length:
            audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), 'constant')

        return audio_data

    def transcribe_audio(self, audio_data: np.ndarray) -> Dict:
        """
        Transcribe audio using Whisper

        Args:
            audio_data: Preprocessed audio data

        Returns:
            Dictionary containing transcription results
        """
        # Move audio to appropriate device
        if self.device != "cpu":
            audio_tensor = torch.from_numpy(audio_data).to(self.device)
        else:
            audio_tensor = torch.from_numpy(audio_data)

        # Perform transcription
        result = self.model.transcribe(
            audio_tensor,
            language=self.language,
            task=self.task,
            word_timestamps=True  # Get word-level timing for better processing
        )

        return result

    def detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """
        Simple voice activity detection

        Args:
            audio_chunk: Audio chunk to analyze

        Returns:
            True if voice activity detected, False otherwise
        """
        # Calculate energy of the audio chunk
        energy = np.mean(audio_chunk ** 2)
        return energy > self.energy_threshold

    def process_streaming_audio(self, audio_generator) -> List[Dict]:
        """
        Process streaming audio for real-time voice commands

        Args:
            audio_generator: Generator yielding audio chunks

        Returns:
            List of processed transcription results
        """
        results = []
        accumulated_audio = np.array([])

        for audio_chunk in audio_generator:
            # Preprocess the chunk
            processed_chunk = self.preprocess_audio(audio_chunk)

            # Detect voice activity
            if self.detect_voice_activity(processed_chunk):
                # Accumulate audio until silence is detected
                accumulated_audio = np.concatenate([accumulated_audio, processed_chunk])
            else:
                # Silence detected - process accumulated audio if sufficient length
                if len(accumulated_audio) > self.sample_rate * 0.5:  # At least 0.5 seconds
                    if np.any(accumulated_audio != 0):  # Check if there's actual audio
                        try:
                            result = self.transcribe_audio(accumulated_audio)
                            results.append(result)

                            # Reset accumulator
                            accumulated_audio = np.array([])
                        except Exception as e:
                            print(f"Error processing audio: {e}")
                            accumulated_audio = np.array([])

        return results

# Example usage for humanoid robot
class HumanoidVoiceController:
    def __init__(self):
        self.voice_processor = WhisperVoiceProcessor(model_size="base", device="cuda")
        self.command_parser = CommandParser()

    def process_voice_command(self, audio_data: np.ndarray) -> Optional[Dict]:
        """
        Process voice command and return structured action

        Args:
            audio_data: Raw audio data from robot's microphones

        Returns:
            Structured action command or None if no valid command found
        """
        # Preprocess and transcribe
        processed_audio = self.voice_processor.preprocess_audio(audio_data)
        transcription = self.voice_processor.transcribe_audio(processed_audio)

        # Extract text
        text = transcription.get("text", "").strip()

        if text:
            print(f"Recognized: {text}")

            # Parse command from text
            structured_command = self.command_parser.parse(text)

            return structured_command

        return None
```

### Optimizing Whisper for Real-Time Robotics

For real-time robotics applications, several optimizations can improve performance:

```python
# Example: Optimized Whisper pipeline for robotics
import whisper
import torch
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque

class OptimizedWhisperProcessor:
    def __init__(self, model_size: str = "base"):
        # Load model once and keep in memory
        self.model = whisper.load_model(model_size, device="cuda")

        # Use smaller model for faster processing if accuracy permits
        self.model_size = model_size

        # Audio buffering for streaming
        self.audio_buffer = deque(maxlen=48000)  # Buffer 3 seconds at 16kHz

        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Configuration for robotics
        self.energy_threshold = 0.01
        self.min_speech_duration = 0.3  # Minimum speech duration in seconds
        self.max_buffer_duration = 10.0  # Maximum to prevent overflow

    def async_transcribe(self, audio_data: np.ndarray) -> asyncio.Future:
        """
        Perform transcription asynchronously to prevent blocking
        """
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor,
            self._transcribe_sync,
            audio_data
        )
        return future

    def _transcribe_sync(self, audio_data: np.ndarray) -> Dict:
        """
        Synchronous transcription method for thread execution
        """
        try:
            result = self.model.transcribe(
                audio_data,
                language="en",
                task="transcribe",
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            return result
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": "", "segments": []}

    def add_audio_chunk(self, chunk: np.ndarray):
        """
        Add audio chunk to buffer
        """
        # Extend buffer with new chunk
        chunk_list = chunk.tolist()
        self.audio_buffer.extend(chunk_list)

        # Limit buffer size to prevent overflow
        max_len = int(self.max_buffer_duration * 16000)  # Convert to samples
        while len(self.audio_buffer) > max_len:
            self.audio_buffer.popleft()

    def get_speech_segment(self) -> Optional[np.ndarray]:
        """
        Extract speech segment from buffer if voice activity detected
        """
        if len(self.audio_buffer) == 0:
            return None

        # Convert buffer to numpy array
        buffer_array = np.array(list(self.audio_buffer))

        # Check for voice activity
        energy = np.mean(buffer_array ** 2)

        if energy > self.energy_threshold and len(buffer_array) > self.min_speech_duration * 16000:
            # Return the current buffer as a speech segment
            return buffer_array
        else:
            return None

    def clear_buffer(self):
        """
        Clear the audio buffer
        """
        self.audio_buffer.clear()
```

## Converting Voice Commands into Structured Intents

### Intent Classification for Robotics

Converting voice commands into structured intents requires understanding both the linguistic structure and the intended robotic action. The process involves:

1. **Command Parsing**: Identifying action verbs, objects, and parameters
2. **Semantic Mapping**: Converting natural language to formal action specifications
3. **Validation**: Ensuring the command is feasible and safe
4. **Parameter Extraction**: Extracting numerical values, locations, and other parameters

### Command Parser Implementation

```python
# Example: Command parser for voice-to-action conversion
import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    NAVIGATE = "navigate"
    GRASP = "grasp"
    DROP = "drop"
    FOLLOW = "follow"
    SPEAK = "speak"
    PERCEIVE = "perceive"
    WAIT = "wait"

@dataclass
class CommandIntent:
    action_type: ActionType
    target: Optional[str] = None
    location: Optional[str] = None
    duration: Optional[float] = None
    parameters: Dict[str, Union[str, float, bool]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

class CommandParser:
    def __init__(self):
        # Define command patterns and their corresponding actions
        self.patterns = [
            # Navigation commands
            (r"move to (.+)", ActionType.NAVIGATE),
            (r"go to (.+)", ActionType.NAVIGATE),
            (r"walk to (.+)", ActionType.NAVIGATE),
            (r"navigate to (.+)", ActionType.NAVIGATE),
            (r"go (.+)", ActionType.NAVIGATE),  # "go kitchen"

            # Grasping commands
            (r"pick up (.+)", ActionType.GRASP),
            (r"grab (.+)", ActionType.GRASP),
            (r"take (.+)", ActionType.GRASP),
            (r"get (.+)", ActionType.GRASP),
            (r"lift (.+)", ActionType.GRASP),

            # Dropping commands
            (r"put down (.+)", ActionType.DROP),
            (r"release (.+)", ActionType.DROP),
            (r"drop (.+)", ActionType.DROP),
            (r"place (.+)", ActionType.DROP),

            # Following commands
            (r"follow (.+)", ActionType.FOLLOW),
            (r"come with (.+)", ActionType.FOLLOW),
            (r"stay with (.+)", ActionType.FOLLOW),

            # Speaking commands
            (r"say (.+)", ActionType.SPEAK),
            (r"speak (.+)", ActionType.SPEAK),
            (r"tell (.+)", ActionType.SPEAK),

            # Perception commands
            (r"look at (.+)", ActionType.PERCEIVE),
            (r"see (.+)", ActionType.PERCEIVE),
            (r"detect (.+)", ActionType.PERCEIVE),
            (r"find (.+)", ActionType.PERCEIVE),

            # Waiting commands
            (r"wait (.+) seconds?", ActionType.WAIT),
            (r"pause for (.+) seconds?", ActionType.WAIT),
            (r"stop for (.+) seconds?", ActionType.WAIT),
        ]

        # Location mappings
        self.location_mappings = {
            "kitchen": [0.0, 5.0, 0.0],
            "living room": [0.0, 0.0, 0.0],
            "bedroom": [3.0, 2.0, 0.0],
            "office": [-2.0, 1.0, 0.0],
            "dining room": [1.0, -1.0, 0.0],
        }

        # Object mappings
        self.object_mappings = {
            "red cup": "red_cup_01",
            "blue bottle": "blue_bottle_01",
            "green box": "green_box_01",
            "apple": "fruit_apple_01",
            "banana": "fruit_banana_01",
        }

    def parse(self, text: str) -> Optional[CommandIntent]:
        """
        Parse text command into structured intent

        Args:
            text: Natural language command text

        Returns:
            Structured command intent or None if parsing fails
        """
        # Normalize text
        text = text.lower().strip()

        # Try to match patterns
        for pattern, action_type in self.patterns:
            match = re.search(pattern, text)
            if match:
                # Extract the target/object/location
                target_text = match.group(1).strip()

                # Create command intent
                intent = CommandIntent(action_type=action_type)

                # Process the target based on action type
                if action_type == ActionType.NAVIGATE:
                    intent.location = self._process_location(target_text)
                elif action_type in [ActionType.GRASP, ActionType.DROP]:
                    intent.target = self._process_object(target_text)
                elif action_type == ActionType.FOLLOW:
                    intent.target = target_text
                elif action_type == ActionType.SPEAK:
                    intent.parameters["text"] = target_text
                elif action_type == ActionType.PERCEIVE:
                    intent.target = self._process_object(target_text)
                elif action_type == ActionType.WAIT:
                    intent.duration = self._extract_duration(target_text)
                else:
                    intent.target = target_text

                return intent

        # If no pattern matches, try to identify generic commands
        return self._parse_generic_command(text)

    def _process_location(self, location_text: str) -> Optional[str]:
        """
        Process location text and return standardized location
        """
        # Check for direct location match
        if location_text in self.location_mappings:
            return location_text

        # Check for partial matches or aliases
        for location in self.location_mappings.keys():
            if location_text in location or location in location_text:
                return location

        # Return the original text if no match found
        return location_text

    def _process_object(self, object_text: str) -> Optional[str]:
        """
        Process object text and return standardized object identifier
        """
        # Check for direct object match
        if object_text in self.object_mappings:
            return self.object_mappings[object_text]

        # Check for partial matches
        for obj_key, obj_val in self.object_mappings.items():
            if object_text in obj_key or obj_key in object_text:
                return obj_val

        # Return the original text if no match found
        return object_text

    def _extract_duration(self, duration_text: str) -> Optional[float]:
        """
        Extract duration from text (e.g., "5 seconds" -> 5.0)
        """
        # Look for number patterns
        numbers = re.findall(r'\d+(?:\.\d+)?', duration_text)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass

        return None

    def _parse_generic_command(self, text: str) -> Optional[CommandIntent]:
        """
        Try to parse commands that don't match predefined patterns
        """
        # Simple heuristics for common patterns
        if "stop" in text or "halt" in text or "pause" in text:
            return CommandIntent(action_type=ActionType.WAIT, duration=0.0)

        if "hello" in text or "hi" in text:
            return CommandIntent(action_type=ActionType.SPEAK, parameters={"text": "Hello!"})

        # If no pattern matches, return None
        return None

# Example usage
parser = CommandParser()

# Test various commands
test_commands = [
    "Go to the kitchen",
    "Pick up the red cup",
    "Tell me your name",
    "Look at the blue bottle",
    "Wait 5 seconds",
    "Follow John"
]

for cmd in test_commands:
    intent = parser.parse(cmd)
    if intent:
        print(f"Command: '{cmd}' -> Action: {intent.action_type}, Target: {getattr(intent, 'target', 'N/A')}, Location: {getattr(intent, 'location', 'N/A')}")
    else:
        print(f"Command: '{cmd}' -> Could not parse")
```

### Intent Validation and Safety Checking

Before executing voice commands, it's crucial to validate them for safety and feasibility:

```python
# Example: Intent validation for robotics safety
class IntentValidator:
    def __init__(self):
        # Define safety constraints
        self.safe_locations = {"kitchen", "living room", "bedroom", "office", "dining room"}
        self.dangerous_objects = {"knife", "blade", "sharp", "hot", "fire", "poison"}
        self.forbidden_actions = {"shoot", "kill", "hurt", "damage", "break"}

        # Define robot capabilities
        self.max_reachable_distance = 2.0  # meters
        self.max_lift_weight = 1.0  # kg
        self.max_operation_time = 300.0  # seconds

    def validate_intent(self, intent: CommandIntent, robot_state: Dict) -> Tuple[bool, List[str]]:
        """
        Validate intent for safety and feasibility

        Args:
            intent: Command intent to validate
            robot_state: Current state of the robot

        Returns:
            Tuple of (is_valid, list_of_validation_errors)
        """
        errors = []

        # Check action safety
        if self._is_action_forbidden(intent):
            errors.append(f"Action '{intent.action_type}' is forbidden")

        # Check location safety
        if intent.location and not self._is_location_safe(intent.location):
            errors.append(f"Location '{intent.location}' may be unsafe")

        # Check object safety
        if intent.target and not self._is_object_safe(intent.target):
            errors.append(f"Object '{intent.target}' may be dangerous")

        # Check feasibility
        if not self._is_action_feasible(intent, robot_state):
            errors.append(f"Action '{intent.action_type}' is not feasible given current robot state")

        # Check duration validity
        if intent.duration and intent.duration > self.max_operation_time:
            errors.append(f"Requested duration {intent.duration}s exceeds maximum {self.max_operation_time}s")

        return len(errors) == 0, errors

    def _is_action_forbidden(self, intent: CommandIntent) -> bool:
        """Check if action is in forbidden list"""
        action_str = intent.action_type.value.lower()
        return action_str in self.forbidden_actions

    def _is_location_safe(self, location: str) -> bool:
        """Check if location is in safe list"""
        # For now, assume all predefined locations are safe
        # In practice, you'd check against a map of safe/unsafe areas
        return location in self.safe_locations

    def _is_object_safe(self, obj: str) -> bool:
        """Check if object is safe to interact with"""
        obj_lower = obj.lower()
        for danger in self.dangerous_objects:
            if danger in obj_lower:
                return False
        return True

    def _is_action_feasible(self, intent: CommandIntent, robot_state: Dict) -> bool:
        """Check if action is feasible given robot state"""
        # This is a simplified check - in practice, you'd check:
        # - robot's current position vs target location
        # - robot's payload capacity vs object weight
        # - robot's battery level
        # - etc.

        if intent.action_type == ActionType.NAVIGATE and intent.location:
            # Check if location is reachable
            # This would involve checking robot's current position vs target
            pass

        elif intent.action_type == ActionType.GRASP and intent.target:
            # Check if object is graspable
            # This would involve checking object properties
            pass

        return True  # Simplified - assume all actions are feasible
```

## Practical Voice-to-Action Pipeline Implementation

### Complete Pipeline Example

```python
# Example: Complete voice-to-action pipeline for humanoid robot
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import AudioData
from typing import Optional
import threading
import queue

class CompleteVoiceToActionPipeline:
    def __init__(self):
        # Initialize components
        self.voice_processor = WhisperVoiceProcessor(model_size="base", device="cuda")
        self.command_parser = CommandParser()
        self.intent_validator = IntentValidator()

        # ROS publishers and subscribers
        self.cmd_pub = rospy.Publisher('/robot/command', String, queue_size=10)
        self.audio_sub = rospy.Subscriber('/audio_input', AudioData, self.audio_callback)
        self.nav_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        # Internal state
        self.audio_queue = queue.Queue()
        self.robot_state = {
            'position': [0.0, 0.0, 0.0],
            'battery_level': 100.0,
            'payload': 0.0
        }

        # Processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_queue, daemon=True)
        self.processing_thread.start()

        rospy.loginfo("Complete Voice-to-Action Pipeline initialized")

    def audio_callback(self, msg: AudioData):
        """Handle incoming audio data from microphone"""
        # Convert audio message to numpy array
        audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Add to processing queue
        self.audio_queue.put(audio_data)

    def process_audio_queue(self):
        """Process audio data from queue in separate thread"""
        accumulated_audio = np.array([])

        while not rospy.is_shutdown():
            try:
                # Get audio from queue (non-blocking with timeout)
                audio_chunk = self.audio_queue.get(timeout=0.1)

                # Add to accumulator
                accumulated_audio = np.concatenate([accumulated_audio, audio_chunk])

                # Process if we have sufficient audio
                if len(accumulated_audio) > 16000 * 0.5:  # Half second worth of audio
                    self.process_accumulated_audio(accumulated_audio)
                    accumulated_audio = np.array([])  # Reset accumulator

            except queue.Empty:
                # Timeout occurred, continue loop
                continue
            except Exception as e:
                rospy.logerr(f"Error processing audio: {e}")
                accumulated_audio = np.array([])  # Reset on error

    def process_accumulated_audio(self, audio_data: np.ndarray):
        """Process accumulated audio data into actions"""
        try:
            # Transcribe audio
            transcription = self.voice_processor.transcribe_audio(audio_data)
            text = transcription.get("text", "").strip()

            if text:
                rospy.loginfo(f"Heard: {text}")

                # Parse command
                intent = self.command_parser.parse(text)

                if intent:
                    # Validate intent
                    is_valid, errors = self.intent_validator.validate_intent(intent, self.robot_state)

                    if is_valid:
                        # Execute action
                        self.execute_intent(intent)
                    else:
                        rospy.logwarn(f"Invalid intent: {errors}")
                        self.speak_response(f"I cannot do that because: {', '.join(errors)}")
                else:
                    rospy.logwarn(f"Could not parse command: {text}")
                    self.speak_response("I didn't understand that command.")

        except Exception as e:
            rospy.logerr(f"Error processing audio: {e}")

    def execute_intent(self, intent: CommandIntent):
        """Execute the parsed intent"""
        rospy.loginfo(f"Executing intent: {intent.action_type.value}")

        if intent.action_type == ActionType.NAVIGATE:
            self.execute_navigation(intent.location)
        elif intent.action_type == ActionType.GRASP:
            self.execute_grasp(intent.target)
        elif intent.action_type == ActionType.DROP:
            self.execute_drop(intent.target)
        elif intent.action_type == ActionType.FOLLOW:
            self.execute_follow(intent.target)
        elif intent.action_type == ActionType.SPEAK:
            self.execute_speak(intent.parameters.get("text", ""))
        elif intent.action_type == ActionType.PERCEIVE:
            self.execute_perceive(intent.target)
        elif intent.action_type == ActionType.WAIT:
            self.execute_wait(intent.duration or 1.0)

    def execute_navigation(self, location: str):
        """Execute navigation to location"""
        # In a real implementation, this would:
        # 1. Look up the coordinates for the location
        # 2. Create a PoseStamped message
        # 3. Publish to navigation system

        # For demo, assume we have coordinates
        if location in self.command_parser.location_mappings:
            coords = self.command_parser.location_mappings[location]
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x = coords[0]
            goal.pose.position.y = coords[1]
            goal.pose.position.z = coords[2]
            goal.pose.orientation.w = 1.0  # No rotation

            self.nav_pub.publish(goal)
            self.speak_response(f"Navigating to {location}")
        else:
            self.speak_response(f"I don't know where {location} is.")

    def execute_grasp(self, target: str):
        """Execute grasping action"""
        # In a real implementation, this would:
        # 1. Locate the object using perception system
        # 2. Plan grasping motion
        # 3. Execute grasping with manipulator
        self.speak_response(f"Attempting to grasp {target}")

    def execute_drop(self, target: str):
        """Execute dropping action"""
        # In a real implementation, this would:
        # 1. Find appropriate drop location
        # 2. Execute dropping motion
        self.speak_response(f"Dropping {target}")

    def execute_follow(self, target: str):
        """Execute following action"""
        # In a real implementation, this would:
        # 1. Track the target person
        # 2. Maintain safe distance while following
        self.speak_response(f"Following {target}")

    def execute_speak(self, text: str):
        """Execute speaking action"""
        # In a real implementation, this would:
        # 1. Use text-to-speech system
        # 2. Play audio through speakers
        rospy.loginfo(f"Speaking: {text}")

    def execute_perceive(self, target: str):
        """Execute perception action"""
        # In a real implementation, this would:
        # 1. Use vision system to locate object
        # 2. Return location information
        self.speak_response(f"Looking for {target}")

    def execute_wait(self, duration: float):
        """Execute waiting action"""
        # In a real implementation, this would:
        # 1. Pause current actions for specified duration
        # 2. Resume listening after wait period
        self.speak_response(f"Waiting for {duration} seconds")
        rospy.sleep(duration)

    def speak_response(self, text: str):
        """Speak a response to the user"""
        # Publish response text
        self.cmd_pub.publish(text)
        rospy.loginfo(f"Response: {text}")

def main():
    """Main function to run the voice-to-action pipeline"""
    rospy.init_node('voice_to_action_pipeline')

    pipeline = CompleteVoiceToActionPipeline()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down voice-to-action pipeline")

if __name__ == '__main__':
    main()
```

## Troubleshooting Voice-to-Action Systems

### Common Issues and Solutions

#### 1. Poor Speech Recognition
**Problem**: Whisper doesn't recognize voice commands accurately.
**Solutions**:
- Check microphone quality and positioning
- Verify audio input format matches Whisper expectations (16kHz, mono, float32)
- Adjust energy threshold for voice activity detection
- Use noise reduction preprocessing
- Consider acoustic environment factors

#### 2. Command Parsing Failures
**Problem**: Voice commands are recognized but not parsed into valid intents.
**Solutions**:
- Expand command pattern matching
- Improve natural language understanding
- Add more robust error handling
- Implement fallback mechanisms

#### 3. Performance Issues
**Problem**: System is slow to respond to voice commands.
**Solutions**:
- Use smaller Whisper models for faster inference
- Implement asynchronous processing
- Optimize audio buffering
- Use GPU acceleration where available

### Performance Optimization Techniques

```python
# Example: Performance monitoring for voice-to-action systems
import time
import psutil
from collections import deque

class VoiceSystemMonitor:
    def __init__(self, window_size: int = 100):
        self.response_times = deque(maxlen=window_size)
        self.cpu_usage_history = deque(maxlen=window_size)
        self.memory_usage_history = deque(maxlen=window_size)

        self.window_size = window_size
        self.start_time = time.time()

    def record_response_time(self, response_time: float):
        """Record response time for a command"""
        self.response_times.append(response_time)

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        if not self.response_times:
            return {}

        avg_response_time = sum(self.response_times) / len(self.response_times)
        max_response_time = max(self.response_times)
        min_response_time = min(self.response_times)

        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        self.cpu_usage_history.append(cpu_percent)
        self.memory_usage_history.append(memory_percent)

        return {
            'avg_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'min_response_time': min_response_time,
            'response_count': len(self.response_times),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'uptime': time.time() - self.start_time
        }

    def is_system_overloaded(self) -> bool:
        """Check if system is overloaded based on metrics"""
        metrics = self.get_performance_metrics()

        if not metrics:
            return False

        # Check if response times are too high (>2 seconds)
        if metrics['avg_response_time'] > 2.0:
            return True

        # Check if CPU usage is too high (>80%)
        if metrics['cpu_percent'] > 80:
            return True

        return False
```

## Summary and Next Steps

Voice-to-action pipelines form the foundation of natural human-robot interaction in humanoid systems. By combining advanced speech recognition with OpenAI Whisper and robust intent classification, we can create intuitive interfaces that allow humans to control robots using natural language.

In the next chapter, we'll explore how to integrate large language models for cognitive planning, transforming natural language goals into structured action sequences that can be executed by ROS 2 systems.

[Continue to Chapter 2: Cognitive Planning with LLMs](./cognitive-planning-llms.md)

## Learning Objectives

By the end of this chapter, you should be able to:
- Implement speech recognition systems using OpenAI Whisper for robotics
- Convert voice commands into structured intents using natural language processing
- Design safe and validated voice-to-action pipelines for humanoid robots
- Optimize voice processing for real-time robotics applications
- Troubleshoot common issues in voice-to-action systems