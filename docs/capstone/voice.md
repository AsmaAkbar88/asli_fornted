---
sidebar_label: 'Voice Processing Integration'
---

# Voice Processing Integration

This section covers the integration of voice processing capabilities into the Physical AI system, enabling natural human-robot interaction through spoken commands.

## Voice Processing Pipeline

The voice processing component of our integrated Physical AI system follows this pipeline:

``` 
Voice Input → Audio Preprocessing → Speech Recognition → Natural Language Understanding → Command Validation → Task Planning Interface
```

Each stage is critical for converting natural language commands into executable robotic actions.

## Audio Preprocessing

### Noise Reduction and Enhancement:
```python
import numpy as np
import librosa
from scipy import signal

class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = 16000
        self.frame_size = 1024
        self.hop_length = 512

    def preprocess_audio(self, audio_data):
        # Apply noise reduction
        enhanced_audio = self.reduce_noise(audio_data)

        # Apply voice activity detection
        voice_segments = self.detect_voice_activity(enhanced_audio)

        # Normalize audio levels
        normalized_audio = self.normalize_audio(voice_segments)

        return normalized_audio

    def reduce_noise(self, audio):
        # Spectral subtraction for noise reduction
        D = librosa.stft(audio)
        magnitude, phase = librosa.magphase(D)

        # Estimate noise profile and subtract
        noise_profile = np.mean(magnitude[:, :100], axis=1, keepdims=True)
        enhanced_magnitude = np.maximum(magnitude - noise_profile * 0.3, 0)

        enhanced_D = enhanced_magnitude * phase
        enhanced_audio = librosa.istft(enhanced_D)

        return enhanced_audio

    def detect_voice_activity(self, audio):
        # Simple energy-based voice activity detection
        frame_length = 256
        hop_length = 128

        # Calculate frame energies
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.sum(frames**2, axis=0)

        # Apply threshold-based detection
        threshold = np.mean(frame_energies) * 0.5
        voice_mask = frame_energies > threshold

        # Reconstruct audio with only voice segments
        voice_segments = []
        for i, is_voice in enumerate(voice_mask):
            if is_voice:
                start = i * hop_length
                end = min(start + frame_length, len(audio))
                voice_segments.append(audio[start:end])

        if voice_segments:
            return np.concatenate(voice_segments)
        else:
            return audio

    def normalize_audio(self, audio):
        # Normalize to standard level
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            normalized = audio / max_amplitude
            return normalized * 0.8  # Leave some headroom
        return audio
```

## Speech Recognition

### Integration with ROS 2:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import speech_recognition as sr

class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('speech_recognition_node')

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Adjust for environment
        self.recognizer.dynamic_energy_threshold = True

        # Audio input subscription
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Text output publisher
        self.text_pub = self.create_publisher(
            String,
            'recognized_text',
            10
        )

        # Timer for continuous listening
        self.listen_timer = self.create_timer(0.1, self.continuous_listen)

    def audio_callback(self, msg):
        # Process audio data from microphone
        audio_data = np.frombuffer(msg.data, dtype=np.int16)

        # Convert to audio segment for recognition
        audio_segment = sr.AudioData(
            audio_data.tobytes(),
            sample_rate=16000,
            sample_width=2
        )

        try:
            # Use Google Speech Recognition (or local alternative)
            text = self.recognizer.recognize_google(audio_segment)
            self.publish_recognized_text(text)
        except sr.UnknownValueError:
            self.get_logger().info('Speech Recognition could not understand audio')
        except sr.RequestError as e:
            self.get_logger().error(f'Could not request results from Speech Recognition service; {e}')

    def continuous_listen(self):
        # Implementation for continuous listening
        pass

    def publish_recognized_text(self, text):
        msg = String()
        msg.data = text
        self.text_pub.publish(msg)
```

### Alternative: Local Speech Recognition:
```python
# Using Vosk for local speech recognition
from vosk import Model, KaldiRecognizer
import json

class LocalSpeechRecognizer:
    def __init__(self, model_path="model"):
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, 16000)
        self.rec.SetWords(True)

    def recognize_audio(self, audio_data):
        if self.rec.AcceptWaveform(audio_data):
            result = self.rec.Result()
            result_dict = json.loads(result)
            return result_dict.get('text', '')
        else:
            # Partial result
            partial_result = self.rec.PartialResult()
            partial_dict = json.loads(partial_result)
            return partial_dict.get('partial', '')

    def is_final_result_ready(self):
        return self.rec.FinalResult() is not None
```

## Natural Language Understanding

### Intent Classification:
```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class IntentClassifier:
    def __init__(self, model_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = IntentClassificationModel()

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

    def classify_intent(self, text):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            predictions = torch.softmax(outputs, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1)

        intent_labels = ['navigation', 'manipulation', 'information', 'system_control']
        return intent_labels[predicted_class.item()], predictions[0][predicted_class].item()

class IntentClassificationModel(nn.Module):
    def __init__(self, num_intents=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_intents)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)
```

### Named Entity Recognition for Robotics:
```python
class RoboticsNER:
    def __init__(self):
        # Define entity types relevant to robotics
        self.entity_types = {
            'object': ['cup', 'book', 'bottle', 'box', 'chair', 'table'],
            'location': ['kitchen', 'bedroom', 'living room', 'office', 'counter', 'shelf'],
            'action': ['pick', 'place', 'move', 'bring', 'take', 'put'],
            'attribute': ['red', 'blue', 'large', 'small', 'heavy', 'light']
        }

    def extract_entities(self, text):
        entities = []
        words = text.lower().split()

        for word in words:
            for entity_type, entity_list in self.entity_types.items():
                if word in entity_list:
                    entities.append({
                        'text': word,
                        'type': entity_type,
                        'start': text.lower().find(word),
                        'end': text.lower().find(word) + len(word)
                    })

        return entities

    def parse_command(self, text):
        intent_classifier = IntentClassifier()
        ner = RoboticsNER()

        intent, confidence = intent_classifier.classify_intent(text)
        entities = ner.extract_entities(text)

        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'raw_text': text
        }
```

## Voice Command Validation

### Grammar and Constraint Checking:
```python
class CommandValidator:
    def __init__(self):
        # Define valid command patterns
        self.valid_patterns = [
            r'go to (the )?(?P<location>\w+)',  # Navigation commands
            r'pick up (the )?(?P<attribute>\w+ )?(?P<object>\w+)',  # Manipulation commands
            r'bring (the )?(?P<object>\w+) to (the )?(?P<location>\w+)',  # Complex commands
            r'what is (on )?(the )?(?P<location>\w+)',  # Information queries
        ]

    def validate_command(self, parsed_command):
        # Check if command matches valid patterns
        command_text = parsed_command['raw_text']

        for pattern in self.valid_patterns:
            import re
            match = re.search(pattern, command_text.lower())
            if match:
                # Extract matched groups
                groups = match.groupdict()

                # Validate entities exist in the environment
                if self.validate_environment_entities(groups):
                    return True, "Valid command"

        return False, "Command does not match valid patterns"

    def validate_environment_entities(self, entities):
        # Check if mentioned entities exist in current environment
        # This would interface with perception system
        return True  # Simplified for example
```

## Integration with Task Planning

### Voice Command to Task Conversion:
```python
class VoiceToTaskConverter:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.ner = RoboticsNER()
        self.validator = CommandValidator()

    def convert_voice_command(self, text):
        # Parse the voice command
        parsed_command = self.ner.parse_command(text)

        # Validate the command
        is_valid, validation_msg = self.validator.validate_command(parsed_command)
        if not is_valid:
            return None, validation_msg

        # Convert to task representation
        task = self.create_task_from_command(parsed_command)

        return task, "Task created successfully"

    def create_task_from_command(self, parsed_command):
        intent = parsed_command['intent']
        entities = parsed_command['entities']

        if intent == 'navigation':
            return self.create_navigation_task(entities)
        elif intent == 'manipulation':
            return self.create_manipulation_task(entities)
        elif intent == 'information':
            return self.create_information_task(entities)
        else:
            return self.create_default_task(parsed_command)

    def create_navigation_task(self, entities):
        location_entity = next((e for e in entities if e['type'] == 'location'), None)
        if location_entity:
            return {
                'type': 'navigation',
                'target_location': location_entity['text'],
                'priority': 1
            }
        return None

    def create_manipulation_task(self, entities):
        object_entity = next((e for e in entities if e['type'] == 'object'), None)
        location_entity = next((e for e in entities if e['type'] == 'location'), None)

        task = {'type': 'manipulation', 'priority': 2}

        if object_entity:
            task['target_object'] = object_entity['text']
        if location_entity:
            task['target_location'] = location_entity['text']

        return task

    def create_information_task(self, entities):
        location_entity = next((e for e in entities if e['type'] == 'location'), None)
        return {
            'type': 'information',
            'query_location': location_entity['text'] if location_entity else 'current_location',
            'priority': 3
        }

    def create_default_task(self, parsed_command):
        return {
            'type': 'unknown',
            'raw_command': parsed_command['raw_text'],
            'priority': 4
        }
```

## Error Handling and Feedback

### Voice Command Error Recovery:
```python
class VoiceCommandRecovery:
    def __init__(self):
        self.command_history = []
        self.error_count = 0
        self.max_errors = 3

    def handle_recognition_error(self, error_type, context):
        if error_type == "recognition_failure":
            return self.request_clarification(context)
        elif error_type == "validation_failure":
            return self.suggest_alternatives(context)
        elif error_type == "execution_failure":
            return self.offer_help(context)
        else:
            return self.general_error_response()

    def request_clarification(self, context):
        # Ask user to repeat or clarify command
        clarification_request = {
            'type': 'request_clarification',
            'message': 'I didn\'t understand that. Could you please repeat or rephrase your command?',
            'context': context
        }
        return clarification_request

    def suggest_alternatives(self, context):
        # Provide alternative interpretations or commands
        alternatives = [
            'Did you mean: "Go to the kitchen"?',
            'Or perhaps: "Pick up the red cup"?',
            'Here are some things I can help with: navigation, object manipulation, information retrieval'
        ]

        suggestion = {
            'type': 'suggestion',
            'alternatives': alternatives,
            'context': context
        }
        return suggestion

    def offer_help(self, context):
        # Provide help information
        help_info = {
            'type': 'help',
            'message': 'I can help with navigation, object manipulation, and information. Try commands like "Go to the kitchen" or "Pick up the book".',
            'context': context
        }
        return help_info

    def general_error_response(self):
        return {
            'type': 'error',
            'message': 'I encountered an error processing your request. Please try again.',
            'context': None
        }
```

## Performance Optimization

### Real-time Voice Processing:
```python
import asyncio
import threading
from queue import Queue

class RealTimeVoiceProcessor:
    def __init__(self):
        self.audio_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=5)
        self.is_running = False

        # Initialize components
        self.preprocessor = AudioPreprocessor()
        self.speech_recognizer = LocalSpeechRecognizer()
        self.intent_classifier = IntentClassifier()
        self.task_converter = VoiceToTaskConverter()

    def start_processing(self):
        self.is_running = True

        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio_stream)
        processing_thread.start()

    def process_audio_stream(self):
        while self.is_running:
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()

                # Process the audio chunk
                processed_audio = self.preprocessor.preprocess_audio(audio_chunk)
                recognized_text = self.speech_recognizer.recognize_audio(processed_audio)

                if recognized_text:
                    # Convert to task
                    task, status = self.task_converter.convert_voice_command(recognized_text)

                    if task:
                        self.result_queue.put({
                            'task': task,
                            'text': recognized_text,
                            'timestamp': time.time()
                        })

    def add_audio_chunk(self, audio_data):
        if not self.audio_queue.full():
            self.audio_queue.put(audio_data)
```

## Testing and Validation

### Voice Processing Tests:
```python
def test_voice_processing_pipeline():
    # Test audio preprocessing
    preprocessor = AudioPreprocessor()
    test_audio = np.random.random(16000)  # 1 second of random audio
    processed = preprocessor.preprocess_audio(test_audio)
    assert len(processed) > 0

    # Test speech recognition
    recognizer = LocalSpeechRecognizer()  # Assuming local model
    # This would require actual audio data

    # Test NLP components
    ner = RoboticsNER()
    entities = ner.extract_entities("Pick up the red cup")
    assert any(e['text'] == 'red' and e['type'] == 'attribute' for e in entities)
    assert any(e['text'] == 'cup' and e['type'] == 'object' for e in entities)

    print("All voice processing tests passed!")

# Run tests
# test_voice_processing_pipeline()
```

The voice processing component serves as the primary interface between human users and the Physical AI system, converting natural language commands into structured tasks that can be processed by the planning and execution layers. Proper integration of this component is crucial for creating intuitive and accessible robotic systems.