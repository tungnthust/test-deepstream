from kafka import KafkaProducer
import json
import time

# Create producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:8912'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Test message
test_message = {
    "timestamp": time.time(),
    "event_type": "test",
    "message": "Test message for intrusion detection system",
    "source": "kafka_test_script"
}

# Send message
future = producer.send('intrusion_detection', test_message)
result = future.get(timeout=60)

print(f"Message sent successfully: {test_message}")
print(f"Result: {result}")

producer.close()