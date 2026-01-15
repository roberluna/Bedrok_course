from rag import handler
import json

event = {
    'body': json.dumps({'question': 'in spanhish: what is GDPR in a very short words'})
}

response = handler(event, {})
print(response)


