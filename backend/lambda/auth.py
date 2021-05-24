def handler(event, context):
  event['response']['autoConfirmUser'] = True
  return event