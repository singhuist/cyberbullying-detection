from twython import Twython

APP_KEY = 'qBrELN3sXKQEY9bUz9WYNkqZs'
APP_SECRET = 'vA212zWI7RCde3BiUE6JzokwpgqUcFEOLQEQ8QxQJrwM68Qn7n'

twitter = Twython(APP_KEY,APP_SECRET)
auth = twitter.get_authentication_tokens(callback_url=None)

print(twitter.search(q='#NOTA'))


