'''
curl -X POST \
  'http://127.0.0.1:5000/scrub' \
  --header 'Accept: */*' \
  --header 'User-Agent: Thunder Client (https://www.thunderclient.com)' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "text" : "Playing, played, plays, player, play, Wanda",
  "options" : {
      "lower_case": true,  
      "remove_punctuation": true, 
      "remove_stop_words": true, 
      "remove_numbers": true, 
      "normalize_spaces": true,  
      "remove_emails": true, 
      "remove_urls": true,
      "lemmatize": true,  
      "remove_entities_tokens": true, 
      "unique_tokens": true,
      "remove_custom": ""
  }
}'
'''