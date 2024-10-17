keyword=$1
ps aux | grep python | grep "$keyword" | awk '{print $2}' | xargs kill