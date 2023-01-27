import datetime

import pytz


def get_str_currentdate():
    now = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
    return now.strftime("%Y%m%d-%H%M%S")
