import datetime


def current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")
