from datetime import timedelta

def parse_timedelta(delta_str):
    try:
        # Initialize days and microseconds to zero
        days = 0
        microseconds = 0

        # Check if the string includes days
        if ', ' in delta_str:
            days_str, time_str = delta_str.split(', ')
            days = int(days_str.split()[0])  # Extract the number of days
        else:
            time_str = delta_str

        # Check if the string includes microseconds
        if '.' in time_str:
            time_str, microseconds_str = time_str.split('.')
            microseconds = int(microseconds_str)

        # Extract hours, minutes, seconds from the time part
        hours, minutes, seconds = [int(part) for part in time_str.split(':')]

        return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
    except Exception as e:
        print('timedelta parsing failed due to following error: {e}')
        return timedelta(days=0, hours=0, minutes=0, seconds=0, microseconds=0)