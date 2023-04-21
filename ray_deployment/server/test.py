import sys
import traceback

try:
    test = [1,2,3,4]
    print(test[100])
except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback_str = traceback.format_exc()
    print("Exception Type:", exc_type.__name__)
    print("Description:", str(exc_value))
    print("Traceback:")
    print(type(traceback_str))
    print(traceback_str)
