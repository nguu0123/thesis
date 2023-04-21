def not_approximate(a, b):
    if abs(a - b) < 10:
        return False
    else:
        return True


def get_prediction(list_pre, i):
    pre_item = list_pre[i]
    keys = list(pre_item.keys())
    return pre_item[keys[0]]


def extract_dict(dict, keys):
    result = {}
    for key in keys:
        result[key] = dict[key]
    return result


def compare_box(box1, box2):
    for key in box1:
        if not_approximate(box1[key], box2[key]):
            return False
    return True


def agg_mean(predictions):
    pre_list = []
    agg_prediction = []
    object_count = 0
    for key in predictions:
        pre_list += predictions[key]
    while pre_list:
        pre_item = get_prediction(pre_list, 0)
        class1 = extract_dict(pre_item[0], ["name"])
        box1 = extract_dict(pre_item[0], ["xmin", "ymin", "xmax", "ymax"])
        duplicate = []
        for i in range(1, len(pre_list)):
            data2 = get_prediction(pre_list, i)[0]
            box2 = extract_dict(data2, ["xmin", "ymin", "xmax", "ymax"])
            class2 = extract_dict(data2, ["name"])
            if compare_box(box1, box2):
                if class1 == class2:
                    pre_item += get_prediction(pre_list, i)
                    duplicate.append(i)
                else:
                    raise Exception("bounding box intercept")
        for item in duplicate:
            pre_list.pop(item)
        mean_item = pre_item[0]
        for item in pre_item[1:]:
            mean_item["confidence"] += item["confidence"]
        mean_item["confidence"] /= len(pre_item)
        detect_obj = {f"object_{object_count}": mean_item}
        pre_list.pop(0)
        object_count += 1
        agg_prediction.append(detect_obj)
    return agg_prediction


def agg_max(predictions):
    pre_list = []
    agg_prediction = []
    object_count = 0
    for key in predictions:
        pre_list += predictions[key]
    while pre_list:
        pre_item = get_prediction(pre_list, 0)
        box1 = extract_dict(pre_item[0], ["xmin", "ymin", "xmax", "ymax"])
        duplicate = []
        for i in range(1, len(pre_list)):
            box2 = extract_dict(
                get_prediction(pre_list, i)[0], ["xmin", "ymin", "xmax", "ymax"]
            )
            if compare_box(box1, box2):
                pre_item += get_prediction(pre_list, i)
                duplicate.append(i)
        for item in duplicate:
            pre_list.pop(item)
        max_item = pre_item[0]
        for item in pre_item:
            if item["confidence"] > max_item["confidence"]:
                max_item = item
        detect_obj = {f"object_{object_count}": max_item}
        pre_list.pop(0)
        object_count += 1
        agg_prediction.append(detect_obj)
    return agg_prediction


data = {
    "yolov5n": [
        {
            "object_0": [
                {
                    "xmin": 283.8602600097656,
                    "ymin": 176.068603515625,
                    "xmax": 629.1629638671875,
                    "ymax": 384.3131103515625,
                    "confidence": 0.7132321000099182,
                    "class": 1,
                    "name": "bicycle",
                }
            ]
        },
        {
            "object_1": [
                {
                    "xmin": 371.38299560546875,
                    "ymin": 44.06036376953125,
                    "xmax": 534.8262939453125,
                    "ymax": 376.6901550292969,
                    "confidence": 0.6608238816261292,
                    "class": 0,
                    "name": "person",
                }
            ]
        },
    ],
    "yolov8n": [
        {
            "object_0": [
                {
                    "xmin": 283.0,
                    "ymin": 167.0,
                    "xmax": 634.0,
                    "ymax": 394.0,
                    "confidence": 0.9223847985267639,
                    "class": 1.0,
                    "name": "bicycle",
                }
            ]
        },
        {
            "object_1": [
                {
                    "xmin": 365.0,
                    "ymin": 40.0,
                    "xmax": 516.0,
                    "ymax": 365.0,
                    "confidence": 0.8321550488471985,
                    "class": 0.0,
                    "name": "person",
                }
            ]
        },
        {
            "object_2": [
                {
                    "xmin": 503.0,
                    "ymin": 148.0,
                    "xmax": 640.0,
                    "ymax": 237.0,
                    "confidence": 0.3270763158798218,
                    "class": 2.0,
                    "name": "car",
                }
            ]
        },
    ],
}
agg_mean(data)
