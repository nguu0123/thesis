import copy
def not_approximate(a, b):
    if abs(a - b) < 10:
        return False
    else:
        return True


def get_prediction(list_pre, i):
    pre_item = copy.deepcopy(list_pre[i])
    keys = list(pre_item.keys())
    return pre_item[keys[0]]


def extract_dict(dict, keys):
    result = {}
    for key in keys:
        result[key] = dict[key]
    return result


def compare_box(box1, box2, threshold):
    x1 = box1['xmin']
    y1 = box1['ymin']
    x2 = box1['xmax']
    y2 = box1['ymax']

    x3 = box2['xmin']
    y3 = box2['ymin']
    x4 = box2['xmax']
    y4 = box2['ymax']

    area_box1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area_box2 = (x4 - x3 + 1) * (y4 - y3 + 1)

    x_intersection = max(x1, x3)
    y_intersection = max(y1, y3)
    x_intersection_end = min(x2, x4)
    y_intersection_end = min(y2, y4)

    area_intersection = max(0, x_intersection_end - x_intersection + 1) * max(0, y_intersection_end - y_intersection + 1)

    area_union = area_box1 + area_box2 - area_intersection

    iou = area_intersection / float(area_union)

    if iou >= threshold:
        return True 
    else:
        return False

def agg_mean(predictions):
    pre_list = []
    agg_prediction = []
    object_count = 0
    for model in predictions:
        for i in range(len(predictions[model])):
            for key,value in predictions[model][i].items():
                pre_list += value 
    while pre_list:
        pre_item = [pre_list[0]]
        class1 = extract_dict(pre_item[0], ["name"])
        box1 = extract_dict(pre_item[0], ["xmin", "ymin", "xmax", "ymax"])
        duplicate = []
        for i in range(1, len(pre_list)):
            data2 = pre_list[i]
            box2 = extract_dict(data2, ["xmin", "ymin", "xmax", "ymax"])
            class2 = extract_dict(data2, ["name"])
            if compare_box(box1, box2, 0.8):
                if class1 == class2:
                    pre_item += [pre_list[i]]
                    duplicate.append(i)
        pre_list = [pre_list[i] for i in range(len(pre_list)) if i not in duplicate]
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
    #print(predictions, '\n')
    pre_list = []
    for model in predictions:
        for i in range(len(predictions[model])):
            for key,value in predictions[model][i].items():
                pre_list += value 
    #print(pre_list, '\n')
    while pre_list:
        pre_item = [pre_list[0]]
        box1 = extract_dict(pre_item[0], ["xmin", "ymin", "xmax", "ymax"])
        class1 = extract_dict(pre_item[0], ["name"])
        duplicate = []
        for i in range(1, len(pre_list)):
            box2 = extract_dict(
                pre_list[i], ["xmin", "ymin", "xmax", "ymax"]
            )
            class2 = extract_dict(pre_list[i], ["name"])
            if compare_box(box1, box2, 0.8) and class1 == class2:
                pre_item += [pre_list[i]]
                duplicate.append(i)
        #print(pre_list, '\n')
        pre_list = [pre_list[i] for i in range(len(pre_list)) if i not in duplicate]
        #print(pre_list, '\n')
        max_item = pre_item[0]
        for item in pre_item:
            if item["confidence"] > max_item["confidence"]:
                max_item = item
        detect_obj = {f"object_{object_count}": max_item}
        pre_list.pop(0)
        object_count += 1
        agg_prediction.append(detect_obj)
    return agg_prediction

#data = {'yolov5n': [{'object_0': [{'xmin': 63.36903381347656, 'ymin': 97.7342529296875, 'xmax': 479.23687744140625, 'ymax': 640.0, 'confidence': 0.5499464869499207, 'class': 21, 'name': 'bear'}]}], 'yolov8n': [{'object_0': [{'xmin': 0.0, 'ymin': 102.0, 'xmax': 480.0, 'ymax': 640.0, 'confidence': 0.5317056775093079, 'class': 21.0, 'name': 'bear'}]}, {'object_1': [{'xmin': 73.0, 'ymin': 89.0, 'xmax': 480.0, 'ymax': 639.0, 'confidence': 0.38965287804603577, 'class': 16.0, 'name': 'dog'}]}]}
#print(agg_max(data))
