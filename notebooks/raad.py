def calculate_area(box):
    """Calculate the area of a bounding box."""
    x1, y1, x2, y2 = box
    return max(0, (x2 - x1)) * max(0, (y2 - y1))

def intersection_area(box1, box2):
    """Calculate intersection area between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return calculate_area((x1, y1, x2, y2))

def total_area(boxes):
    """Calculate total area considering overlaps."""
    total = 0
    for i, box1 in enumerate(boxes):
        area = calculate_area(box1)
        for j, box2 in enumerate(boxes):
            if i != j:
                overlap = intersection_area(box1, box2)
                area -= overlap
        total += max(0, area)
    return total

def raad_metric(pred_boxes, true_boxes):
    """
    Calculate the Relative Affected Area Difference (RAAD) between predicted and true bounding boxes.
    
    Parameters:
        pred_boxes (list of tuples): Predicted bounding boxes [(x1, y1, x2, y2), ...]
        true_boxes (list of tuples): Ground truth bounding boxes [(x1, y1, x2, y2), ...]
    
    Returns:
        float: RAAD metric value
    """
    true_area = total_area(true_boxes)
    pred_area = total_area(pred_boxes)
    return abs(pred_area - true_area) / max(true_area, 1e-6)

