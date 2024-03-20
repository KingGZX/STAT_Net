import pandas as pd
import openpyxl


class Config:
    train_rate = 0.7

    # take "Center of Mass" as additional joint
    # so when this is True, please append this joint to the nodes
    use_CoM = False

    # including spines
    nodes = ["Pelvis", "L3 Spine", "T12 Spine", "T8 Spine",
             "Neck", "Head", "Right Shoulder", "Right Upper Arm",
             "Right Forearm", "Right Hand", "Left Shoulder", "Left Upper Arm",
             "Left Forearm", "Left Hand", "Right Upper Leg", "Right Lower Leg",
             "Right Foot", "Right Toe", "Left Upper Leg", "Left Lower Leg",
             "Left Foot", "Left Toe"]

    joints = len(nodes)

    # for swapping [affected side, unaffected side] layout
    sides = [[6, 7, 8, 9, 10, 11, 12, 13], [10, 11, 12, 13, 6, 7, 8, 9],
             [14, 15, 16, 17, 18, 19, 20, 21], [18, 19, 20, 21, 14, 15, 16, 17]]

    label_fp = "dataset/label/Wisconsin Gait Scale.xlsx"
    labels = pd.read_excel(label_fp, engine="openpyxl")

    affected_fp = "dataset/label/Stroke subject.xlsx"
    affected_side = pd.read_excel(affected_fp, engine="openpyxl").iloc[:, 11].to_list()
    # the simulation files are not labeled, randomly use L as the affected leg
    for i in range(48, 52):
        affected_side[i] = 'L'

    label_dict = {"item1": "Use of hand-held gait aid",
                  "item2": "Stance time on impaired side",
                  "item3": "Step length of unaffected side",
                  "item4": "Weight shift to the affected side with or without gait aid",
                  "item5": "Stance width",
                  "item6": "Guardedness",
                  "item7": "Hip extension of the affected leg",
                  "item8": "External rotation during initial swing",
                  "item9": "Circumduction at mid-swing",
                  "item10": "Hip hiking at mid-swing",
                  "item11": "Knee flexion from toe off to mid-swing",
                  "item12": "Toe clearance",
                  "item13": "Pelvic rotation at terminal swing",
                  "item14": "Initial foot contact"
                  }

    classes = [5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3]

    segment_sheets = ["Segment Position", "Segment Velocity","Segment Acceleration", 
                      # angles, can be extracted for multi-modal (ensemble)
                      "Segment Angular Velocity", "Segment Angular Acceleration",
                      "Segment Orientation - Euler", "Joint Angles ZXY"]

    # corresponding to the segment sheets
    # by default, we only use position information as features,
    # so the input features should be in shape [3, frames, joints = (nodes num)]
    # with one additional sheet used, there'll be 3 more channels.
    segment_sheet_idx = [0, 1, 2, 6]

    spine_segment = ["L5"]

    # since the spine segments are estimated by IMU, I firstly decide not to use it
    ignore_spine = True

    # data augmentation
    augment = True

    # extract gait cycles method
    # option:   4   or   1
    # use 1 to generate more gait cycles
    time_split = 1

    # batch training
    batch_size = 16

    # whether perform lttb padding
    # when this is true, batch_size can be set greater than 1
    padding = True

    # test which item
    # item = [12, 9, 7, 8, 11, 14, 5, 6, 10, 3]
    item = [3, 8, 10, 12, 14]

    num_classes = list()
    for j in item:
        num_classes.append(classes[j - 1])

    # input channel
    in_channels = 12

    # avg_frames (padding goal)
    avg_frames = 120
